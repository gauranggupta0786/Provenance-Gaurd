"""The v2 protected runtime: v1's complete-mediation runtime, generalised from Workspace to all four AgentDojo suites.

A fork of ``pg.agentdojo_adapter.ProtectedWorkspaceRuntime`` (which stays untouched and is still what the v1 configurations run).
The mediation path is the same call for call -- signed proposal, policy, approval card, reviewer, gateway revalidation, atomic
consume, receipts, lifecycle graph -- and is parameterised by a ``SuiteSpec`` instead of Workspace constants.  What v2 adds:

* every card carries ``card.pgx_context`` (the trusted request, the suite spec, the canonical call, the read sources, the
  request capabilities, the live environment) for ``pgx.guards``;
* reads are logged for content provenance in every suite (``ReadLogV2``);
* a denial can tell the agent which CATEGORY of rule declined it (E4, ``feedback=True``), never the matched words or sources.

``tests/test_pgx_runtime.py`` replays the same tool calls through this runtime and through v1's on Workspace and requires the
same mediated decisions and the same final environment.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from agentdojo.functions_runtime import Function, FunctionCallArgTypes, FunctionReturnType, FunctionsRuntime, TaskEnvironment

from pg.agentdojo_adapter import RAW_RESULT_MISSING, MediatedCall, UnsupportedWorkspaceToolError, _jsonable, _policy_outcome
from pg.approval import ApprovalService
from pg.builder import ProposalRejected, ProposalService
from pg.card_context import copied_sources
from pg.confidentiality import ConfidentialityGuard, GuardedOutput, ReadAuthorizationPolicy
from pg.diagnostics import ApprovalTimeoutError, EpisodeStage, RuntimeLimits
from pg.gateway import ExecutionGateway
from pg.lifecycle_graph import LifecycleCheckpointStore, LifecycleGraph, LifecycleServices
from pg.policy import Decision
from pg.policy_ablation import evaluate_for as policy_evaluate_for
from pg.schema_adapters import receipt_record_from_legacy
from pg.trusted_models import (AgentRunState, CallStatus, DataSensitivity, EvaluationRecord, EvaluationStatus, ExecutionRecord,
                               ExecutionStatus, ModelCallRecord, PolicyDecision, ToolIntent)

from .capabilities import CapabilitySet
from .content_v2 import ReadLogV2
from .guards import denial_message, last_category
from .registry import install
from .suites import SuiteSpec


class SuiteAdapter:
    """Gateway adapters for one suite: decode the signed canonical arguments, run AgentDojo's stock function, keep the raw result."""

    def __init__(self, functions: Sequence[Function], spec: SuiteSpec) -> None:
        self._runtime = FunctionsRuntime(functions)
        self._spec = spec
        self._raw: dict[str, Any] = {}

    def pop_raw_result(self, function_name: str) -> Any:
        return self._raw.pop(function_name, RAW_RESULT_MISSING)

    def gateway_adapters(self) -> dict[str, Callable[..., Any]]:
        return {canonical: self._make(canonical) for canonical in self._spec.writes.values()}

    def _make(self, canonical: str) -> Callable[..., Any]:
        def _adapter(environment: TaskEnvironment, **arguments: Any) -> Any:
            function, kwargs = self._spec.to_agentdojo(canonical, arguments)
            result, error = self._runtime.run_function(environment, function, kwargs, raise_on_error=True)
            if error is not None:  # pragma: no cover - raise_on_error guarantees this
                raise RuntimeError(error)
            self._raw[function] = result
            return _jsonable(result)

        return _adapter


class ProtectedRuntime(FunctionsRuntime):
    """Drop-in AgentDojo runtime with complete mediation of every write in ``spec``."""

    def __init__(
        self,
        functions: Sequence[Function],
        *,
        spec: SuiteSpec,
        proposal_service: ProposalService,
        approvals: ApprovalService,
        gateway: ExecutionGateway,
        user_goal: str,
        model_id: str,
        session_id: str,
        principal_id: str = "user_gaurang",
        source_context: Mapping[str, str] | None = None,
        reviewer: Callable[[Any], bool] | None = None,
        runtime_limits: RuntimeLimits | None = None,
        diagnostic_timings: list[dict[str, Any]] | None = None,
        raw_result_for: Callable[[str], Any] | None = None,
        capabilities: CapabilitySet | None = None,
        annotate_sources: bool = True,
        feedback: bool = False,
    ) -> None:
        install()
        super().__init__(functions)
        spec.classify(set(self.functions))
        self._spec = spec
        self._read_log = ReadLogV2(spec.name)
        self._raw_result_for = raw_result_for
        self._direct_reads = FunctionsRuntime([f for f in functions if f.name in spec.reads])
        self._proposal_service, self._approvals, self._gateway = proposal_service, approvals, gateway
        self._user_goal, self._model_id, self._session_id, self._principal_id = user_goal, model_id, session_id, principal_id
        self._source_context = dict(source_context or {})
        self._reviewer = reviewer or (lambda _card: True)
        self._runtime_limits = runtime_limits or RuntimeLimits()
        self._diagnostic_timings = diagnostic_timings if diagnostic_timings is not None else []
        self._capabilities = capabilities
        self._annotate = annotate_sources
        self._feedback = feedback
        self.mediated_calls: list[MediatedCall] = []
        self.lifecycle_runs: list[str] = []
        self.denial_categories: list[str | None] = []
        self._confidentiality_guard = ConfidentialityGuard(
            graph_run_id=f"confidentiality-{session_id}"[:160],
            policy=ReadAuthorizationPolicy(principal_id=principal_id, allowed_tools=frozenset(spec.reads),
                                           allowed_read_sensitivities=frozenset(DataSensitivity)),
            evidence_sink=self._proposal_service._store.append_sensitivity_evidence,
            enabled=self._proposal_service._store.layers.output_guard,
        )

    @property
    def read_sources(self):
        return self._read_log.sources

    def run_function(self, env: TaskEnvironment | None, function: str, kwargs: Mapping[str, FunctionCallArgTypes],
                     raise_on_error: bool = False) -> tuple[FunctionReturnType, str | None]:
        spec = self._spec
        if function in spec.reads:
            result, error = self._direct_reads.run_function(env, function, kwargs, raise_on_error=raise_on_error)
            if error is None:
                self._confidentiality_guard.observe_read(principal_id=self._principal_id, tool_name=function,
                                                         arguments=kwargs, result=result)
                self._read_log.observe(function, result, kwargs)
            return result, error
        if function not in spec.writes:
            error = f"UnsupportedWorkspaceToolError: blocked unclassified function {function!r}"
            if raise_on_error:
                raise UnsupportedWorkspaceToolError(error)
            return "", error
        if env is None:
            error = "ValueError: protected call requires an environment"
            if raise_on_error:
                raise ValueError(error)
            return "", error
        try:
            canonical_tool, arguments = spec.to_canonical(function, kwargs)
        except ValueError as exc:  # an argument that cannot be encoded (e.g. a non-numeric amount) is a schema denial
            self.mediated_calls.append(MediatedCall(function, spec.writes[function], None, "SCHEMA_DENY", "BLOCKED", str(exc)))
            if raise_on_error:
                raise
            return "", f"LifecycleError: {exc}"
        if self._raw_result_for is not None:
            self._raw_result_for(function)
        graph_run_id = str(uuid.uuid4())
        self.lifecycle_runs.append(graph_run_id)
        call_context = dict(self._source_context)
        call_context["agentdojo_tool_request"] = json.dumps({"function": function, "arguments": arguments}, sort_keys=True,
                                                           default=str)
        context: dict[str, Any] = {"outcome": None, "approval": None, "result": None, "denied": False}
        tool_intent = ToolIntent(call_id=f"agentdojo-{graph_run_id}", tool_name=canonical_tool, arguments=_jsonable(arguments),
                                 raw_arguments=_jsonable(dict(kwargs)), parse_error=None)

        def build_proposal(_state: AgentRunState) -> str:
            outcome = self._proposal_service.propose_action(
                session_id=self._session_id, principal_id=self._principal_id, user_goal=self._user_goal,
                model_id=self._model_id, tool_name=canonical_tool, arguments=arguments, source_labels=call_context)
            context["outcome"] = outcome
            return outcome.proposal.proposal_id

        def obtain_approval(state: AgentRunState) -> str | None:
            started = time.perf_counter()
            outcome = context["outcome"]
            if outcome.policy_decision == Decision.ALLOW:
                approval = self._approvals.auto_approve(state.proposal_id)
            else:
                card = self._approvals.request_approval(state.proposal_id)
                card.pgx_context = {
                    "spec": spec, "request": self._user_goal, "canonical": canonical_tool, "function": function,
                    "arguments": arguments, "env": env, "read_sources": self._read_log.sources,
                    "capabilities": self._capabilities,
                    "copied_from": (copied_sources(env, canonical_tool, arguments)
                                    if self._annotate and spec.name == "workspace" else []),
                }
                card.copied_from = card.pgx_context["copied_from"]
                decision = "approve" if self._reviewer(card) else "deny"
                approval = self._approvals.decide(state.proposal_id, reviewer_id="mentor",
                                                  session_token="sess-tok-mentor-001", decision=decision)
                context["denied"] = decision == "deny"
            elapsed = time.perf_counter() - started
            self._diagnostic_timings.append({"stage": EpisodeStage.OBTAIN_APPROVAL.value, "function": function,
                                             "graph_run_id": graph_run_id, "node": "OBTAIN_APPROVAL",
                                             "latency_ms": round(elapsed * 1000),
                                             "error": "ReviewerDenied" if context["denied"] else None})
            if elapsed > self._runtime_limits.approval_seconds:
                raise ApprovalTimeoutError(f"approval simulation exceeded {self._runtime_limits.approval_seconds:.3f}s limit "
                                           f"({elapsed:.3f}s elapsed); decision retained and not retried")
            context["approval"] = approval
            return None if context["denied"] else approval.approval_id

        def execute_write(state: AgentRunState):
            started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            result = self._gateway.execute(state.proposal_id, state.approval_id)
            completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            context["result"] = result
            execution = ExecutionRecord(
                attempt_id=result.receipt.attempt_id, proposal_id=state.proposal_id, approval_id=state.approval_id,
                idempotency_key=result.receipt.idempotency_key, adapter=f"agentdojo:{function}",
                status=ExecutionStatus.SUCCEEDED if result.receipt.result_status == "SUCCEEDED" else ExecutionStatus.FAILED,
                started_at=started_at, completed_at=completed_at, error=result.error)
            receipt = receipt_record_from_legacy(result.receipt, proposal_id=state.proposal_id, detail=result.error or "",
                                                 created_at=completed_at)
            return execution, receipt

        services = LifecycleServices(
            call_model=lambda state: ModelCallRecord(
                call_id=f"model-call-{graph_run_id}", graph_run_id=graph_run_id,
                profile_sha256=hashlib.sha256(self._model_id.encode()).hexdigest(), stage="CALL_MODEL", attempt=1,
                request_evidence_ref=f"evidence://{graph_run_id}/conversation",
                response_evidence_ref=f"evidence://{graph_run_id}/agentdojo-tool-call", latency_ms=0,
                status=CallStatus.SUCCEEDED),
            parse_tool_intent=lambda state: tool_intent,
            label_sources=lambda state: (),
            evaluate_policy=lambda state: _policy_outcome(policy_evaluate_for(
                self._proposal_service._store.layers, principal_id=self._principal_id, tool_name=canonical_tool,
                arguments=arguments)),
            build_proposal=build_proposal, obtain_approval=obtain_approval, execute_read=lambda state: None,
            execute_write=execute_write,
            evaluate_final=lambda state: EvaluationRecord(
                graph_run_id=graph_run_id, scenario_id=self._session_id, arm="protected", seed=0,
                status=EvaluationStatus.EVALUABLE, completed=True,
                final_state_digest=(state.receipt.result_digest if state.receipt else hashlib.sha256(b"rejected").hexdigest()),
                evaluator_version="pgx-protected-runtime-v2"),
            read_tools=frozenset(spec.reads),
        )
        checkpoint_store = LifecycleCheckpointStore(f"{self._proposal_service._store.db_path}.lifecycle.sqlite")
        try:
            graph = LifecycleGraph(services, checkpoint_store)
            final_state = graph.invoke(AgentRunState(
                graph_run_id=graph_run_id, scenario_id=self._session_id, arm="protected", seed=0, session_id=self._session_id,
                principal_id=self._principal_id, model_profile_sha256=hashlib.sha256(self._model_id.encode()).hexdigest(),
                scenario_evidence_ref=f"evidence://{graph_run_id}/scenario"))
            self._diagnostic_timings.extend({"stage": "GRAPH_NODE", "graph_run_id": item.graph_run_id, "node": item.node.value,
                                             "latency_ms": item.latency_ms, "error": item.error,
                                             "evidence_ref": item.evidence_ref}
                                            for item in checkpoint_store.diagnostics(graph_run_id))
        finally:
            checkpoint_store.close()

        outcome = context["outcome"]
        proposal_id = outcome.proposal.proposal_id if outcome is not None else None
        if final_state.policy_outcome and final_state.policy_outcome.decision is PolicyDecision.DENY:
            detail = final_state.policy_outcome.reason
            self.mediated_calls.append(MediatedCall(function, canonical_tool, None, PolicyDecision.DENY.value, "BLOCKED", detail))
            if raise_on_error:
                raise ProposalRejected(f"policy denied: {detail}", policy_decision="DENY")
            return "", f"ProposalRejected: policy denied: {detail}"
        if context["denied"]:
            category = last_category(self._reviewer)
            self.denial_categories.append(category)
            self.mediated_calls.append(MediatedCall(function, canonical_tool, proposal_id, outcome.policy_decision.value,
                                                    "DENIED", category or ""))
            return "", denial_message(category, self._feedback)
        if final_state.status.value != "SUCCEEDED":
            detail = final_state.last_error or "lifecycle failed closed"
            self.mediated_calls.append(MediatedCall(function, canonical_tool, proposal_id,
                                                    outcome.policy_decision.value if outcome is not None else "SCHEMA_DENY",
                                                    "BLOCKED", detail))
            if raise_on_error:
                if isinstance(detail, str) and detail.startswith("policy denied"):
                    raise ProposalRejected(detail)
                raise RuntimeError(detail)
            return "", f"LifecycleError: {detail}"

        result = context["result"]
        self.mediated_calls.append(MediatedCall(function, canonical_tool, proposal_id, outcome.policy_decision.value,
                                                result.receipt.result_status, result.error or ""))
        if result.error is not None:
            if raise_on_error:
                raise RuntimeError(result.error)
            return "", result.error
        value = result.tool_result
        if self._raw_result_for is not None:
            raw = self._raw_result_for(function)
            if raw is not RAW_RESULT_MISSING:
                value = raw
        if function in spec.read_content_writes:
            self._read_log.observe(function, _jsonable(value), kwargs)
        return value, None

    @property
    def sensitivity_evidence(self):
        return self._confidentiality_guard.evidence

    def guard_final_output(self, text: str) -> GuardedOutput:
        guarded = self._confidentiality_guard.guard_output(text)
        self._proposal_service._store.append_output_guard_record(guarded.record)
        return guarded
