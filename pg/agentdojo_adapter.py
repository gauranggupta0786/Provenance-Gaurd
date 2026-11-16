"""Week 5 integration boundary for AgentDojo's Workspace suite.

AgentDojo remains the simulator and evaluator.  It is not an authorization
boundary.  Read-only Workspace functions use AgentDojo's stock runtime;
every state-changing function is translated to an explicit ProvenanceGuard
tool contract and can execute only through the signed proposal, canonical
approval, and gateway path.
"""
from __future__ import annotations

import json
import hashlib
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from pydantic import BaseModel

from agentdojo.functions_runtime import (
    Function,
    FunctionCallArgTypes,
    FunctionReturnType,
    FunctionsRuntime,
    TaskEnvironment,
)

from .approval import ApprovalService
from .card_context import copied_sources, describe_target
from .content_provenance import ReadLog, unlicensed_sources
from .builder import ProposalRejected, ProposalService
from .diagnostics import ApprovalTimeoutError, EpisodeStage, RuntimeLimits
from .gateway import ExecutionGateway
from .policy import Decision, POLICY_VERSION
from .policy_ablation import evaluate_for as policy_evaluate_for
from .lifecycle_graph import LifecycleCheckpointStore, LifecycleGraph, LifecycleServices
from .schema_adapters import receipt_record_from_legacy
from .trusted_models import (
    AgentRunState, CallStatus, EvaluationRecord, EvaluationStatus, ExecutionRecord,
    ExecutionStatus, ModelCallRecord, PolicyDecision, PolicyOutcome, ToolIntent,
)
from .confidentiality import ConfidentialityGuard, GuardedOutput, ReadAuthorizationPolicy
from .trusted_models import DataSensitivity


AGENTDOJO_VERSION = "0.1.35"
BENCHMARK_VERSION = "v1.2.2"
SUITE_NAME = "workspace"

# These functions do not mutate Workspace state in AgentDojo v1.2.2.
READ_ONLY_FUNCTIONS = frozenset({
    "get_sent_emails",
    "get_received_emails",
    "get_draft_emails",
    "search_emails",
    "search_contacts_by_name",
    "search_contacts_by_email",
    "get_current_day",
    "search_calendar_events",
    "get_day_calendar_events",
    "search_files_by_filename",
    "get_file_by_id",
    "list_files",
    "search_files",
})

# AgentDojo name -> signed ProvenanceGuard contract.  get_unread_emails is
# included because AgentDojo marks messages read as a side effect.
WRITE_FUNCTION_MAP = {
    "send_email": "workspace_send_email",
    "delete_email": "workspace_delete_email",
    "get_unread_emails": "workspace_get_unread_emails",
    "create_calendar_event": "workspace_create_calendar_event",
    "cancel_calendar_event": "workspace_cancel_calendar_event",
    "reschedule_calendar_event": "workspace_reschedule_calendar_event",
    "add_calendar_event_participants": "workspace_add_calendar_event_participants",
    "append_to_file": "workspace_append_to_file",
    "create_file": "workspace_create_file",
    "delete_file": "workspace_delete_file",
    "share_file": "workspace_share_file",
}
CANONICAL_TO_AGENTDOJO = {canonical: original for original, canonical in WRITE_FUNCTION_MAP.items()}

# Write-classified tools whose *result* is nonetheless read content the agent can copy into a later send.  get_unread_emails
# is mediated because it mutates read-state, so it never reaches the read branch above; its returned e-mails must still enter
# the read log or the content-provenance check is blind to them (Week 11d in-loop run: the one surviving exfiltration read the
# secret e-mail through this tool).
READ_CONTENT_WRITE_FUNCTIONS = frozenset({"get_unread_emails"})


class UnsupportedWorkspaceToolError(ValueError):
    pass


def _policy_outcome(result: Any) -> PolicyOutcome:
    return PolicyOutcome(
        phase="proposal",
        decision=PolicyDecision(result.decision.value),
        reason=result.reason,
        policy_version=POLICY_VERSION,
        evaluated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


def to_canonical_call(function: str, arguments: Mapping[str, FunctionCallArgTypes]) -> tuple[str, dict[str, Any]]:
    """Translate a model-requested AgentDojo call without executing it."""
    canonical = WRITE_FUNCTION_MAP.get(function)
    if canonical is None:
        raise UnsupportedWorkspaceToolError(f"{function!r} is not a mediated Workspace write function")
    args = dict(arguments)
    if function == "send_email":
        args["recipients"] = list(args.get("recipients") or [])
        args["attachments"] = list(args.get("attachments") or [])
        args["cc"] = list(args.get("cc") or [])
        args["bcc"] = list(args.get("bcc") or [])
    elif function == "create_calendar_event":
        args["description"] = args.get("description") or ""
        args["participants"] = list(args.get("participants") or [])
        args["location"] = args.get("location")
    elif function == "reschedule_calendar_event":
        args["new_end_time"] = args.get("new_end_time")
    return canonical, args


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(v) for v in value]
    return value


RAW_RESULT_MISSING = object()


class AgentDojoWorkspaceAdapter:
    """Privileged gateway adapter backed by AgentDojo's stock simulator.

    The gateway signs and stores a JSON-able copy of each result. The stock
    object is also kept (once, per function) so the model-facing runtime can hand
    AgentDojo exactly what the baseline arm would have returned. Without this,
    list-valued tools such as ``get_unread_emails`` came back as a list of dicts
    and AgentDojo's formatter raised ``TypeError`` (found in the Week 7 pilot).
    """

    def __init__(self, functions: Sequence[Function]):
        self._runtime = FunctionsRuntime(functions)
        self._raw_results: dict[str, Any] = {}

    def pop_raw_result(self, function_name: str) -> Any:
        return self._raw_results.pop(function_name, RAW_RESULT_MISSING)

    def gateway_adapters(self) -> dict[str, Callable[..., Any]]:
        adapters: dict[str, Callable[..., Any]] = {}
        for canonical_name, agentdojo_name in CANONICAL_TO_AGENTDOJO.items():
            adapters[canonical_name] = self._make_adapter(agentdojo_name)
        return adapters

    def _make_adapter(self, function_name: str) -> Callable[..., Any]:
        def _adapter(environment: TaskEnvironment, **arguments: Any) -> Any:
            result, error = self._runtime.run_function(
                environment, function_name, arguments, raise_on_error=True,
            )
            if error is not None:  # pragma: no cover - raise_on_error guarantees this
                raise RuntimeError(error)
            self._raw_results[function_name] = result
            return _jsonable(result)

        return _adapter


@dataclass(frozen=True)
class MediatedCall:
    agentdojo_function: str
    canonical_tool: str
    proposal_id: str | None
    decision: str
    result_status: str
    detail: str = ""


class ProtectedWorkspaceRuntime(FunctionsRuntime):
    """Drop-in AgentDojo runtime with complete mediation of Workspace writes.

    `ToolsExecutor` can only call `run_function`; it never receives the
    privileged adapter or gateway.  This keeps the model-facing benchmark API
    stock-compatible while eliminating the direct write path in the protected
    arm.
    """

    def __init__(
        self,
        functions: Sequence[Function],
        *,
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
        describe_targets: bool = False,
        annotate_sources: bool = False,
        check_provenance: bool = False,
    ) -> None:
        super().__init__(functions)
        self._describe_targets = describe_targets
        self._annotate_sources = annotate_sources
        self._check_provenance = check_provenance
        self._read_log = ReadLog()
        self._raw_result_for = raw_result_for
        self._direct_reads = FunctionsRuntime([f for f in functions if f.name in READ_ONLY_FUNCTIONS])
        self._proposal_service = proposal_service
        self._approvals = approvals
        self._gateway = gateway
        self._user_goal = user_goal
        self._model_id = model_id
        self._session_id = session_id
        self._principal_id = principal_id
        self._source_context = dict(source_context or {})
        self._reviewer = reviewer or (lambda _card: True)
        self._runtime_limits = runtime_limits or RuntimeLimits()
        self._diagnostic_timings = diagnostic_timings if diagnostic_timings is not None else []
        self.mediated_calls: list[MediatedCall] = []
        self.lifecycle_runs: list[str] = []
        self._confidentiality_run_id = f"confidentiality-{session_id}"[:160]
        self._confidentiality_guard = ConfidentialityGuard(
            graph_run_id=self._confidentiality_run_id,
            policy=ReadAuthorizationPolicy(
                principal_id=principal_id,
                allowed_tools=READ_ONLY_FUNCTIONS,
                allowed_read_sensitivities=frozenset(DataSensitivity),
            ),
            evidence_sink=self._proposal_service._store.append_sensitivity_evidence,
            enabled=self._proposal_service._store.layers.output_guard,
        )

        known = READ_ONLY_FUNCTIONS | WRITE_FUNCTION_MAP.keys()
        unexpected = set(self.functions) - known
        if unexpected:
            raise UnsupportedWorkspaceToolError(
                f"AgentDojo {AGENTDOJO_VERSION} exposed unclassified Workspace tools: {sorted(unexpected)}"
            )

    def run_function(
        self,
        env: TaskEnvironment | None,
        function: str,
        kwargs: Mapping[str, FunctionCallArgTypes],
        raise_on_error: bool = False,
    ) -> tuple[FunctionReturnType, str | None]:
        if function in READ_ONLY_FUNCTIONS:
            result, error = self._direct_reads.run_function(
                env, function, kwargs, raise_on_error=raise_on_error,
            )
            if error is None:
                self._confidentiality_guard.observe_read(
                    principal_id=self._principal_id,
                    tool_name=function,
                    arguments=kwargs,
                    result=result,
                )
                self._read_log.observe(function, result, kwargs)
            return result, error

        if function not in WRITE_FUNCTION_MAP:
            error = f"UnsupportedWorkspaceToolError: blocked unclassified function {function!r}"
            if raise_on_error:
                raise UnsupportedWorkspaceToolError(error)
            return "", error
        if env is None:
            error = "ValueError: protected Workspace call requires an environment"
            if raise_on_error:
                raise ValueError(error)
            return "", error

        canonical_tool, arguments = to_canonical_call(function, kwargs)
        if self._raw_result_for is not None:
            self._raw_result_for(function)  # discard any stale stock result
        graph_run_id = str(uuid.uuid4())
        self.lifecycle_runs.append(graph_run_id)
        call_context = dict(self._source_context)
        call_context["agentdojo_tool_request"] = json.dumps(
            {"function": function, "arguments": arguments}, sort_keys=True, default=str,
        )
        context: dict[str, Any] = {"outcome": None, "approval": None, "result": None, "denied": False}
        tool_intent = ToolIntent(
            call_id=f"agentdojo-{graph_run_id}", tool_name=canonical_tool,
            arguments=_jsonable(arguments), raw_arguments=_jsonable(dict(kwargs)), parse_error=None,
        )

        def build_proposal(_state: AgentRunState) -> str:
            outcome = self._proposal_service.propose_action(
                session_id=self._session_id,
                principal_id=self._principal_id,
                user_goal=self._user_goal,
                model_id=self._model_id,
                tool_name=canonical_tool,
                arguments=arguments,
                source_labels=call_context,
            )
            context["outcome"] = outcome
            return outcome.proposal.proposal_id

        def obtain_approval(state: AgentRunState) -> str | None:
            approval_started = time.perf_counter()
            outcome = context["outcome"]
            if outcome.policy_decision == Decision.ALLOW:
                approval = self._approvals.auto_approve(state.proposal_id)
            else:
                card = self._approvals.request_approval(state.proposal_id)
                if self._describe_targets:
                    card.target_description = describe_target(env, canonical_tool, arguments)
                if self._annotate_sources:
                    card.copied_from = copied_sources(env, canonical_tool, arguments)
                    if self._check_provenance and canonical_tool == "workspace_send_email":
                        hits = unlicensed_sources(str(arguments.get("body") or ""), self._user_goal, self._read_log.sources)
                        card.provenance_hits = [{"name": h.name, "kind": h.kind,
                                                 "unlicensed_words": list(h.unlicensed_words), "reason": h.reason()}
                                                for h in hits]
                decision = "approve" if self._reviewer(card) else "deny"
                approval = self._approvals.decide(
                    state.proposal_id,
                    reviewer_id="mentor",
                    session_token="sess-tok-mentor-001",
                    decision=decision,
                )
                context["denied"] = decision == "deny"
            elapsed = time.perf_counter() - approval_started
            self._diagnostic_timings.append({
                "stage": EpisodeStage.OBTAIN_APPROVAL.value,
                "function": function,
                "graph_run_id": graph_run_id,
                "node": "OBTAIN_APPROVAL",
                "latency_ms": round(elapsed * 1000),
                "error": "ReviewerDenied" if context["denied"] else None,
            })
            if elapsed > self._runtime_limits.approval_seconds:
                raise ApprovalTimeoutError(
                    f"approval simulation exceeded {self._runtime_limits.approval_seconds:.3f}s limit "
                    f"({elapsed:.3f}s elapsed); decision retained and not retried"
                )
            context["approval"] = approval
            return None if context["denied"] else approval.approval_id

        def execute_write(state: AgentRunState):
            started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            result = self._gateway.execute(state.proposal_id, state.approval_id)
            completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            context["result"] = result
            execution = ExecutionRecord(
                attempt_id=result.receipt.attempt_id,
                proposal_id=state.proposal_id,
                approval_id=state.approval_id,
                idempotency_key=result.receipt.idempotency_key,
                adapter=f"agentdojo:{function}",
                status=(ExecutionStatus.SUCCEEDED if result.receipt.result_status == "SUCCEEDED"
                        else ExecutionStatus.FAILED),
                started_at=started_at,
                completed_at=completed_at,
                error=result.error,
            )
            receipt = receipt_record_from_legacy(
                result.receipt, proposal_id=state.proposal_id,
                detail=result.error or "", created_at=completed_at,
            )
            return execution, receipt

        graph_services = LifecycleServices(
            call_model=lambda state: ModelCallRecord(
                call_id=f"model-call-{graph_run_id}", graph_run_id=graph_run_id,
                profile_sha256=hashlib.sha256(self._model_id.encode()).hexdigest(),
                stage="CALL_MODEL", attempt=1,
                request_evidence_ref=f"evidence://{graph_run_id}/conversation",
                response_evidence_ref=f"evidence://{graph_run_id}/agentdojo-tool-call",
                latency_ms=0, status=CallStatus.SUCCEEDED,
            ),
            parse_tool_intent=lambda state: tool_intent,
            label_sources=lambda state: (),
            evaluate_policy=lambda state: _policy_outcome(
                policy_evaluate_for(
                    self._proposal_service._store.layers,
                    principal_id=self._principal_id,
                    tool_name=canonical_tool,
                    arguments=arguments,
                ),
            ),
            build_proposal=build_proposal,
            obtain_approval=obtain_approval,
            execute_read=lambda state: None,
            execute_write=execute_write,
            evaluate_final=lambda state: EvaluationRecord(
                graph_run_id=graph_run_id,
                scenario_id=self._session_id,
                arm="protected", seed=0,
                status=EvaluationStatus.EVALUABLE,
                completed=True,
                final_state_digest=(state.receipt.result_digest if state.receipt else hashlib.sha256(b"rejected").hexdigest()),
                evaluator_version="protected-workspace-runtime-v1",
            ),
            read_tools=READ_ONLY_FUNCTIONS,
        )
        checkpoint_store = LifecycleCheckpointStore(
            f"{self._proposal_service._store.db_path}.lifecycle.sqlite"
        )
        try:
            graph = LifecycleGraph(graph_services, checkpoint_store)
            final_state = graph.invoke(AgentRunState(
                graph_run_id=graph_run_id,
                scenario_id=self._session_id,
                arm="protected", seed=0,
                session_id=self._session_id,
                principal_id=self._principal_id,
                model_profile_sha256=hashlib.sha256(self._model_id.encode()).hexdigest(),
                scenario_evidence_ref=f"evidence://{graph_run_id}/scenario",
            ))
            self._diagnostic_timings.extend({
                "stage": "GRAPH_NODE",
                "graph_run_id": item.graph_run_id,
                "node": item.node.value,
                "latency_ms": item.latency_ms,
                "error": item.error,
                "evidence_ref": item.evidence_ref,
            } for item in checkpoint_store.diagnostics(graph_run_id))
        finally:
            checkpoint_store.close()

        outcome = context["outcome"]
        proposal_id = outcome.proposal.proposal_id if outcome is not None else None
        if final_state.policy_outcome and final_state.policy_outcome.decision is PolicyDecision.DENY:
            detail = final_state.policy_outcome.reason
            self.mediated_calls.append(MediatedCall(
                function, canonical_tool, None, PolicyDecision.DENY.value, "BLOCKED", detail,
            ))
            if raise_on_error:
                raise ProposalRejected(f"policy denied: {detail}", policy_decision="DENY")
            return "", f"ProposalRejected: policy denied: {detail}"
        if context["denied"]:
            self.mediated_calls.append(MediatedCall(
                function, canonical_tool, proposal_id, outcome.policy_decision.value, "DENIED",
            ))
            return "", "ReviewerDenied: canonical action was denied"
        if final_state.status.value != "SUCCEEDED":
            detail = final_state.last_error or "lifecycle failed closed"
            self.mediated_calls.append(MediatedCall(
                function, canonical_tool, proposal_id,
                outcome.policy_decision.value if outcome is not None else "SCHEMA_DENY",
                "BLOCKED", detail,
            ))
            if raise_on_error:
                if isinstance(detail, str) and detail.startswith("policy denied"):
                    raise ProposalRejected(detail)
                raise RuntimeError(detail)
            return "", f"LifecycleError: {detail}"

        result = context["result"]
        self.mediated_calls.append(MediatedCall(
            function, canonical_tool, proposal_id, outcome.policy_decision.value,
            result.receipt.result_status, result.error or "",
        ))
        if result.error is not None:
            if raise_on_error:
                raise RuntimeError(result.error)
            return "", result.error
        if self._raw_result_for is not None:
            raw = self._raw_result_for(function)
            if raw is not RAW_RESULT_MISSING:
                if function in READ_CONTENT_WRITE_FUNCTIONS:
                    self._read_log.observe(function, _jsonable(raw), kwargs)
                return raw, None
        if function in READ_CONTENT_WRITE_FUNCTIONS:
            self._read_log.observe(function, result.tool_result, kwargs)
        return result.tool_result, None

    @property
    def sensitivity_evidence(self):
        return self._confidentiality_guard.evidence

    def guard_final_output(self, text: str) -> GuardedOutput:
        guarded = self._confidentiality_guard.guard_output(text)
        self._proposal_service._store.append_output_guard_record(guarded.record)
        return guarded
