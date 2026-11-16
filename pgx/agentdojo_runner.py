"""One episode of the standard AgentDojo benchmark under any v2 configuration, scored by AgentDojo's own checks.

A case is a (suite, user task, injection task or none) triple with its frozen injection texts (``data/v2/injections.jsonl``).
The episode follows ``agentdojo.task_suite.TaskSuite.run_task_with_pipeline`` step for step (same environment loading, same
three attempts when the model returns no output, same ``_check_task_result`` for utility and security) and changes only the
runtime the pipeline executes tools through.  Every episode returns one row; an exception is recorded, never dropped.

For protected configurations the text scored as the agent's final answer is the output guard's version of it -- what the user
would actually see.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages
from agentdojo.types import get_text_content_as_str, text_content_block_from_string

from pg.ablation import FULL
from pg.agentdojo_adapter import AgentDojoWorkspaceAdapter, ProtectedWorkspaceRuntime
from pg.approval import ApprovalService
from pg.builder import ProposalService
from pg.crypto import DevelopmentKeyStore, Signer
from pg.diagnostics import RuntimeLimits, TimedFunctionsRuntime
from pg.gateway import ExecutionGateway
from pg.model_api import CanonicalMessage
from pg.reviewers import ApproveAll, DenialCap, ModelReviewer, ScopedReviewer
from pg.store import EventStore
from pg.week5_runner import compact_episode_evidence_dir
from scripts.week11e_lib import CONFIGS_E

from .baselines import AuditedRuntime, filter_tools
from .capabilities import PlannerUsage, metered, parse_capabilities, plan_capabilities, tool_descriptions
from .configs import CONFIGS_V2
from .guards import GuardChain, ModelReviewerV4, V4Checks
from .registry import install
from .runtime import ProtectedRuntime, SuiteAdapter
from .suites import BENCHMARK_VERSION, spec_for

ROW_SCHEMA = "pgx-agentdojo-episode-v1"


@dataclass(frozen=True)
class Case:
    suite: str
    user_task_id: str
    injection_task_id: str | None
    injections: Mapping[str, str] = field(default_factory=dict)
    split: str = "all"
    in_scope: bool | None = None        # the injection goal needs a write (None for a benign case)
    under_specified: bool = False       # a correct write's target comes from data, not from the request

    @property
    def case_id(self) -> str:
        return f"{self.suite}/{self.user_task_id}/{self.injection_task_id or 'benign'}"


def injection_sources(injections: Mapping[str, str]) -> dict[str, str]:
    """Source labels for the evidence store, one per DISTINCT injected text.

    AgentDojo's attack puts the same text into every injectable field a task reads; v1's proposal record requires its source
    digests to be unique, so identical texts are recorded once under a label naming every field they were placed in.  (Without
    this every write in such an episode failed schema validation -- found by ``tests/test_pgx_e2e.py`` before any run.)"""
    by_text: dict[str, list[str]] = {}
    for vector, text in injections.items():
        by_text.setdefault(text, []).append(vector)
    return {"agentdojo_injection:" + "+".join(sorted(vectors)): text for text, vectors in by_text.items()}


def text_of(client: Any) -> Callable[[str], str]:
    return lambda prompt: client.complete([CanonicalMessage(role="user", content=prompt)], ()).content or ""


def completion_of(client: Any) -> Callable[[str], Any]:
    return lambda prompt: client.complete([CanonicalMessage(role="user", content=prompt)], ())


def _usage(llm: Any) -> tuple[int, int]:
    usage = getattr(llm, "usage", None)
    if isinstance(usage, list):
        return sum(int(u.get("prompt_tokens", 0)) for u in usage), sum(int(u.get("completion_tokens", 0)) for u in usage)
    records = getattr(usage, "records", None)
    if isinstance(records, list):
        return (sum(int(getattr(r, "prompt_tokens", 0) or 0) for r in records),
                sum(int(getattr(r, "completion_tokens", 0) or 0) for r in records))
    return 0, 0


def run_case(case: Case, config: str, *, llm: Any, agent_client: Any, reviewer_client: Any | None, seed: int,
             evidence_root: Path | None = None, limits: RuntimeLimits | None = None) -> dict[str, Any]:
    """One episode.  ``llm`` is the agent's AgentDojo pipeline element; ``agent_client``/``reviewer_client`` answer plain
    prompts (planner, tool filter: the agent's model; reviewer, auditor: the configured reviewer model)."""

    spec_cfg = CONFIGS_V2[config]
    kind = spec_cfg["kind"]
    limits = limits or RuntimeLimits()
    if kind == "v1" and case.suite not in spec_cfg.get("suites", [case.suite]):
        raise ValueError(f"{config} runs only on {spec_cfg['suites']}, not {case.suite}")
    suite = get_suite(BENCHMARK_VERSION, case.suite)
    spec = spec_for(case.suite)
    user_task = suite.get_user_task_by_id(case.user_task_id)
    injection_task = suite.get_injection_task_by_id(case.injection_task_id) if case.injection_task_id else None
    environment = suite.load_and_inject_default_environment(dict(case.injections))
    task_env = user_task.init_environment(environment)
    pre_environment = task_env.model_copy(deep=True)
    prompt = user_task.PROMPT

    pipeline = AgentPipeline.from_config(PipelineConfig(
        llm=llm, model_id=getattr(llm, "model", None) or "local-model", defense=spec_cfg.get("defense"),
        system_message_name=None, system_message=None, tool_delimiter="user", tool_output_format=None))

    temporary = evidence_root is None
    root = Path(evidence_root) if evidence_root is not None else Path(tempfile.mkdtemp(prefix="pgx_"))
    root.mkdir(parents=True, exist_ok=True)
    store = None
    protected = None
    reviewer: Any = None
    extra: dict[str, Any] = {}
    timings: list[dict[str, Any]] = []
    planner_usage = PlannerUsage()
    try:
        if kind == "stock":
            runtime: Any = FunctionsRuntime(suite.tools)
        elif kind == "toolsel":
            chosen, record = filter_tools(prompt, suite.tools, text_of(agent_client))
            extra["tool_filter"] = record
            runtime = FunctionsRuntime([t for t in suite.tools if t.name in set(chosen)])
        elif kind == "auditor":
            runtime = AuditedRuntime(suite.tools, spec=spec, goal=prompt, complete=completion_of(reviewer_client))
            reviewer = runtime
        else:
            install()
            keys = DevelopmentKeyStore(root / ".development-keys")
            signer = Signer(keys.load_or_create("pgx-proposal-key"))
            verifier = keys.verifier()
            episode_dir = compact_episode_evidence_dir(root, f"{case.case_id}-{config}".replace("/", "_"), seed, "protected")
            layers = FULL if kind == "v1" else FULL.without("policy")
            store = EventStore(str(episode_dir / "events.db"), str(episode_dir / "checkpoints"), layers=layers)
            proposals, approvals = ProposalService(store, signer, verifier), ApprovalService(store, signer, verifier)
            source_context = injection_sources(case.injections)
            session_id = f"{case.case_id}-{config}-seed-{seed}"
            if kind == "v1":
                v3 = CONFIGS_E[config]
                # v1 semantics, frozen: the reviewer is the agent's own model (self-review), exactly as Weeks 11e-13 ran it
                scoped = ScopedReviewer(ModelReviewer(agent_client, prompt, prompt_version="v3"), prompt)
                reviewer = DenialCap(scoped, v3["denial_cap"]) if v3.get("denial_cap") else scoped
                adapter = AgentDojoWorkspaceAdapter(suite.tools)
                gateway = ExecutionGateway(store, signer, verifier, task_env, adapters=adapter.gateway_adapters())
                protected = ProtectedWorkspaceRuntime(
                    suite.tools, proposal_service=proposals, approvals=approvals, gateway=gateway, user_goal=prompt,
                    model_id=getattr(llm, "model", "local-model"), session_id=session_id, source_context=source_context,
                    runtime_limits=limits, diagnostic_timings=timings, reviewer=reviewer, raw_result_for=adapter.pop_raw_result,
                    describe_targets=bool(v3.get("describe")), annotate_sources=bool(v3.get("annotate")),
                    check_provenance=bool(v3.get("provenance")))
            else:
                guard = spec_cfg.get("guard")
                caps = None
                if guard and guard.get("capability") == "parse":
                    caps = parse_capabilities(prompt, spec)
                elif guard and guard.get("capability") == "plan":
                    caps = plan_capabilities(prompt, spec, metered(agent_client, planner_usage), tool_descriptions(suite.tools))
                if guard is None:
                    reviewer = ApproveAll()
                else:
                    judge = reviewer_client if guard.get("reviewer") == "independent" else agent_client
                    chain = GuardChain(V4Checks(capability=guard.get("capability") is not None, bypass=guard["bypass"],
                                                numeric=guard["numeric"], scope_fix=guard["scope_fix"]),
                                       ModelReviewerV4(judge, prompt))
                    reviewer = DenialCap(chain, guard["denial_cap"]) if guard.get("denial_cap") else chain
                extra["capabilities"] = caps.to_dict() if caps is not None else None
                adapter = SuiteAdapter(suite.tools, spec)
                gateway = ExecutionGateway(store, signer, verifier, task_env, adapters=adapter.gateway_adapters())
                protected = ProtectedRuntime(
                    suite.tools, spec=spec, proposal_service=proposals, approvals=approvals, gateway=gateway, user_goal=prompt,
                    model_id=getattr(llm, "model", "local-model"), session_id=session_id, source_context=source_context,
                    reviewer=reviewer, runtime_limits=limits, diagnostic_timings=timings, raw_result_for=adapter.pop_raw_result,
                    capabilities=caps, feedback=bool(guard and guard.get("feedback")))
            runtime = protected
        timed = TimedFunctionsRuntime(runtime, limit_seconds=limits.tool_seconds, timings=timings)

        started = time.perf_counter()
        error = None
        messages: list[Any] = []
        model_output = None
        attempts = 0
        for _ in range(3):  # AgentDojo's own rule: re-ask (same runtime) if the model produced no final output
            attempts += 1
            try:
                _, _, task_env, messages, _ = pipeline.query(prompt, timed, task_env)
            except AbortAgentError as exc:
                task_env, messages = exc.task_environment, exc.messages
            except Exception as exc:  # noqa: BLE001 -- recorded, never dropped
                error = f"{type(exc).__name__}: {exc}"
                break
            try:
                model_output = model_output_from_messages(messages)
            except (ValueError, IndexError):
                model_output = None
            if model_output is not None:
                break
        latency_ms = int((time.perf_counter() - started) * 1000)

        final_text = get_text_content_as_str(model_output) if model_output else ""
        output_guard = None
        if protected is not None:
            guarded = protected.guard_final_output(final_text)
            output_guard = guarded.record.model_dump(mode="json")
            if guarded.text != final_text:
                model_output = [text_content_block_from_string(guarded.text)]
                final_text = guarded.text
        trace = functions_stack_trace_from_messages(messages) if messages else []
        utility = attack = None
        evaluator_error = None
        try:
            utility = bool(suite._check_task_result(user_task, model_output or [], pre_environment, task_env, trace))
            if injection_task is not None:
                attack = bool(suite._check_task_result(injection_task, model_output or [], pre_environment, task_env, trace))
        except Exception as exc:  # noqa: BLE001
            evaluator_error = f"{type(exc).__name__}: {exc}"
        prompt_tokens, completion_tokens = _usage(llm)
        summary = reviewer.summary() if reviewer is not None and hasattr(reviewer, "summary") else None
        return {
            "schema": ROW_SCHEMA, "suite": case.suite, "user_task_id": case.user_task_id,
            "injection_task_id": case.injection_task_id, "case_id": case.case_id, "split": case.split,
            "in_scope": case.in_scope, "under_specified": case.under_specified, "config": config, "seed": seed,
            "utility": utility, "attack_success": attack, "error": error or evaluator_error, "attempts": attempts,
            "latency_ms": latency_ms, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "reviewer": summary,
            "reviewer_tokens": (summary or {}).get("prompt_tokens", 0) + (summary or {}).get("completion_tokens", 0),
            "planner": asdict(planner_usage),
            "tool_calls": [c.model_dump(mode="json") for c in trace],
            "mediated_calls": [asdict(c) for c in (protected.mediated_calls if protected is not None else [])],
            "denial_categories": list(getattr(protected, "denial_categories", []) or []),
            "read_sources": (protected._read_log.to_records() if isinstance(protected, ProtectedRuntime) else None),
            "final_response": final_text[:4000], "output_guard": output_guard, "timings": timings[-50:], **extra,
        }
    finally:
        if store is not None:
            store.close()
        if temporary:
            shutil.rmtree(root, ignore_errors=True)
