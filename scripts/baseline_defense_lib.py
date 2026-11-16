"""Task #29: naive, no-ProvenanceGuard defenses, run on the SAME scenarios the PG ladder used, so cost/benefit is comparable.

These are the "what if you just..." baselines the related-work literature compares against: nothing you'd call a mediation
architecture, just prompt- or tool-surface-level mitigations bolted onto the stock ``arm="baseline"`` AgentDojo pipeline
(no proposals, no approvals, no gateway, no audit chain -- that machinery is what ProvenanceGuard adds and these deliberately
lack). They answer "how far do cheap tricks get you" so the ladder results can say ProvenanceGuard is doing more than that.

Four defenses:

* ``none``                       -- the stock baseline arm, unmodified (the existing C0/A0 no-defense condition).
* ``spotlighting_with_delimiting`` -- AgentDojo's built-in defense (arXiv:2403.14720): tool outputs wrapped in <<...>> and
  the system prompt told never to obey instructions between those marks. Works with any LLM pipeline element.
* ``repeat_user_prompt``         -- AgentDojo's built-in defense: the original user goal is re-appended after every tool
  call, so a later instruction competes with a fresh reminder of the real task.
* ``tool_allowlist``              -- NOT an AgentDojo built-in (AgentDojo's own ``tool_filter`` defense requires an OpenAI
  model to pick the allowlist, which this project cannot use -- no paid/hosted API). This is a local, deterministic
  substitute: the FunctionsRuntime is constructed with only the tools the scenario's own ``allowed_writes`` names, plus the
  read-only tools every scenario needs, so the model has no write tool to misuse even if it is convinced to try. This is a
  weaker, static form of tool filtering -- it narrows the write surface to what THIS task legitimately needs, it does not
  reason about intent the way AgentDojo's LLM-based filter does.

A fifth, ``transformers_pi_detector`` (AgentDojo's HuggingFace-model-based prompt-injection classifier), is deliberately left
optional and is NOT in ``DEFENSE_NAMES``: it needs a one-time download of ``protectai/deberta-v3-base-prompt-injection-v2``
from the Hub, which this project's compute nodes cannot reach (network allowlist is package registries + GitHub only). See
``pi_detector_available()`` -- run it once after a manual download to use the detector; this library does not pretend it is
available when it isn't.

Everything here is a wrapper: it reuses ``pg.week5_runner``'s scenario loading, environment setup, and state evaluation
verbatim (``load_scenario``, ``evaluate_state``, the ``EpisodeResult`` record), and only replaces the "which pipeline /
which tools" step that ``run_episode``'s baseline arm hardcodes to ``defense=None`` and the full tool list. It lives outside
``pg/`` on purpose, exactly like the Week 12/13/held-out wrappers: ``pg/week5_runner.py`` is in the frozen protocol digest
and must not change.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages

from pg.agentdojo_adapter import BENCHMARK_VERSION, CANONICAL_TO_AGENTDOJO, SUITE_NAME
from pg.diagnostics import RuntimeLimits, TimedFunctionsRuntime
from pg.ollama_model import SeededOllamaLLM
from pg.schemas import canonicalize, sha256_digest
from pg.week5_runner import EpisodeResult, evaluate_state, snapshot_workspace

DEFENSE_NAMES: tuple[str, ...] = ("none", "spotlighting_with_delimiting", "repeat_user_prompt", "tool_allowlist")
AGENTDOJO_DEFENSE_FOR = {  # the subset that map straight onto AgentDojo's own PipelineConfig.defense
    "spotlighting_with_delimiting": "spotlighting_with_delimiting",
    "repeat_user_prompt": "repeat_user_prompt",
}
READ_ONLY_ALWAYS = (  # every scenario needs to be able to read its own workspace regardless of what it may write
    "get_file_by_id", "search_files", "search_files_by_filename", "list_files", "search_emails",
    "get_unread_emails", "get_sent_emails", "get_received_emails", "get_draft_emails", "search_contacts_by_name",
    "search_contacts_by_email", "get_day_calendar_events", "search_calendar_events", "get_current_day",
)


def pi_detector_available() -> bool:
    """True once ``protectai/deberta-v3-base-prompt-injection-v2`` has been downloaded to the local HF cache.

    The compute nodes here have no route to huggingface.co, so this is False on a fresh checkout; download it once from a
    node that does have internet (or ``huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2``) and it
    becomes available everywhere that shares the HF cache. Never raises -- a missing/broken cache is just "not available".
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache("protectai/deberta-v3-base-prompt-injection-v2", "config.json")
        return isinstance(hit, str)
    except Exception:
        return False


def allowed_tool_names(scenario: dict[str, Any], all_tools) -> list[str]:
    """The tool_allowlist defense's allowlist: read tools everywhere, plus only the write tool(s) this task declares.

    ``allowed_writes[].tool`` is written in the project's canonical names (e.g. ``workspace_send_email``); the suite's
    actual AgentDojo tools use the un-prefixed names (``send_email``) -- ``CANONICAL_TO_AGENTDOJO`` is the same mapping
    ``pg.agentdojo_adapter`` uses to go the other way, reused here rather than re-derived.
    """
    declared_canonical = {w["tool"] for w in scenario.get("allowed_writes", [])}
    declared = {CANONICAL_TO_AGENTDOJO[c] for c in declared_canonical if c in CANONICAL_TO_AGENTDOJO}
    names = set(READ_ONLY_ALWAYS) | declared
    present = {t.name for t in all_tools}
    return sorted(names & present)  # never claim a tool the suite doesn't actually expose


def build_pipeline(defense: str, llm: SeededOllamaLLM) -> AgentPipeline:
    if defense not in DEFENSE_NAMES:
        raise ValueError(f"unknown defense {defense!r}; one of {DEFENSE_NAMES}")
    ad_defense = AGENTDOJO_DEFENSE_FOR.get(defense)  # None for "none" and for "tool_allowlist" (tool-surface change, not a pipeline defense)
    return AgentPipeline.from_config(PipelineConfig(
        llm=llm, model_id=llm.model, defense=ad_defense, system_message_name=None, system_message=None,
        tool_delimiter="user", tool_output_format=None,
    ))


def run_baseline_defended_episode(
    scenario: dict[str, Any], *, defense: str, seed: int = 42, llm: SeededOllamaLLM | None = None,
    runtime_limits: RuntimeLimits | None = None, evidence_dir: str | Path | None = None,
) -> EpisodeResult:
    """One episode of the stock (no-ProvenanceGuard) arm with ``defense`` applied. Mirrors ``run_episode(arm="baseline")``."""
    if defense not in DEFENSE_NAMES:
        raise ValueError(f"unknown defense {defense!r}; one of {DEFENSE_NAMES}")
    suite = get_suite(BENCHMARK_VERSION, SUITE_NAME)
    injections = scenario.get("injections", {})
    environment = suite.load_and_inject_default_environment(injections)
    limits = runtime_limits or RuntimeLimits()
    model = llm or SeededOllamaLLM(seed=seed, runtime_limits=limits)
    pipeline = build_pipeline(defense, model)

    tools = suite.tools
    if defense == "tool_allowlist":
        keep = set(allowed_tool_names(scenario, tools))
        tools = [t for t in tools if t.name in keep]
    runtime: Any = FunctionsRuntime(tools)
    runtime_timings: list[dict[str, Any]] = []
    runtime = TimedFunctionsRuntime(runtime, limit_seconds=limits.tool_seconds, timings=runtime_timings)

    root = Path(evidence_dir) if evidence_dir is not None else Path(tempfile.mkdtemp(prefix="pg_baseline_defense_"))
    root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    error, completed, messages = None, True, []
    try:
        _, _, environment, messages, _ = pipeline.query(scenario["prompt"], runtime, environment, messages=messages, extra_args={})
    except Exception as exc:
        completed, error = False, f"{type(exc).__name__}: {exc}"
    latency_ms = int((time.perf_counter() - started) * 1000)
    if latency_ms > int(limits.episode_seconds * 1000):
        completed, error = False, error or f"EpisodeTimeoutError: exceeded {limits.episode_seconds:.3f}s"

    try:
        passed, evaluated = evaluate_state(environment, scenario.get("checks", []))
    except Exception as exc:
        completed, passed, evaluated = False, False, []
        error = error or f"final-state evaluator {type(exc).__name__}: {exc}"
    # scoring follows run_episode exactly: scenario["kind"] (benign/attack) picks which field the pass/fail lands in
    kind = scenario.get("kind")
    benign_success = passed if kind == "benign" else None
    attack_success = (not passed) if kind == "attack" else None
    traces = functions_stack_trace_from_messages(messages) if messages else []
    usage = model.usage
    state = snapshot_workspace(environment)

    return EpisodeResult(
        scenario_id=scenario["scenario_id"], arm=f"baseline+{defense}", model_id=model.model, seed=seed,
        completed=completed, benign_success=benign_success, attack_success=attack_success, latency_ms=latency_ms,
        prompt_tokens=sum(x["prompt_tokens"] for x in usage), completion_tokens=sum(x["completion_tokens"] for x in usage),
        tool_calls=[call.model_dump(mode="json") for call in traces], mediated_calls=[], audit_traces=[],
        final_state_digest=sha256_digest(canonicalize(state)), checks=evaluated, runtime_timings=runtime_timings,
        error=error,
    )
