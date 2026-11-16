"""Week 1 runtime diagnostics, timeout budgets, and evidence persistence.

The diagnostic layer is deliberately independent from authorization logic.
Only temporary model infrastructure failures are retryable. Schema, policy,
signature, replay, and other security decisions are retained as evidence and
are never reinterpreted or replayed.
"""
from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar


class FailureCategory(str, Enum):
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_SERVER = "MODEL_SERVER"
    MALFORMED_TOOL_CALL = "MALFORMED_TOOL_CALL"
    INVALID_STRUCTURED_OUTPUT = "INVALID_STRUCTURED_OUTPUT"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    SCHEMA_REJECTION = "SCHEMA_REJECTION"
    POLICY_REJECTION = "POLICY_REJECTION"
    SIGNATURE_REJECTION = "SIGNATURE_REJECTION"
    REPLAY_REJECTION = "REPLAY_REJECTION"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    APPROVAL_TIMEOUT = "APPROVAL_TIMEOUT"
    EPISODE_TIMEOUT = "EPISODE_TIMEOUT"
    AGENTDOJO_RUNTIME = "AGENTDOJO_RUNTIME"
    ADAPTER_FAILURE = "ADAPTER_FAILURE"
    SQLITE_FAILURE = "SQLITE_FAILURE"
    EVALUATOR_FAILURE = "EVALUATOR_FAILURE"
    UNKNOWN = "UNKNOWN"


class EpisodeStage(str, Enum):
    HEALTHCHECK = "HEALTHCHECK"
    WARMUP = "WARMUP"
    LOAD_SCENARIO = "LOAD_SCENARIO"
    CALL_MODEL = "CALL_MODEL"
    PARSE_TOOL_CALL = "PARSE_TOOL_CALL"
    CLASSIFY_TOOL = "CLASSIFY_TOOL"
    VALIDATE_ARGUMENTS = "VALIDATE_ARGUMENTS"
    EVALUATE_POLICY = "EVALUATE_POLICY"
    CREATE_PROPOSAL = "CREATE_PROPOSAL"
    OBTAIN_APPROVAL = "OBTAIN_APPROVAL"
    GATEWAY_CHECK = "GATEWAY_CHECK"
    EXECUTE_TOOL = "EXECUTE_TOOL"
    STORE_RECEIPT = "STORE_RECEIPT"
    EVALUATE_STATE = "EVALUATE_STATE"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class RuntimeLimits:
    """Independent runtime limits used by measured diagnostic episodes."""

    model_seconds: float = 300.0
    tool_seconds: float = 30.0
    approval_seconds: float = 30.0
    episode_seconds: float = 1800.0
    infrastructure_retries: int = 1

    def __post_init__(self) -> None:
        for name in ("model_seconds", "tool_seconds", "approval_seconds", "episode_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.infrastructure_retries not in (0, 1):
            raise ValueError("infrastructure_retries must be 0 or 1")


class ToolTimeoutError(TimeoutError):
    pass


class ApprovalTimeoutError(TimeoutError):
    pass


class EpisodeTimeoutError(TimeoutError):
    pass


RETRYABLE_CATEGORIES = frozenset({FailureCategory.MODEL_TIMEOUT, FailureCategory.MODEL_SERVER})


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def classify_failure(error: BaseException | str, stage: EpisodeStage | str) -> FailureCategory:
    """Classify an exception or retained error without changing its meaning."""

    stage_value = EpisodeStage(stage) if not isinstance(stage, EpisodeStage) else stage
    name = type(error).__name__.lower() if isinstance(error, BaseException) else ""
    message = str(error).lower()
    combined = f"{name} {message}"
    if isinstance(error, EpisodeTimeoutError) or "episodetimeouterror" in combined:
        return FailureCategory.EPISODE_TIMEOUT
    if isinstance(error, ApprovalTimeoutError) or "approvaltimeouterror" in combined:
        return FailureCategory.APPROVAL_TIMEOUT
    if isinstance(error, ToolTimeoutError) or "tooltimeouterror" in combined:
        return FailureCategory.TOOL_TIMEOUT
    if "context" in combined and any(token in combined for token in ("length", "window", "overflow", "too long")):
        return FailureCategory.CONTEXT_OVERFLOW
    if isinstance(error, TimeoutError) or any(token in combined for token in (
        "apitimeouterror", "readtimeout", "connecttimeout", "model timeout", "timed out",
    )):
        return FailureCategory.MODEL_TIMEOUT if stage_value in {
            EpisodeStage.HEALTHCHECK, EpisodeStage.WARMUP, EpisodeStage.CALL_MODEL,
        } else FailureCategory.TOOL_TIMEOUT
    if any(token in combined for token in (
        "apiconnectionerror", "connectionerror", "connection refused", "service unavailable",
        "internalservererror", "model server", "ollama", "http 502", "http 503", "http 504",
    )):
        return FailureCategory.MODEL_SERVER
    if any(token in combined for token in ("replay", "already consumed", "already executed")):
        return FailureCategory.REPLAY_REJECTION
    if any(token in combined for token in ("signature", "unknown key", "invalid key")):
        return FailureCategory.SIGNATURE_REJECTION
    if any(token in combined for token in ("policy", "proposalrejected", "reviewerdenied")):
        return FailureCategory.POLICY_REJECTION
    if "validationerror" in combined or "schemaerror" in combined:
        return FailureCategory.SCHEMA_REJECTION
    if stage_value == EpisodeStage.PARSE_TOOL_CALL or any(token in combined for token in (
        "jsondecodeerror", "malformed tool", "invalid tool", "empty function name",
    )):
        return FailureCategory.MALFORMED_TOOL_CALL
    if "structured output" in combined:
        return FailureCategory.INVALID_STRUCTURED_OUTPUT
    if "sqlite" in combined or "database is locked" in combined:
        return FailureCategory.SQLITE_FAILURE
    if stage_value == EpisodeStage.EVALUATE_STATE:
        return FailureCategory.EVALUATOR_FAILURE
    if "adapter" in combined:
        return FailureCategory.ADAPTER_FAILURE
    if stage_value in {EpisodeStage.EXECUTE_TOOL, EpisodeStage.CLASSIFY_TOOL, EpisodeStage.VALIDATE_ARGUMENTS}:
        return FailureCategory.AGENTDOJO_RUNTIME
    return FailureCategory.UNKNOWN


def is_retryable(category: FailureCategory | str) -> bool:
    return FailureCategory(category) in RETRYABLE_CATEGORIES


T = TypeVar("T")


def run_with_infrastructure_retry(
    operation: Callable[[], T],
    *,
    stage: EpisodeStage,
    max_retries: int = 1,
    on_failure: Callable[[BaseException, FailureCategory, int, bool], None] | None = None,
) -> T:
    """Run once plus at most one retry for temporary model infrastructure failures."""

    if max_retries not in (0, 1):
        raise ValueError("max_retries must be 0 or 1")
    attempt = 1
    while True:
        try:
            return operation()
        except BaseException as exc:
            category = classify_failure(exc, stage)
            will_retry = is_retryable(category) and attempt <= max_retries
            if on_failure is not None:
                on_failure(exc, category, attempt, will_retry)
            if not will_retry:
                raise
            attempt += 1


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


@dataclass
class EpisodeDiagnostic:
    scenario_id: str
    arm: str
    seed: int
    model_id: str
    completed: bool
    evaluable: bool
    failing_stage: str | None
    failure_category: str | None
    retryable: bool
    retry_count: int
    exception_type: str | None
    error: str | None
    validation_errors: list[str] = field(default_factory=list)
    model_attempts: list[dict[str, Any]] = field(default_factory=list)
    raw_model_responses: list[Any] = field(default_factory=list)
    parsed_tool_calls: list[Any] = field(default_factory=list)
    model_latency_ms: int = 0
    tool_latency_ms: int = 0
    approval_latency_ms: int = 0
    episode_latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    recorded_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


class TimedFunctionsRuntime:
    """Record tool durations and fail closed after an in-process overrun.

    Python cannot safely pre-empt an in-process state-changing function. The
    wrapper records the overrun after the call returns, raises a non-retryable
    timeout, and prevents subsequent pipeline work.
    """

    def __init__(self, runtime: Any, *, limit_seconds: float, timings: list[dict[str, Any]]) -> None:
        self._runtime = runtime
        self.functions = runtime.functions
        self._limit_seconds = limit_seconds
        self._timings = timings

    def run_function(self, env: Any, function: str, kwargs: Mapping[str, Any], raise_on_error: bool = False):
        started = time.perf_counter()
        error: str | None = None
        try:
            result, error = self._runtime.run_function(env, function, kwargs, raise_on_error=raise_on_error)
            return result, error
        finally:
            elapsed = time.perf_counter() - started
            self._timings.append({
                "stage": EpisodeStage.EXECUTE_TOOL.value,
                "function": function,
                "latency_ms": round(elapsed * 1000),
                "error": error,
            })
            if elapsed > self._limit_seconds:
                raise ToolTimeoutError(
                    f"tool {function!r} exceeded {self._limit_seconds:.3f}s limit "
                    f"({elapsed:.3f}s elapsed); outcome requires reconciliation"
                )


def build_episode_diagnostic(
    *,
    result: Mapping[str, Any],
    scenario_id: str,
    arm: str,
    seed: int,
    model_id: str,
    model_calls: list[dict[str, Any]],
    runtime_timings: list[dict[str, Any]],
) -> EpisodeDiagnostic:
    error = result.get("error")
    failed_calls = [call for call in model_calls if call.get("error")]
    validation_errors = [
        str(item["error"]) for item in runtime_timings
        if item.get("error") and "validation" in str(item["error"]).lower()
    ]
    if error and failed_calls:
        failing_stage = EpisodeStage.CALL_MODEL
        exception_type = failed_calls[-1].get("exception_type")
        category = FailureCategory(failed_calls[-1]["failure_category"])
    elif error:
        failing_stage = EpisodeStage.EVALUATE_STATE if "final-state" in str(error).lower() else EpisodeStage.EXECUTE_TOOL
        exception_type = str(error).split(":", 1)[0]
        category = classify_failure(str(error), failing_stage)
    else:
        failing_stage = None
        exception_type = None
        category = None
    retry_count = sum(1 for call in failed_calls if call.get("will_retry"))
    return EpisodeDiagnostic(
        scenario_id=scenario_id,
        arm=arm,
        seed=seed,
        model_id=model_id,
        completed=bool(result.get("completed")),
        evaluable=bool(result.get("completed")) and not bool(error),
        failing_stage=failing_stage.value if failing_stage else None,
        failure_category=category.value if category else None,
        retryable=is_retryable(category) if category else False,
        retry_count=retry_count,
        exception_type=exception_type,
        error=str(error) if error else None,
        validation_errors=validation_errors,
        model_attempts=model_calls,
        raw_model_responses=[call.get("raw_response") for call in model_calls if call.get("raw_response") is not None],
        parsed_tool_calls=[call.get("parsed_tool_calls") for call in model_calls if call.get("parsed_tool_calls")],
        model_latency_ms=sum(int(call.get("latency_ms", 0)) for call in model_calls),
        tool_latency_ms=sum(
            int(item.get("latency_ms", 0)) for item in runtime_timings
            if item.get("stage") == EpisodeStage.EXECUTE_TOOL.value
        ),
        approval_latency_ms=sum(
            int(item.get("latency_ms", 0)) for item in runtime_timings
            if item.get("stage") == EpisodeStage.OBTAIN_APPROVAL.value
        ),
        episode_latency_ms=int(result.get("latency_ms", 0)),
        prompt_tokens=int(result.get("prompt_tokens", 0)),
        completion_tokens=int(result.get("completion_tokens", 0)),
    )


class DiagnosticRecorder:
    """Retain per-episode JSON evidence plus aggregate CSV/JSON/Markdown reports."""

    CSV_FIELDS = (
        "scenario_id", "arm", "seed", "model_id", "failure_category", "failing_stage",
        "exception_type", "error", "retryable", "retry_count", "model_latency_ms",
        "tool_latency_ms", "approval_latency_ms", "episode_latency_ms", "prompt_tokens",
        "completion_tokens", "evidence_path",
    )

    def __init__(self, *, evidence_dir: str | Path, reports_dir: str | Path) -> None:
        self.evidence_dir = Path(evidence_dir)
        self.reports_dir = Path(reports_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.reports_dir / "episode_diagnostics.jsonl"
        self.episodes_csv_path = self.reports_dir / "episode_diagnostics.csv"
        self.csv_path = self.reports_dir / "failures.csv"
        self._lock = threading.Lock()

    @staticmethod
    def _safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def record(self, diagnostic: EpisodeDiagnostic) -> Path:
        payload = diagnostic.to_dict()
        name = f"{self._safe(diagnostic.scenario_id)}__{diagnostic.arm}__seed-{diagnostic.seed}.json"
        evidence_path = self.evidence_dir / name
        temporary = evidence_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(evidence_path)
        relative_evidence = str(evidence_path).replace("\\", "/")
        with self._lock:
            self._append_line(self.jsonl_path, json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            write_episode_header = not self.episodes_csv_path.exists()
            episode_row = {field: payload.get(field) for field in self.CSV_FIELDS}
            episode_row["evidence_path"] = relative_evidence
            with self.episodes_csv_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
                if write_episode_header:
                    writer.writeheader()
                writer.writerow(episode_row)
                handle.flush()
                os.fsync(handle.fileno())
            if diagnostic.failure_category:
                write_header = not self.csv_path.exists()
                row = {field: payload.get(field) for field in self.CSV_FIELDS}
                row["evidence_path"] = relative_evidence
                with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
        return evidence_path

    def load(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        return [json.loads(line) for line in self.jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def write_reports(self) -> dict[str, Any]:
        rows = self.load()
        failures = [row for row in rows if row.get("failure_category")]
        categories = Counter(row["failure_category"] for row in failures)
        evaluable = sum(bool(row.get("evaluable")) for row in rows)
        summary = {
            "schema_version": "week1-diagnostic-summary-v1",
            "generated_at": utc_now(),
            "scheduled_episodes": len(rows),
            "evaluable_episodes": evaluable,
            "evaluable_rate": evaluable / len(rows) if rows else 0.0,
            "failures": len(failures),
            "failure_categories": dict(sorted(categories.items())),
            "completion_target": 0.98,
            "completion_target_met": bool(rows) and evaluable / len(rows) >= 0.98,
            "retry_policy": "one retry only for MODEL_TIMEOUT or MODEL_SERVER; security decisions never retried",
        }
        (self.reports_dir / "failure_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        category_lines = [f"- `{name}`: {count}" for name, count in sorted(categories.items())]
        report = [
            "# Week 1 Runtime Diagnostic Report", "", f"Generated: {summary['generated_at']}", "",
            f"- Scheduled episodes: **{len(rows)}**",
            f"- Valid, evaluable outcomes: **{evaluable}/{len(rows)}**",
            f"- Runtime failures: **{len(failures)}**",
            f"- 98% completion criterion met: **{summary['completion_target_met']}**", "",
            "## Failure taxonomy", "", *(category_lines or ["- No failures recorded."]), "",
            "## Retry semantics", "",
            "Only temporary model timeout/server failures receive one controlled retry. "
            "Schema rejections, policy denials, invalid signatures, replay attempts, tool overruns, "
            "and other security-relevant outcomes are never retried.", "",
            "## Evidence", "",
            "Each episode has a JSON evidence record and a row in `episode_diagnostics.csv`. Failed episodes "
            "are also indexed in `failures.csv`; the complete stream is `episode_diagnostics.jsonl`.", "",
        ]
        (self.reports_dir / "FAILURE_ROOT_CAUSE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
        return summary
