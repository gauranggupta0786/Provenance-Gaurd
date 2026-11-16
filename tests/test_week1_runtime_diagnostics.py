import json
import time

import pytest

from pg.diagnostics import (
    DiagnosticRecorder,
    EpisodeDiagnostic,
    EpisodeStage,
    FailureCategory,
    RuntimeLimits,
    TimedFunctionsRuntime,
    ToolTimeoutError,
    build_episode_diagnostic,
    classify_failure,
    is_retryable,
    run_with_infrastructure_retry,
)
from pg.week5_runner import compact_episode_evidence_dir


@pytest.mark.parametrize(
    ("error", "stage", "expected"),
    [
        (TimeoutError("model timed out"), EpisodeStage.CALL_MODEL, FailureCategory.MODEL_TIMEOUT),
        ("connection refused by Ollama", EpisodeStage.CALL_MODEL, FailureCategory.MODEL_SERVER),
        ("maximum context length exceeded", EpisodeStage.CALL_MODEL, FailureCategory.CONTEXT_OVERFLOW),
        ("ValidationError: missing recipient", EpisodeStage.VALIDATE_ARGUMENTS, FailureCategory.SCHEMA_REJECTION),
        ("ProposalRejected: policy denied", EpisodeStage.EVALUATE_POLICY, FailureCategory.POLICY_REJECTION),
        ("invalid signature", EpisodeStage.GATEWAY_CHECK, FailureCategory.SIGNATURE_REJECTION),
        ("approval already consumed: replay", EpisodeStage.GATEWAY_CHECK, FailureCategory.REPLAY_REJECTION),
        ("sqlite3.OperationalError: database is locked", EpisodeStage.STORE_RECEIPT, FailureCategory.SQLITE_FAILURE),
    ],
)
def test_failure_taxonomy_is_explicit(error, stage, expected):
    assert classify_failure(error, stage) == expected


def test_only_temporary_model_infrastructure_failures_are_retryable():
    assert is_retryable(FailureCategory.MODEL_TIMEOUT)
    assert is_retryable(FailureCategory.MODEL_SERVER)
    assert not is_retryable(FailureCategory.SCHEMA_REJECTION)
    assert not is_retryable(FailureCategory.POLICY_REJECTION)
    assert not is_retryable(FailureCategory.REPLAY_REJECTION)


def test_infrastructure_failure_receives_exactly_one_retry():
    attempts = []

    def operation():
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("temporary model timeout")
        return "ok"

    assert run_with_infrastructure_retry(operation, stage=EpisodeStage.CALL_MODEL) == "ok"
    assert len(attempts) == 2


def test_security_rejection_is_never_retried():
    attempts = []

    def operation():
        attempts.append(1)
        raise ValueError("ProposalRejected: policy denied")

    with pytest.raises(ValueError, match="policy denied"):
        run_with_infrastructure_retry(operation, stage=EpisodeStage.EVALUATE_POLICY)
    assert len(attempts) == 1


def test_runtime_limits_reject_invalid_values_and_multiple_retries():
    with pytest.raises(ValueError, match="model_seconds"):
        RuntimeLimits(model_seconds=0)
    with pytest.raises(ValueError, match="infrastructure_retries"):
        RuntimeLimits(infrastructure_retries=2)


def test_timed_runtime_records_tool_latency_and_fails_closed_after_overrun():
    class SlowRuntime:
        functions = {}

        @staticmethod
        def run_function(_env, _function, _kwargs, raise_on_error=False):
            time.sleep(0.01)
            return "result", None

    timings = []
    runtime = TimedFunctionsRuntime(SlowRuntime(), limit_seconds=0.001, timings=timings)
    with pytest.raises(ToolTimeoutError, match="requires reconciliation"):
        runtime.run_function(None, "slow", {})
    assert timings[0]["stage"] == EpisodeStage.EXECUTE_TOOL.value
    assert timings[0]["latency_ms"] >= 1


def _diagnostic(*, failed=False):
    return EpisodeDiagnostic(
        scenario_id="heldout-001",
        arm="protected",
        seed=42,
        model_id="test-model",
        completed=not failed,
        evaluable=not failed,
        failing_stage=EpisodeStage.CALL_MODEL.value if failed else None,
        failure_category=FailureCategory.MODEL_TIMEOUT.value if failed else None,
        retryable=failed,
        retry_count=1 if failed else 0,
        exception_type="TimeoutError" if failed else None,
        error="model timed out" if failed else None,
        raw_model_responses=[{"id": "response-1"}],
        parsed_tool_calls=[[{"function": "send_email"}]],
    )


def test_recorder_retains_jsonl_csv_episode_evidence_and_reports(tmp_path):
    recorder = DiagnosticRecorder(evidence_dir=tmp_path / "evidence", reports_dir=tmp_path / "reports")
    evidence = recorder.record(_diagnostic(failed=True))
    recorder.record(_diagnostic(failed=False))
    summary = recorder.write_reports()

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["raw_model_responses"][0]["id"] == "response-1"
    assert payload["parsed_tool_calls"][0][0]["function"] == "send_email"
    assert len(recorder.load()) == 2
    assert len(recorder.episodes_csv_path.read_text(encoding="utf-8").splitlines()) == 3
    assert "MODEL_TIMEOUT" in recorder.csv_path.read_text(encoding="utf-8")
    assert summary["scheduled_episodes"] == 2
    assert summary["evaluable_episodes"] == 1
    assert (tmp_path / "reports" / "FAILURE_ROOT_CAUSE_REPORT.md").exists()


def test_episode_diagnostic_captures_recovered_retry_raw_response_timings_and_tokens():
    result = {
        "completed": True,
        "error": None,
        "latency_ms": 123,
        "prompt_tokens": 17,
        "completion_tokens": 4,
    }
    model_calls = [
        {
            "error": "timed out", "will_retry": True, "failure_category": "MODEL_TIMEOUT",
            "exception_type": "TimeoutError", "latency_ms": 50,
        },
        {
            "error": None, "will_retry": False, "latency_ms": 60,
            "raw_response": {"id": "ok"}, "parsed_tool_calls": [{"function": "list_files"}],
        },
    ]
    diagnostic = build_episode_diagnostic(
        result=result,
        scenario_id="heldout-002",
        arm="baseline",
        seed=7,
        model_id="model",
        model_calls=model_calls,
        runtime_timings=[{"stage": "EXECUTE_TOOL", "latency_ms": 9, "error": None}],
    )
    assert diagnostic.failure_category is None
    assert diagnostic.evaluable
    assert diagnostic.retry_count == 1
    assert diagnostic.model_attempts[0]["failure_category"] == FailureCategory.MODEL_TIMEOUT.value
    assert diagnostic.raw_model_responses == [{"id": "ok"}]
    assert diagnostic.parsed_tool_calls == [[{"function": "list_files"}]]
    assert diagnostic.model_latency_ms == 110
    assert diagnostic.tool_latency_ms == 9
    assert diagnostic.prompt_tokens == 17


def test_compact_evidence_path_avoids_windows_max_path_and_preserves_identity(tmp_path):
    episode_dir = compact_episode_evidence_dir(
        tmp_path, "heldout-benign-001", 42, "protected",
    )
    assert episode_dir.name.startswith("r-")
    assert len(episode_dir.name) == 14
    metadata = json.loads((episode_dir / "episode_identity.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "episode-evidence-location-v1"
    assert metadata["scenario_id"] == "heldout-benign-001"
    assert len(str(episode_dir / "events.db.lifecycle.sqlite")) < 260
