from __future__ import annotations

from pathlib import Path

from pg.lifecycle_graph import LifecycleCheckpointStore, LifecycleGraph, LifecycleServices
from pg.trusted_models import (
    AgentRunState, EvaluationRecord, EvaluationStatus, ExecutionRecord, ExecutionStatus,
    CallStatus, ModelCallRecord, PolicyDecision, PolicyOutcome, ReceiptRecord, RunStatus, ToolIntent,
    WorkflowNode,
)


HEX = "a" * 64


def initial(run_id="run-1"):
    return AgentRunState(
        graph_run_id=run_id, scenario_id="scenario-1", arm="protected", seed=42,
        session_id="session-1", principal_id="trusted-user", model_profile_sha256=HEX,
        scenario_evidence_ref="evidence://scenario-1",
    )


def services(*, approve=True, timeout_once=False, evaluator_fails=False, writes=None):
    attempts = {"model": 0}
    writes = writes if writes is not None else []

    def call_model(state):
        attempts["model"] += 1
        if timeout_once and attempts["model"] == 1:
            raise TimeoutError("model deadline")
        return ModelCallRecord(
            call_id=f"call-{attempts['model']}", graph_run_id=state.graph_run_id,
            profile_sha256=HEX, stage="CALL_MODEL", attempt=min(attempts["model"], 2),
            request_evidence_ref="evidence://request", response_evidence_ref="evidence://response",
            latency_ms=1, status=CallStatus.SUCCEEDED,
        )

    def parse(state):
        return ToolIntent(
            call_id="tool-1", tool_name="send_email",
            arguments={"recipient": "a@example.com", "subject": "s", "body": "b"},
            raw_arguments={"recipient": "a@example.com", "subject": "s", "body": "b"},
            parse_error=None,
        )

    def execute(state):
        writes.append(state.proposal_id)
        return (
            ExecutionRecord(
                attempt_id="attempt-1", proposal_id=state.proposal_id, approval_id=state.approval_id,
                idempotency_key=HEX, adapter="gateway", status=ExecutionStatus.SUCCEEDED,
                started_at="2026-10-09T00:00:00Z", completed_at="2026-10-09T00:00:01Z",
            ),
            ReceiptRecord(
                attempt_id="attempt-1", proposal_id=state.proposal_id, idempotency_key=HEX,
                action_digest=HEX, result_status="SUCCEEDED", result_digest="b" * 64,
                created_at="2026-10-09T00:00:01Z",
            ),
        )

    def evaluate(state):
        if evaluator_fails:
            raise RuntimeError("evaluator unavailable")
        return EvaluationRecord(
            graph_run_id=state.graph_run_id, scenario_id=state.scenario_id, arm=state.arm,
            seed=state.seed, status=EvaluationStatus.EVALUABLE, completed=True,
            benign_success=True, final_state_digest="c" * 64, evaluator_version="evaluator-v1",
        )

    return LifecycleServices(
        call_model=call_model,
        parse_tool_intent=parse,
        label_sources=lambda state: (),
        evaluate_policy=lambda state: PolicyOutcome(
            phase="proposal", decision=PolicyDecision.REQUIRE_APPROVAL, reason="write requires approval",
            policy_version="policy-v1", evaluated_at="2026-10-09T00:00:00Z",
        ),
        build_proposal=lambda state: "proposal-1",
        obtain_approval=lambda state: "approval-1" if approve else None,
        execute_read=lambda state: None,
        execute_write=execute,
        evaluate_final=evaluate,
    )


def test_visible_typed_path_and_node_diagnostics(tmp_path: Path):
    checkpoint = LifecycleCheckpointStore(tmp_path / "graph.db")
    graph = LifecycleGraph(services(), checkpoint)
    result = graph.invoke(initial())
    assert result.status is RunStatus.SUCCEEDED
    assert result.path[0] is WorkflowNode.LOAD_SCENARIO
    assert result.path[-1] is WorkflowNode.END
    assert WorkflowNode.GATEWAY_REVALIDATE in result.path
    assert WorkflowNode.CONSUME_AND_RESERVE in result.path
    assert len(checkpoint.diagnostics("run-1")) == len(result.path)
    assert "BUILD_AND_SIGN_PROPOSAL" in graph.mermaid()


def test_timeout_retries_once_on_validated_route(tmp_path: Path):
    graph = LifecycleGraph(services(timeout_once=True), LifecycleCheckpointStore(tmp_path / "graph.db"))
    result = graph.invoke(initial())
    assert result.status is RunStatus.SUCCEEDED
    assert result.retry_count == 1
    assert result.path.count(WorkflowNode.CALL_MODEL) == 2


def test_approval_interruption_can_resume_without_duplicate_write(tmp_path: Path):
    writes = []
    checkpoint = LifecycleCheckpointStore(tmp_path / "graph.db")
    first = LifecycleGraph(services(approve=False, writes=writes), checkpoint).invoke(initial())
    assert first.status is RunStatus.INTERRUPTED
    assert writes == []
    resumed = LifecycleGraph(services(approve=True, writes=writes), checkpoint).resume("run-1")
    assert resumed.status is RunStatus.SUCCEEDED
    assert writes == ["proposal-1"]
    terminal = LifecycleGraph(services(approve=True, writes=writes), checkpoint).resume("run-1")
    assert terminal.status is RunStatus.SUCCEEDED
    assert writes == ["proposal-1"]


def test_evaluator_failure_is_fail_closed_and_checkpointed(tmp_path: Path):
    checkpoint = LifecycleCheckpointStore(tmp_path / "graph.db")
    result = LifecycleGraph(services(evaluator_fails=True), checkpoint).invoke(initial())
    assert result.status is RunStatus.FAILED
    assert result.current_node is WorkflowNode.EVALUATE_FINAL_STATE
    assert "evaluator unavailable" in result.last_error
    assert checkpoint.load("run-1").status is RunStatus.FAILED


def test_adapter_failure_stops_before_receipt_and_is_not_retried(tmp_path: Path):
    configured = services()
    calls = []

    def fail_adapter(state):
        calls.append(state.proposal_id)
        raise RuntimeError("adapter failure")

    configured.execute_write = fail_adapter
    checkpoint = LifecycleCheckpointStore(tmp_path / "graph.db")
    graph = LifecycleGraph(configured, checkpoint)
    result = graph.invoke(initial())
    assert result.status is RunStatus.FAILED
    assert result.current_node is WorkflowNode.EXECUTE_ADAPTER
    assert result.receipt is None
    assert calls == ["proposal-1"]
    resumed = graph.resume("run-1")
    assert resumed.status is RunStatus.FAILED
    assert calls == ["proposal-1"]
