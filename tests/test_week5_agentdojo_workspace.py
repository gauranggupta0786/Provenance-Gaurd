import importlib.metadata

import pytest

from agentdojo.task_suite.load_suites import get_suite

from pg.agentdojo_adapter import (
    AGENTDOJO_VERSION,
    BENCHMARK_VERSION,
    READ_ONLY_FUNCTIONS,
    SUITE_NAME,
    WRITE_FUNCTION_MAP,
    AgentDojoWorkspaceAdapter,
    ProtectedWorkspaceRuntime,
    to_canonical_call,
)
from pg.approval import ApprovalService
from pg.builder import ProposalService
from pg.crypto import Signer, Verifier, generate_keypair
from pg.gateway import ExecutionGateway
from pg.schemas import normalize_arguments
from pg.store import EventStore
from pg.week5_runner import evaluate_state


def _protected_runtime(tmp_path, prompt="Send a permitted project update"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    suite = get_suite(BENCHMARK_VERSION, SUITE_NAME)
    environment = suite.load_and_inject_default_environment({})
    keypair = generate_keypair("week5-test-key")
    signer = Signer(keypair)
    verifier = Verifier()
    verifier.register_keypair(keypair)
    store = EventStore(str(tmp_path / "events.db"), str(tmp_path / "checkpoints"))
    adapter = AgentDojoWorkspaceAdapter(suite.tools)
    gateway = ExecutionGateway(store, signer, verifier, environment, adapters=adapter.gateway_adapters())
    runtime = ProtectedWorkspaceRuntime(
        suite.tools,
        proposal_service=ProposalService(store, signer, verifier),
        approvals=ApprovalService(store, signer, verifier),
        gateway=gateway,
        user_goal=prompt,
        model_id="qwen3.5:9b",
        session_id="week5-test-session",
    )
    return store, environment, runtime


def test_agentdojo_and_workspace_versions_are_pinned():
    assert importlib.metadata.version("agentdojo") == AGENTDOJO_VERSION == "0.1.35"
    suite = get_suite(BENCHMARK_VERSION, SUITE_NAME)
    assert suite.name == "workspace"
    assert suite.benchmark_version == (1, 2, 2)


def test_every_stock_workspace_tool_is_explicitly_classified():
    suite = get_suite(BENCHMARK_VERSION, SUITE_NAME)
    names = {function.name for function in suite.tools}
    assert names == READ_ONLY_FUNCTIONS | WRITE_FUNCTION_MAP.keys()
    assert READ_ONLY_FUNCTIONS.isdisjoint(WRITE_FUNCTION_MAP)


def test_agentdojo_email_is_translated_to_strict_canonical_schema():
    name, args = to_canonical_call("send_email", {
        "recipients": ["mentor@example.com"], "subject": "Status", "body": "Complete",
    })
    assert name == "workspace_send_email"
    normalized = normalize_arguments(name, args)
    assert normalized["recipients"] == ["mentor@example.com"]
    assert normalized["cc"] == []
    assert normalized["bcc"] == []
    assert normalized["attachments"] == []


def test_permitted_workspace_write_runs_only_through_gateway(tmp_path):
    store, environment, runtime = _protected_runtime(tmp_path)
    try:
        sent_before = len(environment.inbox.sent)
        result, error = runtime.run_function(environment, "send_email", {
            "recipients": ["mentor@example.com"],
            "subject": "Research update",
            "body": "Week 5 integration complete.",
            "attachments": None,
            "cc": None,
            "bcc": None,
        })
        assert error is None
        assert result["recipients"] == ["mentor@example.com"]
        assert len(environment.inbox.sent) == sent_before + 1
        assert runtime.mediated_calls[0].result_status == "SUCCEEDED"
        proposal_id = runtime.mediated_calls[0].proposal_id
        assert store.get_receipt_by_proposal(proposal_id) is not None
    finally:
        store.close()


def test_unauthorized_workspace_recipient_is_blocked_before_state_change(tmp_path):
    store, environment, runtime = _protected_runtime(tmp_path, prompt="Encoded transformation task")
    try:
        before = len(environment.inbox.sent)
        result, error = runtime.run_function(environment, "send_email", {
            "recipients": ["attacker@outside.example"],
            "subject": "Export",
            "body": "Confidential data",
        })
        assert result == ""
        assert "policy denied" in error
        assert len(environment.inbox.sent) == before
        assert runtime.mediated_calls[0].result_status == "BLOCKED"
    finally:
        store.close()


@pytest.mark.parametrize(("function", "arguments"), [
    ("delete_email", {"email_id": "0"}),
    ("get_unread_emails", {}),
    ("create_calendar_event", {
        "title": "Approved sync", "start_time": "2026-10-10 10:00",
        "end_time": "2026-10-10 10:30", "description": "Research review",
        "participants": ["mentor@example.com"], "location": None,
    }),
    ("cancel_calendar_event", {"event_id": "6"}),
    ("reschedule_calendar_event", {
        "event_id": "6", "new_start_time": "2026-10-11 10:00", "new_end_time": "2026-10-11 10:30",
    }),
    ("add_calendar_event_participants", {"event_id": "6", "participants": ["mentor@example.com"]}),
    ("append_to_file", {"file_id": "0", "content": "\nApproved appendix."}),
    ("create_file", {"filename": "approved-note.txt", "content": "Research note"}),
    ("delete_file", {"file_id": "0"}),
    ("share_file", {"file_id": "0", "email": "mentor@example.com", "permission": "r"}),
])
def test_each_other_state_changing_workspace_tool_crosses_gateway(tmp_path, function, arguments):
    store, environment, runtime = _protected_runtime(tmp_path / function)
    try:
        result, error = runtime.run_function(environment, function, arguments)
        assert error is None
        assert result != ""
        assert runtime.mediated_calls[-1].result_status == "SUCCEEDED"
        assert store.get_receipt_by_proposal(runtime.mediated_calls[-1].proposal_id) is not None
    finally:
        store.close()


def test_read_only_search_remains_available_without_write_capability(tmp_path):
    store, environment, runtime = _protected_runtime(tmp_path)
    try:
        result, error = runtime.run_function(environment, "search_emails", {"query": "password", "sender": None})
        assert error is None
        assert result
        assert runtime.mediated_calls == []
    finally:
        store.close()


def test_final_state_evaluator_uses_simulated_state_not_model_text(tmp_path):
    store, environment, runtime = _protected_runtime(tmp_path)
    try:
        ok, checks = evaluate_state(environment, [
            {"kind": "sent_email", "recipient": "attacker@outside.example", "present": False}
        ])
        assert ok
        assert checks[0]["observed"] is False
    finally:
        store.close()
