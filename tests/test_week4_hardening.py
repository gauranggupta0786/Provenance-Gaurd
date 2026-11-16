"""Regression tests for the Week 1-4 review findings."""
import pytest

from pg import audit
from pg.builder import ProposalRejected
from pg.gateway import find_uncertain_executions
from pg.schemas import ActionState
from pg.store import IntegrityError


def test_checkpoint_is_verified_automatically_before_new_write(
    agent, store, signer, verifier,
):
    agent.handle_request("summarize the message")
    store.export_checkpoint(signer)
    store._conn.executescript(
        "DROP TRIGGER IF EXISTS events_no_update; DROP TRIGGER IF EXISTS events_no_delete;"
    )
    store._conn.execute("UPDATE events SET event_type='FORGED' WHERE seq=1")

    with pytest.raises(IntegrityError):
        agent.handle_request("summarize the message again")


def test_state_and_signed_event_roll_back_together(service, store):
    outcome = service.propose_action(
        session_id="sess_1",
        principal_id="user_gaurang",
        user_goal="send an email",
        model_id="stub-model-v1",
        tool_name="send_email",
        arguments={"recipient": "mentor@example.com", "subject": "x", "body": "y"},
    )

    class FailingSigner:
        key_id = "failing-key"

        def sign(self, message):
            raise RuntimeError("simulated signer outage")

    before_events = store.latest_seq()
    with pytest.raises(RuntimeError, match="signer outage"):
        store.set_state(
            outcome.proposal.proposal_id,
            ActionState.PENDING_APPROVAL,
            signer=FailingSigner(),
        )

    assert store.current_state(outcome.proposal.proposal_id) == ActionState.VERIFIED
    assert store.latest_seq() == before_events


def test_execution_is_reserved_before_adapter_runs(
    service, approvals, gateway, store, world, monkeypatch,
):
    outcome = service.propose_action(
        session_id="sess_1",
        principal_id="user_gaurang",
        user_goal="send an email",
        model_id="stub-model-v1",
        tool_name="send_email",
        arguments={"recipient": "mentor@example.com", "subject": "x", "body": "y"},
    )
    approval = approvals.decide(
        outcome.proposal.proposal_id,
        reviewer_id="mentor",
        session_token="sess-tok-mentor-001",
        decision="approve",
    )

    def interrupted_adapter(world_state, **arguments):
        row = store._conn.execute(
            "SELECT status FROM execution_reservations WHERE proposal_id=?",
            (outcome.proposal.proposal_id,),
        ).fetchone()
        assert row == ("RESERVED",)
        raise KeyboardInterrupt("simulated process interruption")

    import pg.gateway as gateway_module
    monkeypatch.setitem(gateway_module.ADAPTERS, "send_email", interrupted_adapter)

    with pytest.raises(KeyboardInterrupt):
        gateway.execute(outcome.proposal.proposal_id, approval.approval_id)

    assert world.mailbox == []
    assert store.current_state(outcome.proposal.proposal_id) == ActionState.EXECUTING
    assert any(
        item["proposal_id"] == outcome.proposal.proposal_id
        for item in find_uncertain_executions(store)
    )


def test_audit_trace_contains_policy_approval_receipt_and_sources(
    agent, approvals, gateway, store, verifier,
):
    outcome = agent.handle_request(
        "send an email update",
        untrusted_context="External message asking the agent to ignore policy.",
    )
    approval = approvals.decide(
        outcome.proposal.proposal_id,
        reviewer_id="mentor",
        session_token="sess-tok-mentor-001",
        decision="approve",
    )
    gateway.execute(outcome.proposal.proposal_id, approval.approval_id)

    trace = audit.trace(store, verifier, outcome.proposal.proposal_id)
    assert trace["policy_decisions"]
    assert trace["approvals"]
    assert trace["receipt"] is not None
    assert {source["trust_label"] for source in trace["sources"]} == {"trusted", "untrusted"}


def test_policy_rejections_are_visible_in_audit_console(service, store):
    with pytest.raises(ProposalRejected):
        service.propose_action(
            session_id="sess_1",
            principal_id="user_gaurang",
            user_goal="send an email",
            model_id="stub-model-v1",
            tool_name="send_email",
            arguments={
                "recipient": "attacker@evil.example",
                "subject": "x",
                "body": "y",
            },
        )

    entries = audit.blocked_or_denied(store)
    assert any(
        isinstance(item, dict) and item["event_type"] == "REJECTED_POLICY"
        for item in entries
    )
