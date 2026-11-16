"""Week 4 Day 5 exit checks: legitimate execution, forgery rejection,
replay rejection, duplicate/concurrent attempts, policy drift, and an
incident-investigation trace -- plus reconciliation and the Day 3 freeze.
"""
import threading
import time

import pytest

from pg.gateway import (
    ExecutionGateway, ForgedTokenError, TokenExpiredError, TokenScopeError,
    PolicyDriftError, ReplayRejected, FrozenPrincipalError, find_uncertain_executions,
)
from pg.store import DuplicateExecutionError
from pg.schemas import ActionState, SchemaError
from pg import audit


def _approved_send_email(service, approvals, **overrides):
    kwargs = dict(
        session_id="sess_1", principal_id="user_gaurang", user_goal="send an email",
        model_id="stub-model-v1", tool_name="send_email",
        arguments={"recipient": "mentor@example.com", "subject": "Status", "body": "All good."},
    )
    kwargs.update(overrides)
    outcome = service.propose_action(**kwargs)
    result = approvals.decide(outcome.proposal.proposal_id, reviewer_id="mentor",
                               session_token="sess-tok-mentor-001", decision="approve")
    return outcome, result


# -- Day 1-2: legitimate end-to-end execution ---------------------------------

def test_legitimate_execution_changes_world_and_records_receipt(service, approvals, gateway, world):
    outcome, result = _approved_send_email(service, approvals)
    exec_result = gateway.execute(outcome.proposal.proposal_id, result.approval_id)

    assert exec_result.receipt.result_status == "SUCCEEDED"
    assert len(world.mailbox) == 1
    assert world.mailbox[0]["recipient"] == "mentor@example.com"
    assert gateway._store.current_state(outcome.proposal.proposal_id) == ActionState.SUCCEEDED


def test_every_completed_write_links_proposal_policy_approval_receipt(service, approvals, gateway, store, verifier):
    outcome, result = _approved_send_email(service, approvals)
    gateway.execute(outcome.proposal.proposal_id, result.approval_id)

    t = audit.trace(store, verifier, outcome.proposal.proposal_id)
    assert t["proposal_signature_valid"] is True
    assert t["current_state"] == "SUCCEEDED"
    assert t["receipt"]["result_status"] == "SUCCEEDED"
    assert any(h["state"] == "APPROVED" for h in t["state_history"])


# -- Day 2: duplicate / concurrent execution attempts -------------------------

def test_second_execution_of_same_approval_is_rejected_as_replay(service, approvals, gateway, world):
    outcome, result = _approved_send_email(service, approvals)
    gateway.execute(outcome.proposal.proposal_id, result.approval_id)

    with pytest.raises(ReplayRejected):
        gateway.execute(outcome.proposal.proposal_id, result.approval_id)
    assert len(world.mailbox) == 1  # no second side effect


def test_concurrent_execution_attempts_only_one_succeeds(service, approvals, gateway, world):
    outcome, result = _approved_send_email(service, approvals)

    outcomes = []
    barrier = threading.Barrier(2)

    def _attempt():
        barrier.wait()
        try:
            gateway.execute(outcome.proposal.proposal_id, result.approval_id)
            outcomes.append("ok")
        except Exception as e:
            outcomes.append(type(e).__name__)

    threads = [threading.Thread(target=_attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count("ok") == 1
    assert len(world.mailbox) == 1  # exactly one side effect, never zero or two


# -- Day 1: forged/expired/scoped tokens --------------------------------------

def test_forged_approval_signature_rejected(service, approvals, gateway, store):
    outcome, result = _approved_send_email(service, approvals)
    # Tamper the stored approval's signature directly (simulating an
    # attacker with raw DB access), bypassing the append-only trigger via
    # the guarded consumed-only path is not possible -- so we drop the
    # trigger the same way the Week 2 integrity tests do.
    store._conn.executescript("DROP TRIGGER IF EXISTS approvals_consume_only;")
    store._conn.execute("UPDATE approvals SET signature='AAAAforged' WHERE approval_id=?",
                         (result.approval_id,))
    with pytest.raises(ForgedTokenError):
        gateway.execute(outcome.proposal.proposal_id, result.approval_id)


def test_expired_approval_rejected(service, approvals, gateway):
    outcome, _ = _approved_send_email(service, approvals)
    # Re-decide isn't allowed twice on the same proposal, so build a
    # fresh proposal and issue a 1-second token this time.
    outcome2 = service.propose_action(
        session_id="sess_1", principal_id="user_gaurang", user_goal="send an email",
        model_id="stub-model-v1", tool_name="send_email",
        arguments={"recipient": "mentor@example.com", "subject": "Status2", "body": "x"},
    )
    result2 = approvals.decide(outcome2.proposal.proposal_id, reviewer_id="mentor",
                                session_token="sess-tok-mentor-001", decision="approve",
                                ttl_seconds=1)
    time.sleep(1.2)
    with pytest.raises(TokenExpiredError):
        gateway.execute(outcome2.proposal.proposal_id, result2.approval_id)


def test_approval_for_a_different_proposal_is_scope_rejected(service, approvals, gateway):
    outcome_a, result_a = _approved_send_email(service, approvals, arguments={
        "recipient": "mentor@example.com", "subject": "A", "body": "a"})
    outcome_b, _ = _approved_send_email(service, approvals, arguments={
        "recipient": "mentor@example.com", "subject": "B", "body": "b"})
    # Approval A is validly signed and unconsumed, but bound to proposal A's
    # digest -- using it against proposal B (a changed/different action)
    # must fail even though both are legitimately-issued tokens.
    with pytest.raises(TokenScopeError):
        gateway.execute(outcome_b.proposal.proposal_id, result_a.approval_id)


# -- Day 1: policy rechecked live, not trusted from approval time ------------

def test_policy_drift_after_approval_blocks_execution(service, approvals, gateway, monkeypatch):
    outcome, result = _approved_send_email(service, approvals)

    import pg.gateway as gw
    from pg.policy import PolicyResult, Decision as D
    monkeypatch.setattr(gw, "policy_evaluate", lambda **kw: PolicyResult(D.DENY, "policy bundle downgraded"))

    with pytest.raises(PolicyDriftError):
        gateway.execute(outcome.proposal.proposal_id, result.approval_id)
    assert gateway._store.current_state(outcome.proposal.proposal_id) == ActionState.REVOKED


# -- Day 3: repeated-probe freeze ---------------------------------------------

def test_repeated_rejections_freeze_capability_minting(service, approvals, gateway):
    from pg.builder import ProposalRejected
    for _ in range(6):
        try:
            service.propose_action(
                session_id="sess_1", principal_id="user_gaurang", user_goal="send an email",
                model_id="stub-model-v1", tool_name="send_email",
                arguments={"recipient": "attacker@evil.example", "subject": "x", "body": "x"},
            )
        except ProposalRejected:
            pass

    outcome, result = _approved_send_email(service, approvals)
    with pytest.raises(FrozenPrincipalError):
        gateway.execute(outcome.proposal.proposal_id, result.approval_id)


# -- Day 2/5: reconciliation, never silent auto-retry -------------------------

def test_interrupted_execution_is_flagged_not_auto_retried(service, approvals, gateway, store, signer):
    outcome, result = _approved_send_email(service, approvals)
    # Simulate a crash *after* the state flips to EXECUTING but *before*
    # the receipt is written, by consuming the approval and moving state
    # directly (bypassing gateway.execute, which would run to completion).
    store.consume_approval(result.approval_id)
    store.set_state(outcome.proposal.proposal_id, ActionState.EXECUTING, signer=signer)

    uncertain = find_uncertain_executions(store)
    assert any(u["proposal_id"] == outcome.proposal.proposal_id for u in uncertain)

    # And gateway.execute on the same approval now correctly reports replay
    # (already consumed) rather than silently re-running the side effect.
    with pytest.raises(ReplayRejected):
        gateway.execute(outcome.proposal.proposal_id, result.approval_id)


def test_missing_or_unavailable_approval_fails_closed(gateway, service):
    outcome = service.propose_action(
        session_id="sess_1", principal_id="user_gaurang", user_goal="send an email",
        model_id="stub-model-v1", tool_name="send_email",
        arguments={"recipient": "mentor@example.com", "subject": "x", "body": "x"},
    )
    with pytest.raises(SchemaError):
        gateway.execute(outcome.proposal.proposal_id, "no-such-approval-id")


def test_update_record_and_summarize_adapters_also_work(service, approvals, gateway, world):
    o1 = service.propose_action(
        session_id="sess_1", principal_id="user_gaurang", user_goal="update a record",
        model_id="stub-model-v1", tool_name="update_record",
        arguments={"record_id": "rec_1001", "field": "status", "new_value": "closed"},
    )
    r1 = approvals.decide(o1.proposal.proposal_id, reviewer_id="mentor",
                           session_token="sess-tok-mentor-001", decision="approve")
    exec1 = gateway.execute(o1.proposal.proposal_id, r1.approval_id)
    assert exec1.receipt.result_status == "SUCCEEDED"
    assert world.records["rec_1001"]["status"] == "closed"

    o2 = service.propose_action(
        session_id="sess_1", principal_id="user_gaurang", user_goal="summarize the message",
        model_id="stub-model-v1", tool_name="summarize",
        arguments={"message_id": "msg_1", "max_words": 5},
    )
    r2 = approvals.auto_approve(o2.proposal.proposal_id)
    exec2 = gateway.execute(o2.proposal.proposal_id, r2.approval_id)
    assert exec2.receipt.result_status == "SUCCEEDED"
    assert exec2.tool_result["summary"]
