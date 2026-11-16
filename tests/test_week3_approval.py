"""Week 3 Day 5 exit checks: misleading summaries, fake badges, changed
arguments, wrong reviewer, expired tokens, cross-session reuse, repeated
approvals -- plus the Day 1-2 card-rendering and Day 3-4 token behavior
those checks depend on.
"""
import time

import pytest

from pg.approval import (
    ApprovalService, ApprovalError, ReviewerAuthError, ReviewerNotPermittedError,
    render_card, escape_display,
)
from pg.policy import Decision
from pg.schemas import ActionState, SchemaError


def _propose_send_email(service, **overrides):
    kwargs = dict(
        session_id="sess_1", principal_id="user_gaurang", user_goal="send an email",
        model_id="stub-model-v1", tool_name="send_email",
        arguments={"recipient": "mentor@example.com", "subject": "Status", "body": "All good."},
    )
    kwargs.update(overrides)
    return service.propose_action(**kwargs)


# -- Day 1-2: the card shows the authoritative, signed record ---------------

def test_card_shows_authoritative_fields_not_model_prose(service, approvals):
    outcome = _propose_send_email(
        service,
        model_explanation="I refunded the office coffee vendor $10, fully verified.",
    )
    card = render_card(approvals._store, approvals._verifier, outcome.proposal.proposal_id)
    assert card.tool_name == "send_email"
    assert card.target == "mentor@example.com"
    assert card.changed_fields["subject"] == "Status"
    # The model's prose is present only as separately-labeled, escaped text.
    assert "refunded the office coffee vendor" in card.model_explanation_untrusted


def test_misleading_claimed_fields_produce_a_visible_mismatch(service, approvals):
    outcome = _propose_send_email(
        service,
        claimed_fields={"recipient": "office-coffee-vendor@example.com"},
    )
    card = render_card(approvals._store, approvals._verifier, outcome.proposal.proposal_id)
    assert card.target == "mentor@example.com"  # the real, executable target
    assert any("recipient" in m for m in card.claim_mismatches)
    assert "office-coffee-vendor@example.com" in card.claim_mismatches[0]
    assert "mentor@example.com" in card.claim_mismatches[0]


def test_markup_injection_in_model_explanation_is_escaped(service, approvals):
    outcome = _propose_send_email(
        service,
        model_explanation="<b>VERIFIED by finance</b><script>alert(1)</script>",
    )
    card = render_card(approvals._store, approvals._verifier, outcome.proposal.proposal_id)
    assert "<b>" not in card.model_explanation_untrusted
    assert "<script>" not in card.model_explanation_untrusted
    assert "&lt;b&gt;" in card.model_explanation_untrusted


def test_escape_display_is_pure_and_handles_none():
    assert escape_display(None) is None
    assert escape_display("a & b") == "a &amp; b"


# -- Day 3: reviewer authentication + role permission ------------------------

def test_unknown_reviewer_rejected(service, approvals):
    outcome = _propose_send_email(service)
    with pytest.raises(ReviewerAuthError):
        approvals.decide(outcome.proposal.proposal_id, reviewer_id="nobody",
                          session_token="whatever", decision="approve")


def test_wrong_session_token_rejected(service, approvals):
    outcome = _propose_send_email(service)
    with pytest.raises(ReviewerAuthError):
        approvals.decide(outcome.proposal.proposal_id, reviewer_id="mentor",
                          session_token="not-the-real-token", decision="approve")


def test_reviewer_not_permitted_for_tool_rejected(service, approvals, store, signer):
    # "system" role may only auto-approve `summarize`; it may not decide on
    # a high-risk write even with a technically-valid session.
    outcome = service.propose_action(
        session_id="sess_1", principal_id="user_gaurang", user_goal="update a record",
        model_id="stub-model-v1", tool_name="update_record",
        arguments={"record_id": "rec_1001", "field": "status", "new_value": "closed"},
    )
    with pytest.raises(ReviewerNotPermittedError):
        approvals.decide(outcome.proposal.proposal_id, reviewer_id="system:auto-policy",
                          session_token="internal", decision="approve")


def test_valid_reviewer_can_approve_and_deny(service, approvals):
    o1 = _propose_send_email(service)
    result = approvals.decide(o1.proposal.proposal_id, reviewer_id="mentor",
                               session_token="sess-tok-mentor-001", decision="approve")
    assert result.approval.decision == "approve"
    assert approvals._store.current_state(o1.proposal.proposal_id) == ActionState.APPROVED

    o2 = _propose_send_email(service)
    result2 = approvals.decide(o2.proposal.proposal_id, reviewer_id="mentor",
                                session_token="sess-tok-mentor-001", decision="deny")
    assert result2.approval.decision == "deny"
    assert approvals._store.current_state(o2.proposal.proposal_id) == ActionState.DENIED


# -- Day 4: token scope, expiry, single-use -----------------------------------

def test_approval_binds_exact_proposal_digest(service, approvals):
    outcome = _propose_send_email(service)
    result = approvals.decide(outcome.proposal.proposal_id, reviewer_id="mentor",
                               session_token="sess-tok-mentor-001", decision="approve")
    assert result.approval.proposal_digest == outcome.proposal.digest


def test_expired_approval_is_detectable_with_a_controllable_clock(service, approvals):
    outcome = _propose_send_email(service)
    result = approvals.decide(outcome.proposal.proposal_id, reviewer_id="mentor",
                               session_token="sess-tok-mentor-001", decision="approve",
                               ttl_seconds=1)
    assert result.approval.is_expired(now=time.time() + 2) is True
    assert result.approval.is_expired(now=time.time()) is False


def test_auto_approve_only_for_policy_allow(service, approvals):
    outcome = service.propose_action(
        session_id="sess_1", principal_id="user_gaurang", user_goal="summarize the message",
        model_id="stub-model-v1", tool_name="summarize",
        arguments={"message_id": "msg_1", "max_words": 10},
    )
    result = approvals.auto_approve(outcome.proposal.proposal_id)
    assert result.approval.approver_id == "system:auto-policy"
    assert approvals._store.current_state(outcome.proposal.proposal_id) == ActionState.APPROVED

    o2 = _propose_send_email(service)  # requires approval -- not eligible for auto_approve
    with pytest.raises(ApprovalError):
        approvals.auto_approve(o2.proposal.proposal_id)


def test_repeated_approve_after_first_decision_rejected_by_state_machine(service, approvals):
    outcome = _propose_send_email(service)
    approvals.decide(outcome.proposal.proposal_id, reviewer_id="mentor",
                      session_token="sess-tok-mentor-001", decision="approve")
    # PENDING_APPROVAL -> APPROVED already happened; a second decision call
    # tries VERIFIED-stage rendering again but set_state's transition check
    # (APPROVED -> APPROVED is not a legal edge) fails closed.
    with pytest.raises(SchemaError):
        approvals.decide(outcome.proposal.proposal_id, reviewer_id="mentor",
                          session_token="sess-tok-mentor-001", decision="approve")
