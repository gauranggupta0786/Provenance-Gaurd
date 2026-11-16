"""Week 1 exit checks (see docs/WEEK1_DELIVERABLES.md):

- One valid proposal for each of the three supported actions.
- Expected rejection for malformed, unknown-tool, and unauthorized-target examples.
- A (stubbed) model produces one structured tool request.
"""
import pytest

from pg.builder import ProposalRejected
from pg.schemas import SchemaError, build_proposal


VALID_CASES = [
    ("summarize", {"message_id": "msg_1", "max_words": 10}),
    ("send_email", {"recipient": "mentor@example.com", "subject": "Hi", "body": "Status update."}),
    ("update_record", {"record_id": "rec_1001", "field": "status", "new_value": "closed"}),
]


@pytest.mark.parametrize("tool_name,arguments", VALID_CASES)
def test_valid_proposal_for_each_supported_action(service, tool_name, arguments):
    outcome = service.propose_action(
        session_id="sess_1", principal_id="user_gaurang", user_goal="do the thing",
        model_id="stub-model-v1", tool_name=tool_name, arguments=arguments,
    )
    p = outcome.proposal
    assert p.tool_name == tool_name
    assert p.digest is not None and p.signature is not None
    assert p.proposal_id


def test_malformed_request_rejected_unknown_tool(service):
    with pytest.raises(ProposalRejected):
        service.propose_action(
            session_id="sess_1", principal_id="user_gaurang", user_goal="do a bad thing",
            model_id="stub-model-v1", tool_name="delete_everything", arguments={},
        )


def test_malformed_request_rejected_missing_field(service):
    with pytest.raises(ProposalRejected):
        service.propose_action(
            session_id="sess_1", principal_id="user_gaurang", user_goal="send an email",
            model_id="stub-model-v1", tool_name="send_email",
            arguments={"recipient": "mentor@example.com", "subject": "Hi"},  # missing body
        )


def test_malformed_request_rejected_extra_field(service):
    with pytest.raises(ProposalRejected):
        service.propose_action(
            session_id="sess_1", principal_id="user_gaurang", user_goal="send an email",
            model_id="stub-model-v1", tool_name="send_email",
            arguments={"recipient": "mentor@example.com", "subject": "Hi", "body": "x",
                       "trusted": True},  # model-supplied trusted flag must not be accepted
        )


def test_unauthorized_target_rejected_recipient(service):
    with pytest.raises(ProposalRejected) as ei:
        service.propose_action(
            session_id="sess_1", principal_id="user_gaurang", user_goal="send an email",
            model_id="stub-model-v1", tool_name="send_email",
            arguments={"recipient": "attacker@evil.example", "subject": "Hi", "body": "x"},
        )
    assert "recipient" in str(ei.value)


def test_unauthorized_target_rejected_record_ownership(service):
    with pytest.raises(ProposalRejected) as ei:
        service.propose_action(
            session_id="sess_1", principal_id="user_gaurang", user_goal="update a record",
            model_id="stub-model-v1", tool_name="update_record",
            arguments={"record_id": "rec_9999", "field": "status", "new_value": "closed"},
        )
    assert "own" in str(ei.value)


def test_unknown_principal_denied(service):
    with pytest.raises(ProposalRejected):
        service.propose_action(
            session_id="sess_1", principal_id="unknown_user", user_goal="summarize",
            model_id="stub-model-v1", tool_name="summarize",
            arguments={"message_id": "msg_1", "max_words": 5},
        )


def test_model_supplied_trusted_flag_grants_no_privilege(service):
    """Model-supplied trusted=true or user IDs grant no privileges (architecture contract)."""
    with pytest.raises(ProposalRejected):
        service.propose_action(
            session_id="sess_1", principal_id="user_gaurang", user_goal="update a record",
            model_id="stub-model-v1", tool_name="update_record",
            arguments={"record_id": "rec_9999", "field": "status", "new_value": "closed",
                       "trusted": True},
        )


def test_stub_model_produces_one_structured_tool_request(agent):
    """Deterministic stub stands in for the real model milestone (kept open until Week 5)."""
    outcome = agent.handle_request("Please summarize the latest message")
    assert outcome.proposal.tool_name == "summarize"
    assert outcome.proposal.arguments["message_id"] == "msg_1"
