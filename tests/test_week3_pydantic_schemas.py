from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pg.trusted_models import (
    AgentRunState,
    ApprovalRecord,
    ModelCallRecord,
    ProposalRecord,
    RunStatus,
    ToolIntent,
    WorkflowNode,
    schema_manifest,
)


HEX = "a" * 64


def proposal(**updates):
    value = {
        "proposal_id": "proposal-1", "session_id": "session-1", "principal_id": "user-1",
        "user_goal_hash": HEX, "model_id": "model-1", "tool_name": "send_email",
        "operation": "send_email", "arguments": {"recipient": "a@example.com"},
        "source_digests": ("b" * 64,), "policy_version": "policy-v1", "risk_level": "high",
        "created_at": "2026-10-09T00:00:00Z", "key_id": "key-1",
    }
    value.update(updates)
    return value


@pytest.mark.parametrize("change", [
    {"proposal_id": None},
    {"unexpected": "field"},
    {"arguments": "not-an-object"},
    {"schema_version": "2.0"},
    {"tool_name": "send_email", "operation": "update_record"},
    {"source_digests": ("b" * 64, "b" * 64)},
    {"arguments": {str(i): i for i in range(17)}},
])
def test_proposal_rejects_malformed_boundary_data(change):
    with pytest.raises(ValidationError):
        ProposalRecord.model_validate(proposal(**change))


def test_tool_intent_rejects_ambiguous_and_malformed_calls():
    with pytest.raises(ValidationError):
        ToolIntent(call_id="c", tool_name="x", arguments={}, raw_arguments={}, parse_error="bad")
    with pytest.raises(ValidationError):
        ToolIntent(call_id="c", tool_name="x", arguments=None, raw_arguments={}, parse_error=None)
    with pytest.raises(ValidationError):
        ToolIntent.model_validate({
            "schema_version": "tool-intent-v2", "call_id": "c", "tool_name": "x",
            "arguments": {}, "raw_arguments": {}, "parse_error": None,
        })


def test_strict_types_do_not_coerce():
    with pytest.raises(ValidationError):
        ModelCallRecord(
            call_id="c", graph_run_id="r", profile_sha256=HEX, stage="CALL_MODEL",
            attempt="1", request_evidence_ref="evidence://request", latency_ms=1,
            status="SUCCEEDED",
        )


def test_invalid_state_and_approval_cross_fields_are_rejected():
    with pytest.raises(ValidationError):
        AgentRunState(
            graph_run_id="r", scenario_id="s", arm="protected", seed=1,
            current_node=WorkflowNode.CALL_MODEL, status=RunStatus.SUCCEEDED,
            session_id="session", principal_id="principal", model_profile_sha256=HEX,
            scenario_evidence_ref="evidence://scenario",
        )
    with pytest.raises(ValidationError):
        ApprovalRecord(
            approval_id="a", proposal_id="p", proposal_digest=HEX, session_id="s",
            principal_id="u", approver_id="reviewer", decision="approve",
            issued_at="2026-10-09T01:00:00Z", expires_at="2026-10-09T00:00:00Z",
            nonce="n", audience="gateway", policy_version="p1", renderer_version="r1",
            key_id="k", digest=HEX, signature="x" * 32,
        )


def test_generated_schema_manifest_hashes_verify_and_forbid_extra():
    manifest = schema_manifest()
    assert len(manifest["schemas"]) >= 11
    schema_path = next(
        item["file"] for item in manifest["schemas"] if item["model"] == "ToolIntent"
    )
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "v1" / schema_path).read_text()
    )
    assert schema["additionalProperties"] is False
