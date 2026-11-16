from __future__ import annotations

import json
import os
import uuid

import pytest

from pg.confidentiality import (
    ConfidentialityGuard,
    ReadAuthorizationError,
    ReadAuthorizationPolicy,
)
from pg.crypto import DevelopmentKeyStore, Signer
from pg.gateway import SourceDriftError
from pg.provenance import SourceRecord
from pg.store import EventStore, IntegrityError
from pg.trusted_models import DataSensitivity, OutputGuardDecision


def policy(**updates):
    values = {
        "principal_id": "user-1",
        "allowed_tools": frozenset({"get_file_by_id"}),
        "allowed_read_sensitivities": frozenset(DataSensitivity),
    }
    values.update(updates)
    return ReadAuthorizationPolicy(**values)


def test_read_evidence_is_field_sensitive_and_direct_output_is_blocked():
    guard = ConfidentialityGuard(graph_run_id="run-1", policy=policy())
    records = guard.observe_read(
        principal_id="user-1",
        tool_name="get_file_by_id",
        arguments={"file_id": "file-1"},
        result={"id": "file-1", "filename": "payroll.txt", "content": "SALARY-SECRET-9000"},
    )
    protected = [item for item in records if item.sensitivity is DataSensitivity.CONFIDENTIAL]
    assert protected and protected[0].field_path == "result.content"
    assert not hasattr(protected[0], "raw_value")

    guarded = guard.guard_output("The file says SALARY-SECRET-9000")
    assert guarded.record.decision is OutputGuardDecision.BLOCK
    assert guarded.disclosure_attempted
    assert not guarded.confidentiality_violation
    assert "SALARY-SECRET-9000" not in guarded.text


def test_explicit_read_authorization_denies_wrong_principal_and_tool():
    guard = ConfidentialityGuard(graph_run_id="run-1", policy=policy())
    with pytest.raises(ReadAuthorizationError, match="principal"):
        guard.observe_read(
            principal_id="intruder", tool_name="get_file_by_id", arguments={}, result={"content": "secret"},
        )
    with pytest.raises(ReadAuthorizationError, match="not authorized"):
        guard.observe_read(
            principal_id="user-1", tool_name="send_email", arguments={}, result={"content": "secret"},
        )


def test_persistent_development_key_survives_restart_and_tracks_rotation(tmp_path):
    first_store = DevelopmentKeyStore(tmp_path / "keys")
    first = first_store.load_or_create("dev-key-v1")
    signature = Signer(first).sign(b"restart evidence")

    reopened = DevelopmentKeyStore(tmp_path / "keys")
    same = reopened.load_or_create("ignored-after-creation")
    assert same.key_id == first.key_id
    assert reopened.verifier().verify(first.key_id, b"restart evidence", signature)

    second = reopened.rotate("dev-key-v2")
    metadata = reopened.metadata()
    assert metadata["active_key_id"] == second.key_id
    assert metadata["rotation_counter"] == 2
    assert reopened.verifier().known_key_ids() == ["dev-key-v1", "dev-key-v2"]
    assert next(item for item in metadata["keys"] if item["key_id"] == "dev-key-v1")["status"] == "retired"


def _checkpoint_store(tmp_path):
    keys = DevelopmentKeyStore(tmp_path / "keys")
    pair = keys.load_or_create("checkpoint-key")
    signer = Signer(pair)
    verifier = keys.verifier()
    store = EventStore(str(tmp_path / "events.db"), str(tmp_path / "checkpoints"))
    store.register_integrity_verifier(verifier)
    for index in range(3):
        store.append_event("TEST", {"index": str(index)}, str(index) * 64, signer)
        store.export_checkpoint(signer)
    return store, verifier


def test_checkpoint_middle_deletion_and_editing_are_detected(tmp_path):
    store, verifier = _checkpoint_store(tmp_path)
    middle = tmp_path / "checkpoints" / "cp_000002.json"
    middle.unlink()
    with pytest.raises(IntegrityError, match="gap|middle|anchor"):
        store.verify_checkpoint_history(verifier)

    # Restore a fresh history, then edit a signed middle checkpoint.
    other = tmp_path / "edit"
    other.mkdir()
    edited, edited_verifier = _checkpoint_store(other)
    path = other / "checkpoints" / "cp_000002.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["head_hash"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntegrityError, match="signature"):
        edited.verify_checkpoint_history(edited_verifier)


def test_checkpoint_tail_deletion_detected_and_restart_verifies(tmp_path):
    store, verifier = _checkpoint_store(tmp_path)
    db_path, checkpoint_dir = store.db_path, store.checkpoint_dir
    store.close()
    reopened = EventStore(db_path, checkpoint_dir)
    reopened.verify_chain(verifier)
    reopened.verify_against_checkpoint(verifier)
    os.unlink(os.path.join(checkpoint_dir, "cp_000003.json"))
    with pytest.raises(IntegrityError, match="tail deletion|anchor"):
        reopened.verify_checkpoint_history(verifier)
    reopened.close()


def test_gateway_revalidates_source_version_immediately_before_execution(
    service, approvals, gateway, store,
):
    outcome = service.propose_action(
        session_id="source-drift-session",
        principal_id="user_gaurang",
        user_goal="send approved update",
        model_id="test-model",
        tool_name="send_email",
        arguments={"recipient": "mentor@example.com", "subject": "Update", "body": "Approved"},
    )
    approved = approvals.decide(
        outcome.proposal.proposal_id,
        reviewer_id="mentor",
        session_token="sess-tok-mentor-001",
        decision="approve",
    )
    bound = store.sources_by_digest(outcome.proposal.source_digests)[0]
    store.append_source_record(SourceRecord(
        source_id=str(uuid.uuid4()), tenant_id=bound["tenant_id"],
        source_type=bound["source_type"], source_identity=bound["source_identity"],
        trust_label=bound["trust_label"], content_digest="d" * 64,
        version=int(bound["version"]) + 1, observed_at="2026-10-09T00:00:00Z",
    ))
    with pytest.raises(SourceDriftError, match="source version drift"):
        gateway.execute(outcome.proposal.proposal_id, approved.approval_id)
