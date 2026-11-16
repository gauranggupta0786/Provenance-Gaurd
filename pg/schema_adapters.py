"""Validated adapters between Pydantic records, legacy services, AgentDojo, and SQLite."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from agentdojo.functions_runtime import FunctionCall

from .policy import Decision, PolicyResult
from .provenance import SourceRecord
from .schemas import Approval, Proposal, Receipt, canonicalize, sha256_digest
from .trusted_models import (
    ApprovalDecision,
    ApprovalRecord,
    EvaluationRecord,
    EvaluationStatus,
    PolicyDecision,
    PolicyOutcome,
    ProposalRecord,
    ReceiptRecord,
    SourceEvidence,
    ToolIntent,
)


def proposal_record_from_legacy(proposal: Proposal) -> ProposalRecord:
    return ProposalRecord(
        schema_version=proposal.schema_version,
        proposal_id=proposal.proposal_id,
        session_id=proposal.session_id,
        principal_id=proposal.principal_id,
        user_goal_hash=proposal.user_goal_hash,
        model_id=proposal.model_id,
        tool_name=proposal.tool_name,
        operation=proposal.operation,
        arguments=proposal.arguments,
        source_digests=tuple(proposal.source_digests),
        policy_version=proposal.policy_version,
        risk_level=proposal.risk_level,
        created_at=proposal.created_at,
        key_id=proposal.key_id,
        model_explanation=proposal.model_explanation,
        claimed_fields=proposal.claimed_fields,
        digest=proposal.digest,
        signature=proposal.signature,
    )


def proposal_legacy_from_record(record: ProposalRecord) -> Proposal:
    return Proposal(
        schema_version=record.schema_version,
        proposal_id=record.proposal_id,
        session_id=record.session_id,
        principal_id=record.principal_id,
        user_goal_hash=record.user_goal_hash,
        model_id=record.model_id,
        tool_name=record.tool_name,
        operation=record.operation,
        arguments=dict(record.arguments),
        source_digests=list(record.source_digests),
        policy_version=record.policy_version,
        risk_level=record.risk_level,
        created_at=record.created_at,
        key_id=record.key_id,
        model_explanation=record.model_explanation,
        claimed_fields=dict(record.claimed_fields) if record.claimed_fields is not None else None,
        digest=record.digest,
        signature=record.signature,
    )


def proposal_record_from_sqlite(row: Mapping[str, Any]) -> ProposalRecord:
    payload = dict(row["payload"])
    payload.update({
        "model_explanation": row.get("model_explanation"),
        "claimed_fields": row.get("claimed_fields"),
        "digest": row["digest"],
        "signature": row["signature"],
    })
    payload["source_digests"] = tuple(payload["source_digests"])
    return ProposalRecord.model_validate(payload)


def approval_record_from_legacy(
    approval: Approval,
    *,
    approval_id: str,
    proposal_id: str,
    consumed: bool = False,
    consumed_at: str | None = None,
) -> ApprovalRecord:
    if approval.digest is None or approval.signature is None or approval.key_id is None:
        raise ValueError("approval must be signed before conversion")
    return ApprovalRecord(
        approval_id=approval_id,
        proposal_id=proposal_id,
        proposal_digest=approval.proposal_digest,
        session_id=approval.session_id,
        principal_id=approval.principal_id,
        approver_id=approval.approver_id,
        decision=ApprovalDecision(approval.decision),
        issued_at=approval.issued_at,
        expires_at=approval.expires_at,
        nonce=approval.nonce,
        audience=approval.audience,
        policy_version=approval.policy_version,
        renderer_version=approval.renderer_version,
        key_id=approval.key_id,
        digest=approval.digest,
        signature=approval.signature,
        consumed=consumed,
        consumed_at=consumed_at,
    )


def approval_record_from_sqlite(row: Mapping[str, Any]) -> ApprovalRecord:
    payload = dict(row["payload"])
    return ApprovalRecord(
        approval_id=str(row["approval_id"]),
        proposal_id=str(row["proposal_id"]),
        proposal_digest=str(row["proposal_digest"]),
        session_id=str(payload["session_id"]),
        principal_id=str(payload["principal_id"]),
        approver_id=str(payload["approver_id"]),
        decision=ApprovalDecision(str(row["decision"])),
        issued_at=str(payload["issued_at"]),
        expires_at=str(row["expires_at"]),
        nonce=str(row["nonce"]),
        audience=str(payload["audience"]),
        policy_version=str(payload["policy_version"]),
        renderer_version=str(payload["renderer_version"]),
        key_id=str(row["key_id"]),
        digest=sha256_digest(canonicalize(payload)),
        signature=str(row["signature"]),
        consumed=bool(row["consumed"]),
        consumed_at=row.get("consumed_at"),
    )


def receipt_record_from_legacy(
    receipt: Receipt,
    *,
    proposal_id: str,
    detail: str,
    created_at: str,
) -> ReceiptRecord:
    return ReceiptRecord(
        attempt_id=receipt.attempt_id,
        proposal_id=proposal_id,
        idempotency_key=receipt.idempotency_key,
        action_digest=receipt.action_digest,
        result_status=receipt.result_status,
        result_digest=receipt.result_digest,
        detail=detail,
        created_at=created_at,
    )


def receipt_record_from_sqlite(row: Mapping[str, Any]) -> ReceiptRecord:
    return ReceiptRecord.model_validate(dict(row))


def source_evidence_from_legacy(record: SourceRecord, *, raw_evidence_ref: str | None = None) -> SourceEvidence:
    return SourceEvidence(
        source_id=record.source_id,
        tenant_id=record.tenant_id,
        source_type=record.source_type,
        source_identity=record.source_identity,
        trust_label=record.trust_label,
        content_digest=record.content_digest,
        version=record.version,
        observed_at=record.observed_at,
        raw_evidence_ref=raw_evidence_ref,
    )


def policy_outcome_from_legacy(
    result: PolicyResult,
    *,
    phase: str,
    policy_version: str,
    evaluated_at: str,
    proposal_id: str | None = None,
) -> PolicyOutcome:
    return PolicyOutcome(
        proposal_id=proposal_id,
        phase=phase,
        decision=PolicyDecision(result.decision.value),
        reason=result.reason,
        policy_version=policy_version,
        evaluated_at=evaluated_at,
    )


def tool_intent_from_agentdojo(call: "FunctionCall", *, response_id: str | None = None) -> ToolIntent:
    return ToolIntent(
        call_id=call.id or f"agentdojo-{call.function}",
        tool_name=call.function,
        arguments=json.loads(json.dumps(dict(call.args), default=str, allow_nan=False)),
        raw_arguments=json.loads(json.dumps(dict(call.args), default=str, allow_nan=False)),
        source_response_id=response_id,
        parse_error=None,
    )


def tool_intent_to_agentdojo(intent: ToolIntent) -> "FunctionCall":
    from agentdojo.functions_runtime import FunctionCall

    if not intent.valid:
        raise ValueError(f"cannot cross trusted boundary with malformed tool intent: {intent.parse_error}")
    return FunctionCall(function=intent.tool_name, args=dict(intent.arguments or {}), id=intent.call_id)


def evaluation_record_from_episode(
    episode: Mapping[str, Any],
    *,
    graph_run_id: str,
    evaluator_version: str,
) -> EvaluationRecord:
    error = episode.get("error")
    return EvaluationRecord(
        graph_run_id=graph_run_id,
        scenario_id=str(episode["scenario_id"]),
        arm=str(episode["arm"]),
        seed=int(episode["seed"]),
        status=EvaluationStatus.ERROR if error else EvaluationStatus.EVALUABLE,
        completed=bool(episode["completed"]),
        benign_success=episode.get("benign_success"),
        attack_success=episode.get("attack_success"),
        final_state_digest=episode.get("final_state_digest") or None,
        evaluator_version=evaluator_version,
        error=error,
    )
