"""Week 2 Day 1-2: propose_action() -- the sole tool-request interface.

The agent calls `propose_action`. It never touches `pg.tools`, never holds
a `Signer`, and never writes to the store directly. This module:
  1. Assigns trusted identity/time (never taken from the caller's claims).
  2. Normalizes and validates arguments (rejects malformed/ambiguous input).
  3. Runs the Week-1 policy check.
  4. Builds, signs, and durably stores the proposal.
  5. Records an audit event for every attempt -- including rejections --
     so the wrapper "records all attempted calls" per the Week 2 exit check.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from .schemas import build_proposal, Proposal, SchemaError, sha256_digest, canonicalize, ActionState
from .policy import evaluate as policy_evaluate, Decision, POLICY_VERSION
from .policy_ablation import evaluate_disabled as policy_evaluate_disabled
from .crypto import Signer, Verifier
from .store import EventStore
from .provenance import SourceLabeler
from .schema_adapters import proposal_record_from_legacy
from .trusted_models import ToolIntent
from .store import UNSIGNED_PLACEHOLDER

# Fixed stand-in recorded when the provenance layer is ablated: proposals must
# carry at least one source digest, but it identifies nothing.
PROVENANCE_DISABLED_DIGEST = sha256_digest(b"provenance-layer-disabled")


class ProposalRejected(Exception):
    def __init__(self, reason: str, policy_decision: Optional[str] = None):
        super().__init__(reason)
        self.reason = reason
        self.policy_decision = policy_decision


@dataclass
class ProposalOutcome:
    proposal: Proposal
    policy_decision: Decision
    policy_reason: str


class ProposalService:
    """Trusted boundary component. Only trusted application code should
    construct one of these (it needs a Signer, i.e. private key access)."""

    def __init__(self, store: EventStore, signer: Signer, verifier: Verifier):
        self._store = store
        self._signer = signer
        self._store.register_integrity_verifier(verifier)
        self._source_labeler = SourceLabeler()
        self._layers = store.layers

    def propose_action(
        self,
        *,
        session_id: str,
        principal_id: str,
        user_goal: str,
        model_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        source_labels: Optional[dict[str, str]] = None,
        model_explanation: Optional[str] = None,
        claimed_fields: Optional[dict[str, Any]] = None,
    ) -> ProposalOutcome:
        """`source_labels` maps a short name -> raw text; we store only
        digests (Week 1 note: keep source references, not raw content, as
        part of the signed record). `model_explanation`/`claimed_fields`
        are optional, UNSIGNED, untrusted text the model may attach to its
        own request -- Week 3's card renderer displays them only as
        clearly-labeled, escaped commentary, never as authoritative."""
        source_labels = source_labels or {}
        if self._layers.provenance:
            source_records = [
                self._source_labeler.label(
                    tenant_id=principal_id,
                    source_type="authenticated_user",
                    source_identity=principal_id,
                    content=user_goal,
                )
            ]
            for label, content in source_labels.items():
                if label == "user_goal" and content == user_goal:
                    continue
                source_records.append(self._source_labeler.label(
                    tenant_id=principal_id,
                    source_type="external_content",
                    source_identity=label,
                    content=content,
                ))
            source_digests = [record.content_digest for record in source_records]
        else:
            source_records = []
            source_digests = [PROVENANCE_DISABLED_DIGEST]
        user_goal_hash = sha256_digest(user_goal.encode("utf-8"))
        attempt_id = str(uuid.uuid4())

        try:
            # The model-facing mapping is converted into a strict record before
            # schema normalization, policy evaluation, signing, or persistence.
            intent = ToolIntent(
                call_id=attempt_id,
                tool_name=tool_name,
                arguments=arguments,
                raw_arguments=arguments,
                parse_error=None,
            )
            proposal = build_proposal(
                session_id=session_id,
                principal_id=principal_id,
                user_goal_hash=user_goal_hash,
                model_id=model_id,
                tool_name=intent.tool_name,
                arguments=dict(intent.arguments or {}),
                source_digests=source_digests,
                policy_version=POLICY_VERSION,
                key_id=self._signer.key_id,
                model_explanation=model_explanation,
                claimed_fields=claimed_fields,
                validate=self._layers.schema,
            )
        except SchemaError as e:
            self._record_attempt(attempt_id, principal_id, tool_name, status="REJECTED_SCHEMA", detail=str(e))
            raise ProposalRejected(f"schema rejection: {e}") from e

        policy_result = (policy_evaluate if self._layers.policy else policy_evaluate_disabled)(
            principal_id=principal_id, tool_name=tool_name, arguments=proposal.arguments,
        )
        if policy_result.decision == Decision.DENY:
            self._record_attempt(attempt_id, principal_id, tool_name, status="REJECTED_POLICY",
                                  detail=policy_result.reason, proposal_digest=proposal.compute_digest())
            raise ProposalRejected(f"policy denied: {policy_result.reason}", policy_decision=policy_result.decision.value)

        proposal.digest = proposal.compute_digest()
        proposal.signature = (
            self._signer.sign(canonicalize(proposal.signed_view()))
            if self._layers.signed_proposal else UNSIGNED_PLACEHOLDER
        )
        proposal_record_from_legacy(proposal)

        self._store.persist_verified_proposal(
            proposal,
            attempt_id=attempt_id,
            policy_decision=policy_result.decision.value,
            policy_reason=policy_result.reason,
            signer=self._signer,
            source_records=source_records,
        )
        return ProposalOutcome(proposal=proposal, policy_decision=policy_result.decision, policy_reason=policy_result.reason)

    def _record_attempt(self, attempt_id: str, principal_id: str, tool_name: str, *,
                         status: str, detail: str, proposal_digest: Optional[str] = None) -> None:
        digest = proposal_digest or sha256_digest(f"{principal_id}|{tool_name}|{detail}".encode("utf-8"))
        self._store.append_event(
            event_type=status,
            record_ids={"attempt_id": attempt_id, "principal_id": principal_id, "tool_name": tool_name,
                        "detail": detail[:200]},
            content_digest=digest,
            signer=self._signer,
        )
        # Week 4 Day 3: counted on trusted principal_id + a fixed window,
        # never on the wording of the request, so a rephrased attack can't
        # reset it. The gateway refuses to mint capabilities once this
        # crosses the threshold (see gateway.MAX_REJECTIONS_PER_WINDOW).
        self._store.increment_budget(principal_id, "*", "rejected", window_seconds=600)
