"""Week 3 deliverable: the human approves the logged action, not the
agent's narrative.

Three pieces, matching the plan's Week 3 day-by-day breakdown:

  1. `render_card` -- a server-side renderer that reads ONLY the verified,
     stored, signed proposal (reloaded via `store.load_verified_proposal`,
     never a caller-held object) plus the policy decision. Any optional
     model-written explanation or "claimed fields" are shown separately,
     HTML-escaped, and explicitly labeled untrusted; the renderer diffs
     claimed fields against the authoritative arguments and surfaces any
     mismatch instead of hiding it.

  2. Reviewer authentication + role mapping -- a tiny stand-in for "a local
     application session" (Day 3). No free-form reviewer identity is ever
     accepted from the agent/model.

  3. `ApprovalService` -- issues short-lived, single-use "tokens" (signed
     `Approval` records bound to proposal digest, reviewer, session,
     audience, policy version, and a nonce), persists approval state, and
     enforces expiry/scope/consumption via the store (Day 4).
"""
from __future__ import annotations

import html
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .schemas import Approval, ActionState, SchemaError, TOOL_CONTRACTS, canonicalize
from .policy import evaluate as policy_evaluate, Decision, POLICY_VERSION
from .policy_ablation import evaluate_disabled as policy_evaluate_disabled
from .crypto import Signer, Verifier
from .store import EventStore

RENDERER_VERSION = "card-v1-week3"
DEFAULT_APPROVAL_TTL_SECONDS = 300  # "short-lived"
DEFAULT_AUDIENCE = "gateway"


class ApprovalError(Exception):
    """Base class for every Week 3 rejection -- always fail closed."""


class ReviewerAuthError(ApprovalError):
    pass


class ReviewerNotPermittedError(ApprovalError):
    pass


class CardRenderError(ApprovalError):
    pass


# ---------------------------------------------------------------------------
# Reviewer authentication + role mapping (Day 3)
#
# Stand-in for "authenticate the reviewer through a local application
# session." Real deployments would replace REVIEWER_SESSIONS with an actual
# session store (cookie/JWT-backed); the *shape* of the check -- a trusted
# session token maps to a role, and the role gates which tools a reviewer
# may decide on -- is what the gateway and tests depend on.
# ---------------------------------------------------------------------------

REVIEWER_SESSIONS: dict[str, dict[str, str]] = {
    "mentor": {"session_token": "sess-tok-mentor-001", "role": "approver"},
    "reviewer_gaurang": {"session_token": "sess-tok-reviewer-001", "role": "approver"},
    "system:auto-policy": {"session_token": "internal", "role": "system"},
}

# Role -> set of tool names that role may decide approve/deny on.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "approver": {name for name in TOOL_CONTRACTS
                 if name not in {"summarize", "workspace_get_unread_emails"}},
    # The system role only ever auto-approves policy-ALLOW (no-approval-
    # required) actions -- see ApprovalService.auto_approve.
    "system": {"summarize", "workspace_get_unread_emails"},
}


def authenticate_reviewer(reviewer_id: str, session_token: str) -> str:
    """Returns the reviewer's role, or raises ReviewerAuthError."""
    session = REVIEWER_SESSIONS.get(reviewer_id)
    if session is None or session["session_token"] != session_token:
        raise ReviewerAuthError(f"reviewer {reviewer_id!r} failed session authentication")
    return session["role"]


def _check_reviewer_permitted(role: str, tool_name: str) -> None:
    allowed = ROLE_PERMISSIONS.get(role, set())
    if tool_name not in allowed:
        raise ReviewerNotPermittedError(f"role {role!r} is not permitted to decide on tool {tool_name!r}")


# ---------------------------------------------------------------------------
# Canonical approval card (Day 1-2)
# ---------------------------------------------------------------------------

def escape_display(text: Optional[str]) -> Optional[str]:
    """HTML-escape untrusted text before it is ever shown next to
    authoritative fields, so markup/badge-injection attempts render as
    inert literal text instead of formatting."""
    if text is None:
        return None
    return html.escape(text, quote=True)


def _target_and_changes(tool_name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """What the card calls "complete target" and "changed fields", derived
    only from the verified, normalized, signed arguments -- never from any
    free-form agent text."""
    if tool_name == "send_email":
        return arguments["recipient"], {
            "subject": arguments["subject"], "body": arguments["body"],
        }
    if tool_name == "update_record":
        return arguments["record_id"], {
            "field": arguments["field"], "new_value": arguments["new_value"],
        }
    if tool_name == "summarize":
        return arguments["message_id"], {"max_words": arguments["max_words"]}
    if tool_name == "workspace_send_email":
        destinations = arguments["recipients"] + arguments["cc"] + arguments["bcc"]
        return ", ".join(destinations), {
            "subject": arguments["subject"], "body": arguments["body"],
            "attachments": arguments["attachments"], "cc": arguments["cc"], "bcc": arguments["bcc"],
        }
    if tool_name == "workspace_delete_email":
        return arguments["email_id"], {"effect": "move email to trash"}
    if tool_name == "workspace_get_unread_emails":
        return "Workspace inbox", {"effect": "return unread email and mark it read"}
    if tool_name == "workspace_create_calendar_event":
        return arguments["title"], {k: arguments[k] for k in (
            "description", "start_time", "end_time", "location", "participants")}
    if tool_name in {"workspace_cancel_calendar_event", "workspace_reschedule_calendar_event",
                     "workspace_add_calendar_event_participants"}:
        return arguments["event_id"], {k: v for k, v in arguments.items() if k != "event_id"}
    if tool_name == "workspace_create_file":
        return arguments["filename"], {"content": arguments["content"]}
    if tool_name in {"workspace_append_to_file", "workspace_delete_file", "workspace_share_file"}:
        return arguments["file_id"], {k: v for k, v in arguments.items() if k != "file_id"}
    raise CardRenderError(f"no card layout for tool {tool_name!r}")


@dataclass
class ApprovalCard:
    proposal_id: str
    proposal_digest: str
    tool_name: str
    operation: str
    risk_level: str
    target: str
    changed_fields: dict[str, Any]
    policy_decision: str
    policy_reason: str
    policy_version: str
    source_digests: list[str]
    created_at: str
    principal_id: str
    # Untrusted, clearly labeled, HTML-escaped. Never used in verification.
    model_explanation_untrusted: Optional[str]
    # Any claimed_fields the model supplied, diffed against the
    # authoritative arguments above. Non-empty => a model wrote something
    # that does not match what will actually execute.
    claim_mismatches: list[str] = field(default_factory=list)
    # Week 11b: what the target id refers to, looked up by trusted code in the live environment (never from model text).
    target_description: Optional[str] = None
    # Week 11c: drive files whose text an e-mail body copies verbatim (trusted, from the live environment). Not rendered.
    copied_from: Optional[list[str]] = None
    # Week 11d: read sources (named + content-word hits) the body draws on but the request did not name. Not rendered.
    provenance_hits: Optional[list[dict]] = None

    def render_text(self, *, describe: bool = True) -> str:
        """A plain-text rendering suitable for a CLI/demo "screenshot"."""
        lines = [
            "HIGH-RISK ACTION - VERIFIED RECORD" if self.risk_level == "high" else "VERIFIED RECORD",
            f"Tool: {self.tool_name}",
            f"Target: {self.target}",
        ]
        if describe and self.target_description:
            lines.append(f"Target is: {self.target_description}")
        for k, v in self.changed_fields.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"Policy: {self.policy_decision} ({self.policy_reason})")
        lines.append(f"Proposal digest: {self.proposal_digest[:16]}...")
        if self.claim_mismatches:
            lines.append("MISMATCH: model's own explanation does not match the recorded action:")
            for m in self.claim_mismatches:
                lines.append(f"  - {m}")
        if self.model_explanation_untrusted:
            lines.append(f"Unverified model commentary (escaped, not authoritative): "
                          f"{self.model_explanation_untrusted}")
        return "\n".join(lines)


def render_card(store: EventStore, verifier: Verifier, proposal_id: str) -> ApprovalCard:
    """The Week 3 "trusted renderer": reloads the signed proposal from the
    store, re-verifies its signature, re-evaluates policy live (so the
    card always reflects *current* policy, not a stale decision cached by
    the agent), and only then builds the card. Nothing here is sourced
    from an in-memory object the agent could have influenced."""
    proposal = store.load_verified_proposal(proposal_id, verifier)
    policy_result = (policy_evaluate if store.layers.policy else policy_evaluate_disabled)(
        principal_id=proposal.principal_id, tool_name=proposal.tool_name, arguments=proposal.arguments,
    )
    target, changed = _target_and_changes(proposal.tool_name, proposal.arguments)

    mismatches: list[str] = []
    claimed = proposal.claimed_fields or {}
    explanation = escape_display(proposal.model_explanation)
    if not store.layers.grounded_approval:
        # Ablation: the card repeats what the model says it is doing.  The
        # claimed values replace the recorded ones and nothing is flagged.
        if "target" in claimed:
            target = str(claimed["target"])
        changed = {k: (claimed[k] if k in claimed else v) for k, v in changed.items()}
        if "recipient" in claimed and proposal.tool_name == "send_email":
            target = str(claimed["recipient"])
        explanation = proposal.model_explanation
    else:
        for k, claimed_v in claimed.items():
            actual_v = target if k == "target" else proposal.arguments.get(k, "<no such field in the record>")
            if str(claimed_v) != str(actual_v):
                mismatches.append(f"model claimed {k}={claimed_v!r} but the signed record shows {k}={actual_v!r}")

    return ApprovalCard(
        proposal_id=proposal.proposal_id,
        proposal_digest=proposal.digest,
        tool_name=proposal.tool_name,
        operation=proposal.operation,
        risk_level=proposal.risk_level,
        target=target,
        changed_fields=changed,
        policy_decision=policy_result.decision.value,
        policy_reason=policy_result.reason,
        policy_version=proposal.policy_version,
        source_digests=list(proposal.source_digests),
        created_at=proposal.created_at,
        principal_id=proposal.principal_id,
        model_explanation_untrusted=explanation,
        claim_mismatches=mismatches,
    )


# ---------------------------------------------------------------------------
# Approval / token service (Day 3-4)
# ---------------------------------------------------------------------------

@dataclass
class ApprovalOutcome:
    approval_id: str
    approval: Approval
    card: ApprovalCard


class ApprovalService:
    def __init__(self, store: EventStore, signer: Signer, verifier: Verifier):
        self._store = store
        self._signer = signer
        self._verifier = verifier
        self._store.register_integrity_verifier(verifier)

    # -- Day 1: move a verified proposal into the review queue --------------

    def request_approval(self, proposal_id: str) -> ApprovalCard:
        card = render_card(self._store, self._verifier, proposal_id)
        self._store.set_state(proposal_id, ActionState.PENDING_APPROVAL, signer=self._signer)
        return card

    # -- auto-approval for policy-ALLOW (no human decision needed) ----------

    def auto_approve(self, proposal_id: str, *, ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
                      audience: str = DEFAULT_AUDIENCE) -> ApprovalOutcome:
        card = render_card(self._store, self._verifier, proposal_id)
        if card.policy_decision != Decision.ALLOW.value:
            raise ApprovalError(
                f"auto_approve is only for policy ALLOW actions; proposal {proposal_id!r} is "
                f"{card.policy_decision}"
            )
        return self._issue(
            proposal_id=proposal_id, card=card, approver_id="system:auto-policy",
            reviewer_role="system", decision="approve", ttl_seconds=ttl_seconds, audience=audience,
        )

    # -- Day 3: authenticated human decision ---------------------------------

    def decide(self, proposal_id: str, *, reviewer_id: str, session_token: str, decision: str,
               ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS, audience: str = DEFAULT_AUDIENCE
               ) -> ApprovalOutcome:
        if decision not in ("approve", "deny"):
            raise ApprovalError(f"decision must be 'approve' or 'deny', got {decision!r}")

        role = authenticate_reviewer(reviewer_id, session_token)  # raises ReviewerAuthError
        card = render_card(self._store, self._verifier, proposal_id)  # fresh read, every time
        _check_reviewer_permitted(role, card.tool_name)  # raises ReviewerNotPermittedError

        # A valid approval can never override a policy denial (plan
        # constraint from Week 3 Day 2 / the deck's "Primary claim").
        if card.policy_decision == Decision.DENY.value:
            raise ApprovalError(f"cannot approve: policy denies this action ({card.policy_reason})")

        return self._issue(
            proposal_id=proposal_id, card=card, approver_id=reviewer_id, reviewer_role=role,
            decision=decision, ttl_seconds=ttl_seconds, audience=audience,
        )

    def _issue(self, *, proposal_id: str, card: ApprovalCard, approver_id: str, reviewer_role: str,
               decision: str, ttl_seconds: int, audience: str) -> ApprovalOutcome:
        # A decision always passes through PENDING_APPROVAL on its way to
        # APPROVED/DENIED (VERIFIED -> APPROVED is not a legal direct edge
        # in the Week 1 state machine). Calling request_approval()
        # separately first is optional -- decide()/auto_approve() enter
        # the queue implicitly if they're the first thing to touch this
        # proposal.
        if self._store.current_state(proposal_id) == ActionState.VERIFIED:
            self._store.set_state(proposal_id, ActionState.PENDING_APPROVAL, signer=self._signer)

        proposal = self._store.load_verified_proposal(proposal_id, self._verifier)  # fresh, verified
        now = time.time()
        approval = Approval(
            proposal_digest=proposal.digest,
            session_id=proposal.session_id,
            principal_id=proposal.principal_id,
            approver_id=approver_id,
            decision=decision,
            issued_at=_iso(now),
            expires_at=_iso(now + ttl_seconds),
            nonce=str(uuid.uuid4()),
            audience=audience,
            policy_version=proposal.policy_version,
            renderer_version=RENDERER_VERSION,
            key_id=self._signer.key_id,
        )
        approval.digest = approval.compute_digest()
        approval.signature = self._signer.sign(canonicalize(approval.signed_view()))
        approval_id = str(uuid.uuid4())
        self._store.append_approval(approval, approval_id, proposal_id)

        target_state = ActionState.APPROVED if decision == "approve" else ActionState.DENIED
        self._store.set_state(
            proposal_id, target_state, signer=self._signer,
            event_extra={"approval_id": approval_id, "approver_id": approver_id, "role": reviewer_role},
        )
        return ApprovalOutcome(approval_id=approval_id, approval=approval, card=card)


def _iso(epoch_seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_seconds))
