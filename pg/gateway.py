"""Week 4 deliverable: the execution gateway is the sole write-tool caller.

Every real side effect goes through `ExecutionGateway.execute`, which:
  1. Reloads and re-verifies the proposal from the store (never trusts a
     caller-held object).
  2. Reloads and re-verifies the approval "token", checking signature,
     expiry, exact proposal-digest binding, audience/session scope, and
     decision.
  3. Rechecks policy LIVE -- a stale approval cannot survive a policy
     downgrade between approval-time and execution-time.
  4. Atomically consumes the approval (single-use) and reserves the
     execution slot (a second, independent UNIQUE-constraint reservation
     keyed on the proposal digest) before any tool adapter runs.
  5. Dispatches to exactly one of the three simulated tool adapters.
  6. Durably records a receipt and the resulting terminal state, linked
     back to the proposal, policy decision, and approval.

Fail-closed throughout: any missing, inconsistent, expired, unverifiable,
or already-consumed input raises before the adapter is ever called.
"""
from __future__ import annotations

import contextlib
import time
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from .schemas import (
    Proposal, Receipt, ActionState, SchemaError, canonicalize, sha256_digest,
)
from .policy import evaluate as policy_evaluate, Decision, POLICY_VERSION
from .policy_ablation import evaluate_disabled as policy_evaluate_disabled
from .crypto import Signer, Verifier
from .store import EventStore, DuplicateExecutionError, ReplayError, IntegrityError
from . import tools as _tools

MAX_REJECTIONS_PER_WINDOW = 5
REJECTION_WINDOW_SECONDS = 600


class GatewayError(Exception):
    """Base class -- every rejection here is a fail-closed DENY, never a
    silent fallback execution path."""


class FrozenPrincipalError(GatewayError):
    pass


class ForgedTokenError(GatewayError):
    pass


class TokenExpiredError(GatewayError):
    pass


class TokenScopeError(GatewayError):
    pass


class PolicyDriftError(GatewayError):
    pass


class SourceDriftError(GatewayError):
    pass


class ReplayRejected(GatewayError):
    pass


# Tool name -> adapter callable. Each adapter receives the WorldState and
# the proposal's exact, normalized, signed arguments -- nothing else, and
# nothing from any free-form agent text (least privilege, invariant I6).
ADAPTERS = {
    "summarize": lambda world, **kw: {"summary": _tools.summarize_impl(world, **kw)},
    "send_email": lambda world, **kw: _tools.send_email_impl(world, **kw),
    "update_record": lambda world, **kw: _tools.update_record_impl(world, **kw),
}


@dataclass
class ExecutionResult:
    receipt: Receipt
    tool_result: Optional[dict]
    error: Optional[str]


class ExecutionGateway:
    def __init__(self, store: EventStore, signer: Signer, verifier: Verifier, world: Any,
                 adapters: Optional[dict[str, Any]] = None):
        self._store = store
        self._signer = signer
        self._verifier = verifier
        self._world = world
        self._adapters = adapters if adapters is not None else ADAPTERS
        self._execution_lock = threading.RLock()
        self._layers = store.layers
        self._store.register_integrity_verifier(verifier)

    def execute(self, proposal_id: str, approval_id: str) -> ExecutionResult:
        # The process-level lock is part of the atomic-consume layer: with the
        # layer ablated, concurrent callers are not serialised here either.
        guard = self._execution_lock if self._layers.atomic_consume else contextlib.nullcontext()
        with guard:
            return self._execute_locked(proposal_id, approval_id)

    def _execute_locked(self, proposal_id: str, approval_id: str) -> ExecutionResult:
        proposal = self._store.load_verified_proposal(proposal_id, self._verifier)

        if self._layers.gateway_revalidation:
            self._revalidate(proposal, proposal_id, approval_id)
        else:
            # Ablation: execute whatever the store says is approved.  The
            # approval must still exist and the state machine still applies.
            if self._store.get_approval(approval_id) is None:
                raise SchemaError(f"no such approval: {approval_id!r}")

        # Atomically consume approval, reserve idempotency, and enter EXECUTING
        # before the adapter can change simulated world state.
        attempt_id = str(uuid.uuid4())
        consumed = self._store.consume_approval_and_reserve_execution(
            approval_id=approval_id,
            proposal_id=proposal_id,
            idempotency_key=proposal.digest,
            attempt_id=attempt_id,
            signer=self._signer,
        )
        if not consumed:
            raise ReplayRejected(f"approval {approval_id!r} was already consumed (replay attempt)")

        # -- Day 5: dispatch to exactly one adapter, least privilege ---------
        adapter = self._adapters.get(proposal.tool_name)
        if adapter is None:
            raise GatewayError(f"no adapter registered for tool {proposal.tool_name!r}")

        try:
            tool_result = adapter(self._world, **proposal.arguments)
            result_status = "SUCCEEDED"
            error = None
        except Exception as e:  # simulated tool failure -- still durably recorded, never silent
            tool_result = None
            result_status = "FAILED"
            error = f"{type(e).__name__}: {e}"

        result_digest = sha256_digest(canonicalize({"result": tool_result, "error": error}))
        receipt = Receipt(
            action_digest=proposal.digest,
            attempt_id=attempt_id,
            result_status=result_status,
            result_digest=result_digest,
            idempotency_key=proposal.digest,  # one receipt per proposal, ever
        )
        self._store.complete_reserved_execution(
            receipt=receipt,
            proposal_id=proposal_id,
            detail=(error or ""),
            signer=self._signer,
            approval_id=approval_id,
        )
        return ExecutionResult(receipt=receipt, tool_result=tool_result, error=error)


    def _revalidate(self, proposal: Proposal, proposal_id: str, approval_id: str) -> None:
        """Everything the gateway re-checks before it lets an action run."""
        # -- Day 3: repeated-probe freeze, checked before minting anything --
        rejection_count = self._store.get_budget_count(
            proposal.principal_id, "*", "rejected", window_seconds=REJECTION_WINDOW_SECONDS,
        )
        if rejection_count > MAX_REJECTIONS_PER_WINDOW:
            raise FrozenPrincipalError(
                f"principal {proposal.principal_id!r} is frozen: {rejection_count} rejected "
                f"proposals in the last {REJECTION_WINDOW_SECONDS}s"
            )

        try:
            approval_row = self._store.get_approval(approval_id)
        except ValidationError as exc:
            raise ForgedTokenError(
                f"approval {approval_id!r} is not a valid trusted record"
            ) from exc
        if approval_row is None:
            raise SchemaError(f"no such approval: {approval_id!r}")

        # -- Day 1: verify signature, decision, expiry, exact binding --------
        if not self._verifier.verify(approval_row["key_id"], canonicalize(approval_row["payload"]),
                                      approval_row["signature"]):
            raise ForgedTokenError(f"approval {approval_id!r} fails signature verification")

        payload = approval_row["payload"]
        if approval_row["decision"] != "approve":
            raise GatewayError(f"approval {approval_id!r} is a {approval_row['decision']!r} decision, not approve")

        if _now() >= _parse_iso(approval_row["expires_at"]):
            raise TokenExpiredError(f"approval {approval_id!r} expired at {approval_row['expires_at']}")

        if approval_row["proposal_digest"] != proposal.digest:
            raise TokenScopeError(
                "approval is bound to a different proposal digest -- the action changed after "
                "approval; a new proposal and a new approval are required"
            )
        # Week 11: the digest column is not covered by the signature, the payload is.  Compare the *signed*
        # digest too, otherwise a genuine approval can be re-pointed at another signed proposal by editing
        # the unsigned columns (Week 10 finding approval_rebinding).
        if payload.get("proposal_digest") != proposal.digest:
            raise TokenScopeError(
                "the signed approval payload is bound to a different proposal digest (approval re-pointing)"
            )
        if approval_row["proposal_id"] != proposal.proposal_id:
            raise TokenScopeError("approval's proposal_id does not match the proposal being executed")
        if payload.get("session_id") != proposal.session_id:
            raise TokenScopeError("approval session does not match the proposal's session (cross-session reuse)")
        if payload.get("audience") != "gateway":
            raise TokenScopeError(f"approval audience {payload.get('audience')!r} is not valid for the gateway")

        # -- Day 1: recheck policy live -- approval cannot outlive a policy change --
        if proposal.policy_version != POLICY_VERSION:
            raise PolicyDriftError(
                f"proposal policy version {proposal.policy_version!r} is not current {POLICY_VERSION!r}"
            )
        if self._layers.provenance:  # without source records there is nothing to compare
            try:
                self._store.assert_source_versions_current(proposal.source_digests)
            except IntegrityError as exc:
                raise SourceDriftError(str(exc)) from exc
        live_policy = (policy_evaluate if self._layers.policy else policy_evaluate_disabled)(
            principal_id=proposal.principal_id, tool_name=proposal.tool_name, arguments=proposal.arguments,
        )
        self._store.append_policy_decision(
            proposal_id,
            "execution",
            live_policy.decision.value,
            live_policy.reason,
            proposal.policy_version,
            self._signer,
        )
        if live_policy.decision == Decision.DENY:
            self._store.set_state(proposal_id, ActionState.REVOKED, signer=self._signer,
                                   event_extra={"reason": f"policy drift: {live_policy.reason}"})
            raise PolicyDriftError(f"policy now denies this action: {live_policy.reason}")


# ---------------------------------------------------------------------------
# Reconciliation (Day 2/5): a crash between "reserved" and "receipt written"
# must be visible and must NOT be silently auto-retried.
# ---------------------------------------------------------------------------

def find_uncertain_executions(store: EventStore) -> list[dict]:
    """Proposals whose latest state is EXECUTING but which have no receipt
    -- either genuinely still running (unlikely in this synchronous
    prototype) or interrupted mid-execution. These require human
    reconciliation, never an automatic retry (plan: "An uncertain external
    outcome needs reconciliation rather than automatic retry")."""
    uncertain = []
    for proposal_id in store.proposals_in_state(ActionState.EXECUTING):
        if store.get_receipt_by_proposal(proposal_id) is None:
            uncertain.append({
                "proposal_id": proposal_id,
                "state_history": store.state_history(proposal_id),
            })
    return uncertain


def _now() -> float:
    return time.time()


def _parse_iso(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
