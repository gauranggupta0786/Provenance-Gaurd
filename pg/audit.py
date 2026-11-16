"""Week 4 Day 4 deliverable: the audit console.

This is deliberately a set of plain query functions over the store rather
than a web UI -- the plan's exit check is "a reviewer can reconstruct the
original request, exact approved values, actual execution, and incident
disposition from one trace," which `trace()` below satisfies directly and
which a thin UI could render as-is.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .schemas import ActionState
from .store import EventStore, IntegrityError
from .crypto import Verifier
from .gateway import find_uncertain_executions

REPEATED_PROBE_THRESHOLD = 5
REPEATED_PROBE_WINDOW_SECONDS = 600


@dataclass
class Alert:
    kind: str
    detail: str
    severity: str = "high"


def pending_approvals(store: EventStore) -> list[str]:
    return store.proposals_in_state(ActionState.PENDING_APPROVAL)


def blocked_or_denied(store: EventStore) -> list:
    return (
        store.proposals_in_state(ActionState.BLOCKED)
        + store.proposals_in_state(ActionState.DENIED)
        + store.rejected_attempts()
    )


def missing_receipts(store: EventStore) -> list[dict]:
    """Proposals stuck in EXECUTING with no receipt -- see gateway module
    docstring: these need reconciliation, not auto-retry."""
    return find_uncertain_executions(store)


def integrity_check(store: EventStore, verifier: Verifier) -> Alert | None:
    """Runs the same checks the demo/tests use, but returns an Alert
    instead of raising, for console display."""
    try:
        store.verify_chain(verifier)
        store.verify_against_checkpoint(verifier)
    except IntegrityError as e:
        return Alert(kind="log_integrity_failure", detail=str(e), severity="critical")
    return None


def repeated_probe_alerts(store: EventStore, principal_ids: list[str]) -> list[Alert]:
    """High-confidence alert condition: a burst of denied/rejected
    proposals from one principal. Mirrors the freeze threshold the
    gateway itself enforces (`gateway.MAX_REJECTIONS_PER_WINDOW`) so an
    operator sees the alert at (or slightly before) the point capability
    minting actually stops."""
    alerts = []
    for pid in principal_ids:
        count = store.get_budget_count(pid, "*", "rejected", window_seconds=REPEATED_PROBE_WINDOW_SECONDS)
        if count >= REPEATED_PROBE_THRESHOLD:
            alerts.append(Alert(
                kind="repeated_probe",
                detail=f"principal {pid!r} has {count} rejected proposals in the last "
                       f"{REPEATED_PROBE_WINDOW_SECONDS}s",
                severity="high",
            ))
    return alerts


def trace(store: EventStore, verifier: Verifier, proposal_id: str) -> dict:
    """Full propose-to-receipt drill-down for one proposal: original
    request, exact approved values, actual execution, and current
    disposition -- everything a reviewer needs from a single call."""
    proposal_row = store.get_proposal(proposal_id)
    history = store.state_history(proposal_id)
    receipt = store.get_receipt_by_proposal(proposal_id)
    approvals = store.approvals_for_proposal(proposal_id)
    policy_decisions = store.policy_decisions_for_proposal(proposal_id)

    signature_ok = None
    if proposal_row is not None:
        from .schemas import canonicalize
        signature_ok = verifier.verify(
            proposal_row["key_id"], canonicalize(proposal_row["payload"]), proposal_row["signature"],
        )

    return {
        "proposal_id": proposal_id,
        "proposal": proposal_row,
        "proposal_signature_valid": signature_ok,
        "state_history": history,
        "current_state": history[-1]["state"] if history else None,
        "receipt": receipt,
        "approvals": approvals,
        "policy_decisions": policy_decisions,
        "sources": store.sources_by_digest(
            proposal_row["payload"].get("source_digests", []) if proposal_row else []
        ),
    }


def summary(store: EventStore, verifier: Verifier, principal_ids: list[str]) -> dict:
    """One-shot console snapshot -- what a reviewer or the security monitor
    would want on a dashboard."""
    return {
        "pending_approvals": pending_approvals(store),
        "blocked_or_denied": blocked_or_denied(store),
        "missing_receipts": missing_receipts(store),
        "integrity": integrity_check(store, verifier),
        "alerts": repeated_probe_alerts(store, principal_ids),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
