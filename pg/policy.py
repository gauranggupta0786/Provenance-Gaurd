"""Week 1 Day 2 + 4 deliverable: initial policy configuration.

This is the "tool-name rules and keyword heuristics" starting point made
explicit and versioned, scoped to exactly what Week 1 asks for: allowed
recipients, record ownership, editable fields, and task scope. Per-user
ABAC, rate limits, and source-provenance checks are scheduled for Weeks
1-4 in the full plan and are intentionally NOT all implemented here --
this module documents the initial (not final) policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .schemas import TOOL_CONTRACTS, RiskLevel

POLICY_VERSION = "v2-week5-agentdojo-workspace"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


@dataclass
class PolicyResult:
    decision: Decision
    reason: str


# Static v1 configuration. In a real deployment this is loaded from a
# versioned, signed policy bundle; here it is an explicit in-memory table
# so the Week-1 exit checks can demonstrate ALLOW/REQUIRE_APPROVAL/DENY.
PRINCIPAL_PERMISSIONS: dict[str, dict[str, object]] = {
    "user_gaurang": {
        "allowed_tools": set(TOOL_CONTRACTS),
        "allowed_recipients": {
            "mentor@example.com", "reviewer@example.com",
            # Stock AgentDojo Workspace identities used by the reproducible
            # benchmark fixtures. Attacker-controlled external addresses are
            # intentionally absent.
            "emma.johnson@gmail.com", "james.smith@gmail.com",
            "sarah.lee@gmail.com", "taylor.johnson@gmail.com",
        },
        "owned_record_ids": {"rec_1001", "rec_1002"},
    },
}


def evaluate(*, principal_id: str, tool_name: str, arguments: dict) -> PolicyResult:
    """Read-and-write are both explicitly policed, per the plan's Week 1 note
    that "all writes are approval-requiring; give read/output access
    explicit policies too."""
    perms = PRINCIPAL_PERMISSIONS.get(principal_id)
    if perms is None:
        return PolicyResult(Decision.DENY, f"unknown principal {principal_id!r}")

    if tool_name not in TOOL_CONTRACTS:
        return PolicyResult(Decision.DENY, f"unknown tool {tool_name!r}")

    if tool_name not in perms["allowed_tools"]:
        return PolicyResult(Decision.DENY, f"principal not permitted to use {tool_name!r}")

    if tool_name == "summarize":
        # Read/output action: explicit policy, but low risk -> no approval needed.
        return PolicyResult(Decision.ALLOW, "summarize is read-only and explicitly permitted")

    if tool_name == "send_email":
        if arguments.get("recipient") not in perms["allowed_recipients"]:
            return PolicyResult(Decision.DENY, "recipient not in allowed list")
        return PolicyResult(Decision.REQUIRE_APPROVAL, "send_email is a write action -> approval required")

    if tool_name == "update_record":
        if arguments.get("record_id") not in perms["owned_record_ids"]:
            return PolicyResult(Decision.DENY, "principal does not own this record")
        return PolicyResult(Decision.REQUIRE_APPROVAL, "update_record is a write action -> approval required")

    if tool_name == "workspace_get_unread_emails":
        return PolicyResult(Decision.ALLOW, "reading unread email is permitted; read-state change is mediated")

    if tool_name == "workspace_send_email":
        destinations = arguments["recipients"] + arguments["cc"] + arguments["bcc"]
        denied = sorted(set(destinations) - set(perms["allowed_recipients"]))
        if denied:
            return PolicyResult(Decision.DENY, f"email destination not allowed: {denied}")
        return PolicyResult(Decision.REQUIRE_APPROVAL, "sending Workspace email requires approval")

    if tool_name in {"workspace_create_calendar_event", "workspace_add_calendar_event_participants"}:
        denied = sorted(set(arguments["participants"]) - set(perms["allowed_recipients"]))
        if denied:
            return PolicyResult(Decision.DENY, f"calendar participant not allowed: {denied}")
        return PolicyResult(Decision.REQUIRE_APPROVAL, "calendar participant change requires approval")

    if tool_name == "workspace_share_file":
        if arguments["email"] not in perms["allowed_recipients"]:
            return PolicyResult(Decision.DENY, "file-share destination not in allowed list")
        return PolicyResult(Decision.REQUIRE_APPROVAL, "sharing a Workspace file requires approval")

    if tool_name.startswith("workspace_"):
        return PolicyResult(Decision.REQUIRE_APPROVAL, "Workspace state-changing action requires approval")

    return PolicyResult(Decision.DENY, "no matching policy rule")  # fail closed (I7)
