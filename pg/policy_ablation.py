"""Week 10: the policy layer's off-switch, kept out of ``pg/policy.py``.

``pg/policy.py`` is hash-pinned by the Week 6 frozen-protocol manifest, so the
ablation variant of the policy lives here and ``pg/policy.py`` stays
byte-identical to what Weeks 5-9 ran.
"""
from __future__ import annotations

from .ablation import FULL, Layers
from .policy import Decision, PolicyResult, evaluate
from .schemas import TOOL_CONTRACTS

# Read tools are routed to ALLOW even with the layer off: routing by tool
# class is plumbing, allowlist / ownership rules are the defence.
UNPOLICED_READ_TOOLS = frozenset({"summarize", "workspace_get_unread_emails"})


def evaluate_disabled(*, principal_id: str, tool_name: str, arguments: dict) -> PolicyResult:
    """The policy layer switched off.

    Nothing is ever DENY except an unregistered tool: reads are ALLOW and every
    other tool is REQUIRE_APPROVAL, with no recipient, participant,
    destination or ownership check.
    """
    if tool_name not in TOOL_CONTRACTS:
        return PolicyResult(Decision.DENY, f"unknown tool {tool_name!r}")
    if tool_name in UNPOLICED_READ_TOOLS:
        return PolicyResult(Decision.ALLOW, "policy layer disabled (ablation): read tool routed to ALLOW")
    return PolicyResult(
        Decision.REQUIRE_APPROVAL,
        "policy layer disabled (ablation): no allowlist or ownership check, approval routing only",
    )


def evaluate_for(layers: Layers = FULL, *, principal_id: str, tool_name: str, arguments: dict) -> PolicyResult:
    """`policy.evaluate`, honouring the policy-layer switch."""
    if layers.policy:
        return evaluate(principal_id=principal_id, tool_name=tool_name, arguments=arguments)
    return evaluate_disabled(principal_id=principal_id, tool_name=tool_name, arguments=arguments)
