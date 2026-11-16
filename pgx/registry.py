"""Register the new suites' tools in v1's tool tables at runtime, without editing ``pg/``.

v1 keeps its tool contracts, reviewer roles and card layouts in module-level tables (``pg.schemas.TOOL_CONTRACTS``,
``pg.approval.ROLE_PERMISSIONS``, ``pg.approval._target_and_changes``).  ``install()`` ADDS the new canonical tools to those
tables and wraps the card-layout function so that new tools get a layout and every v1 tool is delegated, untouched, to v1's
own function.  It never changes or removes a v1 entry; ``tests/test_pgx_registry.py`` checks that every v1 tool validates and
renders byte-identically before and after.

Under v1's static policy (``pg.policy.evaluate``) any new tool falls through to "no matching policy rule" and is DENIED (fail
closed), so v2 configurations run the generic policy (``Layers(policy=False)``: unknown tools denied, every write needs approval,
no hand-written allow-list).  That choice is deliberate and stated in the v2 plan.
"""
from __future__ import annotations

from typing import Any

import pg.approval as _approval
import pg.policy as _policy
import pg.schemas as _schemas

from .suites import NEW_CONTRACTS

_INSTALLED = False
_V1_LAYOUT = _approval._target_and_changes


def _layout(tool_name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if tool_name not in NEW_CONTRACTS:
        return _V1_LAYOUT(tool_name, arguments)
    a = arguments
    if tool_name in ("travel_send_email",):
        dest = a["recipients"] + a["cc"] + a["bcc"]
        return ", ".join(dest), {"subject": a["subject"], "body": a["body"], "attachments": a["attachments"],
                                 "cc": a["cc"], "bcc": a["bcc"]}
    if tool_name in ("banking_send_money", "banking_schedule_transaction"):
        return a["recipient"], {k: v for k, v in a.items() if k != "recipient"}
    if tool_name == "banking_update_scheduled_transaction":
        return f"scheduled transaction {a['id']}", {k: v for k, v in a.items() if k != "id" and v is not None}
    if tool_name == "banking_update_password":
        return "account password", {"effect": "replace the account password", "new_password": a["password"]}
    if tool_name == "banking_update_user_info":
        return "account holder details", {k: v for k, v in a.items() if v is not None}
    if tool_name == "slack_add_user_to_channel":
        return f"{a['user']} -> #{a['channel']}", {"effect": "add the user to the channel"}
    if tool_name == "slack_send_direct_message":
        return a["recipient"], {"body": a["body"]}
    if tool_name == "slack_send_channel_message":
        return f"#{a['channel']}", {"body": a["body"]}
    if tool_name == "slack_invite_user_to_slack":
        return a["user"], {"user_email": a["user_email"], "effect": "invite to the workspace"}
    if tool_name == "slack_remove_user_from_slack":
        return a["user"], {"effect": "remove from the workspace"}
    if tool_name == "slack_post_webpage":
        return a["url"], {"content": a["content"]}
    if tool_name == "slack_get_webpage":
        return a["url"], {"effect": "fetch this URL (the URL itself leaves the system)"}
    if tool_name == "travel_create_calendar_event":
        return a["title"], {k: a[k] for k in ("description", "start_time", "end_time", "location", "participants")}
    if tool_name == "travel_cancel_calendar_event":
        return a["event_id"], {"effect": "cancel the event"}
    if tool_name == "travel_reserve_hotel":
        return a["hotel"], {"start_day": a["start_day"], "end_day": a["end_day"]}
    if tool_name == "travel_reserve_car_rental":
        return a["company"], {"start_time": a["start_time"], "end_time": a["end_time"]}
    if tool_name == "travel_reserve_restaurant":
        return a["restaurant"], {"start_time": a["start_time"]}
    raise _approval.CardRenderError(f"no card layout for tool {tool_name!r}")  # pragma: no cover - every new tool is listed


def install() -> None:
    """Idempotent.  Adds the new tools; refuses if a new name already means something else in v1."""
    global _INSTALLED
    if _INSTALLED:
        return
    clash = [n for n in NEW_CONTRACTS if n in _schemas.TOOL_CONTRACTS and _schemas.TOOL_CONTRACTS[n] is not NEW_CONTRACTS[n]]
    if clash:
        raise RuntimeError(f"pgx tool names already defined in v1: {clash}")
    _schemas.TOOL_CONTRACTS.update(NEW_CONTRACTS)
    _approval.ROLE_PERMISSIONS["approver"].update(NEW_CONTRACTS)
    _policy.PRINCIPAL_PERMISSIONS["user_gaurang"]["allowed_tools"].update(NEW_CONTRACTS)
    _approval._target_and_changes = _layout
    _INSTALLED = True


def installed() -> bool:
    return _INSTALLED
