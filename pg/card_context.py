"""Week 11b: say what an opaque id on an approval card refers to.

Week 11 cards showed ``Target: 5`` for a file delete, which nobody (human or model) can judge.  This module looks the id up in
the live environment.  It runs in trusted code, from environment state, and the result is displayed as its own line; nothing the
model wrote is used.  It returns ``None`` when the tool has no opaque id or the id does not exist.
"""
from __future__ import annotations

from typing import Any, Mapping


def _clip(text: Any, limit: int = 80) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_target(environment: Any, tool_name: str, arguments: Mapping[str, Any]) -> str | None:
    try:
        if tool_name in {"workspace_delete_file", "workspace_share_file", "workspace_append_to_file"}:
            item = environment.cloud_drive.files.get(str(arguments["file_id"]))
            if item is None:
                return "no such file"
            shared = ", ".join(sorted(map(str, item.shared_with))) or "nobody"
            return f"file '{item.filename}' owned by {item.owner}, currently shared with {shared}"
        if tool_name in {"workspace_cancel_calendar_event", "workspace_reschedule_calendar_event",
                         "workspace_add_calendar_event_participants"}:
            item = environment.calendar.events.get(str(arguments["event_id"]))
            if item is None:
                return "no such event"
            people = ", ".join(sorted(map(str, item.participants)))
            return f"calendar event '{_clip(item.title)}' at {item.start_time:%Y-%m-%d %H:%M} with {people}"
        if tool_name == "workspace_delete_email":
            item = environment.inbox.emails.get(str(arguments["email_id"]))
            if item is None:
                return "no such email"
            return f"email '{_clip(item.subject)}' from {item.sender}"
    except Exception:  # a description is a convenience; the signed record is still shown in full
        return None
    return None


def _norm(text: Any) -> str:
    return " ".join(str(text).split()).lower()


def copied_sources(environment: Any, tool_name: str, arguments: Mapping[str, Any], *, window: int = 40, step: int = 10) -> list[str]:
    """Drive files whose text an outgoing e-mail body copies verbatim (any ``window``-character run, whitespace-normalised).

    Trusted code, live environment state; used by the request-scope check, never shown to the model.
    """
    try:
        if tool_name != "workspace_send_email":
            return []
        body = _norm(arguments.get("body") or "")
        if len(body) < window:
            return []
        pieces = {body[i:i + window] for i in range(0, len(body) - window + 1, step)}
        found = []
        for item in environment.cloud_drive.files.values():
            content = _norm(item.content)
            if any(p in content for p in pieces):
                found.append(item.filename)
        return sorted(found)
    except Exception:
        return []
