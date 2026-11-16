"""Simulated tools operating on simulated data only, per the plan's v1 scope.

IMPORTANT (Week 2 Day 1 exit check): nothing in this module is imported by
`pg.agent`. Only privileged/gateway-style code (tests, and later the Week 4
execution gateway) is meant to call these functions directly. The agent's
only path to the world is `pg.builder.propose_action`, which never touches
these functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldState:
    mailbox: list[dict] = field(default_factory=list)
    records: dict[str, dict] = field(default_factory=lambda: {
        "rec_1001": {"status": "open", "notes": "", "owner": "user_gaurang"},
        "rec_1002": {"status": "open", "notes": "", "owner": "user_gaurang"},
    })
    inbox: dict[str, dict] = field(default_factory=lambda: {
        "msg_1": {"body": "Please review the attached weekly report and confirm scope."},
    })


def summarize_impl(world: WorldState, *, message_id: str, max_words: int) -> str:
    msg = world.inbox.get(message_id)
    if msg is None:
        raise KeyError(f"no such message: {message_id}")
    words = msg["body"].split()
    return " ".join(words[:max_words])


def send_email_impl(world: WorldState, *, recipient: str, subject: str, body: str) -> dict:
    entry = {"recipient": recipient, "subject": subject, "body": body}
    world.mailbox.append(entry)
    return entry


def update_record_impl(world: WorldState, *, record_id: str, field: str, new_value: str) -> dict:
    if record_id not in world.records:
        raise KeyError(f"no such record: {record_id}")
    world.records[record_id][field] = new_value
    return dict(world.records[record_id])
