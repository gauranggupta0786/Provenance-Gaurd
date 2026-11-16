"""Deterministic agent stub for Week 1/2 control development.

Per the plan: "If model access is delayed, use a deterministic stub for
control development and keep the model milestone explicitly open." This
stub decides which of the three v1 actions to take from a simple keyword
rule (a real model integration replaces `decide` in Week 5).

Access-boundary note: this module intentionally does NOT import
`pg.tools` or `pg.crypto`. It only ever calls a `propose_channel`
callable handed to it at construction time -- never the underlying
ProposalService, store, or signer. This is a process-level convention
enforced by code review / the negative-access test in Week 2, not yet a
hard security boundary; true isolation is the Week 4-5 "isolated
gateway" and "agent adapter" work.
"""
from __future__ import annotations

from typing import Callable, Any


ProposeChannel = Callable[..., Any]


def make_propose_channel(service) -> ProposeChannel:
    """Wrap a ProposalService in a plain closure so the object handed to
    the agent exposes nothing but a call surface -- no .signer/.store
    attributes for the agent to stumble onto."""

    def _channel(**kwargs):
        return service.propose_action(**kwargs)

    return _channel


class DeterministicAgent:
    def __init__(self, *, session_id: str, principal_id: str, model_id: str,
                 propose_channel: ProposeChannel):
        self.session_id = session_id
        self.principal_id = principal_id
        self.model_id = model_id
        self._propose = propose_channel  # the ONLY way this agent can act

    def handle_request(self, user_goal: str, *, message_id: str = "msg_1",
                        untrusted_context: str = ""):
        """Very simple deterministic rule set standing in for a real model:
        decides one of the three supported tool calls from keywords in the
        (trusted) user goal. `untrusted_context` represents external
        content (e.g. an email body) that may try to steer the agent --
        it is passed through as a source label, never as trusted identity
        or as a reason to bypass policy."""
        goal_lower = user_goal.lower()
        source_labels = {"user_goal": user_goal}
        if untrusted_context:
            source_labels["untrusted_context"] = untrusted_context

        if "summarize" in goal_lower:
            tool_name, arguments = "summarize", {"message_id": message_id, "max_words": 12}
        elif "email" in goal_lower or "send" in goal_lower:
            tool_name, arguments = "send_email", {
                "recipient": "mentor@example.com",
                "subject": "Weekly update",
                "body": "Summary of this week's progress is attached.",
            }
        elif "update" in goal_lower or "record" in goal_lower:
            tool_name, arguments = "update_record", {
                "record_id": "rec_1001", "field": "status", "new_value": "in_review",
            }
        else:
            tool_name, arguments = "summarize", {"message_id": message_id, "max_words": 12}

        return self._propose(
            session_id=self.session_id,
            principal_id=self.principal_id,
            user_goal=user_goal,
            model_id=self.model_id,
            tool_name=tool_name,
            arguments=arguments,
            source_labels=source_labels,
        )
