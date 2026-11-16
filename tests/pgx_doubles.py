"""Test doubles for v2: a scripted AgentDojo agent and scripted plain-prompt model clients.  No network, no model.

``ScriptedAgent`` replays tool calls computed from AgentDojo tasks' own ground truth (an injected agent = the injection task's
ground truth, then the user task's) one call per turn, then answers with the user task's ``GROUND_TRUTH_OUTPUT``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.types import ChatAssistantMessage, text_content_block_from_string


class ScriptedAgent(BasePipelineElement):
    name = "scripted-agent"

    def __init__(self, tasks: Sequence[Any], final_text: str | None = None, extra_calls: Sequence[Any] = ()) -> None:
        self.model = "scripted-agent"
        self._tasks = list(tasks)
        self._final = final_text
        self._extra = list(extra_calls)
        self._calls: list[Any] | None = None
        self.usage: list[dict[str, int]] = []

    def query(self, query, runtime, env=None, messages=(), extra_args=None):
        extra_args = {} if extra_args is None else extra_args
        if self._calls is None:
            self._calls = list(self._extra)
            for task in self._tasks:
                self._calls += list(task.ground_truth(env.model_copy(deep=True)))
        done = sum(1 for m in messages if m["role"] == "assistant")
        self.usage.append({"prompt_tokens": 10, "completion_tokens": 2})
        if done < len(self._calls):
            msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")],
                                       tool_calls=[self._calls[done]])
        else:
            final = self._final if self._final is not None else getattr(self._tasks[-1], "GROUND_TRUTH_OUTPUT", "")
            msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(final)], tool_calls=None)
        return query, runtime, env, [*messages, msg], extra_args


@dataclass
class Completion:
    content: str
    prompt_tokens: int = 7
    completion_tokens: int = 1


class ScriptedClient:
    """``.complete(messages, tools)`` answering with ``answer(prompt)``; records every prompt."""

    def __init__(self, answer: Callable[[str], str]) -> None:
        self._answer = answer
        self.prompts: list[str] = []

    def complete(self, messages, tools=()):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        return Completion(self._answer(prompt))


def deny_if(*markers: str) -> ScriptedClient:
    """A reviewer that approves unless the prompt mentions one of ``markers`` (an attacker's address, IBAN...)."""
    return ScriptedClient(lambda p: "DENY" if any(m.lower() in p.lower() for m in markers) else "APPROVE")
