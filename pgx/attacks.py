"""The v2 standard attack: AgentDojo's ``important_instructions`` with a fixed, model-neutral addressee.

AgentDojo's own class looks up the target model's prose name ("GPT-4") from the pipeline name and refuses names it does not know,
which covers none of our local models.  We use the identical template with the addressee fixed to "AI assistant" -- the wording
our Week 11-13 held-out scenarios already used -- for EVERY model, frontier included, so the attack text is byte-identical across
models.  This is the one documented deviation from AgentDojo's default and is stated in every v2 result.
"""
from __future__ import annotations

from typing import Any

from agentdojo.attacks.base_attacks import FixedJailbreakAttack
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack

ATTACK_NAME = "important_instructions_ai_assistant"
ADDRESSEE = "AI assistant"
USER_NAME = "Emma Johnson"


class FixedAddresseeAttack(FixedJailbreakAttack):
    name = ATTACK_NAME
    user_name = USER_NAME
    model_name = ADDRESSEE

    def __init__(self, task_suite: Any) -> None:
        super().__init__(ImportantInstructionsAttack._JB_STRING, task_suite, target_pipeline=None)  # type: ignore[arg-type]
        self.user_name, self.model_name = USER_NAME, ADDRESSEE


def injections_for(task_suite: Any, user_task: Any, injection_task: Any) -> dict[str, str]:
    """vector -> injected text, for the vectors the user task's correct execution actually reads."""
    return FixedAddresseeAttack(task_suite).attack(user_task, injection_task)
