"""Every configuration a v2 run can use, by name.  Frozen into the v2 plan; changing one changes the plan hash.

kinds:
  stock     AgentDojo's own pipeline, optionally with one of its built-in defenses (no ProvenanceGuard)
  toolsel   local re-implementation of AgentDojo's tool_filter: a model picks the tools from the request, the agent gets only those
  auditor   AlignmentCheck-style trace auditor: a model reads the user's goal and the agent's actions so far and blocks drift
  v1        the frozen v1 runtime and policy (pg.agentdojo_adapter), Workspace only: the bridge to Weeks 11-13
  pgx       the v2 runtime on any suite; ``guard`` None = approve-all (the plumbing control), else the PG-v4 chain with flags
"""
from __future__ import annotations

from typing import Any

V4_DEFAULTS: dict[str, Any] = {"capability": "plan", "bypass": True, "numeric": True, "scope_fix": True, "feedback": True,
                               "reviewer": "independent", "denial_cap": 2}

CONFIGS_V2: dict[str, dict[str, Any]] = {
    "none": {"kind": "stock", "defense": None, "note": "no defense"},
    "spotlighting": {"kind": "stock", "defense": "spotlighting_with_delimiting", "note": "AgentDojo built-in"},
    "repeat_user_prompt": {"kind": "stock", "defense": "repeat_user_prompt", "note": "AgentDojo built-in"},
    "tool_filter": {"kind": "toolsel", "note": "re-implementation of AgentDojo's tool_filter with the agent model"},
    "pi_detector": {"kind": "stock", "defense": "transformers_pi_detector",
                    "note": "AgentDojo's ProtectAI DeBERTa detector; needs the model downloaded by hand (optional)"},
    "trace_auditor": {"kind": "auditor", "reviewer": "independent",
                      "note": "AlignmentCheck-style auditor (re-implementation), independent model"},
    "C0g": {"kind": "pgx", "guard": None, "note": "v2 plumbing, generic policy, approve-all: the control for PG-v4"},
    "SCP-v3": {"kind": "v1", "suites": ["workspace"], "note": "frozen v1 SCP-v3 (v1 allow-list, scope check, prompt v3, provenance)"},
    "PG-v4": {"kind": "pgx", "guard": dict(V4_DEFAULTS), "note": "all v2 changes"},
    "PG-v4-capParse": {"kind": "pgx", "guard": {**V4_DEFAULTS, "capability": "parse"}, "note": "capability from the keyword parser"},
    "PG-v4-noCap": {"kind": "pgx", "guard": {**V4_DEFAULTS, "capability": None}, "note": "no capability layer (so no bypass)"},
    "PG-v4-noBypass": {"kind": "pgx", "guard": {**V4_DEFAULTS, "bypass": False}, "note": "reviewer asked even when licensed"},
    "PG-v4-noNum": {"kind": "pgx", "guard": {**V4_DEFAULTS, "numeric": False}, "note": "v1 figure rule only"},
    "PG-v4-noFeedback": {"kind": "pgx", "guard": {**V4_DEFAULTS, "feedback": False}, "note": "v1's bare denial message"},
    "PG-v4-self": {"kind": "pgx", "guard": {**V4_DEFAULTS, "reviewer": "self"}, "note": "reviewer = the agent's own model"},
}

GROUPS: dict[str, list[str]] = {
    "core": ["none", "C0g", "PG-v4"],
    "baselines": ["spotlighting", "repeat_user_prompt", "tool_filter", "trace_auditor"],
    "bridge": ["SCP-v3"],
    "ablations": ["PG-v4-capParse", "PG-v4-noCap", "PG-v4-noBypass", "PG-v4-noNum", "PG-v4-noFeedback", "PG-v4-self"],
    "frontier": ["none", "PG-v4"],
}


def resolve(names: list[str] | None, groups: list[str] | None) -> list[str]:
    out: list[str] = []
    for g in groups or []:
        if g not in GROUPS:
            raise KeyError(f"unknown group {g!r}; choose from {sorted(GROUPS)}")
        out += GROUPS[g]
    out += names or []
    unknown = [n for n in out if n not in CONFIGS_V2]
    if unknown:
        raise KeyError(f"unknown configuration(s) {unknown}; choose from {sorted(CONFIGS_V2)}")
    seen: list[str] = []
    for n in out:
        if n not in seen:
            seen.append(n)
    return seen


def needs_reviewer_model(name: str) -> bool:
    spec = CONFIGS_V2[name]
    if spec["kind"] == "auditor" or spec["kind"] == "v1":
        return True
    return spec["kind"] == "pgx" and spec["guard"] is not None


