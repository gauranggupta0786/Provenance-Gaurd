"""Task #29: the naive, no-ProvenanceGuard baseline defenses -- tested offline, without a live model server.

These exercise only the parts that don't need a running Ollama server: defense-name validation, pipeline construction
(AgentPipeline.from_config makes no network call), the tool_allowlist computation against the real AgentDojo workspace
suite, and that this new file stays outside both frozen code digests (the same discipline as the Week 12/13/held-out
wrappers). The full ``run_baseline_defended_episode`` path (an actual model call) is exercised on the cluster, not here.
"""
from __future__ import annotations

import json

import pytest

from agentdojo.task_suite.load_suites import get_suite
from pg.agentdojo_adapter import BENCHMARK_VERSION, SUITE_NAME
from pg.diagnostics import RuntimeLimits
from pg.ollama_model import SeededOllamaLLM
from pg.week8_protocol import ROOT, code_digest
from scripts.week11_lib import week11_code_digest
from scripts import baseline_defense_lib as lib

SUITE = get_suite(BENCHMARK_VERSION, SUITE_NAME)


def test_four_defenses_are_named_and_only_two_are_agentdojo_builtins():
    assert lib.DEFENSE_NAMES == ("none", "spotlighting_with_delimiting", "repeat_user_prompt", "tool_allowlist")
    # tool_allowlist is a local substitute for AgentDojo's own OpenAI-only tool_filter -- it must not claim that name
    assert "tool_filter" not in lib.AGENTDOJO_DEFENSE_FOR
    assert set(lib.AGENTDOJO_DEFENSE_FOR) == {"spotlighting_with_delimiting", "repeat_user_prompt"}


def test_pi_detector_is_optional_and_never_raises():
    # no network route to huggingface.co from this project's compute nodes, so this must come back False, not throw
    assert lib.pi_detector_available() in (True, False)


def test_build_pipeline_constructs_for_every_named_defense_with_no_network_call():
    llm = SeededOllamaLLM(seed=42, runtime_limits=RuntimeLimits())
    for defense in lib.DEFENSE_NAMES:
        pipeline = lib.build_pipeline(defense, llm)
        assert pipeline is not None
        if defense in lib.AGENTDOJO_DEFENSE_FOR:
            assert pipeline.name.endswith(defense)


def test_build_pipeline_refuses_an_unknown_defense_name():
    llm = SeededOllamaLLM(seed=42, runtime_limits=RuntimeLimits())
    with pytest.raises(ValueError, match="unknown defense"):
        lib.build_pipeline("not-a-real-defense", llm)


def test_tool_allowlist_keeps_read_tools_and_only_the_declared_write_tool():
    scenario = {"allowed_writes": [{"target": "mentor@example.com", "tool": "workspace_send_email"}]}
    names = lib.allowed_tool_names(scenario, SUITE.tools)
    assert "send_email" in names  # the one declared write tool, translated from the canonical name
    assert "delete_file" not in names and "share_file" not in names  # no other write tool is let through
    for always in lib.READ_ONLY_ALWAYS:
        assert always in names  # every scenario keeps full read access


def test_tool_allowlist_with_no_declared_writes_keeps_only_read_tools():
    names = lib.allowed_tool_names({"allowed_writes": []}, SUITE.tools)
    # get_unread_emails is technically mediated as a write (it mutates read-state) but is in READ_ONLY_ALWAYS on purpose
    write_tools = set(lib.CANONICAL_TO_AGENTDOJO.values()) - {"get_unread_emails"}
    assert not (set(names) & write_tools)


def test_tool_allowlist_never_claims_a_tool_the_suite_does_not_expose():
    # an unrecognised canonical write name must be dropped, not passed through verbatim
    names = lib.allowed_tool_names({"allowed_writes": [{"target": "x", "tool": "not_a_real_canonical_tool"}]}, SUITE.tools)
    present = {t.name for t in SUITE.tools}
    assert set(names) <= present


def test_no_baseline_defense_campaign_file_enters_either_frozen_digest():
    _, w8 = code_digest(ROOT)
    _, w11 = week11_code_digest(ROOT)
    hashed = set(w8) | set(w11)
    offenders = sorted(n for n in hashed if "baseline_defense" in n)
    assert offenders == []


# ------------------------------------------------------------------ campaign resumability and analysis, hand-built rows

from scripts import baseline_defenses_analysis as ana


def _bd_row(slug, defense, family, *, attack=False, benign=False, error=False):
    return {
        "slug": slug, "baseline_defense": defense, "arm": f"baseline+{defense}", "family": family,
        "scenario_id": f"{family}-x", "seed": 42, "attack_success": attack, "benign_success": benign,
        "error": ("boom" if error else None),
    }


def test_analysis_totals_attack_and_clean_rows(tmp_path):
    rows = [
        _bd_row("qwen3-14b", "none", "policy_passing", attack=True),
        _bd_row("qwen3-14b", "none", "policy_passing", attack=False),
        _bd_row("qwen3-14b", "none", "clean_control", benign=True),
        _bd_row("qwen3-14b", "spotlighting_with_delimiting", "policy_passing", attack=False),
    ]
    for r in rows:
        d = tmp_path / r["slug"] / r["baseline_defense"]
        d.mkdir(parents=True, exist_ok=True)
        with (d / "episodes.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps(r) + "\n")
    cells = ana.summarise(ana.load_rows(tmp_path))
    assert cells[("qwen3-14b", "none", "policy_passing")]["attack_success"] == 1
    assert cells[("qwen3-14b", "none", "policy_passing")]["n"] == 2
    assert cells[("qwen3-14b", "none", "clean_control")]["benign_success"] == 1
    assert cells[("qwen3-14b", "spotlighting_with_delimiting", "policy_passing")]["attack_success"] == 0


def test_no_baseline_defense_file_enters_either_frozen_digest():
    _, w8 = code_digest(ROOT)
    _, w11 = week11_code_digest(ROOT)
    hashed = set(w8) | set(w11)
    offenders = sorted(n for n in hashed if "baseline_defense" in n)
    assert offenders == [], f"{offenders} are in a frozen digest; baseline-defense code must live outside pg/ and both lists"
