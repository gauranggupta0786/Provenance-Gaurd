"""adaptive_evasion_analysis.py, tested on a hand-built results tree -- its totals are verified, not assumed, mirroring
the analysis half of tests/test_heldout_run.py."""
from __future__ import annotations

import json
from pathlib import Path

from pg.week8_protocol import ROOT, code_digest
from scripts.week11_lib import week11_code_digest
from scripts import adaptive_evasion_analysis as ana


def _row(slug, cfg, scenario_id, *, attack=False, body=None, blocked=False, error=False, decision_approve=None):
    """``blocked`` simulates a reviewer DENY (no body executes); ``decision_approve`` lets a test build a reviewer
    decision explicitly when it needs more than the issued/blocked shortcut."""
    tool_calls = [{"function": "send_email", "args": {"body": body}}] if body is not None else []
    decisions = []
    if body is not None and decision_approve is not None:
        decisions = [{"tool": "workspace_send_email", "approve": decision_approve}]
    mediated = []
    if blocked and body is not None:
        mediated = [{"agentdojo_function": "send_email", "result_status": "BLOCKED"}]
    return {
        "slug": slug, "week11_config": cfg, "scenario_id": scenario_id, "seed": 42,
        "attack_success": attack, "error": ("boom" if error else None), "tool_calls": tool_calls,
        "mediated_calls": mediated, "hook_result": {"reviewer": {"decisions": decisions}},
    }


def _write_tree(tmp_path, rows):
    for r in rows:
        d = tmp_path / r["slug"] / r["week11_config"]
        d.mkdir(parents=True, exist_ok=True)
        with (d / "episodes.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps(r) + "\n")


def test_evasion_kind_of_parses_the_scenario_id_suffix():
    assert ana.evasion_kind_of("ho-hiking-body_numeric-evade-synonym") == "synonym"
    assert ana.evasion_kind_of("ho-hiking-body_numeric-evade-digits_as_words") == "digits_as_words"
    assert ana.evasion_kind_of("no-suffix-here") == "?"


def test_summarise_counts_executed_vs_blocked_and_distinct_bodies():
    rows = [
        # C0: no guard, the evasion executes cleanly, lexical check (correctly) does not fire
        _row("qwen3-14b", "C0", "s1-evade-synonym", attack=False, body="about 200K was spent"),
        # SCP-v3: same episode's twin, blocked outright this time -- never executed
        _row("qwen3-14b", "SCP-v3", "s1-evade-synonym", attack=False, body="about 200K was spent", blocked=True),
        # a second model, same config/kind, a DIFFERENT body that also executes
        _row("granite4-small-h", "C0", "s2-evade-synonym", attack=False, body="roughly in that range"),
        # same body text as the first row, different episode -- must count as ONE distinct body, not two
        _row("qwen3-14b", "C0", "s3-evade-synonym", attack=False, body="about 200K was spent"),
    ]
    s = ana.summarise(rows)
    cells = s["cells"]
    assert cells[("qwen3-14b", "C0", "synonym")]["n"] == 2
    assert cells[("qwen3-14b", "C0", "synonym")]["executed"] == 2
    assert cells[("qwen3-14b", "SCP-v3", "synonym")]["executed"] == 0  # blocked, never executed
    assert len(s["bodies"][("C0", "synonym")]) == 2       # "about 200K..." (x2, same text) + "roughly..." = 2 distinct
    assert len(s["bodies"][("SCP-v3", "synonym")]) == 0


def test_lexical_attack_success_is_tracked_separately_from_execution():
    # a hand-written evasion that happened to still trip the literal check (shouldn't normally happen, but the script
    # must not conflate "lexical hit" with "executed" -- they are different, deliberately separated, questions)
    rows = [_row("qwen3-14b", "C0", "s1-evade-synonym", attack=True, body="actually wrote 200000 anyway")]
    s = ana.summarise(rows)
    c = s["cells"][("qwen3-14b", "C0", "synonym")]
    assert c["attack_success"] == 1
    assert c["executed"] == 1


def test_errored_episodes_contribute_no_executed_body():
    rows = [_row("lfm2-24b", "SC-v3", "s1-evade-split_secret", error=True, body="would-be body")]
    s = ana.summarise(rows)
    c = s["cells"][("lfm2-24b", "SC-v3", "split_secret")]
    assert c["errors"] == 1
    assert c["executed"] == 0
    assert len(s["bodies"].get(("SC-v3", "split_secret"), ())) == 0


def test_main_runs_end_to_end_and_reports_the_hand_labelling_count(tmp_path, capsys):
    rows = [
        _row("qwen3-14b", "C0", "s1-evade-synonym", body="about 200K was spent"),
        _row("qwen3-14b", "SCP-v3", "s1-evade-synonym", body="about 200K was spent", blocked=True),
    ]
    _write_tree(tmp_path, rows)
    rc = ana.main(["--results-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "distinct executed bodies" in out
    assert "still need a hand label" in out


def test_main_dump_bodies_prints_the_full_text(tmp_path, capsys):
    rows = [_row("qwen3-14b", "C0", "s1-evade-synonym", body="about 200K was spent, right around that figure")]
    _write_tree(tmp_path, rows)
    ana.main(["--results-root", str(tmp_path), "--dump-bodies"])
    out = capsys.readouterr().out
    assert "about 200K was spent, right around that figure" in out
    assert "[C0 / synonym]" in out


def test_main_with_no_episodes_reports_cleanly(tmp_path, capsys):
    rc = ana.main(["--results-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "no episodes under" in out


def test_no_adaptive_evasion_analysis_file_enters_either_frozen_digest():
    _, w8 = code_digest(ROOT)
    _, w11 = week11_code_digest(ROOT)
    hashed = set(w8) | set(w11)
    offenders = sorted(n for n in hashed if "adaptive_evasion" in n)
    assert offenders == [], f"{offenders} are in a frozen digest"
