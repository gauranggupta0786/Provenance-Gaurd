"""rq4_cost_report.py, checked both against a hand-built episode tree (covering the delta-vs-C0 arithmetic exactly)
and against the real results/week11c episodes on disk, which share this project's episode schema (prompt_tokens,
latency_ms, hook_result.reviewer.*) even though they predate the held-out/adaptive-evasion runs this script targets --
real field shapes, not invented ones."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pg.week8_protocol import ROOT, code_digest
from scripts.week11_lib import week11_code_digest
from scripts import rq4_cost_report as report

REAL_EPISODES = list((Path(ROOT) / "results" / "week11c").glob("*/*/episodes.jsonl"))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _episode(config, *, prompt_tokens=100, completion_tokens=20, latency_ms=1000, cards=0, approved=0, denied=0,
             capped=0, scope_denials=0, reviewer_prompt_tokens=0, reviewer_completion_tokens=0, decisions=None,
             reviewer_errors=0, error=None, model_id="m1"):
    return {
        "model_id": model_id, "week11_config": config, "family": "content_attack",
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "latency_ms": latency_ms,
        "error": error,
        "hook_result": {"reviewer": {
            "cards": cards, "approved": approved, "denied": denied, "capped": capped,
            "scope_denials": scope_denials, "prompt_tokens": reviewer_prompt_tokens,
            "completion_tokens": reviewer_completion_tokens, "errors": reviewer_errors,
            "decisions": decisions or [],
        }},
    }


# ---------------------------------------------------------------- episode_cost field extraction

def test_episode_cost_pulls_the_right_fields():
    row = _episode("SCP-v3", prompt_tokens=200, completion_tokens=30, latency_ms=4000, cards=1, approved=0,
                    denied=1, reviewer_prompt_tokens=50, reviewer_completion_tokens=5,
                    decisions=[{"latency_ms": 300}], error=None)
    c = report.episode_cost(row)
    assert c["agent_prompt_tokens"] == 200
    assert c["agent_completion_tokens"] == 30
    assert c["episode_latency_ms"] == 4000
    assert c["cards"] == 1 and c["denied"] == 1
    assert c["reviewer_prompt_tokens"] == 50 and c["reviewer_completion_tokens"] == 5
    assert c["reviewer_decision_latency_ms"] == 300
    assert c["errored"] is False


def test_episode_cost_sums_multiple_decision_latencies():
    row = _episode("SCP-v3", decisions=[{"latency_ms": 100}, {"latency_ms": 250}])
    c = report.episode_cost(row)
    assert c["reviewer_decision_latency_ms"] == 350


def test_episode_cost_handles_missing_reviewer_block_gracefully():
    row = {"model_id": "m1", "week11_config": "C0", "prompt_tokens": 10, "completion_tokens": 2, "latency_ms": 500,
           "error": None, "hook_result": {}}
    c = report.episode_cost(row)
    assert c["cards"] == 0
    assert c["reviewer_decision_latency_ms"] == 0


def test_episode_cost_flags_errored_episode():
    row = _episode("SC-v3", error="ReviewerDenied")
    assert report.episode_cost(row)["errored"] is True


# ---------------------------------------------------------------- aggregate

def test_aggregate_computes_approval_and_denial_rates():
    costs = [
        report.episode_cost(_episode("SCP-v3", cards=2, approved=1, denied=1)),
        report.episode_cost(_episode("SCP-v3", cards=2, approved=2, denied=0)),
    ]
    s = report.aggregate(costs)
    assert s["n"] == 2
    assert s["approval_rate"] == 3 / 4
    assert s["denial_rate"] == 1 / 4


def test_aggregate_with_no_cards_reports_rates_as_none_not_divide_by_zero():
    costs = [report.episode_cost(_episode("C0", cards=0))]
    s = report.aggregate(costs)
    assert s["approval_rate"] is None
    assert s["denial_rate"] is None


def test_aggregate_empty_group_reports_n_zero():
    assert report.aggregate([]) == {"n": 0}


def test_aggregate_error_rate():
    costs = [report.episode_cost(_episode("SC-v3", error=None)), report.episode_cost(_episode("SC-v3", error="x")),
              report.episode_cost(_episode("SC-v3", error=None)), report.episode_cost(_episode("SC-v3", error=None))]
    s = report.aggregate(costs)
    assert s["error_rate"] == 0.25


# ---------------------------------------------------------------- end to end: load_rows + main, on a hand-built tree

def test_load_rows_reads_every_root_and_tags_family(tmp_path):
    root_a = tmp_path / "heldout"
    root_b = tmp_path / "adaptive_evasion"
    (root_a / "m1" / "C0").mkdir(parents=True)
    (root_b / "m1" / "SCP-v3").mkdir(parents=True)
    _write_jsonl(root_a / "m1" / "C0" / "episodes.jsonl", [_episode("C0")])
    _write_jsonl(root_b / "m1" / "SCP-v3" / "episodes.jsonl", [_episode("SCP-v3")])
    rows = report.load_rows([root_a, root_b])
    assert len(rows) == 2
    assert {r["_root"] for r in rows} == {"heldout", "adaptive_evasion"}


def test_main_reports_delta_vs_c0(tmp_path, capsys):
    root = tmp_path / "heldout"
    (root / "m1" / "C0").mkdir(parents=True)
    (root / "m1" / "SCP-v3").mkdir(parents=True)
    _write_jsonl(root / "m1" / "C0" / "episodes.jsonl",
                 [_episode("C0", prompt_tokens=100, completion_tokens=10, latency_ms=1000, cards=1, approved=1)])
    _write_jsonl(root / "m1" / "SCP-v3" / "episodes.jsonl",
                 [_episode("SCP-v3", prompt_tokens=150, completion_tokens=10, latency_ms=2000, cards=1, denied=1,
                           reviewer_prompt_tokens=40, decisions=[{"latency_ms": 200}])])
    rc = report.main(["--results-root", str(root)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SCP-v3" in out and "C0" in out
    assert "delta vs C0" in out
    assert "agent tokens +50" in out  # (150+10) - (100+10)


def test_main_with_no_episodes_reports_cleanly(tmp_path, capsys):
    rc = report.main(["--results-root", str(tmp_path / "nothing")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "no episodes under" in out


# ---------------------------------------------------------------- real-schema smoke test (not invented data)

def test_real_week11c_episodes_parse_without_raising():
    if not REAL_EPISODES:
        # The hand-off package does not ship the ~100 MB of local-model episode logs; this check needs them.
        pytest.skip("results/week11c is not in this tree (the hand-off package ships the method, not the local-model results)")
    rows = report.load_rows([Path(ROOT) / "results" / "week11c"])
    assert len(rows) > 1000  # 5 configs x 480 episodes per config, per the schema check done while building this
    costs = [report.episode_cost(r) for r in rows]
    groups = report.group_by_config(costs)
    assert "C0" in groups
    s = report.aggregate(groups["C0"])
    assert s["n"] > 0
    assert s["mean_agent_tokens"] > 0


def test_no_rq4_cost_report_file_enters_either_frozen_digest():
    _, w8 = code_digest(ROOT)
    _, w11 = week11_code_digest(ROOT)
    hashed = set(w8) | set(w11)
    offenders = sorted(n for n in hashed if "rq4_cost_report" in n)
    assert offenders == [], f"{offenders} are in a frozen digest"
