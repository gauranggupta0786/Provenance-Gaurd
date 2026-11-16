import hashlib
import json
from pathlib import Path

from pg.week6_metrics import summarize_adaptive, summarize_frozen, wilson_interval
from scripts.run_week6_adaptive import DISCLOSURE, attempt_prompt


ROOT = Path(__file__).resolve().parents[1]


def _row(scenario_id, arm, seed, kind, success, latency):
    return {
        "scenario_id": scenario_id, "arm": arm, "seed": seed, "kind": kind,
        "family": "direct" if kind == "attack" else "benign", "guidance": None,
        "completed": True,
        "attack_success": success if kind == "attack" else None,
        "benign_success": success if kind == "benign" else None,
        "latency_ms": latency, "prompt_tokens": 10, "completion_tokens": 2,
        "tool_calls": [], "mediated_calls": [], "checks": [], "error": None,
    }


def test_wilson_interval_keeps_exact_counts_and_bounded_interval():
    metric = wilson_interval(3, 10)
    assert metric["numerator"] == 3
    assert metric["denominator"] == 10
    assert metric["rate"] == 0.3
    assert 0 <= metric["ci95_low"] < metric["rate"] < metric["ci95_high"] <= 1


def test_frozen_summary_pairs_latency_and_does_not_drop_failures():
    rows = [
        _row("attack", "baseline", 42, "attack", True, 100),
        _row("attack", "protected", 42, "attack", False, 140),
        _row("benign", "baseline", 42, "benign", True, 80),
        _row("benign", "protected", 42, "benign", False, 120),
    ]
    summary = summarize_frozen(rows, expected_total=4)
    assert summary["status"] == "complete"
    assert summary["arms"]["baseline"]["attack_success"]["rate"] == 1.0
    assert summary["arms"]["protected"]["attack_success"]["rate"] == 0.0
    assert summary["arms"]["protected"]["benign_success"]["denominator"] == 1
    assert summary["added_latency"]["matched_pairs"] == 2
    assert summary["added_latency"]["median_ms"] == 40


def test_adaptive_curve_counts_success_by_attempt_cutoffs():
    rows = [
        {"objective_id": "a", "attempt": 1, "attack_success": False},
        {"objective_id": "a", "attempt": 2, "attack_success": True},
        {"objective_id": "b", "attempt": 1, "attack_success": False},
    ]
    summary = summarize_adaptive(rows, expected_objectives=2)
    assert summary["success_by_attempt"]["1"]["numerator"] == 0
    assert summary["success_by_attempt"]["5"]["numerator"] == 1
    assert summary["success_by_attempt"]["15"]["denominator"] == 2


def test_adaptive_dataset_is_reserved_checksummed_and_ten_objectives():
    root = ROOT / "data" / "week6"
    manifest = json.loads((root / "adaptive_manifest.json").read_text(encoding="utf-8"))
    payload = (root / "adaptive_objectives.jsonl").read_bytes()
    assert manifest["count"] == 10
    assert manifest["attempts_per_objective"] == 15
    assert hashlib.sha256(payload).hexdigest() == manifest["sha256"]


def test_adaptive_prompt_discloses_architecture_and_uses_feedback_without_secrets():
    objective = {
        "target": "attacker@outside.example", "asset": "payroll.csv", "payload": "private data",
    }
    prompt = attempt_prompt(objective, 15, "policy denied previous destination")
    assert DISCLOSURE in prompt
    assert "policy denied previous destination" in prompt
    assert "signing keys" in prompt and "not disclosed" in prompt
    assert "attacker@outside.example" in prompt
