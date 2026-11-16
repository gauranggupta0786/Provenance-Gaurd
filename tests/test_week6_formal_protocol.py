from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pg.formal_evaluation import (
    DeterministicFinalStateEvaluator,
    METRIC_SPECIFICATION,
    PAPER_CLAIM_MAP,
    bootstrap_paired_mean_ci,
    summarize_formal_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def row(scenario, arm, *, error=None, completed=True, **values):
    result = {
        "scenario_id": scenario, "arm": arm, "seed": 42,
        "kind": "attack", "completed": completed, "error": error,
        "attack_success": False, "benign_success": None, "utility_success": False,
        "latency_ms": 100, "prompt_tokens": 10, "completion_tokens": 2,
        "mediated_calls": [], "security_dimensions": [],
    }
    result.update(values)
    return result


def test_every_requested_metric_and_paper_claim_is_frozen():
    identifiers = {item["id"] for item in METRIC_SPECIFICATION}
    assert identifiers == {
        "targeted_attack_success_rate", "confidentiality_violation_rate",
        "integrity_violation_rate", "authorization_violation_rate",
        "replay_success_rate", "replay_detection_rate", "audit_tamper_detection_rate",
        "benign_task_success", "utility_under_attack", "false_block_rate",
        "model_runtime_failure_rate", "latency_tokens_storage_throughput_approval_burden",
    }
    mapped = set(",".join(PAPER_CLAIM_MAP.values()).split(","))
    assert identifiers <= mapped


def test_primary_denominators_keep_errors_and_sensitivity_analysis_is_separate():
    rows = [
        row("a", "baseline", attack_success=True),
        row("a", "protected", error="model timeout", completed=False),
        row("b", "baseline", security_dimensions=["confidentiality"], confidentiality_violation=True),
        row("b", "protected", security_dimensions=["confidentiality"], confidentiality_violation=False),
    ]
    summary = summarize_formal_protocol(rows)
    assert summary["primary"]["model_runtime_failure_rate"]["denominator"] == 4
    assert summary["primary"]["targeted_attack_success_rate"]["denominator"] == 4
    assert summary["error_free_sensitivity_analysis"]["excluded_errors"] == 1
    assert summary["error_free_sensitivity_analysis"]["rows"] == 3


def test_paired_bootstrap_is_deterministic_and_retains_pair_count():
    first = bootstrap_paired_mean_ci([10, 20, -5], samples=500)
    second = bootstrap_paired_mean_ci([10, 20, -5], samples=500)
    assert first == second
    assert first["pairs"] == 3
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]


def test_deterministic_evaluator_never_uses_judge_for_security():
    scenario = {"scenario_id": "s", "security_dimensions": ["confidentiality", "authorization"]}
    episode = {
        "checks": [{"passed": True}], "final_state_digest": "a" * 64,
        "confidentiality_violation": False, "attack_success": False,
        "audit_traces": [{}], "output_guard": {"decision": "BLOCK"},
    }
    evaluated = DeterministicFinalStateEvaluator().evaluate(scenario, episode)
    assert evaluated["judge_used"] is False
    assert evaluated["confidentiality_violation"] is False
    assert evaluated["authorization_violation"] is False


def test_frozen_protocol_hashes_context_and_judge_policy_are_self_consistent():
    root = ROOT / "data" / "week6" / "frozen_protocol"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["artifacts"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    context = json.loads((root / "context_profile.json").read_text(encoding="utf-8"))
    assert context["dataset_scenarios_profiled"] == 100
    assert len(context["per_scenario"]) == 100
    assert context["required_context_envelope_tokens"] <= context["selected_common_context_bucket"]
    assert context["selected_common_context_bucket"] == 32768  # revised in Week 7: 8192 ignored read-tool outputs
    assert context["read_tool_output_reserve"]["largest_tools"][0] == "list_files"
    judge = json.loads((root / "semantic_judge_protocol.json").read_text(encoding="utf-8"))
    assert judge["enabled"] is False
    assert manifest["security_judge_dependency"] is False


def test_confidentiality_cases_freeze_direct_disclosure_expectations():
    rows = [json.loads(line) for line in (
        ROOT / "data" / "week6" / "frozen_protocol" / "confidentiality_cases.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert any(row["expected"].get("protected_output_decision") == "BLOCK" for row in rows)
    assert all("security_dimensions" in row for row in rows)
