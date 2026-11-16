"""Every experiment of the report, runnable against a hosted model: the suites.

A suite is a pinned dataset, an ordered list of configurations and an episode kind.  These tests check that the pins are the ones
the frozen libraries use, that the original ladder plans did not move, that each of the four episode kinds (config, defense,
arms, attacker) runs end to end against ``frontier/mock_server.py`` -- a scripted server that is explicitly not a model -- and
that the budget, resume, seed and report behaviour holds across kinds.  The scripted answers themselves are pinned by
``scripts/frontier_selftest.py``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier.campaign import CampaignStop, RunSettings, build_plan, planned_episodes, run_campaign, select
from frontier.mock_server import start_server
from frontier.openai_adapter import RemoteConfig
from frontier.suites import DATASETS, GROUPS, LEGACY_DATASET_ALIASES, SUITES, resolve, suite_for
from pg.week8_protocol import ProtocolViolation
from scripts import frontier_report
from scripts.handoff_verify import DATASET_HASHES
from scripts.week11e_lib import WEEK11D_DATASET_SHA256

KEY = "sk-test-0123456789abcdef"
QUIET = dict(log=lambda message: None)
LADDER_PLAN_SHA = {"heldout": "abe2be6dfd0a6634af99f80e7bc96e3afc71d3910ec067c06df46127fc8284d4",
                   "tuning": "3cedb064dfa1b66b56b083af89587a6c6f826d25e5f3a56d66747fcc3ce29b38"}


@pytest.fixture(scope="module")
def mock():
    server, state = start_server(0, model="mock-model", api_key=KEY, reviewer="strict")
    yield server, state
    server.shutdown()
    server.server_close()


def config_for(server, **overrides) -> RemoteConfig:
    base = dict(api_base=f"http://127.0.0.1:{server.server_address[1]}/v1", model="mock-model", api_key=KEY, max_retries=0)
    base.update(overrides)
    return RemoteConfig(**base)


def saved(root: Path, suite: str) -> list[dict]:
    return [json.loads(line) for path in sorted(root.glob(f"{suite}/*/*/episodes.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- pins and plans

def test_every_dataset_pin_agrees_with_the_verifier_and_the_frozen_libraries():
    assert {v["path"]: v["sha256"] for v in DATASETS.values()} == DATASET_HASHES
    assert DATASETS["tuning"]["sha256"] == WEEK11D_DATASET_SHA256


@pytest.mark.parametrize("key", list(SUITES))
def test_every_suite_builds_a_deterministic_pinned_plan(key):
    first, second = build_plan(key), build_plan(key)
    assert first == second and first["plan_sha256"]
    assert first["configs"] and first.get("suite", key) == key
    assert planned_episodes(select(first)) > 0


@pytest.mark.parametrize("key,sha", LADDER_PLAN_SHA.items())
def test_the_original_ladder_plans_did_not_move(key, sha):
    """The held-out and tuning plans are what the measured results were drawn from; widening the runner must not change them."""

    assert build_plan(key)["plan_sha256"] == sha


def test_suite_sizes_are_the_ones_in_the_report():
    sizes = {key: planned_episodes(select(build_plan(key))) for key in SUITES}
    assert sizes == {"heldout": 648, "tuning": 432, "adaptive_evasion": 648, "baselines": 864, "w11": 504, "w11b": 420, "w11c": 600,
                     "week8": 600, "week8_adaptive": 10}


def test_groups_and_names_resolve_and_unknown_ones_are_refused():
    assert resolve(["pdf"]) == list(SUITES) or set(resolve(["pdf"])) == set(SUITES)
    assert set(GROUPS["main"]) == {"heldout", "adaptive_evasion", "baselines"}
    assert set(GROUPS["main"]) | set(GROUPS["appendix"]) == set(SUITES)
    assert resolve(["heldout", "heldout"]) == ["heldout"]
    assert LEGACY_DATASET_ALIASES["both"] == ("tuning", "heldout")
    with pytest.raises(ProtocolViolation):
        suite_for("no-such-suite")


def test_only_the_ladder_suites_with_a_model_reviewer_use_one():
    assert {k for k, s in SUITES.items() if s.model_reviewer} <= {"heldout", "tuning", "adaptive_evasion", "w11", "w11b", "w11c"}
    assert not SUITES["baselines"].model_reviewer and not SUITES["week8"].model_reviewer and not SUITES["week8_adaptive"].model_reviewer


# --------------------------------------------------------------------------- the four kinds run

@pytest.mark.parametrize("suite,scenario,configs", [
    ("baselines", "ho-hiking-delete_file_hr", ["none", "tool_allowlist"]),           # defense
    ("w11", "pp-email-mail_attacker", ["B0", "P-R1"]),                                # config, Week-11 spec
    ("week8", "heldout-direct-01", ["baseline", "protected"]),                        # arms
])
def test_each_kind_runs_and_records_its_suite(mock, tmp_path, suite, scenario, configs):
    server, _ = mock
    out = run_campaign(suite_key=suite, agent=config_for(server), selection_args={"scenario_ids": [scenario], "seeds": [42], "configs": configs},
                       results_root=tmp_path, **QUIET)
    rows = saved(tmp_path, suite)
    assert len(rows) == len(configs) and not any(r["error"] for r in rows)
    assert {r["week11_config"] for r in rows} == set(configs) and {r["frontier_suite"] for r in rows} == {suite}
    assert out["suite"] == suite and all(r["frontier_usage"]["agent"]["calls"] > 0 for r in rows)


def test_the_cheap_defenses_change_what_they_say_they_change(mock, tmp_path):
    """The allow-list removes the delete tool the task did not declare; spotlighting changes nothing for a scripted agent."""

    server, _ = mock
    run_campaign(suite_key="baselines", agent=config_for(server), selection_args={"scenario_ids": ["ho-hiking-delete_file_hr"], "seeds": [42]},
                 results_root=tmp_path, **QUIET)
    got = {r["week11_config"]: r["attack_success"] for r in saved(tmp_path, "baselines")}
    assert got["none"] and got["spotlighting_with_delimiting"] and got["repeat_user_prompt"] and not got["tool_allowlist"]
    assert all(r["baseline_defense"] for r in saved(tmp_path, "baselines"))


def test_the_stock_arm_is_hijacked_and_the_protected_arm_is_not(mock, tmp_path):
    server, _ = mock
    run_campaign(suite_key="week8", agent=config_for(server), selection_args={"scenario_ids": ["heldout-indirect-01"], "seeds": [42]},
                 results_root=tmp_path, **QUIET)
    got = {r["week11_config"]: r["attack_success"] for r in saved(tmp_path, "week8")}
    assert got == {"baseline": True, "protected": False}


def test_the_adaptive_attacker_runs_up_to_fifteen_attempts_and_resumes(mock, tmp_path):
    server, _ = mock
    args = dict(suite_key="week8_adaptive", agent=config_for(server), selection_args={"scenario_ids": ["adaptive-01"]}, results_root=tmp_path, **QUIET)
    run_campaign(**args)
    rows = saved(tmp_path, "week8_adaptive")
    assert [r["attempt"] for r in rows] == list(range(1, 16)) and not any(r["attack_success"] for r in rows)
    assert {r["scenario_id"] for r in rows} == {"adaptive-01"} and {r["family"] for r in rows} == {"adaptive"}
    again = run_campaign(**args)
    assert len(saved(tmp_path, "week8_adaptive")) == 15 and again["configs"][0]["ran"] == 0


def test_a_reviewer_is_only_distinguished_where_a_suite_uses_one(mock, tmp_path):
    server, _ = mock
    other = config_for(server, model="other-reviewer")
    out = run_campaign(suite_key="baselines", agent=config_for(server), reviewer=other,
                       selection_args={"scenario_ids": ["ho-hiking-clean"], "seeds": [42], "configs": ["none"]}, results_root=tmp_path, **QUIET)
    assert "other-reviewer" not in out["slug"]


# --------------------------------------------------------------------------- seeds, budget, resume

def test_each_campaign_seed_reaches_the_api(mock, tmp_path):
    server, state = mock
    before = len(state.requests)
    run_campaign(suite_key="heldout", agent=config_for(server),
                 selection_args={"scenario_ids": ["ho-hiking-clean"], "seeds": [42, 314, 2718], "configs": ["C0"]}, results_root=tmp_path, **QUIET)
    assert {p.get("seed") for p in state.requests[before:]} == {42, 314, 2718}


def test_the_budget_bounds_the_whole_invocation_not_each_suite(mock, tmp_path):
    server, _ = mock
    agent = config_for(server, price_per_1k_input=0.01, price_per_1k_output=0.02)
    settings = RunSettings(max_cost=0.05)
    sel = {"scenario_ids": ["ho-hiking-clean", "ho-hiking-share_expenses"], "seeds": [42]}
    with pytest.raises(CampaignStop) as stop:
        for suite in ("heldout", "baselines", "adaptive_evasion"):
            out = run_campaign(suite_key=suite, agent=agent, selection_args=sel if suite != "adaptive_evasion" else {"seeds": [42]},
                               results_root=tmp_path, settings=settings, **QUIET)
            settings.carried_cost = out["usage"]["spent_total"]
    assert stop.value.code == 3 and stop.value.usage["spent_total"] >= 0.05


# --------------------------------------------------------------------------- the command line and the report

def cli(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    import os
    base = {k: v for k, v in os.environ.items() if not k.startswith("PROVENANCEGUARD_")}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_frontier_campaign.py"), *args], env={**base, **(env or {})},
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_list_and_the_full_report_dry_run_need_no_key():
    listing = cli(["--list"])
    assert listing.returncode == 0 and all(key in listing.stdout for key in SUITES)
    full = cli(["--suite", "pdf", "--dry-run"])
    assert full.returncode == 0 and "4,716 episodes + 10 attacker objectives" in full.stdout and "nothing was sent" in full.stdout


@pytest.mark.parametrize("args", [
    ["--suite", "heldout", "baselines", "--configs", "C0"],
    ["--suite", "heldout", "baselines", "--scenario-id", "ho-hiking-clean"],
    ["--suite", "no-such-suite", "--dry-run"],
    ["--suite", "heldout", "--dataset", "tuning", "--dry-run"],
])
def test_ambiguous_or_unknown_selections_are_refused(args):
    proc = cli(args)
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_legacy_dataset_spelling_still_works():
    proc = cli(["--dataset", "both", "--dry-run"])
    assert proc.returncode == 0 and "1,080 episodes" in proc.stdout


def test_report_prints_every_kind_and_shows_the_attacker_curve(mock, tmp_path, capsys):
    server, _ = mock
    for suite, sel in (("heldout", {"scenario_ids": ["ho-hiking-clean"]}), ("baselines", {"scenario_ids": ["ho-hiking-clean"]}),
                       ("week8", {"scenario_ids": ["heldout-direct-01", "heldout-benign-001"]}), ("week8_adaptive", {"scenario_ids": ["adaptive-01"]})):
        run_campaign(suite_key=suite, agent=config_for(server), selection_args={**sel, "seeds": [42]}, results_root=tmp_path, **QUIET)
    assert frontier_report.main(["--results-root", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    for suite in ("heldout", "baselines", "week8", "week8_adaptive"):
        assert f"[{suite}]" in text
    assert "attempt" in text.lower() and "none" in text and "tool_allowlist" in text
    capsys.readouterr()
    assert frontier_report.main(["--results-root", str(tmp_path), "--suite", "week8"]) == 0
    only = capsys.readouterr().out
    assert "[week8]" in only and "[baselines]" not in only


def test_stock_rows_never_raise_the_blocked_control_regression(tmp_path, capsys):
    """Without ProvenanceGuard the control legitimately succeeds; only a protected row may be flagged for it."""

    rows = [{"slug": "m", "week11_config": "none", "frontier_suite": "baselines", "frontier_schema": 1, "scenario_id": "s", "seed": 42,
             "family": "mail_attacker_hr", "kind": "attack", "attack_success": True, "benign_success": False, "error": None,
             "frontier_usage": {"agent": {"calls": 1, "input_tokens": 1, "output_tokens": 1}}}]
    path = tmp_path / "baselines" / "m" / "none"
    path.mkdir(parents=True)
    (path / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    frontier_report.main(["--results-root", str(tmp_path)])
    assert "REGRESSION" not in capsys.readouterr().out
