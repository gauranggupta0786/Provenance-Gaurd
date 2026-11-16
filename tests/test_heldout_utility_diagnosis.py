"""The held-out utility-drop diagnostic -- tested offline against hand-built episode rows shaped like the real
hook_result/reviewer schema (pg.reviewers._record uses "why", not "reason" -- this caught that exact bug once already).
"""
from __future__ import annotations

import json
from pathlib import Path

from pg.week8_protocol import ROOT, code_digest
from scripts.week11_lib import week11_code_digest
from scripts import heldout_utility_diagnosis as diag


def _row(slug, config, scenario_id, *, family="clean_control", done=False, decisions=None, legit_denied=0, error=None,
         seed=42):
    return {
        "slug": slug, "week11_config": config, "family": family, "scenario_id": scenario_id, "seed": seed, "error": error,
        "hook_result": {"task_write_done": done, "legit_cards_denied": legit_denied,
                        "reviewer": {"decisions": decisions or []}},
    }


def _write(tmp_path: Path, rows: list[dict]) -> None:
    for r in rows:
        d = tmp_path / r["slug"] / r["week11_config"]
        d.mkdir(parents=True, exist_ok=True)
        with (d / "episodes.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps(r) + "\n")


def test_denial_reasons_reads_the_why_field_not_reason():
    row = _row("x", "SC-v3", "s1", decisions=[{"approve": False, "why": "scope check: boom"}])
    assert diag.denial_reasons(row) == ["scope check: boom"]


def test_failed_clean_episodes_excludes_other_families_and_unprotected_configs():
    rows = [
        _row("x", "SC-v3", "clean-ok", done=True),            # succeeded -- not a failure
        _row("x", "SC-v3", "clean-bad", done=False),           # a real failure
        _row("x", "C0", "clean-bad-c0", done=False),            # C0 is not a protected config -- excluded
        _row("x", "SC-v3", "pp-1", family="policy_passing", done=False),  # wrong family -- excluded
    ]
    out = diag.failed_clean_episodes(rows)
    assert [r["scenario_id"] for r in out] == ["clean-bad"]


def test_scope_check_copies_text_reason_is_bucketed_as_the_hiking_email_hypothesis(tmp_path, capsys):
    rows = [_row("granite4-small-h", "SC-v3", "ho-hiking-x", decisions=[
        {"approve": False, "why": "scope check: the body copies text from hiking-offer.eml, which the request does not mention"},
    ], legit_denied=1)]
    _write(tmp_path, rows)
    diag.main(["--results-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "hiking-email hypothesis" in out
    assert "1/1 have legit_cards_denied > 0" in out


def test_no_denial_recorded_bucket_for_a_model_that_simply_never_tried(tmp_path, capsys):
    rows = [_row("qwen3-14b", "SCP-v3", "ho-x", done=False, decisions=[], legit_denied=0)]
    _write(tmp_path, rows)
    diag.main(["--results-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "model simply never attempted the write" in out


def test_model_filter_restricts_to_named_slugs(tmp_path, capsys):
    rows = [_row("qwen3-14b", "SC-v3", "a", done=False), _row("granite4-small-h", "SC-v3", "b", done=False)]
    _write(tmp_path, rows)
    diag.main(["--results-root", str(tmp_path), "--model", "qwen3-14b"])
    out = capsys.readouterr().out
    assert "qwen3-14b" in out and "granite4-small-h" not in out


def test_no_failures_returns_cleanly(tmp_path, capsys):
    _write(tmp_path, [_row("x", "SC-v3", "ok", done=True)])
    rc = diag.main(["--results-root", str(tmp_path)])
    assert rc == 0
    assert "no failed clean_control episodes" in capsys.readouterr().out


def test_net_regression_splits_pre_existing_from_guard_caused():
    rows = [
        # pre-existing: fails at C0 too -- the agent itself never got there, independent of any guard
        _row("gemma4-e4b", "C0", "ho-hiking-clean", done=False),
        _row("gemma4-e4b", "SC-v3", "ho-hiking-clean", done=False),
        # guard-caused: succeeds with no guard, breaks once SC-v3 turns on -- the real attributable cost
        _row("qwen3-14b", "C0", "ho-events-clean", done=True),
        _row("qwen3-14b", "SC-v3", "ho-events-clean", done=False),
        # a success -- not a failure at all, excluded entirely
        _row("qwen3-14b", "SC-v3", "ho-ok", done=True),
        # no matching C0 row recorded for this episode at all
        _row("lfm2-24b", "SCP-v3", "ho-orphan", done=False),
    ]
    split = diag.net_regression(rows)
    assert [r["scenario_id"] for r in split["pre_existing"]] == ["ho-hiking-clean"]
    assert [r["scenario_id"] for r in split["guard_caused"]] == ["ho-events-clean"]
    assert [r["scenario_id"] for r in split["unknown"]] == ["ho-orphan"]


def test_net_regression_keys_by_slug_scenario_and_seed_not_scenario_alone():
    # same scenario_id, different seeds: each must be matched against its OWN seed's C0 row, not any C0 row that exists
    rows = [
        _row("x", "C0", "s", done=True, seed=42),
        _row("x", "SC-v3", "s", done=False, seed=42),   # matches seed=42's C0 (done=True) -> guard_caused
        _row("x", "C0", "s", done=False, seed=314),
        _row("x", "SC-v3", "s", done=False, seed=314),  # matches seed=314's C0 (done=False) -> pre_existing
    ]
    split = diag.net_regression(rows)
    assert [r["seed"] for r in split["guard_caused"]] == [42]
    assert [r["seed"] for r in split["pre_existing"]] == [314]


def test_c0_task_done_only_reads_clean_control_c0_rows():
    rows = [
        _row("x", "C0", "a", done=True),
        _row("x", "C0", "b", done=True, family="policy_passing"),  # wrong family -- excluded
        _row("x", "SC-v3", "c", done=True),                          # wrong config -- excluded
    ]
    done = diag.c0_task_done(rows)
    assert list(done.keys()) == [diag.episode_key(rows[0])]


def test_print_net_regression_reports_split_and_per_model_counts(capsys):
    rows = [
        _row("gemma4-e4b", "C0", "ho-hiking-clean", done=False),
        _row("gemma4-e4b", "SC-v3", "ho-hiking-clean", done=False),
        _row("qwen3-14b", "C0", "ho-events-clean", done=True),
        _row("qwen3-14b", "SC-v3", "ho-events-clean", done=False),
    ]
    diag.print_net_regression(rows)
    out = capsys.readouterr().out
    assert "failed at C0 too" in out
    assert "guard-attributable utility cost" in out
    assert "gemma4-e4b" in out and "qwen3-14b" in out


def test_print_net_regression_empty_filter_reports_cleanly(capsys):
    rows = [_row("gemma4-e4b", "C0", "a", done=False), _row("gemma4-e4b", "SC-v3", "a", done=False)]
    diag.print_net_regression(rows, models=["qwen3-14b"])
    assert "no failed clean_control episodes" in capsys.readouterr().out


def test_cli_net_regression_flag_invokes_the_split_report(tmp_path, capsys):
    rows = [
        _row("gemma4-e4b", "C0", "ho-hiking-clean", done=False),
        _row("gemma4-e4b", "SC-v3", "ho-hiking-clean", done=False),
        _row("qwen3-14b", "C0", "ho-events-clean", done=True),
        _row("qwen3-14b", "SC-v3", "ho-events-clean", done=False),
    ]
    _write(tmp_path, rows)
    rc = diag.main(["--results-root", str(tmp_path), "--net-regression"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "failed at C0 too" in out
    # the plain (non-net-regression) output's denial-reason breakdown header must NOT appear
    assert "denial-reason breakdown" not in out


def test_cli_net_regression_respects_model_filter(tmp_path, capsys):
    rows = [
        _row("gemma4-e4b", "C0", "ho-hiking-clean", done=False),
        _row("gemma4-e4b", "SC-v3", "ho-hiking-clean", done=False),
        _row("qwen3-14b", "C0", "ho-events-clean", done=True),
        _row("qwen3-14b", "SC-v3", "ho-events-clean", done=False),
    ]
    _write(tmp_path, rows)
    diag.main(["--results-root", str(tmp_path), "--net-regression", "--model", "qwen3-14b"])
    out = capsys.readouterr().out
    assert "qwen3-14b" in out and "gemma4-e4b" not in out


def test_no_heldout_utility_diagnosis_file_enters_either_frozen_digest():
    _, w8 = code_digest(ROOT)
    _, w11 = week11_code_digest(ROOT)
    hashed = set(w8) | set(w11)
    assert sorted(n for n in hashed if "heldout_utility" in n) == []
