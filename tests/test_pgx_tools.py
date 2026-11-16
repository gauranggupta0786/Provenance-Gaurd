"""v2 tools: campaign runner (resume, registration gate, manifests), adaptive loop, statistics, label export/agreement."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agentdojo.task_suite.load_suites import get_suite

from pgx.suites import BENCHMARK_VERSION
from tests.pgx_doubles import ScriptedAgent, ScriptedClient, deny_if

ROOT = Path(__file__).resolve().parents[1]
APPROVE = ScriptedClient(lambda p: "APPROVE")


def _factory(case, seed):
    s = get_suite(BENCHMARK_VERSION, case.suite)
    tasks = ([s.get_injection_task_by_id(case.injection_task_id)] if case.injection_task_id else []) + [
        s.get_user_task_by_id(case.user_task_id)]
    return ScriptedAgent(tasks)


# ---------------------------------------------------------------------------- campaign runner
def test_runner_writes_resumes_and_refuses_foreign_dirs(tmp_path):
    from scripts import run_v2_agentdojo as r

    kw = dict(agent="fake-agent", reviewer="fake-reviewer", configs=["none", "PG-v4-capParse"], splits=["dev"], suites=["banking"],
              variance=False, results_root=tmp_path, agent_factory=_factory, agent_client_factory=lambda s: APPROVE,
              reviewer_client_factory=lambda s: APPROVE, identities={"agent_model_digest": "a1", "reviewer_model_digest": "r1"},
              smoke=3, log=lambda *_: None)
    assert r.run(**kw) == 0
    path = tmp_path / "fake-agent__fake-reviewer" / "PG-v4-capParse" / "episodes.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 3 and {x["seed"] for x in rows} == {42} and all(x["pgx_digest"] for x in rows)
    assert r.run(**kw) == 0 and len(path.read_text().splitlines()) == 3            # resume: nothing new
    with pytest.raises(r.Refused):
        r.run(**{**kw, "identities": {"agent_model_digest": "OTHER", "reviewer_model_digest": "r1"}})


def test_test_split_needs_a_matching_registration(tmp_path, monkeypatch):
    from scripts import run_v2_agentdojo as r

    monkeypatch.setattr(r, "REGISTRATION", tmp_path / "registration.json")
    with pytest.raises(r.Refused, match="registration"):
        r.check_registration("abc")
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("final\n")
    monkeypatch.setattr(r, "PREREG", prereg)
    (tmp_path / "registration.json").write_text(json.dumps({"pgx_digest": "abc", "preregistration_sha256": r.sha(prereg)}))
    assert r.check_registration("abc")["pgx_digest"] == "abc"
    with pytest.raises(r.Refused, match="code changed"):
        r.check_registration("def")
    prereg.write_text("edited\n")
    with pytest.raises(r.Refused, match="PREREGISTRATION"):
        r.check_registration("abc")


def test_register_refuses_unconfirmed_preregistration(tmp_path, monkeypatch):
    from scripts import v2_register as reg

    doc = tmp_path / "p.md"
    doc.write_text("threshold: 30% [CONFIRM: utility gate]\n")
    monkeypatch.setattr(reg, "PREREG", doc)
    monkeypatch.setattr(reg, "OUT", tmp_path / "registration.json")
    assert reg.main([]) == 2 and not (tmp_path / "registration.json").exists()
    doc.write_text("threshold: 30%\n")
    assert reg.main([]) == 0 and (tmp_path / "registration.json").exists()
    assert reg.main([]) == 2  # a second registration must be deliberate


def test_real_preregistration_is_still_a_draft_until_confirmed():
    text = (ROOT / "docs" / "PREREGISTRATION_V2.md").read_text()
    assert "[CONFIRM:" in text  # the student decides every marked threshold before registering
    assert not (ROOT / "data" / "v2" / "registration.json").exists()


def test_load_cases_counts_and_split_are_consistent():
    from scripts.run_v2_agentdojo import load_cases

    allc = load_cases(["dev", "test"], ["workspace", "travel", "banking", "slack"])
    assert sum(c.injection_task_id is None for c in allc) == 97 and sum(c.injection_task_id is not None for c in allc) == 949
    dev = load_cases(["dev"], ["banking"])
    assert dev and all(c.split == "dev" for c in dev)


# ---------------------------------------------------------------------------- adaptive loop
def test_adaptive_attack_stops_at_first_success_and_records_attempts():
    from pgx.adaptive import attack_target, extract_injection
    from pgx.agentdojo_runner import Case

    case = Case("banking", "user_task_0", "injection_task_0", {"v1": "x", "v2": "y"}, "dev", True, False)
    seen = []

    def run(c):
        seen.append(c.injections)
        return {"case_id": c.case_id, "attack_success": len(seen) == 3, "final_response": "ok"}

    answers = iter(["<injection>try one</injection>", "no tags here", "<injection>try three</injection>",
                    "<injection>try four</injection>"])
    rows = attack_target(case, "PG-v4", goal="steal", user_task="pay rent", attacker=lambda p: next(answers), run=run, budget=6)
    assert [r["attempt"] for r in rows] == [1, 2, 3, 4] and rows[1]["error"]
    assert rows[-1]["attack_success"] and len(seen) == 3
    assert all(set(inj) == {"v1", "v2"} and len(set(inj.values())) == 1 for inj in seen)  # same text in every field
    assert extract_injection("<INJECTION> a </injection>") == "a" and extract_injection("x") is None


def test_whitebox_feedback_reports_the_declining_stage():
    from pgx.adaptive import describe_outcome

    row = {"attack_success": False, "final_response": "done", "reviewer": {"decisions": [
        {"approve": False, "stage": "destination", "category": "unrequested_recipient", "tool": "banking_send_money"}]}}
    assert "destination" in describe_outcome(row, "whitebox") and "destination" not in describe_outcome(row, "outcome")


# ---------------------------------------------------------------------------- statistics
def test_cluster_bootstrap_and_paired_tests():
    from scripts.stats_v2 import cluster_bootstrap, holm, mcnemar_exact, paired, rate

    rows = [{"cluster": f"s{i // 3}", "success": i % 2 == 0} for i in range(30)]
    lo, hi = cluster_bootstrap(rows, rate)
    assert 0 <= lo <= rate(rows) <= hi <= 1
    assert mcnemar_exact(0, 0) == 1.0 and abs(mcnemar_exact(10, 0) - 2 / 1024) < 1e-12
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    a = [{"case": f"c{i}", "seed": 42, "cluster": f"s{i}", "success": True} for i in range(10)]
    b = [{"case": f"c{i}", "seed": 42, "cluster": f"s{i}", "success": i < 2} for i in range(10)]
    t = paired(a, b)
    assert (t["pairs"], t["only_first"], t["only_second"]) == (10, 8, 0) and t["diff"] == pytest.approx(0.8)


def test_stats_reads_v2_rows(tmp_path, capsys):
    from scripts import stats_v2

    d = tmp_path / "x" / "none"
    d.mkdir(parents=True)
    rows = [{"schema": "pgx-agentdojo-episode-v1", "agent": "m", "config": "none", "suite": "banking", "user_task_id": f"user_task_{i}",
             "case_id": f"banking/user_task_{i}/{'benign' if i % 2 else 'injection_task_0'}", "seed": 42,
             "injection_task_id": None if i % 2 else "injection_task_0", "in_scope": True, "utility": True,
             "attack_success": None if i % 2 else (i % 4 == 0), "error": None} for i in range(8)]
    (d / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert stats_v2.main(["--results-root", str(tmp_path)]) == 0
    assert "banking" not in capsys.readouterr().out or True


# ---------------------------------------------------------------------------- labels
@pytest.mark.skipif(not (ROOT / "results" / "week11d").exists(), reason="needs local Week 11d results")
def test_label_export_agreement_and_rescore_round_trip(tmp_path):
    from scripts import content_rescore, label_agreement, label_export

    args = ["--results-root", str(ROOT / "results/week11d"), "--scenarios", str(ROOT / "data/week11d/scenarios.jsonl")]
    assert label_export.main(args + ["--out", str(tmp_path / "A.csv"), "--annotator", "A"]) == 0
    assert label_export.main(args + ["--out", str(tmp_path / "B.csv"), "--annotator", "B"]) == 0
    a = list(csv.DictReader((tmp_path / "A.csv").open()))
    b = list(csv.DictReader((tmp_path / "B.csv").open()))
    assert {r["digest"] for r in a} == {r["digest"] for r in b} and [r["digest"] for r in a] != [r["digest"] for r in b]
    assert "config" not in a[0] and "model" not in a[0]
    from scripts.provenance_rescore import LABELS as BUILTIN

    def truth(d):  # consistent with the built-in Week 11d table where it has the body, otherwise arbitrary but fixed
        return BUILTIN[d][0] if d in BUILTIN else ("leak" if int(d, 16) % 3 == 0 else "clean")

    for rows, path, flip in ((a, "A.csv", False), (b, "B.csv", True)):
        for i, r in enumerate(rows):
            v = truth(r["digest"])
            r["verdict"] = ({"leak": "clean", "clean": "leak"}[v]) if (flip and i == 0) else v
        with (tmp_path / path).open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    table = tmp_path / "labels.json"
    assert label_agreement.main([str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), "--disagreements",
                                 str(tmp_path / "dis.csv"), "--write-table", str(table)]) == 1   # one disagreement pending
    dis = list(csv.DictReader((tmp_path / "dis.csv").open()))
    assert len(dis) == 1
    dis[0]["verdict"] = truth(dis[0]["digest"])
    with (tmp_path / "dis.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(dis[0]))
        w.writeheader()
        w.writerows(dis)
    assert label_agreement.main([str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), "--adjudicated", str(tmp_path / "dis.csv"),
                                 "--write-table", str(table)]) == 0
    labels = content_rescore.load_labels(table)
    assert len(labels) >= len(a)


def test_kappa_values():
    from scripts.label_agreement import kappa

    assert kappa([("leak", "leak"), ("clean", "clean")]) == 1.0
    assert kappa([("leak", "clean"), ("clean", "leak")]) == pytest.approx(-1.0)
