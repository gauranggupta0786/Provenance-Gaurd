"""Week 8 campaign tooling against the fake server (verifies plumbing and refusals; says nothing about real models)."""
import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_ollama import start_server  # noqa: E402

from pg.week7_models import candidate_by_slug, fetch_server_metadata, profile_from_server_metadata
from pg.week8_protocol import (
    CODE_FILES, ProtocolViolation, ROOT, audit_arm_parity, build_protocol_manifest, canonical_sha256, evidence_token_dir,
    load_frozen_matrix, matrix_digest, portable_fingerprint, prepare_model_session, quarantine_stale_evidence,
    verify_live_server, verify_protocol,
)
from scripts.run_week8_adaptive import run_adaptive
from scripts.run_week8_frozen import load_jsonl, run_campaign

SLUGS = ("qwen35-9b", "gemma4-e4b")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    server, _ = start_server(0)
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    base = tmp_path_factory.mktemp("w8")
    profiles = base / "profiles"
    profiles.mkdir()
    selected = []
    with httpx.Client(base_url=endpoint) as http:
        for slug in SLUGS:
            cand = candidate_by_slug(slug)
            meta = fetch_server_metadata(cand, endpoint=endpoint, client=http)
            profile, record = profile_from_server_metadata(candidate=cand, show=meta["show"], tags_entry=meta["tags_entry"],
                                                           server_version=meta["server_version"], endpoint="http://127.0.0.1:11111")
            (profiles / f"{slug}.json").write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
            selected.append({"slug": slug, "ollama_tag": profile.model_id, "family": cand["family"], "model_digest": profile.model_digest,
                             "profile_sha256": profile.fingerprint, "template_sha256": profile.chat_template_sha256,
                             "renderer": record["renderer"], "parser": record["parser"], "quantization": profile.quantization,
                             "serving_engine": "ollama 0.0.0-fake", "selected_context_length": 32768, "thinking_mode": "disabled",
                             "license_note": None, "reasons": ["ok"]})
    matrix = {"schema_version": "week7-final-model-matrix-v1", "frozen": True, "selected": selected, "needs_attention": [],
              "excluded": [{"slug": "llama31-8b", "reasons": ["excluded for the test"]}], "not_run": [],
              "context": {"common_bucket": 32768}, "family_duplicates_in_selected": {}, "gpu_approval": {"status": "pending"}}
    matrix["matrix_sha256"] = matrix_digest(matrix)
    matrix_path = base / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True))
    approval = base / "approval.json"
    approval.write_text(json.dumps({"approved_by": "Test Person", "approved_at": "2026-10-16T00:00:00Z",
                                    "matrix_sha256": matrix["matrix_sha256"], "note": "test", "decisions": {}}))
    manifest = build_protocol_manifest(root=ROOT, matrix_path=matrix_path, profiles_dir=profiles, approval_path=approval)
    protocol = base / "protocol.json"
    protocol.write_text(json.dumps(manifest, sort_keys=True))
    yield {"endpoint": endpoint, "base": base, "profiles": profiles, "matrix": matrix_path, "approval": approval,
           "protocol": protocol, "manifest": manifest}
    server.shutdown()


def kw(world, results, **extra):
    return dict(results_root=results, protocol_path=world["protocol"], matrix_path=world["matrix"], profiles_dir=world["profiles"],
                log=lambda *_: None, **extra)


def test_protocol_verifies_and_session_identity(world):
    assert verify_protocol(world["protocol"], matrix_path=world["matrix"])["protocol_sha256"] == world["manifest"]["protocol_sha256"]
    session = prepare_model_session("gemma4-e4b", world["endpoint"], protocol_path=world["protocol"], matrix_path=world["matrix"],
                                    profiles_dir=world["profiles"])
    assert session["profile"].endpoint == world["endpoint"] and session["live"]["renderer"] == "gemma4"
    assert portable_fingerprint(session["profile"]) == world["manifest"]["models"]["gemma4-e4b"]["portable_profile_sha256"]


def test_frozen_campaign_runs_resumes_and_audits_parity(world, tmp_path):
    scenarios = [json.loads(l)["scenario_id"] for l in (ROOT / "data/week5/heldout.jsonl").read_text().splitlines()[:1]]
    benign = next(json.loads(l)["scenario_id"] for l in (ROOT / "data/week5/heldout.jsonl").read_text().splitlines() if json.loads(l)["kind"] == "benign")
    attack = next(json.loads(l)["scenario_id"] for l in (ROOT / "data/week5/heldout.jsonl").read_text().splitlines() if json.loads(l)["kind"] == "attack")
    first = run_campaign("qwen35-9b", world["endpoint"], scenario_ids=[benign, attack], seeds=[42], **kw(world, tmp_path))
    assert first["ran"] == 4 and first["errors"] == 0
    rows = load_jsonl(tmp_path / "qwen35-9b" / "frozen_episodes.jsonl")
    assert len(rows) == 4 and {r["arm"] for r in rows} == {"baseline", "protected"}
    assert all(r["model_digest"] == world["manifest"]["models"]["qwen35-9b"]["model_digest"] and r["protocol_sha256"] for r in rows)
    assert all(r["request_log"] and "request_sha256" in r["request_log"][0] for r in rows)
    parity = audit_arm_parity(rows)
    assert parity["passed"] and parity["pairs_checked"] == 2, parity
    again = run_campaign("qwen35-9b", world["endpoint"], scenario_ids=[benign, attack], seeds=[42], **kw(world, tmp_path))
    assert again["selected"] == 0
    more = run_campaign("qwen35-9b", world["endpoint"], scenario_ids=[benign], seeds=[42, 314], **kw(world, tmp_path))
    assert more["ran"] == 2
    keys = [(r["scenario_id"], r["arm"], r["seed"]) for r in load_jsonl(tmp_path / "qwen35-9b" / "frozen_episodes.jsonl")]
    assert len(keys) == len(set(keys)) == 6
    sessions = [json.loads(l) for l in (tmp_path / "qwen35-9b" / "sessions.jsonl").read_text().splitlines()]
    assert [s["event"] for s in sessions].count("start") == 3 and any(s["event"] == "end" for s in sessions)


def test_adaptive_campaign_runs_with_identity_and_request_log(world, tmp_path):
    summary = run_adaptive("gemma4-e4b", world["endpoint"], objective_ids=None, max_objectives=1, **kw(world, tmp_path))
    rows = load_jsonl(tmp_path / "gemma4-e4b" / "adaptive_attempts.jsonl")
    assert summary["objectives_run"] == 1 and 1 <= len(rows) <= 15
    assert rows[0]["model_id"] == "gemma4:e4b" and rows[0]["model_digest"] and rows[0]["request_log"]
    assert run_adaptive("gemma4-e4b", world["endpoint"], max_objectives=1, **kw(world, tmp_path))["objectives_run"] == 1  # next objective


def test_refusals(world, tmp_path):
    prof = world["profiles"] / "gemma4-e4b.json"
    original = prof.read_text()
    prof.write_text(original.replace('"temperature": 0.0', '"temperature": 0.5'))
    with pytest.raises(ProtocolViolation, match="does not match"):
        prepare_model_session("gemma4-e4b", world["endpoint"], protocol_path=world["protocol"], matrix_path=world["matrix"], profiles_dir=world["profiles"])
    prof.write_text(original)
    with pytest.raises(ProtocolViolation, match="not in the frozen"):
        prepare_model_session("llama31-8b", world["endpoint"], protocol_path=world["protocol"], matrix_path=world["matrix"], profiles_dir=world["profiles"])
    # live server serving a different digest
    session = prepare_model_session("gemma4-e4b", world["endpoint"], protocol_path=world["protocol"], matrix_path=world["matrix"], profiles_dir=world["profiles"])
    with httpx.Client(base_url=world["endpoint"]) as http, pytest.raises(ProtocolViolation, match="installed digest"):
        verify_live_server(session["profile"].revalidated(model_digest="0" * 64), session["entry"], http)
    # edited protocol manifest
    edited = tmp_path / "p.json"
    data = json.loads(world["protocol"].read_text())
    data["campaign"]["seeds"] = [1]
    edited.write_text(json.dumps(data))
    with pytest.raises(ProtocolViolation, match="edited"):
        verify_protocol(edited, matrix_path=world["matrix"])
    # approval for another matrix
    from pg.week8_protocol import load_approval
    other = tmp_path / "a.json"
    other.write_text(json.dumps({"approved_by": "x", "matrix_sha256": "0" * 64}))
    with pytest.raises(ProtocolViolation, match="different matrix"):
        load_approval(load_frozen_matrix(world["matrix"]), other)
    # rows from another model in the output file
    out = tmp_path / "res" / "gemma4-e4b"
    out.mkdir(parents=True)
    (out / "frozen_episodes.jsonl").write_text(json.dumps({"scenario_id": "s", "arm": "baseline", "seed": 42, "model_digest": "bad"}) + "\n")
    with pytest.raises(ProtocolViolation, match="different model"):
        run_campaign("gemma4-e4b", world["endpoint"], **kw(world, tmp_path / "res"), seeds=[42])


def test_code_change_after_freeze_is_refused_but_analysis_edits_are_not(world, tmp_path):
    root = tmp_path / "proj"
    shutil.copytree(ROOT / "pg", root / "pg", ignore=shutil.ignore_patterns("__pycache__"))
    for name in CODE_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, root / name)
    for rel in ("data/week5/heldout.jsonl", "data/week6/adaptive_objectives.jsonl"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    manifest = build_protocol_manifest(root=root, matrix_path=world["matrix"], profiles_dir=world["profiles"], approval_path=world["approval"])
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    verify_protocol(path, root=root, matrix_path=world["matrix"])
    with (root / "pg" / "week8_analysis.py").open("a") as handle:
        handle.write("\n# analysis fix after the campaign\n")
    verify_protocol(path, root=root, matrix_path=world["matrix"])
    with (root / "pg" / "gateway.py").open("a") as handle:
        handle.write("\n# measurement code change\n")
    with pytest.raises(ProtocolViolation, match="code changed"):
        verify_protocol(path, root=root, matrix_path=world["matrix"])


def test_interrupted_episode_leftovers_are_moved_not_reused(tmp_path):
    folder = evidence_token_dir(tmp_path, "scn", 42, "protected")
    folder.mkdir(parents=True)
    (folder / "events.db").write_text("stale rejected attempts")
    moved = quarantine_stale_evidence(folder)
    assert not folder.exists() and (moved / "events.db").read_text() == "stale rejected attempts"
    assert quarantine_stale_evidence(folder) is None




def test_parity_audit_detects_a_difference():
    def row(arm, first_hash, tools="t1"):
        return {"scenario_id": "s", "seed": 42, "arm": arm, "scenario_checksum": "c", "model_digest": "d", "portable_profile_sha256": "p",
                "defense_version": "v", "protocol_sha256": "x", "model_id": "m",
                "request_log": [{"call": 1, "request_sha256": first_hash, "tools_sha256": tools}]}
    assert audit_arm_parity([row("baseline", "h"), row("protected", "h")])["passed"]
    bad = audit_arm_parity([row("baseline", "h"), row("protected", "DIFFERENT")])
    assert not bad["passed"] and bad["examples"][0]["field"] == "first_request_sha256"
    tools = audit_arm_parity([row("baseline", "h", "t1"), row("protected", "h", "t2")])
    assert not tools["passed"] and tools["examples"][0]["field"] == "tools_sha256"
    assert not audit_arm_parity([{**row("baseline", "h"), "request_log": []}, row("protected", "h")])["passed"]  # inconclusive is not a pass




