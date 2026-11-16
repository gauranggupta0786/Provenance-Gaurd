import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "week5"


def _rows(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines()]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def test_frozen_heldout_manifest_has_planned_composition_and_unique_ids():
    rows = _rows("heldout.jsonl")
    counts = Counter(row["family"] for row in rows)
    assert len(rows) == 100
    assert sum(row["kind"] == "benign" for row in rows) == 50
    assert sum(row["kind"] == "attack" for row in rows) == 50
    assert counts == {
        "benign": 50, "tip": 30, "direct": 5, "indirect": 5,
        "approval_deception": 5, "composite": 5,
    }
    assert len({row["scenario_id"] for row in rows}) == 100


def test_tip_matrix_contains_all_forms_and_guidance_levels():
    rows = [row for row in _rows("heldout.jsonl") if row["family"] == "tip"]
    forms = {row["scenario_id"].split("-")[-2] for row in rows}
    levels = {row["guidance"] for row in rows}
    assert forms == {"caesar", "morse", "vigenere", "atbash", "phonetic", "t9",
                     "base64", "binary", "riddle", "python"}
    assert levels == {"easy", "medium", "hard"}


def test_scenario_and_file_checksums_are_valid():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for filename, section in [("development.jsonl", "development"), ("heldout.jsonl", "heldout")]:
        payload = (DATA / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == manifest[section]["sha256"]
        for row in _rows(filename):
            checksum = row.pop("checksum")
            assert hashlib.sha256(_canonical(row)).hexdigest() == checksum


def test_development_set_is_separate_and_balanced():
    development = _rows("development.jsonl")
    heldout = _rows("heldout.jsonl")
    assert len(development) == 40
    assert sum(row["kind"] == "benign" for row in development) == 20
    assert sum(row["kind"] == "attack" for row in development) == 20
    assert {row["scenario_id"] for row in development}.isdisjoint(
        {row["scenario_id"] for row in heldout}
    )

