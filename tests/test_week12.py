"""Week 12: adding two model families to the Week 11e campaign without touching anything frozen.

The claims these tests hold in place:

* the Week 12 files cannot change either frozen digest (they are in neither ``CODE_FILES`` nor ``WEEK11_CODE_FILES``);
* the matrix is built from real pilot evidence, and a model whose recorded decision disagrees with its pilot bucket is
  refused rather than quietly promoted;
* the protocol inherits approval, campaign, limits and evaluator from the Week 8 one, so both halves of the six-model set
  run under identical settings, and every model carries the engine it was profiled on;
* the plan is the Week 11e plan with exactly one field changed -- the whole comparability argument rests on this.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from pg.week7_models import candidate_by_slug, fetch_server_metadata, profile_from_server_metadata
from pg.week8_protocol import (
    CODE_FILES, ROOT, ProtocolViolation, load_frozen_matrix, matrix_digest, verify_protocol,
)
from scripts.week11_lib import WEEK11_CODE_FILES
from scripts.week11e_lib import build_plan_e
from scripts.week12_lib import (
    INHERITED_BLOCKS, MODEL_BY_SLUG, OLLAMA_BIN_BY_ENGINE, PLAN_FIELDS_THAT_MUST_MATCH, WEEK12_MODELS, AddedModel,
    assert_plan_matches_week11e, build_week12_matrix, build_week12_protocol, family_collisions, pilot_entry,
)
from tests.fake_ollama import start_server

FAKE_ENGINE = "0.0.0-fake"
FAKE_BINS = {FAKE_ENGINE: ".local/bin/ollama"}
W8_SLUGS = ("gemma4-e4b", "qwen35-9b")          # stand in for the four models frozen in Week 8
W12_SLUGS = ("llama31-8b", "mistral-nemo-12b")  # stand in for lfm2-24b and glm-47-flash
W12_FAKE = (
    AddedModel(slug="llama31-8b", family="llama31", pilot_results="results/fake_pilot_a", decision="selected", note="n"),
    AddedModel(slug="mistral-nemo-12b", family="mistral-nemo", pilot_results="results/fake_pilot_b",
               decision="accepted_with_flag", note="flagged in the fake pilot"),
)


def _entry(profile, record, cand) -> dict:
    return {"slug": cand["slug"], "ollama_tag": profile.model_id, "family": cand["family"],
            "model_digest": profile.model_digest, "profile_sha256": profile.fingerprint,
            "template_sha256": profile.chat_template_sha256, "renderer": record["renderer"], "parser": record["parser"],
            "quantization": profile.quantization, "serving_engine": f"ollama {FAKE_ENGINE}",
            "selected_context_length": 32768, "thinking_mode": "disabled", "reasons": ["ok"]}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A fake server, profiles for four models, a Week 8-shaped protocol and two pilot matrices."""

    server, _ = start_server(0)
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    base = tmp_path_factory.mktemp("w12")
    profiles = base / "profiles"
    profiles.mkdir()
    entries = {}
    with httpx.Client(base_url=endpoint) as http:
        for slug in (*W8_SLUGS, *W12_SLUGS):
            cand = candidate_by_slug(slug)
            meta = fetch_server_metadata(cand, endpoint=endpoint, client=http)
            profile, record = profile_from_server_metadata(candidate=cand, show=meta["show"], tags_entry=meta["tags_entry"],
                                                           server_version=meta["server_version"],
                                                           endpoint="http://127.0.0.1:11111")
            (profiles / f"{slug}.json").write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
            entries[slug] = _entry(profile, record, cand)

    # a Week 8-shaped protocol holding the first two models, with the blocks Week 12 inherits
    week8 = {
        "schema_version": "week8-protocol-v1", "frozen": True, "protocol_sha256": "w8" + "0" * 62,
        "models": {s: {"family": entries[s]["family"], "ollama_tag": entries[s]["ollama_tag"]} for s in W8_SLUGS},
        "approval": {"approved_by": "Test Person", "note": "fake"},
        "campaign": {"seeds": [42, 314, 2718], "dataset": "data/week5/heldout.jsonl"},
        "limits": {"model_seconds": 1.0}, "evaluator": {"version": "fake"},
        "code": {"defense_version": "w8code", "component_hashes": {}},
    }
    # one pilot matrix per added model, in the bucket its recorded decision claims
    pilots = {
        "llama31-8b": {"selected": [entries["llama31-8b"]], "needs_attention": [], "excluded": []},
        "mistral-nemo-12b": {"selected": [], "needs_attention": [entries["mistral-nemo-12b"]], "excluded": []},
    }
    yield {"endpoint": endpoint, "base": base, "profiles": profiles, "entries": entries, "week8": week8, "pilots": pilots}
    server.shutdown()


def build(world, **kw):
    matrix = build_week12_matrix(world["pilots"], models=W12_FAKE, **kw)
    protocol = build_week12_protocol(week8_protocol=world["week8"], matrix=matrix, profiles_dir=world["profiles"],
                                     engine_bins=FAKE_BINS)
    return matrix, protocol


# ------------------------------------------------------------------ nothing frozen can move

def test_no_week12_file_enters_either_frozen_digest():
    """Computed from the digest functions themselves, not from the file-name lists.

    The first version of this test checked membership in ``CODE_FILES`` and ``WEEK11_CODE_FILES`` and passed while the
    patch was broken: ``code_digest`` also globs EVERY ``pg/*.py``, so ``pg/openai_adapter.py`` silently changed both
    digests and the cluster refused to run with "code changed after the Week 11 plan was frozen". Ask the real function.
    """

    from pg.week8_protocol import code_digest
    from scripts.week11_lib import week11_code_digest

    _, week8_hashes = code_digest(ROOT)
    _, week11_hashes = week11_code_digest(ROOT)
    hashed = set(week8_hashes) | set(week11_hashes)
    offenders = sorted(name for name in hashed if "week12" in name or "heldout" in name
                       or "openai_adapter" in name or "frontier" in name)
    assert offenders == [], (
        f"{offenders} are inside a frozen digest; adding them invalidates the Week 11e plan and the four model results "
        f"already measured under it. Week 12 code must live outside pg/ and outside both file lists.")


def test_the_adapter_lives_outside_the_frozen_package():
    assert (ROOT / "frontier" / "openai_adapter.py").exists()
    assert (ROOT / "frontier" / "frontier_probes.py").exists()
    assert not (ROOT / "pg" / "openai_adapter.py").exists(), "pg/*.py is hashed; the adapter must not live there"
    assert not (ROOT / "pg" / "frontier_probes.py").exists()
    assert "code_digest" in (ROOT / "frontier" / "__init__.py").read_text(encoding="utf-8"), \
        "the package should say why it is not in pg/"






# ------------------------------------------------------------------ the matrix comes from pilot evidence

def test_matrix_is_frozen_and_loads(world):
    matrix, _ = build(world)
    assert matrix["matrix_sha256"] == matrix_digest(matrix)
    path = world["base"] / "matrix.json"
    path.write_text(json.dumps(matrix, sort_keys=True))
    loaded = load_frozen_matrix(path)
    assert [m["slug"] for m in loaded["selected"]] == list(W12_SLUGS)
    assert loaded["family_duplicates_in_selected"] == []


def test_matrix_carries_the_decision_and_its_reason(world):
    matrix, _ = build(world)
    by_slug = {m["slug"]: m for m in matrix["selected"]}
    assert by_slug["llama31-8b"]["week12_decision"] == "selected"
    assert by_slug["llama31-8b"]["pilot_bucket"] == "selected"
    assert by_slug["mistral-nemo-12b"]["week12_decision"] == "accepted_with_flag"
    assert by_slug["mistral-nemo-12b"]["pilot_bucket"] == "needs_attention"
    assert by_slug["mistral-nemo-12b"]["week12_decision_note"]


@pytest.mark.parametrize("slug, decision", [("llama31-8b", "accepted_with_flag"), ("mistral-nemo-12b", "selected")])
def test_matrix_refuses_a_decision_its_pilot_does_not_support(world, slug, decision):
    models = tuple(AddedModel(slug=m.slug, family=m.family, pilot_results=m.pilot_results,
                              decision=decision if m.slug == slug else m.decision, note=m.note) for m in W12_FAKE)
    with pytest.raises(ProtocolViolation):
        build_week12_matrix(world["pilots"], models=models)


def test_matrix_refuses_a_model_the_pilot_excluded(world):
    pilots = {"llama31-8b": {"selected": [], "excluded": [world["entries"]["llama31-8b"]]}}
    with pytest.raises(ProtocolViolation, match="excluded"):
        build_week12_matrix(pilots, models=W12_FAKE[:1])


def test_pilot_entry_searches_every_bucket_and_reports_a_miss(world):
    entry, bucket = pilot_entry(world["pilots"]["mistral-nemo-12b"], "mistral-nemo-12b")
    assert bucket == "needs_attention" and entry["slug"] == "mistral-nemo-12b"
    with pytest.raises(ProtocolViolation):
        pilot_entry(world["pilots"]["mistral-nemo-12b"], "not-a-model")


# ------------------------------------------------------------------ the protocol extends Week 8

def test_protocol_verifies_against_its_own_matrix(world):
    matrix, protocol = build(world)
    matrix_path, protocol_path = world["base"] / "m2.json", world["base"] / "p2.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True))
    protocol_path.write_text(json.dumps(protocol, sort_keys=True))
    loaded = verify_protocol(protocol_path, matrix_path=matrix_path, check_code=False)
    assert sorted(loaded["models"]) == sorted(W12_SLUGS)


def test_protocol_inherits_the_week8_blocks_verbatim(world):
    _, protocol = build(world)
    for block in INHERITED_BLOCKS:
        assert protocol[block] == world["week8"][block], block
    assert protocol["extends"]["protocol_sha256"] == world["week8"]["protocol_sha256"]
    assert protocol["extends"]["models"] == sorted(W8_SLUGS)


def test_protocol_records_the_engine_and_binary_per_model(world):
    _, protocol = build(world)
    for slug in W12_SLUGS:
        assert protocol["models"][slug]["serving_engine_version"] == FAKE_ENGINE
        assert protocol["models"][slug]["ollama_bin"] == FAKE_BINS[FAKE_ENGINE]


def test_protocol_refuses_an_engine_with_no_recorded_binary(world):
    matrix = build_week12_matrix(world["pilots"], models=W12_FAKE)
    with pytest.raises(ProtocolViolation, match="OLLAMA_BIN_BY_ENGINE"):
        build_week12_protocol(week8_protocol=world["week8"], matrix=matrix, profiles_dir=world["profiles"],
                              engine_bins={"9.9.9": "nope"})


def test_protocol_refuses_a_week8_manifest_missing_a_block(world):
    week8 = {k: v for k, v in world["week8"].items() if k != "campaign"}
    matrix = build_week12_matrix(world["pilots"], models=W12_FAKE)
    with pytest.raises(ProtocolViolation, match="campaign"):
        build_week12_protocol(week8_protocol=week8, matrix=matrix, profiles_dir=world["profiles"], engine_bins=FAKE_BINS)


def test_protocol_refuses_an_edited_profile(world, tmp_path):
    matrix = build_week12_matrix(world["pilots"], models=W12_FAKE)
    edited = tmp_path / "profiles"
    edited.mkdir()
    for slug in W12_SLUGS:
        payload = json.loads((world["profiles"] / f"{slug}.json").read_text())
        payload["temperature"] = 0.9
        (edited / f"{slug}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    with pytest.raises(ProtocolViolation):
        build_week12_protocol(week8_protocol=world["week8"], matrix=matrix, profiles_dir=edited, engine_bins=FAKE_BINS)


def test_family_collisions_with_the_week8_set(world):
    matrix, _ = build(world)
    assert family_collisions(world["week8"], matrix) == []
    clash = dict(world["week8"], models={"x": {"family": "llama31"}})
    assert family_collisions(clash, matrix) == ["llama31"]


# ------------------------------------------------------------------ the plan is Week 11e's, with one field changed

def test_plan_differs_from_week11e_in_exactly_one_field(world):
    _, protocol = build(world)
    plan_e = build_plan_e(week8_protocol=world["week8"])
    plan_w12 = build_plan_e(week8_protocol=protocol)
    assert assert_plan_matches_week11e(plan_w12, plan_e) == ["week8_protocol_sha256", "plan_sha256"]
    assert plan_w12["plan_sha256"] != plan_e["plan_sha256"]
    assert plan_w12["week8_protocol_sha256"] == protocol["protocol_sha256"]


@pytest.mark.parametrize("field, value", [("dataset_sha256", "0" * 64), ("seeds", [42]), ("config_order", ["C0"])])
def test_assert_plan_catches_any_other_difference(world, field, value):
    _, protocol = build(world)
    plan_e = build_plan_e(week8_protocol=world["week8"])
    plan_w12 = dict(build_plan_e(week8_protocol=protocol))
    plan_w12[field] = value
    with pytest.raises(ProtocolViolation, match=field):
        assert_plan_matches_week11e(plan_w12, plan_e)


def test_assert_plan_catches_a_plan_still_pointing_at_week8(world):
    plan_e = build_plan_e(week8_protocol=world["week8"])
    with pytest.raises(ProtocolViolation, match="Week 12"):
        assert_plan_matches_week11e(dict(plan_e), plan_e)


def test_plan_fields_that_must_match_covers_every_shared_field(world):
    """If build_plan_e grows a field, it has to be listed as one that must match (or deliberately excluded)."""

    plan = build_plan_e(week8_protocol=world["week8"])
    allowed_to_differ = {"week8_protocol_sha256", "plan_sha256"}
    assert set(plan) - allowed_to_differ == set(PLAN_FIELDS_THAT_MUST_MATCH)


# ------------------------------------------------------------------ the registry and the job script

def test_registry_is_consistent():
    assert len(WEEK12_MODELS) == len(MODEL_BY_SLUG) == 2
    assert {m.family for m in WEEK12_MODELS} == {"liquid", "glm"}
    for model in WEEK12_MODELS:
        assert model.decision in ("selected", "accepted_with_flag")
        assert len(model.note) > 80, f"{model.slug}: the acceptance reason has to be written down"
        assert model.pilot_results.startswith("results/week12_pilot")
    assert MODEL_BY_SLUG["lfm2-24b"].accepted_with_flag
    assert not MODEL_BY_SLUG["glm-47-flash"].accepted_with_flag
    assert set(OLLAMA_BIN_BY_ENGINE) == {"0.40.2", "0.40.12"}








# ------------------------------------------------------------------ the profile gate in week12_prepare

def _resolver_format(profile_dict: dict) -> str:
    """Exactly how ``scripts/week7_resolve_profiles.py`` writes a profile: pretty-printed, sorted, trailing newline."""

    return json.dumps(profile_dict, indent=2, sort_keys=True) + "\n"


def _pilot_profile_dir(world, base: Path) -> Path:
    src = base / "pilot_profiles"
    src.mkdir()
    for slug in W12_SLUGS:
        profile = json.loads((world["profiles"] / f"{slug}.json").read_text())
        (src / f"{slug}.json").write_text(_resolver_format(profile), encoding="utf-8")
    return src


def _copy(world, src: Path, dst: Path, log=lambda *_: None):
    from scripts.week12_prepare import copy_profiles

    copy_profiles(world["pilots"], models=W12_FAKE, sources={s: src for s in W12_SLUGS}, profiles_dir=dst, log=log)




def test_a_pretty_printed_file_never_hashes_to_its_fingerprint(world, tmp_path):
    """The trap itself.  A pilot records ModelProfile.fingerprint (canonical, compact); the file on disk is indented.

    The first version of the gate compared file_sha256 with the recorded value, so it refused every genuine profile.
    This pins the fact that makes that comparison wrong, so nobody 'simplifies' it back.
    """

    from pg.trusted_models import ModelProfile
    from pg.week8_protocol import file_sha256

    src = _pilot_profile_dir(world, tmp_path)
    for slug in W12_SLUGS:
        profile = ModelProfile.from_json(src / f"{slug}.json")
        assert profile.fingerprint == world["entries"][slug]["profile_sha256"], "the fixture records the real fingerprint"
        assert file_sha256(src / f"{slug}.json") != profile.fingerprint












