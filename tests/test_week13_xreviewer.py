"""Week 13 cross-model reviewer harness: the wiring, tested offline against the real property, not a proxy.

The one thing this experiment changes from the frozen campaign is that the reviewer's model client is built from the
*reviewer's* profile, not the agent's.  The frozen runner's bug class (Week 12, twice) was a test that checked something
adjacent to the property.  So these tests run the real ``run_pair`` loop with injected sessions and record which profile the
reviewer LLM was actually built from -- and assert no Week 13 file can enter a frozen digest, computed from the digest
functions themselves.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from pg.week8_protocol import ProtocolViolation, ROOT
from scripts import week13_xreviewer as xr
from scripts.week13_lib import (
    MODEL_FAMILY, XREVIEWER_CONFIGS, config_spec, family_of, paths_for_slug, pair_dir_name, same_family,
)


@dataclass
class FakeProfile:
    slug: str

    @property
    def model_id(self) -> str:
        return f"{self.slug}:tag"

    serving_engine_version: str = "0.0.0"

    @property
    def model_digest(self) -> str:
        return f"digest-{self.slug}"


class FakeClient:
    request_log: list = []


@dataclass
class FakeLLM:
    profile: FakeProfile
    model_client: FakeClient = FakeClient()


def fake_session(slug: str) -> dict:
    return {"identity": {"slug": slug, "model_digest": f"digest-{slug}"},
            "profile": FakeProfile(slug),
            "protocol": {"campaign": {"seeds": [42, 314, 2718]}}}


def fake_loader(slug, endpoint, *, http=None):
    return fake_session(slug)


def fake_llm_factory(profile, *, seed, limits, client=None):
    return FakeLLM(profile)


def make_recording_episode_fn(seen: list):
    def episode_fn(scenario, spec, seed, *, agent_llm, review_llm):
        seen.append({"scenario": scenario["scenario_id"], "seed": seed,
                     "agent_profile": agent_llm.profile.slug,
                     "reviewer_profile": None if review_llm is None else review_llm.profile.slug,
                     "provenance": bool(spec.get("provenance"))})
        # a trivial deterministic outcome; the wiring is what is under test, not the verdict
        return {"attack_success": False, "latency_ms": 1, "tool_calls": [], "mediated_calls": [],
                "hook_result": {"task_write_done": True, "injected_cards": 1, "injected_cards_approved": 0}}
    return episode_fn


def run(tmp_path, agent, reviewer, seen, **kw):
    return xr.run_pair(
        agent, "http://agent", reviewer, "http://reviewer",
        family="policy_passing", results_root=tmp_path, session_loader=fake_loader,
        llm_factory=fake_llm_factory, episode_fn=make_recording_episode_fn(seen), log=lambda *_: None, **kw)


# ------------------------------------------------------------------ the property the experiment exists to change

def test_cross_pair_builds_the_reviewer_llm_from_the_reviewer_profile(tmp_path):
    seen: list = []
    run(tmp_path, "glm-47-flash", "qwen3-14b", seen)
    assert seen, "no episodes ran"
    assert all(e["agent_profile"] == "glm-47-flash" for e in seen)
    assert all(e["reviewer_profile"] == "qwen3-14b" for e in seen), "reviewer LLM was not built from the reviewer profile"
    assert {e["reviewer_profile"] for e in seen} != {e["agent_profile"] for e in seen}


def test_self_pair_builds_the_reviewer_llm_from_the_agent_profile(tmp_path):
    seen: list = []
    run(tmp_path, "glm-47-flash", "glm-47-flash", seen)
    assert all(e["agent_profile"] == e["reviewer_profile"] == "glm-47-flash" for e in seen)


def test_both_configs_run_and_only_scp_enables_provenance(tmp_path):
    seen: list = []
    summary = run(tmp_path, "glm-47-flash", "qwen3-14b", seen)
    assert set(summary["configs"]) == set(XREVIEWER_CONFIGS)
    prov = {s["provenance"] for s in seen}
    assert prov == {True, False}, "SC-v3 must have provenance off and SCP-v3 on"


# ------------------------------------------------------------------ guard rails

def test_same_family_cross_pair_is_refused_unless_explicit(tmp_path, monkeypatch):
    # The real six are all distinct families, so inject a second gemma to create a same-family, different-slug pair.
    monkeypatch.setitem(MODEL_FAMILY, "gemma-sibling", "gemma")
    with pytest.raises(ProtocolViolation, match="family"):
        run(tmp_path, "gemma4-e4b", "gemma-sibling", [])
    # explicit opt-in is allowed
    run(tmp_path / "ok", "gemma4-e4b", "gemma-sibling", [], allow_same_family=True)


def test_self_pair_is_allowed_as_the_control(tmp_path):
    # same slug is the self-review control, not a same-family cross pair -- it must never be refused
    summary = run(tmp_path, "glm-47-flash", "glm-47-flash", [])
    assert summary["cross_model"] is False


def test_unknown_slug_is_refused(tmp_path):
    with pytest.raises(ProtocolViolation, match="not one of the six"):
        run(tmp_path, "not-a-model", "qwen3-14b", [])


def test_c0_is_not_part_of_the_experiment(tmp_path):
    with pytest.raises(ProtocolViolation, match="not part of the cross-reviewer"):
        run(tmp_path, "glm-47-flash", "qwen3-14b", [], configs=["C0"])
    with pytest.raises((KeyError, ValueError)):
        config_spec("C0")


def test_resume_is_idempotent(tmp_path):
    seen1: list = []
    run(tmp_path, "glm-47-flash", "qwen3-14b", seen1)
    seen2: list = []
    summary = run(tmp_path, "glm-47-flash", "qwen3-14b", seen2)
    assert seen2 == [], "a completed pair re-ran episodes"
    assert all(c["ran"] == 0 and c["already_saved"] > 0 for c in summary["configs"].values())


def test_rows_carry_the_pair_identity_and_family_filter(tmp_path):
    run(tmp_path, "glm-47-flash", "qwen3-14b", [])
    out = tmp_path / pair_dir_name("glm-47-flash", "qwen3-14b") / "SCP-v3" / "episodes.jsonl"
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows and all(r["agent_slug"] == "glm-47-flash" and r["reviewer_slug"] == "qwen3-14b" for r in rows)
    assert all(r["week13_config"] == "SCP-v3" and r["experiment"] == "cross_model_reviewer" for r in rows)
    assert all(r["family"] == "policy_passing" for r in rows), "the family filter leaked other families in"
    assert all(r["cross_model"] is True for r in rows)
    # 20 policy_passing scenarios x 3 seeds
    assert len(rows) == 60


# ------------------------------------------------------------------ nothing Week 13 can enter a frozen digest

def test_no_week13_file_enters_either_frozen_digest():
    from pg.week8_protocol import code_digest
    from scripts.week11_lib import week11_code_digest

    _, w8 = code_digest(ROOT)
    _, w11 = week11_code_digest(ROOT)
    hashed = set(w8) | set(w11)
    offenders = sorted(n for n in hashed if "week13" in n)
    assert offenders == [], f"{offenders} are in a frozen digest; Week 13 code must live outside pg/ and both file lists"
    assert not (ROOT / "pg" / "week13_lib.py").exists()
    assert not (ROOT / "pg" / "week13_xreviewer.py").exists()


# ------------------------------------------------------------------ small helpers

def test_paths_route_each_model_to_its_frozen_artifacts():
    w12_proto, _, _ = paths_for_slug("glm-47-flash")
    w8_proto, _, _ = paths_for_slug("qwen3-14b")
    assert "week12" in str(w12_proto) and "week8" in str(w8_proto)
    with pytest.raises(KeyError):
        paths_for_slug("not-a-model")


def test_resolve_serving_week12_fields_and_week8_fallback(tmp_path, monkeypatch):
    import scripts.week13_lib as lib

    w12 = tmp_path / "w12.json"
    w12.write_text(json.dumps({"models": {"glm-47-flash": {
        "ollama_tag": "glm-4.7-flash:latest", "ollama_bin": "ollama-0.40.12/bin/ollama", "serving_engine_version": "0.40.12",
        "serving_engine": "ollama 0.40.12"}}}))
    w8 = tmp_path / "w8.json"  # the original four: no ollama_bin, no serving_engine_version -- only the combined string
    w8.write_text(json.dumps({"models": {"qwen3-14b": {
        "ollama_tag": "qwen3:14b", "serving_engine": "ollama 0.40.2"}}}))

    def fake_paths(slug):
        return (w12 if slug == "glm-47-flash" else w8), None, None

    monkeypatch.setattr(lib, "paths_for_slug", fake_paths)
    assert lib.resolve_serving("glm-47-flash") == ("glm-4.7-flash:latest", "ollama-0.40.12/bin/ollama", "0.40.12")
    tag, binpath, engine = lib.resolve_serving("qwen3-14b")
    assert tag == "qwen3:14b" and engine == "0.40.2" and binpath == lib.DEFAULT_OLLAMA_BIN


def test_family_helpers():
    assert family_of("glm-47-flash") == "glm"
    assert not same_family("glm-47-flash", "qwen3-14b")
    assert same_family("glm-47-flash", "glm-47-flash")
    assert set(MODEL_FAMILY) == {"gemma4-e4b", "granite4-small-h", "mistral-nemo-12b", "qwen3-14b", "lfm2-24b", "glm-47-flash"}
    assert len(set(MODEL_FAMILY.values())) == 6, "the six models must be six distinct families"
