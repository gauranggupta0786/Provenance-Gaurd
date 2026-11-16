"""Week 12: add two more model families to the Week 11e campaign without touching anything frozen.

The Week 8 protocol manifest is hash-frozen over exactly four models, and ``scripts/run_week11_policy_passing.py`` is one of
the Week 11 code files the Week 11e plan pins by digest.  Neither may change: editing either would invalidate the Week 11e
plan and with it the four model results already in ``results/week11e``.

So Week 12 adds a *parallel* layer beside them, built from the same code:

* ``data/week12/model_matrix.json``    -- a frozen matrix in the Week 7 shape holding only the new models, built from the
  pilot evidence in ``results/week12_pilot*/final_model_matrix.json`` (not hand-written).
* ``data/week12/protocol_manifest.json`` -- a protocol manifest that *extends* the Week 8 one: approval, campaign (seeds),
  limits and evaluator blocks are copied from it verbatim, so the new models are run under identical settings.
* ``data/week12/plan.json``           -- the Week 11e plan with exactly one field changed, ``week8_protocol_sha256``.
  ``assert_plan_matches_week11e`` proves that claim field by field, and a test asserts it too.

The campaign itself is run by ``scripts/run_week12_campaign.py``, which calls the frozen Week 11 runner with these paths.

Per-model serving engine.  ``verify_live_server`` enforces the engine version recorded in each model's profile, so the two
new models must be served by the engine their pilot used: lfm2 by Ollama 0.40.2, glm-4.7-flash by the side-by-side 0.40.12
(it does not load at all on 0.40.2).  ``scripts/week12_job.sh`` picks the binary per model from ``OLLAMA_BIN_BY_ENGINE``.
The engine version is recorded per model in the matrix, the protocol and the paper.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pg.week8_protocol import (
    ProtocolViolation, canonical_sha256, code_digest, load_frozen_profile, matrix_digest, portable_fingerprint,
    utc_now,
)

MATRIX_PATH_W12 = ROOT / "data" / "week12" / "model_matrix.json"
PROTOCOL_PATH_W12 = ROOT / "data" / "week12" / "protocol_manifest.json"
PROFILES_DIR_W12 = ROOT / "data" / "week12" / "profiles_w12"
PLAN_PATH_W12 = ROOT / "data" / "week12" / "plan.json"
# the engine each new model is served by, and the binary that provides it (paths are relative to $HOME)
OLLAMA_BIN_BY_ENGINE = {
    "0.40.2": ".local/bin/ollama",
    "0.40.12": "ollama-0.40.12/bin/ollama",
}


@dataclass(frozen=True)
class AddedModel:
    """One model joining the campaign in Week 12, with where its pilot evidence lives and why it was accepted."""

    slug: str
    family: str
    pilot_results: str
    decision: str          # "selected" (passed every automated check) or "accepted_with_flag"
    note: str

    @property
    def accepted_with_flag(self) -> bool:
        return self.decision == "accepted_with_flag"


WEEK12_MODELS: tuple[AddedModel, ...] = (
    AddedModel(
        slug="lfm2-24b", family="liquid", pilot_results="results/week12_pilot4", decision="accepted_with_flag",
        note="38/38 evaluable, every probe passed, 30 structured tool calls, 0 malformed, no thinking mode. Flagged for one "
             "escape-stress episode with no tool call; the server log shows the cause is Ollama's own lfm2 tool-call parser "
             "('lfm2 tool call parsing failed: invalid argument: expected =') on a quote/escape-heavy argument, not the "
             "model declining to act. Accepted on the Week 7 precedent that gemma4-e4b, itself flagged, was selected. The "
             "flag is reported in the paper.",
    ),
    AddedModel(
        slug="glm-47-flash", family="glm", pilot_results="results/week12_pilot5", decision="selected",
        note="38/38 evaluable, every probe passed including thinking_control, 35 structured tool calls, 0 malformed. Served "
             "by Ollama 0.40.12: it does not load on 0.40.2. Its first pilot reported 'not installed' because the harness "
             "compares tag names exactly and an untagged pull is listed as <name>:latest; the tag is explicit now.",
    ),
)

MODEL_BY_SLUG = {m.slug: m for m in WEEK12_MODELS}

# fields copied verbatim from the Week 8 protocol so the new models run under identical settings
INHERITED_BLOCKS = ("approval", "campaign", "limits", "evaluator")
# the plan fields that must be identical to Week 11e's; only week8_protocol_sha256 (and the derived plan_sha256) may differ
PLAN_FIELDS_THAT_MUST_MATCH = (
    "schema_version", "dataset", "dataset_sha256", "replicates", "scenario_ids", "seeds", "config_order", "configs",
    "reviewers", "reviewer_prompt_sha256", "approval_seconds", "episodes_per_config", "code", "rules",
)


def pilot_entry(pilot_matrix: Mapping[str, Any], slug: str) -> tuple[dict[str, Any], str]:
    """Find ``slug`` in a Week 7-style pilot matrix; returns (entry, bucket it was found in)."""

    for bucket in ("selected", "needs_attention", "excluded", "dropped_by_decision"):
        for item in pilot_matrix.get(bucket) or []:
            if isinstance(item, Mapping) and item.get("slug") == slug:
                return dict(item), bucket
    raise ProtocolViolation(f"{slug!r} is not in the pilot matrix (buckets: {sorted(pilot_matrix)})")


def build_week12_matrix(pilot_matrices: Mapping[str, Mapping[str, Any]], *, models: Sequence[AddedModel] = WEEK12_MODELS,
                        context_bucket: int = 32768) -> dict[str, Any]:
    """A frozen matrix in the Week 7 shape, holding only the Week 12 models, built from their pilot matrices.

    ``pilot_matrices`` maps slug -> the ``final_model_matrix.json`` its pilot produced.  Every field of each entry is copied
    from that evidence; the only things added here are the acceptance decision and its reason.
    """

    selected: list[dict[str, Any]] = []
    for model in models:
        entry, bucket = pilot_entry(pilot_matrices[model.slug], model.slug)
        if bucket not in ("selected", "needs_attention"):
            raise ProtocolViolation(f"{model.slug}: the pilot put it in {bucket!r}; it cannot join the campaign")
        if model.decision == "selected" and bucket != "selected":
            raise ProtocolViolation(f"{model.slug}: recorded as 'selected' but the pilot bucket is {bucket!r}")
        if model.accepted_with_flag and bucket != "needs_attention":
            raise ProtocolViolation(f"{model.slug}: recorded as flagged but the pilot bucket is {bucket!r}")
        if entry.get("family") != model.family:
            raise ProtocolViolation(f"{model.slug}: pilot family {entry.get('family')!r} != {model.family!r}")
        selected.append({
            **entry,
            "week12_decision": model.decision,
            "week12_decision_note": model.note,
            "pilot_results": model.pilot_results,
            "pilot_bucket": bucket,
        })
    families = [item["family"] for item in selected]
    matrix = {
        "schema_version": "week12-model-matrix-v1",
        "frozen": True,
        "generated_at": utc_now(),
        "note": "Models added to the Week 11e campaign in Week 12. Built from the pilot matrices, not hand-written. "
                "The Week 8 protocol and the four models frozen in it are untouched.",
        "selected": selected,
        "needs_attention": [],
        "excluded": [],
        "not_run": [],
        "family_duplicates_in_selected": sorted({f for f in families if families.count(f) > 1}),
        "context": {
            "common_bucket": context_bucket,
            "per_model_smallest_bucket": {item["slug"]: item.get("selected_context_length") for item in selected},
            "rule": "the Week 8 context bucket, unchanged: the new models are run at the same context as the first four",
        },
        "terminology": {"selected": "joins the Week 11e campaign in Week 12",
                        "week12_decision": "'selected' = every automated pilot check passed; "
                                           "'accepted_with_flag' = passed the >=98% evaluable gate with a recorded flag"},
    }
    matrix["matrix_sha256"] = matrix_digest(matrix)
    return matrix


def build_week12_protocol(*, week8_protocol: Mapping[str, Any], matrix: Mapping[str, Any],
                          profiles_dir: Path = PROFILES_DIR_W12, root: Path = ROOT,
                          engine_bins: Mapping[str, str] = OLLAMA_BIN_BY_ENGINE) -> dict[str, Any]:
    """A protocol manifest for the Week 12 models that extends the Week 8 one.

    Approval, campaign (seeds), limits and evaluator are copied from the Week 8 protocol verbatim, so both halves of the
    six-model set are run under identical settings.  No new GPU approval is claimed: the inherited block is the Week 7 one,
    and these models run on the same already-approved hardware.
    """

    missing = [b for b in INHERITED_BLOCKS if b not in week8_protocol]
    if missing:
        raise ProtocolViolation(f"the Week 8 protocol has no {missing} block(s) to inherit")
    models: dict[str, Any] = {}
    for item in matrix["selected"]:
        slug = item["slug"]
        profile, entry = load_frozen_profile(slug, matrix, profiles_dir)
        added = MODEL_BY_SLUG.get(slug)
        models[slug] = {
            "ollama_tag": profile.model_id, "family": entry["family"], "model_digest": profile.model_digest,
            "profile_file_sha256": profile.fingerprint, "portable_profile_sha256": portable_fingerprint(profile),
            "template_sha256": profile.chat_template_sha256, "renderer": entry.get("renderer"),
            "parser": entry.get("parser"), "quantization": profile.quantization,
            "serving_engine": f"{profile.serving_engine} {profile.serving_engine_version}",
            "serving_engine_version": profile.serving_engine_version,
            "ollama_bin": engine_bins.get(str(profile.serving_engine_version)),
            "selected_context_length": profile.selected_context_length, "max_output_tokens": profile.max_output_tokens,
            "thinking_mode": profile.thinking_mode, "temperature": profile.temperature, "top_p": profile.top_p,
            "top_k": profile.top_k, "min_p": profile.min_p,
            "week12_decision": entry.get("week12_decision"), "week12_decision_note": entry.get("week12_decision_note"),
            "pilot_results": entry.get("pilot_results"),
        }
    unknown_engine = sorted(s for s, m in models.items() if not m["ollama_bin"])
    if unknown_engine:
        raise ProtocolViolation(f"no Ollama binary recorded for the engine of {unknown_engine}; "
                                f"add it to OLLAMA_BIN_BY_ENGINE")
    digest, hashes = code_digest(root)
    manifest = {
        "schema_version": "week12-protocol-v1",
        "frozen": True,
        "created_at": utc_now(),
        "extends": {
            "protocol": "data/week8/protocol_manifest.json",
            "protocol_sha256": week8_protocol["protocol_sha256"],
            "models": sorted(week8_protocol["models"]),
            "note": "The Week 8 protocol and its four models are unchanged. This manifest covers only the models added in "
                    "Week 12 and inherits " + ", ".join(INHERITED_BLOCKS) + " from it verbatim.",
        },
        "week7_matrix": {
            "path": "data/week12/model_matrix.json",
            "matrix_sha256": matrix["matrix_sha256"],
            "selected": [m["slug"] for m in matrix["selected"]],
            "context_bucket": matrix["context"]["common_bucket"],
        },
        **{block: week8_protocol[block] for block in INHERITED_BLOCKS},
        "models": models,
        "code": {"defense_version": digest, "component_hashes": hashes,
                 "week8_defense_version": week8_protocol["code"]["defense_version"],
                 "note": "the Week 11/12 code is pinned by the plan's own Week 11 code digest, which is identical to "
                         "Week 11e's; this block records the Week 8 guard-code digest for both"},
        "rules": [
            "the Week 12 models run the Week 11e plan, dataset, configurations and seeds, unchanged",
            "each model is served by the engine version recorded in its profile; verify_live_server enforces it",
            "glm-4.7-flash runs on Ollama 0.40.12 because it does not load on 0.40.2; the engine is reported per model",
            "a model accepted with a flag carries that flag into the results and the paper",
            "no file frozen by the Week 8 protocol or the Week 11e plan is modified by Week 12",
        ],
    }
    manifest["protocol_sha256"] = canonical_sha256({k: v for k, v in manifest.items()
                                                    if k not in ("created_at", "protocol_sha256")})
    return manifest


def assert_plan_matches_week11e(plan_w12: Mapping[str, Any], plan_e: Mapping[str, Any]) -> list[str]:
    """The Week 12 plan must be the Week 11e plan with exactly one field changed.  Returns the fields that differ."""

    differing = [f for f in PLAN_FIELDS_THAT_MUST_MATCH if plan_w12.get(f) != plan_e.get(f)]
    if differing:
        raise ProtocolViolation(
            f"the Week 12 plan differs from the Week 11e plan in {differing}; it must differ only by "
            f"week8_protocol_sha256, or the two halves of the model set are not running the same experiment")
    if plan_w12["week8_protocol_sha256"] == plan_e["week8_protocol_sha256"]:
        raise ProtocolViolation("the Week 12 plan points at the Week 8 protocol; it must point at the Week 12 one")
    extra = sorted(set(plan_w12) - set(plan_e)) + sorted(set(plan_e) - set(plan_w12))
    if extra:
        raise ProtocolViolation(f"the Week 12 plan has different fields from the Week 11e plan: {extra}")
    return ["week8_protocol_sha256", "plan_sha256"]


def family_collisions(week8_protocol: Mapping[str, Any], matrix: Mapping[str, Any]) -> list[str]:
    """Families already represented by a Week 8 model -- the set keeps one model per family."""

    existing = {str(m.get("family")) for m in week8_protocol["models"].values()}
    return sorted({str(item["family"]) for item in matrix["selected"]} & existing)



