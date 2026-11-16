"""Week 13 shared code: experiments that each vary exactly ONE thing about the frozen Week 11e/12 setup.

Nothing here edits a frozen file.  Each experiment writes to its own results root with its own manifest, and reuses the
frozen dataset, seeds, scenarios, layers and reviewer machinery, so that the single variable under test is the only thing
that differs from the Week 11e/12 campaign.  Like the Week 12 layer, this code lives outside ``pg/`` (``code_digest`` hashes
every ``pg/*.py``) and outside both frozen file lists, so it cannot change the Week 8 defense version or the Week 11e plan.

The first experiment is the **cross-model reviewer**: the agent is one model and the reviewer is a *different* model.  The
frozen campaign builds the reviewer LLM from the agent's own profile (``run_week11_policy_passing.run_config``:
``review_llm = build_llm(profile, ...)``), so every measured result so far is *self-review*.  Week 12 found that two of six
same-model reviewers approved ~10% of policy-passing attacks; this experiment tests whether an independent reviewer closes
that gap.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import re

from pg.week8_protocol import MATRIX_PATH, PROFILES_DIR, PROTOCOL_PATH, ProtocolViolation
from scripts.week11e_lib import CONFIGS_E, DATASET_PATH_E, WEEK11D_DATASET_SHA256
from scripts.week12_lib import MATRIX_PATH_W12, MODEL_BY_SLUG, OLLAMA_BIN_BY_ENGINE, PROFILES_DIR_W12, PROTOCOL_PATH_W12

# the six models in the final set and the family each belongs to (one model per family)
MODEL_FAMILY: dict[str, str] = {
    "gemma4-e4b": "gemma",
    "granite4-small-h": "granite",
    "mistral-nemo-12b": "mistral-nemo",
    "qwen3-14b": "qwen",
    "lfm2-24b": "liquid",
    "glm-47-flash": "glm",
}

DATASET_PATH = DATASET_PATH_E          # the frozen Week 11e dataset, byte-identical to Week 11d (sha 4a01d47d...)
DATASET_SHA256 = WEEK11D_DATASET_SHA256

RESULTS_ROOT_XREVIEWER = ROOT / "results" / "week13_xreviewer"

# only the configs whose verdict can depend on the reviewer MODEL.  C0 is approve-all (reviewer model is irrelevant there),
# so it is excluded by default: running it would measure nothing about the reviewer.
XREVIEWER_CONFIGS: tuple[str, ...] = ("SC-v3", "SCP-v3")


def family_of(slug: str) -> str:
    if slug not in MODEL_FAMILY:
        raise KeyError(f"{slug!r} is not one of the six models {sorted(MODEL_FAMILY)}")
    return MODEL_FAMILY[slug]


def same_family(agent_slug: str, reviewer_slug: str) -> bool:
    return family_of(agent_slug) == family_of(reviewer_slug)


def paths_for_slug(slug: str) -> tuple[Path, Path, Path]:
    """The frozen protocol, matrix and profiles directory a model's identity is verified against.

    The two models added in Week 12 are frozen in the Week 12 manifests; the original four in the Week 8 ones.  Each model
    is therefore checked against the exact artifacts it was frozen in, whichever side of the pair it is on.
    """

    if slug in MODEL_BY_SLUG:
        return PROTOCOL_PATH_W12, MATRIX_PATH_W12, PROFILES_DIR_W12
    if slug in MODEL_FAMILY:
        return PROTOCOL_PATH, MATRIX_PATH, PROFILES_DIR
    raise KeyError(f"{slug!r} is not one of the six models {sorted(MODEL_FAMILY)}")


def pair_dir_name(agent_slug: str, reviewer_slug: str) -> str:
    """A filesystem-safe directory name for one (agent, reviewer) pair.  A self-pair reads 'agent__self'."""

    return f"{agent_slug}__{'self' if agent_slug == reviewer_slug else reviewer_slug}"


def config_spec(name: str) -> dict[str, Any]:
    """The frozen Week 11e configuration, unchanged.  Only the reviewer MODEL differs in this experiment, never the config."""

    if name not in CONFIGS_E:
        raise KeyError(f"{name!r} is not a Week 11e configuration ({sorted(CONFIGS_E)})")
    if name == "C0":
        raise ValueError("C0 is approve-all: the reviewer model has no effect, so it is not part of the cross-reviewer test")
    return dict(CONFIGS_E[name])


REVIEWER_NAME = "S+M-v3"        # the reviewer used by SC-v3 and SCP-v3 (scoped + model reviewer, prompt v3)
DEFAULT_OLLAMA_BIN = ".local/bin/ollama"   # the original four were all served on the stock 0.40.2 binary


def resolve_serving(slug: str, root: Path = ROOT) -> tuple[str, str, str]:
    """How to serve ``slug``: (ollama tag, binary path relative to $HOME, engine version).

    The two Week 12 models record ``ollama_bin`` and ``serving_engine_version`` in their protocol.  The original four were
    frozen before those fields existed, so their engine is read from the ``serving_engine`` string (e.g. 'ollama 0.40.2')
    and the binary defaulted from it.  A pair that mixes engines therefore needs one server per model, each on its own
    binary -- which is exactly why the job script serves the agent and the reviewer separately.
    """

    protocol_path, _, _ = paths_for_slug(slug)
    if not protocol_path.exists():
        raise ProtocolViolation(f"{protocol_path} not found: freeze the protocol first (this runs on the cluster)")
    models = json.loads(protocol_path.read_text(encoding="utf-8")).get("models", {})
    m = models.get(slug)
    if not m:
        raise ProtocolViolation(f"{slug!r} is not in {protocol_path}")
    tag = m["ollama_tag"]
    engine = m.get("serving_engine_version")
    if not engine:
        found = re.search(r"\d+\.\d+\.\d+", str(m.get("serving_engine", "")))
        if not found:
            raise ProtocolViolation(f"{slug}: no serving engine version recorded in {protocol_path}")
        engine = found.group(0)
    ollama_bin = m.get("ollama_bin") or OLLAMA_BIN_BY_ENGINE.get(engine) or DEFAULT_OLLAMA_BIN
    return tag, ollama_bin, engine
