"""Week 10 plan, code identity and the per-episode integrity hook.

Week 10 changes the guard code (pg/ablation.py and the layer switches), so the Week 8 code digest no longer matches the
tree.  That is by design: Week 8 and Week 9 results were produced by, and remain tied to, the code of their time.  Week 10 freezes
its OWN digest in data/week10/plan.json and every Week 10 runner refuses to start unless the tree still matches it.  All other Week 8
checks (frozen matrix, approval, model profile, live server digest/template/engine) still run unchanged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]

from pg.ablation import MODEL_DRIVEN_ORDER, Layers, ladder_added, model_driven_configs
from pg.week8_protocol import ProtocolViolation, canonical_sha256, code_digest, file_sha256

PLAN_PATH = ROOT / "data" / "week10" / "plan.json"
RESULTS_ROOT = ROOT / "results" / "week10"
# Everything below determines what Week 10 measures and is hashed together with pg/*.py and the Week 8 code files.
WEEK10_CODE_FILES = (
    "scripts/week10_lib.py", "scripts/week10_attacks.py", "scripts/run_week10_ladder.py",
    "scripts/run_week10_protocol_attacks.py",
)

CONFIG_NOTES = {
    "A0": "stock AgentDojo: model + benchmark only, no ProvenanceGuard pipeline",
    "A0m": "mediation plumbing only: every write goes through proposal/approval/gateway, every layer off",
    "FULL-policy": "every layer on except the deterministic policy",
}


def week10_code_digest(root: Path = ROOT) -> tuple[str, dict[str, str]]:
    _, hashes = code_digest(root)
    for name in WEEK10_CODE_FILES:
        path = root / name
        if not path.exists():
            raise ProtocolViolation(f"Week 10 code file missing: {name}")
        hashes[name] = file_sha256(path)
    return canonical_sha256(hashes), hashes


def config_spec(name: str, layers: Layers | None) -> dict[str, Any]:
    spec: dict[str, Any] = {"arm": "baseline" if layers is None else "protected",
                            "layers": None if layers is None else layers.to_dict()}
    if name in CONFIG_NOTES:
        spec["note"] = CONFIG_NOTES[name]
    else:
        try:
            spec["adds"] = list(ladder_added(name))
        except KeyError:
            pass
    return spec


def build_plan(*, week8_protocol: Mapping[str, Any], week9_plan: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    digest, hashes = week10_code_digest(root)
    configs = model_driven_configs()
    plan: dict[str, Any] = {
        "schema_version": "week10-ablation-plan-v1",
        "week8_protocol_sha256": week8_protocol["protocol_sha256"],
        "week9_plan_sha256": week9_plan["plan_sha256"],
        "dataset_sha256": week8_protocol["campaign"]["dataset_sha256"],
        "scenario_ids": list(week9_plan["scenario_ids"]),
        "seeds": list(week8_protocol["campaign"]["seeds"]),
        "config_order": list(MODEL_DRIVEN_ORDER),
        "configs": {name: config_spec(name, configs[name]) for name in MODEL_DRIVEN_ORDER},
        "reviewer": "approve-all (the benchmark reviewer; identical in every configuration)",
        "episodes_per_config": len(week9_plan["scenario_ids"]) * len(week8_protocol["campaign"]["seeds"]),
        "code": {"digest": digest, "component_hashes": hashes},
        "rules": [
            "A0 is the stock baseline arm; every other configuration is the protected arm with the stated layers",
            "every configuration runs the same scenarios and seeds with the frozen Week 8 model profile, so episodes pair across configurations",
            "every scheduled episode stays in every denominator; failures are recorded, never dropped",
            "the benchmark reviewer approves every card, so the human-review effect of the grounded-approval layer is not measured here (see Part A)",
            "no code, plan or dataset file may change after this plan is frozen",
        ],
    }
    plan["plan_sha256"] = canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    if not Path(path).exists():
        raise ProtocolViolation(f"{path} not found: run  python scripts/week10_prepare.py  first")
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ProtocolViolation(f"{path}: plan_sha256 does not match its content (the file was edited after it was frozen)")
    return plan





def integrity_hook(context: Mapping[str, Any]) -> dict[str, Any] | None:
    """End-of-episode audit step: verify the chain and write+verify a checkpoint, as far as the layers are enabled.

    Mediated episodes otherwise never export a checkpoint (the pre-write verification only starts once one exists), so
    without this step the event-chain and checkpoint layers would cost only their signatures in a model-driven run.
    """
    store = context.get("store")
    if store is None:
        return None
    verifier, gateway = context["verifier"], context["gateway"]
    layers = store.layers
    started = time.perf_counter()
    result: dict[str, Any] = {"events": store.latest_seq(), "chain_verified": None, "checkpoint_verified": None, "error": None}
    try:
        if layers.event_chain:
            store.verify_chain(verifier)
            result["chain_verified"] = True
        if layers.checkpoints:
            store.export_checkpoint(gateway._signer)
            store.verify_against_checkpoint(verifier)
            result["checkpoint_verified"] = True
    except Exception as exc:  # recorded; an integrity failure here is a finding, not a crash
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["integrity_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result
