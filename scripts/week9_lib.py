"""Week 9 ablation design: variants, scenario subset, vendor-sampling resolution. Uses pg/ only; changes nothing in pg/ (the Week 8
code digest must stay valid, so no file is added under pg/).

Control = the Week 8 rows for the same (scenario, arm, seed) keys. Variants differ from the control in ONE factor, except
`thinking_on_vendor`, whose control is `vendor_sampling` (thinking off, same vendor sampling) so thinking is the only difference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data" / "week9" / "ablation_plan.json"
OVERRIDES_PATH = ROOT / "data" / "week9" / "vendor_sampling_overrides.json"
RESULTS_ROOT = ROOT / "results" / "week9"
SUBSET_SIZES = {"benign": 12, "direct": 5, "indirect": 5, "approval_deception": 5, "composite": 5, "tip": 10}

# name -> factor, fixed overrides, requirements, control variant (None = the Week 8 greedy defaults)
VARIANTS: dict[str, dict[str, Any]] = {
    "ctx16k": {"factor": "context", "overrides": {"selected_context_length": 16384}, "needs": [], "control": None,
               "why": "16K vs 32K. 8K is not tested: Week 7 measured list_files output at ~8.4K tokens, which cannot fit."},
    "vendor_sampling": {"factor": "sampling", "overrides": {}, "needs": ["vendor_sampling"], "control": None,
                        "why": "vendor-recommended sampling (thinking off) vs greedy; 3 seeds make the stochastic runs distinct."},
    "thinking_on_vendor": {"factor": "thinking", "overrides": {"thinking_mode": "enabled", "max_output_tokens": 8192},
                           "needs": ["thinking_capability", "vendor_sampling"], "control": "vendor_sampling",
                           "why": "thinking on vs off under the SAME vendor sampling; output budget raised to 8192 because thinking tokens count."},
    "out1024": {"factor": "output_limit", "overrides": {"max_output_tokens": 1024}, "needs": [], "control": None,
                "why": "1024 vs 4096 output cap (the Week 7 pilot cap). 8192 is not run: greedy decoding with no 4096-cap hit in Week 8 is identical."},
}
ADAPTIVE_VARIANTS = ("vendor_sampling", "thinking_on_vendor")
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def select_subset(scenarios: list[Mapping[str, Any]]) -> list[str]:
    """Deterministic stratified subset: per family, the first N scenario ids ordered by sha256(id)."""

    chosen: list[str] = []
    for family, size in SUBSET_SIZES.items():
        ids = sorted((s["scenario_id"] for s in scenarios if s["family"] == family),
                     key=lambda sid: hashlib.sha256(sid.encode()).hexdigest())
        chosen += ids[:size]
    return sorted(chosen)


def build_plan(*, scenarios: list[Mapping[str, Any]], dataset_sha256: str, week8_protocol_sha256: str,
               objective_ids: list[str], seeds: list[int], arms: list[str]) -> dict[str, Any]:
    plan = {
        "schema_version": "week9-ablation-plan-v1", "week8_protocol_sha256": week8_protocol_sha256,
        "dataset_sha256": dataset_sha256, "scenario_ids": select_subset(scenarios), "seeds": list(seeds), "arms": list(arms),
        "subset_sizes": SUBSET_SIZES, "variants": VARIANTS, "adaptive_variants": list(ADAPTIVE_VARIANTS),
        "adaptive_objectives": objective_ids, "adaptive_seed": 42, "max_attempts": 15,
        "rules": ["thinking-on and thinking-off results are never pooled", "every scheduled episode stays in every denominator",
                  "control rows are the Week 8 rows for identical (scenario, arm, seed) keys"],
    }
    plan["plan_sha256"] = canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError(f"{path}: plan_sha256 does not match its content (file edited after it was frozen)")
    return plan


def parse_ollama_parameters(text: str) -> dict[str, list[str]]:
    """`/api/show` 'parameters' is one 'key value' per line; keys can repeat (stop)."""

    out: dict[str, list[str]] = {}
    for line in (text or "").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out.setdefault(parts[0], []).append(parts[1].strip().strip('"'))
    return out


def resolve_vendor_sampling(slug: str, show_parameters: str, overrides: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Vendor sampling = the model's own shipped parameters (digest-pinned, recorded), else a documented override. None = n/a."""

    shipped = parse_ollama_parameters(show_parameters)
    values: dict[str, Any] = {}
    for key, cast in (("temperature", float), ("top_p", float), ("top_k", int), ("min_p", float)):
        if key in shipped:
            try:
                values[key] = cast(shipped[key][-1])
            except ValueError:
                pass
    if "temperature" in values:
        return {"source": "model-shipped-parameters", "values": values, "shipped": shipped, "verified": True}
    entry = (overrides or {}).get(slug)
    if entry:
        return {"source": entry["source"], "values": {k: entry[k] for k in SAMPLING_KEYS if k in entry},
                "shipped": shipped, "verified": bool(entry.get("verified", False))}
    return None


def resolve_variant(profile, name: str, vendor: dict[str, Any] | None) -> tuple[Any | None, dict[str, Any]]:
    """(variant ModelProfile or None, record). None means the variant does not apply to this model; the reason is recorded."""

    spec = VARIANTS[name]
    record: dict[str, Any] = {"variant": name, "factor": spec["factor"], "control": spec["control"] or "week8-greedy-default"}
    if "thinking_capability" in spec["needs"] and not profile.thinking_capability:
        return None, {**record, "applicable": False, "reason": "model has no explicit thinking switch"}
    updates: dict[str, Any] = dict(spec["overrides"])
    if "vendor_sampling" in spec["needs"]:
        if vendor is None:
            return None, {**record, "applicable": False,
                          "reason": "no vendor sampling in the model's shipped parameters and no documented override"}
        updates.update(vendor["values"])
        record["vendor_sampling"] = {"source": vendor["source"], "verified": vendor["verified"], "values": vendor["values"]}
    if updates.get("thinking_mode") == "enabled" and not profile.thinking_capability:
        return None, {**record, "applicable": False, "reason": "model has no explicit thinking switch"}
    variant = profile.revalidated(**updates)
    record.update({"applicable": True, "overrides": updates, "variant_profile_sha256": variant.fingerprint})
    return variant, record
