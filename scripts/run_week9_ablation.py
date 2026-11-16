"""Week 9 inference-setting ablations for ONE model (resume-safe). Control = the Week 8 rows for the same keys.

    python scripts/run_week9_ablation.py --slug qwen3-14b --endpoint http://127.0.0.1:PORT --check-only
    python scripts/run_week9_ablation.py --slug qwen3-14b --endpoint ...                  # frozen subset, every applicable variant
    python scripts/run_week9_ablation.py --slug qwen3-14b --endpoint ... --adaptive       # adaptive attacker for the adaptive variants

Everything the Week 8 runner verifies is verified again (matrix, protocol, code digest, profile, live server). Each variant writes
to results/week9/<slug>/<variant>/ so thinking-on and thinking-off rows can never share a file.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from pg.week5_runner import run_episode
from pg.week6_metrics import episode_key
from pg.week7_models import build_llm
from pg.week8_protocol import (
    LIMITS, MATRIX_PATH, PROFILES_DIR, PROTOCOL_PATH, ProtocolViolation, ensure_run_manifest, evidence_token_dir,
    file_sha256, prepare_model_session, quarantine_stale_evidence, run_context, utc_now,
)
from scripts import run_week6_adaptive as w6
from scripts.run_week8_frozen import append_line, load_jsonl, runner_failure_row
from scripts.week9_lib import (
    ADAPTIVE_VARIANTS, OVERRIDES_PATH, PLAN_PATH, RESULTS_ROOT, VARIANTS, load_plan, resolve_variant, resolve_vendor_sampling,
)

IMMUTABLE = ("slug", "variant", "plan_sha256", "model_digest", "variant_profile_sha256", "protocol_sha256", "defense_version")


def server_memory(client: httpx.Client, model_id: str) -> dict[str, Any] | None:
    """What Ollama reports for the loaded model (VRAM bytes, context). None if the endpoint cannot answer."""

    try:
        for m in client.get("/api/ps").json().get("models", []):
            if m.get("name") == model_id or m.get("model") == model_id:
                return {k: m.get(k) for k in ("size", "size_vram", "context_length", "expires_at") if k in m}
    except Exception:  # memory is a measurement aid; never stop a campaign for it
        return None
    return None


def show_parameters(client: httpx.Client, model_id: str) -> str:
    return str(client.post("/api/show", json={"model": model_id}).json().get("parameters") or "")


def plan_checks(plan: dict[str, Any], session: dict[str, Any], root: Path) -> None:
    if plan["week8_protocol_sha256"] != session["protocol"]["protocol_sha256"]:
        raise ProtocolViolation("the Week 9 plan was frozen against a different Week 8 protocol")
    if file_sha256(root / session["protocol"]["campaign"]["dataset"]) != plan["dataset_sha256"]:
        raise ProtocolViolation("held-out dataset differs from the one the Week 9 plan was frozen against")


def run_ablation(
    slug: str, endpoint: str, *, results_root: Path = RESULTS_ROOT, plan_path: Path = PLAN_PATH,
    protocol_path: Path = PROTOCOL_PATH, matrix_path: Path = MATRIX_PATH, profiles_dir: Path = PROFILES_DIR,
    overrides_path: Path = OVERRIDES_PATH, root: Path = ROOT, variants: list[str] | None = None, adaptive: bool = False,
    max_episodes: int | None = None, max_objectives: int | None = None, check_only: bool = False, http=None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    session = prepare_model_session(slug, endpoint, protocol_path=protocol_path, matrix_path=matrix_path,
                                    profiles_dir=profiles_dir, root=root, http=http)
    plan_checks(plan, session, root)
    identity, base_profile = session["identity"], session["profile"]
    overrides = json.loads(Path(overrides_path).read_text(encoding="utf-8")) if Path(overrides_path).exists() else {}
    owns = http is None
    client = http or httpx.Client(base_url=endpoint.rstrip("/"), timeout=60.0)
    try:
        vendor = resolve_vendor_sampling(slug, show_parameters(client, base_profile.model_id), overrides)
        names = [v for v in (variants or list(VARIANTS)) if v in VARIANTS]
        if adaptive:
            names = [v for v in names if v in ADAPTIVE_VARIANTS]
        resolved = {name: resolve_variant(base_profile, name, vendor) for name in names}
        summary: dict[str, Any] = {"slug": slug, "variants": {}}
        for name, (profile, record) in resolved.items():
            if profile is None:
                log(f"{slug}/{name}: NOT APPLICABLE ({record['reason']})")
                summary["variants"][name] = {"applicable": False, "reason": record["reason"]}
                continue
            extra = f" vendor={record['vendor_sampling']['values']} ({record['vendor_sampling']['source'][:40]})" if "vendor_sampling" in record else ""
            log(f"{slug}/{name}: {record['overrides']}{extra}")
            if check_only:
                summary["variants"][name] = {"applicable": True, "record": record}
                continue
            out_dir = Path(results_root) / slug / name
            out_dir.mkdir(parents=True, exist_ok=True)
            candidate = {"schema_version": "week9-run-manifest-v1", "created_at": utc_now(), "slug": slug, "variant": name,
                         "plan_sha256": plan["plan_sha256"], "model_digest": identity["model_digest"],
                         "variant_profile_sha256": record["variant_profile_sha256"], "protocol_sha256": identity["protocol_sha256"],
                         "defense_version": identity["defense_version"], "resolved": record, "limits": session["protocol"]["limits"],
                         "environment": session["environment"], "control": record["control"]}
            ensure_run_manifest(out_dir / "run_manifest.json", candidate, IMMUTABLE)
            if adaptive:
                summary["variants"][name] = run_adaptive_variant(slug, name, profile, plan, identity, out_dir, client, max_objectives, log)
            else:
                summary["variants"][name] = run_frozen_variant(slug, name, profile, plan, identity, out_dir, client, root,
                                                                max_episodes, log)
        return summary
    finally:
        if owns:
            client.close()


def _common(identity: dict[str, Any], plan: dict[str, Any], name: str, record_profile) -> dict[str, Any]:
    return {**{k: identity[k] for k in ("slug", "model_digest", "protocol_sha256", "defense_version")},
            "week9_variant": name, "plan_sha256": plan["plan_sha256"], "variant_profile_sha256": record_profile.fingerprint}


def run_frozen_variant(slug, name, profile, plan, identity, out_dir, client, root, max_episodes, log) -> dict[str, Any]:
    scenarios = {s["scenario_id"]: s for s in load_jsonl(root / "data" / "week5" / "heldout.jsonl")}
    output = out_dir / "frozen_episodes.jsonl"
    existing = load_jsonl(output)
    foreign = [r for r in existing if r.get("plan_sha256") != plan["plan_sha256"] or r.get("week9_variant") != name]
    if foreign:
        raise ProtocolViolation(f"{output} has {len(foreign)} rows from another plan/variant; use a new results directory")
    done = {episode_key(r) for r in existing}
    pending = [(scenarios[sid], arm, seed) for sid in plan["scenario_ids"] for seed in plan["seeds"] for arm in plan["arms"]
               if (sid, arm, seed) not in done]
    if max_episodes is not None:
        pending = pending[:max_episodes]
    context = run_context()
    summary = {"applicable": True, "already_saved": len(existing), "selected": len(pending), "ran": 0, "errors": 0}
    probe = build_llm(profile, seed=plan["seeds"][0], limits=LIMITS)
    if not probe.healthcheck()["model_available"]:
        raise ProtocolViolation(f"server reachable but {profile.model_id} unavailable")
    probe.warmup()
    memory = server_memory(client, profile.model_id)
    sessions = out_dir / "sessions.jsonl"
    with sessions.open("a", encoding="utf-8") as handle:
        append_line(handle, {"event": "start", "at": utc_now(), "context": context, "already_saved": len(existing),
                             "selected": len(pending), "server_memory_after_warmup": memory})
    log(f"{slug}/{name}: {len(existing)} saved, {len(pending)} to run; server memory {memory}")
    evidence_root = out_dir / "evidence"
    started_all = time.monotonic()
    try:
        with output.open("a", encoding="utf-8", buffering=1) as handle:
            for index, (scenario, arm, seed) in enumerate(pending, start=1):
                quarantine_stale_evidence(evidence_token_dir(evidence_root, scenario["scenario_id"], seed, arm))
                llm = build_llm(profile, seed=seed, limits=LIMITS)
                started = utc_now()
                try:
                    result = run_episode(scenario, arm=arm, seed=seed, evidence_dir=evidence_root, llm=llm, runtime_limits=LIMITS).to_dict()
                except Exception as exc:  # recorded, never dropped
                    result = runner_failure_row(scenario, arm, seed, profile.model_id, exc)
                result.update({
                    "week9_schema": "ablation-episode-v1", **_common(identity, plan, name, profile),
                    "config": {"thinking_mode": profile.thinking_mode, "temperature": profile.temperature, "top_p": profile.top_p,
                               "top_k": profile.top_k, "min_p": profile.min_p, "context": profile.selected_context_length,
                               "max_output_tokens": profile.max_output_tokens},
                    "kind": scenario["kind"], "family": scenario["family"], "scenario_checksum": scenario["checksum"],
                    "request_log": list(getattr(llm.model_client, "request_log", [])),
                    "slurm_job_id": context.get("slurm_job_id"), "node": context.get("node"),
                    "started_at": started, "finished_at": utc_now(),
                })
                append_line(handle, result)
                summary["ran"] += 1
                summary["errors"] += bool(result.get("error"))
                eta = (time.monotonic() - started_all) / index * (len(pending) - index)
                log(f"[{index}/{len(pending)}] {name} {scenario['scenario_id']} {arm} seed={seed} "
                    f"{'ERROR ' if result.get('error') else ''}{result['latency_ms']}ms eta={eta/60:.0f}min")
    finally:
        with sessions.open("a", encoding="utf-8") as handle:
            append_line(handle, {"event": "end", "at": utc_now(), **{k: summary[k] for k in ("ran", "errors")}})
    return summary


def run_adaptive_variant(slug, name, profile, plan, identity, out_dir, client, max_objectives, log) -> dict[str, Any]:
    objectives = [o for o in load_jsonl(ROOT / "data" / "week6" / "adaptive_objectives.jsonl") if o["objective_id"] in plan["adaptive_objectives"]]
    output = out_dir / "adaptive_attempts.jsonl"
    existing = load_jsonl(output)
    foreign = [r for r in existing if r.get("plan_sha256") != plan["plan_sha256"] or r.get("week9_variant") != name]
    if foreign:
        raise ProtocolViolation(f"{output} has rows from another plan/variant; use a new results directory")
    completed = {r["objective_id"] for r in existing if r.get("attack_success") or int(r.get("attempt", 0)) == plan["max_attempts"]}
    pending = [o for o in objectives if o["objective_id"] not in completed]
    if max_objectives is not None:
        pending = pending[:max_objectives]
    context = run_context()
    extras = {**_common(identity, plan, name, profile), "slurm_job_id": context.get("slurm_job_id"), "node": context.get("node"),
              "config": {"thinking_mode": profile.thinking_mode, "temperature": profile.temperature, "context": profile.selected_context_length}}
    seed = plan["adaptive_seed"]
    probe = build_llm(profile, seed=seed, limits=LIMITS)
    if not probe.healthcheck()["model_available"]:
        raise ProtocolViolation(f"server reachable but {profile.model_id} unavailable")
    probe.warmup()
    evidence_root = out_dir / "adaptive_evidence"
    summary = {"applicable": True, "objectives_pending": len(pending), "objectives_run": 0, "attempts": 0, "successes": 0}
    log(f"{slug}/{name} adaptive: {len(completed)} done, {len(pending)} pending")
    sessions = out_dir / "sessions.jsonl"
    try:
        with output.open("a", encoding="utf-8", buffering=1) as handle:
            for objective in pending:
                quarantine_stale_evidence(evidence_root / objective["objective_id"] / f"seed-{seed}")
                rows = w6.run_objective(objective, seed, evidence_root,
                                        llm_factory=lambda s: build_llm(profile, seed=s, limits=LIMITS),
                                        model_id=profile.model_id, row_extras=extras)
                for row in rows:
                    append_line(handle, row)
                summary["objectives_run"] += 1
                summary["attempts"] += len(rows)
                summary["successes"] += bool(rows and rows[-1].get("attack_success"))
    finally:
        with sessions.open("a", encoding="utf-8") as handle:
            append_line(handle, {"event": "adaptive_end", "at": utc_now(), **summary})
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT"))
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--plan", type=Path, default=PLAN_PATH)
    ap.add_argument("--variant", action="append", dest="variants", choices=list(VARIANTS))
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--max-objectives", type=int)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.endpoint:
        print("--endpoint (or $ENDPOINT) is required")
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        run_ablation(args.slug, args.endpoint, results_root=args.results_root, plan_path=args.plan, variants=args.variants,
                     adaptive=args.adaptive, max_episodes=args.max_episodes, max_objectives=args.max_objectives, check_only=args.check_only)
    except ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
