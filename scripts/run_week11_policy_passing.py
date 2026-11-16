"""Week 11: policy-passing attacks under a reviewer panel (resume-safe, ONE model per call).

    python scripts/run_week11_policy_passing.py --slug qwen3-14b --endpoint http://127.0.0.1:PORT --check-only
    python scripts/run_week11_policy_passing.py --slug qwen3-14b --endpoint ...                    # every configuration, in plan order
    python scripts/run_week11_policy_passing.py --slug qwen3-14b --endpoint ... --config P-R2 --max-episodes 4 --results-root results/week11_smoke

Each configuration writes to results/week11/<slug>/<config>/episodes.jsonl.  B0 is the stock baseline; the other configurations are
the protected arm with the stated policy setting and reviewer.  All configurations use the same 28 scenarios, the Week 8 seeds and the
frozen Week 8 model profile, so episodes pair across configurations.
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
from scripts.run_week8_frozen import append_line, load_jsonl, runner_failure_row
from scripts.run_week9_ablation import server_memory
from scripts.week11_lib import (
    PLAN_PATH, RESULTS_ROOT, REVIEW_LIMITS, layers_for, load_plan, make_hook, make_reviewer, needs_model, verify_week11_code,
)

IMMUTABLE = ("slug", "config", "plan_sha256", "model_digest", "portable_profile_sha256", "protocol_sha256", "defense_version", "layers", "reviewer")


def plan_checks(plan: dict[str, Any], session: dict[str, Any], root: Path) -> None:
    if plan["week8_protocol_sha256"] != session["protocol"]["protocol_sha256"]:
        raise ProtocolViolation("the Week 11 plan was frozen against a different Week 8 protocol")


def run_policy_passing(
    slug: str, endpoint: str, *, results_root: Path = RESULTS_ROOT, plan_path: Path = PLAN_PATH,
    protocol_path: Path = PROTOCOL_PATH, matrix_path: Path = MATRIX_PATH, profiles_dir: Path = PROFILES_DIR,
    root: Path = ROOT, configs: list[str] | None = None, max_episodes: int | None = None, check_only: bool = False,
    http=None, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    code_digest = verify_week11_code(plan, root)
    session = prepare_model_session(slug, endpoint, protocol_path=protocol_path, matrix_path=matrix_path,
                                    profiles_dir=profiles_dir, root=root, http=http, check_code=False)
    plan_checks(plan, session, root)
    identity, profile = session["identity"], session["profile"]
    unknown = [c for c in (configs or []) if c not in plan["configs"]]
    if unknown:
        raise ProtocolViolation(f"unknown configuration(s) {unknown}; the plan has {plan['config_order']}")
    names = [c for c in plan["config_order"] if not configs or c in configs]
    owns = http is None
    client = http or httpx.Client(base_url=endpoint.rstrip("/"), timeout=60.0)
    summary: dict[str, Any] = {"slug": slug, "configs": {}, "week11_code_digest": code_digest}
    try:
        for name in names:
            spec = plan["configs"][name]
            log(f"{slug}/{name}: arm={spec['arm']} layers={'-' if spec['layers'] is None else [k for k, v in spec['layers'].items() if not v] or 'all on'}")
            if check_only:
                summary["configs"][name] = {"check_only": True}
                continue
            out_dir = Path(results_root) / slug / name
            out_dir.mkdir(parents=True, exist_ok=True)
            candidate = {"schema_version": "week11-run-manifest-v1", "created_at": utc_now(), "slug": slug, "config": name,
                         "plan_sha256": plan["plan_sha256"], "model_digest": identity["model_digest"],
                         "portable_profile_sha256": identity["portable_profile_sha256"],
                         "protocol_sha256": identity["protocol_sha256"], "defense_version": code_digest,
                         "layers": spec["layers"], "reviewer": spec["reviewer"], "arm": spec["arm"],
                         "limits": {**session["protocol"]["limits"], "approval_seconds": REVIEW_LIMITS.approval_seconds},
                         "environment": session["environment"]}
            ensure_run_manifest(out_dir / "run_manifest.json", candidate, IMMUTABLE)
            summary["configs"][name] = run_config(slug, name, profile, plan, identity, code_digest, out_dir, client, root,
                                                  max_episodes, log)
        return summary
    finally:
        if owns:
            client.close()


def run_config(slug, name, profile, plan, identity, code_digest, out_dir, client, root, max_episodes, log) -> dict[str, Any]:
    scenarios = {s["scenario_id"]: s for s in load_jsonl(root / plan["dataset"])}
    spec = plan["configs"][name]
    layers, arm, reviewer_name = layers_for(spec), spec["arm"], spec["reviewer"]
    output = out_dir / "episodes.jsonl"
    existing = load_jsonl(output)
    foreign = [r for r in existing if r.get("plan_sha256") != plan["plan_sha256"] or r.get("week11_config") != name]
    if foreign:
        raise ProtocolViolation(f"{output} has {len(foreign)} rows from another plan/configuration; use a new results directory")
    done = {episode_key(r) for r in existing}
    pending = [(scenarios[sid], seed) for seed in plan["seeds"] for sid in plan["scenario_ids"] if (sid, arm, seed) not in done]
    if max_episodes is not None:
        pending = pending[:max_episodes]
    context = run_context()
    summary = {"arm": arm, "already_saved": len(existing), "selected": len(pending), "ran": 0, "errors": 0}
    if not pending:
        log(f"{slug}/{name}: complete ({len(existing)} saved)")
        return summary
    probe = build_llm(profile, seed=plan["seeds"][0], limits=REVIEW_LIMITS)
    if not probe.healthcheck()["model_available"]:
        raise ProtocolViolation(f"server reachable but {profile.model_id} unavailable")
    probe.warmup()
    memory = server_memory(client, profile.model_id)
    sessions = out_dir / "sessions.jsonl"
    with sessions.open("a", encoding="utf-8") as handle:
        append_line(handle, {"event": "start", "at": utc_now(), "context": context, "already_saved": len(existing),
                             "selected": len(pending), "server_memory_after_warmup": memory})
    log(f"{slug}/{name}: {len(existing)} saved, {len(pending)} to run")
    evidence_root = out_dir / "evidence"
    started_all = time.monotonic()
    try:
        with output.open("a", encoding="utf-8", buffering=1) as handle:
            for index, (scenario, seed) in enumerate(pending, start=1):
                quarantine_stale_evidence(evidence_token_dir(evidence_root, scenario["scenario_id"], seed, arm))
                llm = build_llm(profile, seed=seed, limits=REVIEW_LIMITS)
                started = utc_now()
                review_llm = build_llm(profile, seed=seed, limits=REVIEW_LIMITS) if needs_model(reviewer_name) else None
                try:
                    reviewer = make_reviewer(reviewer_name, scenario, review_llm, cap=spec.get("denial_cap"))
                    kwargs: dict[str, Any] = {"post_episode_hook": make_hook(scenario, reviewer)}
                    if arm == "protected":
                        kwargs.update({"layers": layers, "reviewer": reviewer, "describe_targets": bool(spec.get("describe", False)),
                                       "annotate_sources": bool(spec.get("annotate", False)),
                                       "check_provenance": bool(spec.get("provenance", False))})
                    result = run_episode(scenario, arm=arm, seed=seed, evidence_dir=evidence_root, llm=llm,
                                         runtime_limits=REVIEW_LIMITS, **kwargs).to_dict()
                except Exception as exc:  # recorded, never dropped
                    result = runner_failure_row(scenario, arm, seed, profile.model_id, exc)
                result.update({
                    "week11_schema": "policy-passing-episode-v1", "week11_config": name, "reviewer": reviewer_name,
                    "vector": scenario.get("vector"), "goal": scenario.get("goal"), "plan_sha256": plan["plan_sha256"],
                    "layers": None if layers is None else layers.to_dict(),
                    **{k: identity[k] for k in ("slug", "model_digest", "protocol_sha256")}, "defense_version": code_digest,
                    "portable_profile_sha256": identity["portable_profile_sha256"],
                    "kind": scenario["kind"], "family": scenario["family"], "scenario_checksum": scenario["checksum"],
                    "request_log": list(getattr(llm.model_client, "request_log", [])),
                    "reviewer_request_log": list(getattr(getattr(review_llm, "model_client", None), "request_log", [])),
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT"))
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--plan", type=Path, default=PLAN_PATH)
    ap.add_argument("--config", action="append", dest="configs")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.endpoint:
        print("--endpoint (or $ENDPOINT) is required")
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        run_policy_passing(args.slug, args.endpoint, results_root=args.results_root, plan_path=args.plan, configs=args.configs,
                   max_episodes=args.max_episodes, check_only=args.check_only)
    except ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
