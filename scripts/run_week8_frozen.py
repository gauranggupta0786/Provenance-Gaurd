"""Resume-safe Week 8 frozen campaign for ONE model: 100 scenarios x 2 arms x 3 seeds (600 episodes).

    python scripts/run_week8_frozen.py --slug granite4-small-h --endpoint http://127.0.0.1:PORT
    python scripts/run_week8_frozen.py --slug qwen3-14b --endpoint ... --check-only     # verify everything, run nothing
    python scripts/run_week8_frozen.py --slug qwen3-14b --endpoint ... --max-episodes 4 --results-root results/week8_smoke

Refuses to start unless the frozen matrix, the approval record, the Week 8 protocol manifest, the code digest, the model's
profile, and the live server (model digest, template, engine version) all match what was frozen. Re-running the same command
continues where it stopped; finished (scenario, arm, seed) keys are never repeated.
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

from pg.week5_runner import run_episode
from pg.week6_metrics import episode_key
from pg.week7_models import build_llm
from pg.week8_protocol import (
    ARMS, LIMITS, MATRIX_PATH, PROFILES_DIR, PROTOCOL_PATH, RESULTS_ROOT, ProtocolViolation, SEEDS,
    ensure_run_manifest, evidence_token_dir, file_sha256, prepare_model_session, quarantine_stale_evidence,
    run_context, utc_now,
)

IMMUTABLE = ("slug", "protocol_sha256", "model_digest", "portable_profile_sha256", "defense_version", "dataset_sha256")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append_line(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def runner_failure_row(scenario: dict[str, Any], arm: str, seed: int, model_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "scenario_id": scenario["scenario_id"], "arm": arm, "model_id": model_id, "seed": seed, "completed": False,
        "benign_success": False if scenario["kind"] == "benign" else None, "attack_success": None,
        "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0, "tool_calls": [], "mediated_calls": [],
        "audit_traces": [], "final_state_digest": "", "checks": [], "final_response": "", "output_guard": None,
        "disclosure_attempted": False, "confidentiality_violation": False, "sensitivity_evidence": [],
        "runtime_timings": [], "error": f"runner-level {type(exc).__name__}: {exc}",
    }


def run_campaign(
    slug: str, endpoint: str, *, results_root: Path = RESULTS_ROOT, protocol_path: Path = PROTOCOL_PATH,
    matrix_path: Path = MATRIX_PATH, profiles_dir: Path = PROFILES_DIR, root: Path = ROOT,
    scenario_ids: list[str] | None = None, seeds: list[int] | None = None, arms: list[str] | None = None,
    max_episodes: int | None = None, check_only: bool = False, http=None, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    session = prepare_model_session(slug, endpoint, protocol_path=protocol_path, matrix_path=matrix_path,
                                    profiles_dir=profiles_dir, root=root, http=http)
    protocol, profile, identity = session["protocol"], session["profile"], session["identity"]
    campaign = protocol["campaign"]
    seeds = list(seeds or campaign["seeds"])
    arms = list(arms or campaign["arms"])
    if not set(seeds) <= set(campaign["seeds"]) or not set(arms) <= set(campaign["arms"]):
        raise ProtocolViolation(f"seeds/arms must be subsets of the frozen {campaign['seeds']} / {campaign['arms']}")

    dataset = root / campaign["dataset"]
    if file_sha256(dataset) != campaign["dataset_sha256"]:
        raise ProtocolViolation("held-out dataset differs from the frozen checksum")
    scenarios = load_jsonl(dataset)
    if len(scenarios) != campaign["scenarios"]:
        raise ProtocolViolation(f"frozen dataset must contain {campaign['scenarios']} scenarios, found {len(scenarios)}")

    out_dir = Path(results_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = {
        "schema_version": "week8-run-manifest-v1", "created_at": utc_now(), **identity,
        "profile_file_sha256": protocol["models"][slug]["profile_file_sha256"], "dataset_sha256": campaign["dataset_sha256"],
        "arms": campaign["arms"], "seeds": campaign["seeds"], "expected_episodes": campaign["episodes_per_model"],
        "limits": protocol["limits"], "environment": session["environment"],
        "outcome_rule": "AgentDojo final simulated state; model prose or refusal is not the outcome",
        "error_rule": "every scheduled episode stays in every denominator; failures are recorded, never dropped",
    }
    manifest = ensure_run_manifest(out_dir / "run_manifest.json", candidate, IMMUTABLE)

    output = out_dir / "frozen_episodes.jsonl"
    existing = load_jsonl(output)
    foreign = [r for r in existing if r.get("model_digest") != identity["model_digest"] or r.get("protocol_sha256") != identity["protocol_sha256"]]
    if foreign:
        raise ProtocolViolation(f"{output} contains {len(foreign)} rows from a different model/protocol; use a new results directory")
    done = {episode_key(r) for r in existing}
    wanted = set(scenario_ids or [s["scenario_id"] for s in scenarios])
    pending = [(s, a, sd) for s in scenarios if s["scenario_id"] in wanted for sd in seeds for a in arms
               if (s["scenario_id"], a, sd) not in done]
    if max_episodes is not None:
        pending = pending[:max_episodes]

    context = run_context()
    sessions = out_dir / "sessions.jsonl"
    summary = {"slug": slug, "already_saved": len(existing), "selected": len(pending), "ran": 0, "errors": 0}
    with sessions.open("a", encoding="utf-8") as handle:
        append_line(handle, {"event": "start", "at": utc_now(), "context": context, "live_server": session["live"],
                             "already_saved": len(existing), "selected": len(pending), "check_only": check_only})
    log(f"{slug}: {len(existing)}/{manifest['expected_episodes']} saved, {len(pending)} selected "
        f"(protocol {identity['protocol_sha256'][:12]}, digest {identity['model_digest'][:12]})")
    if check_only:
        return summary

    probe = build_llm(profile, seed=seeds[0], limits=LIMITS)
    health = probe.healthcheck()
    if not health["model_available"]:
        raise ProtocolViolation(f"server reachable but {profile.model_id} unavailable")
    probe.warmup()
    evidence_root = out_dir / "evidence"
    started_all = time.monotonic()
    try:
        with output.open("a", encoding="utf-8", buffering=1) as handle:
            for index, (scenario, arm, seed) in enumerate(pending, start=1):
                stale = quarantine_stale_evidence(evidence_token_dir(evidence_root, scenario["scenario_id"], seed, arm))
                if stale:
                    log(f"  moved leftovers of an interrupted episode to {stale.name}")
                llm = build_llm(profile, seed=seed, limits=LIMITS)
                started = utc_now()
                try:
                    result = run_episode(scenario, arm=arm, seed=seed, evidence_dir=evidence_root, llm=llm,
                                         runtime_limits=LIMITS).to_dict()
                except Exception as exc:  # recorded, never dropped
                    result = runner_failure_row(scenario, arm, seed, profile.model_id, exc)
                result.update({
                    "week8_schema": "frozen-episode-v1", **{k: identity[k] for k in (
                        "slug", "model_digest", "portable_profile_sha256", "protocol_sha256", "defense_version")},
                    "kind": scenario["kind"], "family": scenario["family"], "guidance": scenario.get("guidance"),
                    "split": scenario["split"], "scenario_checksum": scenario["checksum"],
                    "request_log": list(getattr(llm.model_client, "request_log", [])),
                    "slurm_job_id": context.get("slurm_job_id"), "node": context.get("node"),
                    "gpu": (context.get("gpus") or [None])[0], "started_at": started, "finished_at": utc_now(),
                })
                append_line(handle, result)
                summary["ran"] += 1
                summary["errors"] += bool(result.get("error"))
                outcome = ("attack=" + str(result.get("attack_success"))) if scenario["kind"] == "attack" else (
                    "benign=" + str(result.get("benign_success")))
                elapsed = time.monotonic() - started_all
                eta = elapsed / index * (len(pending) - index)
                log(f"[{index}/{len(pending)}] {scenario['scenario_id']} {arm} seed={seed} {outcome} "
                    f"{'ERROR ' if result.get('error') else ''}{result['latency_ms']}ms eta={eta/60:.0f}min")
    finally:
        with sessions.open("a", encoding="utf-8") as handle:
            append_line(handle, {"event": "end", "at": utc_now(), **summary})
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT"), help="server URL (default $ENDPOINT)")
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    ap.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    ap.add_argument("--profiles-dir", type=Path, default=PROFILES_DIR)
    ap.add_argument("--scenario", action="append", dest="scenarios")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--arms", nargs="+", choices=list(ARMS))
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.endpoint:
        print("--endpoint (or $ENDPOINT) is required")
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # Slurm time limit: let the session record close
    try:
        run_campaign(args.slug, args.endpoint, results_root=args.results_root, protocol_path=args.protocol,
                     matrix_path=args.matrix, profiles_dir=args.profiles_dir, scenario_ids=args.scenarios,
                     seeds=args.seeds, arms=args.arms, max_episodes=args.max_episodes, check_only=args.check_only)
    except ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
