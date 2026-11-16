"""Week 8 adaptive-attacker campaign for ONE model: 10 objectives x up to 15 attempts, protected arm, seed 42.

    python scripts/run_week8_adaptive.py --slug granite4-small-h --endpoint http://127.0.0.1:PORT

Same identity checks as the frozen campaign. An objective's rows are written only after the whole objective finishes (success or
attempt 15), so an interruption can never leave a misleading partial curve; an interrupted objective starts again from attempt 1
with its half-written event store moved aside (the policy budget must not inherit stale rejected attempts).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pg.week7_models import build_llm
from pg.week8_protocol import (
    ADAPTIVE_SEED, LIMITS, MATRIX_PATH, PROFILES_DIR, PROTOCOL_PATH, RESULTS_ROOT, ProtocolViolation, file_sha256,
    prepare_model_session, quarantine_stale_evidence, run_context, utc_now,
)
from scripts import run_week6_adaptive as w6
from scripts.run_week8_frozen import append_line, load_jsonl


def run_adaptive(
    slug: str, endpoint: str, *, results_root: Path = RESULTS_ROOT, protocol_path: Path = PROTOCOL_PATH,
    matrix_path: Path = MATRIX_PATH, profiles_dir: Path = PROFILES_DIR, root: Path = ROOT,
    objective_ids: list[str] | None = None, max_objectives: int | None = None, http=None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    session = prepare_model_session(slug, endpoint, protocol_path=protocol_path, matrix_path=matrix_path,
                                    profiles_dir=profiles_dir, root=root, http=http)
    protocol, profile, identity = session["protocol"], session["profile"], session["identity"]
    adaptive = protocol["campaign"]["adaptive"]
    dataset = root / "data" / "week6" / "adaptive_objectives.jsonl"
    if file_sha256(dataset) != adaptive["dataset_sha256"]:
        raise ProtocolViolation("adaptive objectives differ from the frozen checksum")
    objectives = load_jsonl(dataset)
    out_dir = Path(results_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "adaptive_attempts.jsonl"
    existing = load_jsonl(output)
    foreign = [r for r in existing if r.get("model_digest") != identity["model_digest"] or r.get("protocol_sha256") != identity["protocol_sha256"]]
    if foreign:
        raise ProtocolViolation(f"{output} contains rows from a different model/protocol; use a new results directory")
    completed = {r["objective_id"] for r in existing if r.get("attack_success") or int(r.get("attempt", 0)) == adaptive["max_attempts"]}
    wanted = set(objective_ids or [o["objective_id"] for o in objectives])
    pending = [o for o in objectives if o["objective_id"] in wanted and o["objective_id"] not in completed]
    if max_objectives is not None:
        pending = pending[:max_objectives]
    context = run_context()
    extras = {**{k: identity[k] for k in ("slug", "model_digest", "portable_profile_sha256", "protocol_sha256", "defense_version")},
              "slurm_job_id": context.get("slurm_job_id"), "node": context.get("node"), "gpu": (context.get("gpus") or [None])[0]}
    probe = build_llm(profile, seed=ADAPTIVE_SEED, limits=LIMITS)
    if not probe.healthcheck()["model_available"]:
        raise ProtocolViolation(f"server reachable but {profile.model_id} unavailable")
    probe.warmup()
    evidence_root = out_dir / "adaptive_evidence"
    summary = {"slug": slug, "objectives_pending": len(pending), "objectives_run": 0, "attempts": 0, "successes": 0}
    sessions = out_dir / "sessions.jsonl"
    with sessions.open("a", encoding="utf-8") as handle:
        append_line(handle, {"event": "adaptive_start", "at": utc_now(), "context": context, "pending": len(pending)})
    log(f"{slug} adaptive: {len(completed)} objectives done, {len(pending)} pending")
    try:
        with output.open("a", encoding="utf-8", buffering=1) as handle:
            for objective in pending:
                stale = quarantine_stale_evidence(evidence_root / objective["objective_id"] / f"seed-{ADAPTIVE_SEED}")
                if stale:
                    log(f"  moved leftovers of an interrupted objective to {stale.name}")
                rows = w6.run_objective(
                    objective, ADAPTIVE_SEED, evidence_root,
                    llm_factory=lambda seed: build_llm(profile, seed=seed, limits=LIMITS),
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
    ap.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    ap.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    ap.add_argument("--profiles-dir", type=Path, default=PROFILES_DIR)
    ap.add_argument("--objective", action="append", dest="objectives")
    ap.add_argument("--max-objectives", type=int)
    args = ap.parse_args()
    if not args.endpoint:
        print("--endpoint (or $ENDPOINT) is required")
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        run_adaptive(args.slug, args.endpoint, results_root=args.results_root, protocol_path=args.protocol,
                     matrix_path=args.matrix, profiles_dir=args.profiles_dir, objective_ids=args.objectives,
                     max_objectives=args.max_objectives)
    except ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
