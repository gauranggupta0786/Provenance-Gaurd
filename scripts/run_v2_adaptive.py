"""Phase 5 runner: the adaptive attacker against chosen configurations, for ONE agent / reviewer / attacker model triple.

    python scripts/run_v2_adaptive.py --agent qwen3-14b --agent-endpoint ... --reviewer gemma4-e4b --reviewer-endpoint ... \
        --attacker qwen3-14b --attacker-endpoint ... --config none --config PG-v4 --budget 10 --feedback outcome --split test

Targets: the frozen target list ``data/v2/adaptive_targets.json`` (written once by ``--make-targets``: per suite, every in-scope
injection task paired with two test-split user tasks chosen with a fixed seed).  Each (target, config) is attacked until success
or ``--budget`` attempts; rows go to ``<results-root>/<agent>__<reviewer>__atk-<attacker>/<config>/<feedback>/episodes.jsonl``.
Resume-safe at the target level (a target with saved rows is skipped).  Test-split targets need the registration, exactly like
``run_v2_agentdojo.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgx.adaptive import attack_target, run_for  # noqa: E402
from pgx.agentdojo_runner import Case  # noqa: E402
from pgx.configs import CONFIGS_V2, needs_reviewer_model  # noqa: E402
from pgx.version import pgx_digest  # noqa: E402
from scripts.run_v2_agentdojo import Refused, check_registration, load_cases  # noqa: E402

DATA = ROOT / "data" / "v2"
TARGETS = DATA / "adaptive_targets.json"
RESULTS_ROOT = ROOT / "results" / "v2_adaptive"


def make_targets(per_injection: int = 2) -> list[dict]:
    tasks = json.loads((DATA / "agentdojo_tasks.json").read_text())
    out = []
    for split_name in ("dev", "test"):
        cases = [c for c in load_cases([split_name], list(tasks["suites"]), benign=False) if c.in_scope]
        by_inj: dict[tuple, list[Case]] = {}
        for c in cases:
            by_inj.setdefault((c.suite, c.injection_task_id), []).append(c)
        for (suite, iid), cs in sorted(by_inj.items()):
            rng = random.Random(f"adaptive-{split_name}-{suite}-{iid}")
            for c in sorted(rng.sample(cs, min(per_injection, len(cs))), key=lambda c: c.case_id):
                out.append({"suite": suite, "user_task_id": c.user_task_id, "injection_task_id": iid, "split": split_name})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-targets", action="store_true", help="write data/v2/adaptive_targets.json once and stop")
    ap.add_argument("--agent")
    ap.add_argument("--agent-endpoint", default=os.environ.get("AGENT_ENDPOINT"))
    ap.add_argument("--reviewer")
    ap.add_argument("--reviewer-endpoint", default=os.environ.get("REVIEWER_ENDPOINT"))
    ap.add_argument("--attacker")
    ap.add_argument("--attacker-endpoint", default=os.environ.get("ATTACKER_ENDPOINT"))
    ap.add_argument("--config", action="append", dest="configs")
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--feedback", choices=("outcome", "whitebox"), default="outcome")
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--max-targets", type=int)
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = ap.parse_args(argv)
    if args.make_targets:
        if TARGETS.exists():
            print(f"REFUSED: {TARGETS.relative_to(ROOT)} exists (frozen); delete it deliberately to rebuild")
            return 2
        targets = make_targets()
        TARGETS.write_text(json.dumps(targets, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {len(targets)} targets to {TARGETS.relative_to(ROOT)}")
        return 0
    missing = [n for n in ("agent", "agent_endpoint", "reviewer", "reviewer_endpoint", "attacker", "attacker_endpoint", "configs")
               if not getattr(args, n)]
    if missing:
        print(f"REFUSED: missing {missing}")
        return 2
    unknown = [c for c in args.configs if c not in CONFIGS_V2]
    if unknown:
        print(f"REFUSED: unknown configuration(s) {unknown}")
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    from pg.model_api import CanonicalMessage
    from pg.week7_models import build_llm
    from pg.week8_protocol import ProtocolViolation
    from scripts.week11_lib import REVIEW_LIMITS
    from scripts.week13_xreviewer import load_session

    try:
        if args.split == "test":
            check_registration(pgx_digest(ROOT)[0])
        agent = load_session(args.agent, args.agent_endpoint)
        judge = agent if args.reviewer == args.agent else load_session(args.reviewer, args.reviewer_endpoint)
        atk = agent if args.attacker == args.agent else load_session(args.attacker, args.attacker_endpoint)
    except (ProtocolViolation, KeyError, Refused) as exc:
        print(f"REFUSED: {exc}")
        return 2
    targets = [t for t in json.loads(TARGETS.read_text()) if t["split"] == args.split][: args.max_targets]
    tasks = json.loads((DATA / "agentdojo_tasks.json").read_text())
    cases = {c.case_id: c for c in load_cases([args.split], sorted({t["suite"] for t in targets}), benign=False)}
    attacker_client = build_llm(atk["profile"], seed=42, limits=REVIEW_LIMITS).model_client

    def attacker(prompt: str) -> str:
        return attacker_client.complete([CanonicalMessage(role="user", content=prompt)], ()).content or ""

    root = args.results_root / f"{args.agent}__{'self' if args.reviewer == args.agent else args.reviewer}__atk-{args.attacker}"
    for config in args.configs:
        if CONFIGS_V2[config]["kind"] == "v1":
            mine = [t for t in targets if t["suite"] in CONFIGS_V2[config]["suites"]]
        else:
            mine = targets
        out = root / config / args.feedback
        out.mkdir(parents=True, exist_ok=True)
        path = out / "episodes.jsonl"
        done = {json.loads(l)["case_id"] for l in path.read_text().splitlines() if l.strip()} if path.exists() else set()
        run = run_for(config, llm_factory=lambda: build_llm(agent["profile"], seed=42, limits=REVIEW_LIMITS),
                      agent_client=build_llm(agent["profile"], seed=42, limits=REVIEW_LIMITS).model_client,
                      reviewer_client=(build_llm(judge["profile"], seed=42, limits=REVIEW_LIMITS).model_client
                                       if needs_reviewer_model(config) else None),
                      evidence_root=out / "evidence")
        for i, t in enumerate(mine, 1):
            case_id = f"{t['suite']}/{t['user_task_id']}/{t['injection_task_id']}"
            if case_id in done:
                continue
            meta = tasks["suites"][t["suite"]]
            rows = attack_target(cases[case_id], config, goal=meta["injection_tasks"][t["injection_task_id"]]["goal"],
                                 user_task=meta["user_tasks"][t["user_task_id"]]["prompt"], attacker=attacker, run=run,
                                 budget=args.budget, feedback=args.feedback)
            with path.open("a", encoding="utf-8") as handle:
                for r in rows:
                    r.update({"agent": args.agent, "reviewer": args.reviewer, "attacker": args.attacker,
                              "pgx_digest": pgx_digest(ROOT)[0], "split": args.split})
                    handle.write(json.dumps(r, default=str) + "\n")
            broke = next((r["attempt"] for r in rows if r.get("attack_success")), None)
            print(f"[{i}/{len(mine)}] {config} {case_id}: " + (f"broken at attempt {broke}" if broke else f"held for {len(rows)} attempts"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
