"""Run the v2 evaluation (standard AgentDojo, four suites) for ONE agent model and ONE reviewer model.  Resume-safe.

    python scripts/run_v2_agentdojo.py --agent qwen3-14b --agent-endpoint http://127.0.0.1:P1 \
        --reviewer gemma4-e4b --reviewer-endpoint http://127.0.0.1:P2 --group core --split dev --check-only
    python scripts/run_v2_agentdojo.py ... --group core --group baselines --split dev            # development runs
    python scripts/run_v2_agentdojo.py ... --group core --split test                            # needs the registration
    python scripts/run_v2_agentdojo.py ... --split test --variance                               # + seeds 314, 2718 on the subset

Cases: every benign user task and every (user task, injection task) pair of the chosen split, in a fixed order; seed 42
everywhere (``--variance`` adds seeds 314 and 2718 on the frozen variance subset of test user tasks).  Rows go to
``<results-root>/<agent>__<reviewer|self>/<config>/episodes.jsonl``; a rerun skips (case, seed) pairs already saved and refuses a
results directory written by another plan, code version or model.

The test split is touched only after ``scripts/v2_register.py`` has recorded the preregistration and the v2 code digest; this
runner refuses a test-split run if the code or the preregistration changed since.  Exit codes: 0 done, 2 refused, 4 stopped by
the consecutive-error breaker, 143 terminated (rerun to resume).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgx.agentdojo_runner import Case, run_case  # noqa: E402
from pgx.configs import CONFIGS_V2, needs_reviewer_model, resolve  # noqa: E402
from pgx.suites import SUITES  # noqa: E402
from pgx.version import pgx_digest  # noqa: E402

DATA = ROOT / "data" / "v2"
RESULTS_ROOT = ROOT / "results" / "v2"
REGISTRATION = DATA / "registration.json"
PREREG = ROOT / "docs" / "PREREGISTRATION_V2.md"
IMMUTABLE = ("plan_sha256", "pgx_digest", "agent", "agent_model_digest", "reviewer", "reviewer_model_digest", "config")


class Refused(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(splits: list[str], suites: list[str], *, benign: bool = True, security: bool = True) -> list[Case]:
    tasks = json.loads((DATA / "agentdojo_tasks.json").read_text())
    split = json.loads((DATA / "split.json").read_text())
    rows = [json.loads(line) for line in (DATA / "injections.jsonl").read_text().splitlines() if line.strip()]
    out: list[Case] = []
    for suite in suites:
        meta = tasks["suites"][suite]
        part = {u: name for name in ("dev", "test") for u in split["suites"][suite][name]}
        users = sorted(meta["user_tasks"], key=lambda u: int(u.split("_")[-1]))
        for uid in users:
            if part[uid] not in splits:
                continue
            under = bool(meta["user_tasks"][uid]["under_specified"])
            if benign:
                out.append(Case(suite, uid, None, {}, part[uid], None, under))
            if security:
                for r in rows:
                    if r["suite"] == suite and r["user_task_id"] == uid:
                        goal_kind = meta["injection_tasks"][r["injection_task_id"]]["goal_kind"]
                        out.append(Case(suite, uid, r["injection_task_id"], r["injections"], part[uid], goal_kind == "write", under))
    return out


def variance_cases(cases: list[Case]) -> set[str]:
    split = json.loads((DATA / "split.json").read_text())
    subset = {(s, u) for s, v in split["suites"].items() for u in v["variance_subset"]}
    return {c.case_id for c in cases if (c.suite, c.user_task_id) in subset}


def check_registration(digest: str) -> dict[str, Any]:
    if not REGISTRATION.exists():
        raise Refused("the test split needs data/v2/registration.json: commit the preregistration, then run "
                      "scripts/v2_register.py (development runs use --split dev)")
    reg = json.loads(REGISTRATION.read_text())
    if reg.get("pgx_digest") != digest:
        raise Refused(f"the v2 code changed after registration (registered {reg.get('pgx_digest', '')[:12]}, now {digest[:12]}); "
                      "test-split results must come from the registered code")
    if not PREREG.exists() or reg.get("preregistration_sha256") != sha(PREREG):
        raise Refused("docs/PREREGISTRATION_V2.md changed after registration")
    return reg


def ensure_manifest(path: Path, candidate: dict[str, Any]) -> None:
    if path.exists():
        old = json.loads(path.read_text())
        diff = [k for k in IMMUTABLE if old.get(k) != candidate.get(k)]
        if diff:
            raise Refused(f"{path.parent} was written with different {diff}; use another --results-root")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def run(*, agent: str, reviewer: str, configs: list[str], splits: list[str], suites: list[str], variance: bool,
        results_root: Path, agent_factory: Callable[[Case, int], Any], agent_client_factory: Callable[[int], Any],
        reviewer_client_factory: Callable[[int], Any], identities: dict[str, Any], max_episodes: int | None = None,
        max_consecutive_errors: int = 8, smoke: int | None = None, log: Callable[[str], None] = print) -> int:
    digest, _ = pgx_digest(ROOT)
    plan_sha = sha(DATA / "plan.json")
    registration = check_registration(digest) if "test" in splits else None
    cases = load_cases(splits, suites)
    if smoke:
        picked: list[Case] = []
        for suite in suites:
            mine = [c for c in cases if c.suite == suite]
            picked += [c for c in mine if c.injection_task_id is None][:1] + [c for c in mine if c.injection_task_id][:smoke - 1]
        cases = picked
    vset = variance_cases(cases) if variance else set()
    pair_dir = results_root / f"{agent}__{'self' if reviewer == agent else reviewer}"
    total_errors = 0
    for config in configs:
        spec = CONFIGS_V2[config]
        mine = [c for c in cases if spec["kind"] != "v1" or c.suite in spec.get("suites", [])]
        out_dir = pair_dir / config
        ensure_manifest(out_dir / "run_manifest.json", {
            "schema": "pgx-v2-run-manifest-v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "plan_sha256": plan_sha, "pgx_digest": digest, "config": config, "config_spec": spec, "agent": agent,
            "agent_model_digest": identities.get("agent_model_digest"), "reviewer": reviewer,
            "reviewer_model_digest": identities.get("reviewer_model_digest"), "splits": splits,
            "registration": registration})
        path = out_dir / "episodes.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done.add((r["case_id"], r["seed"]))
        work = [(c, 42) for c in mine] + [(c, s) for s in (314, 2718) for c in mine if c.case_id in vset]
        work = [(c, s) for c, s in work if (c.case_id, s) not in done]
        if max_episodes is not None:
            work = work[:max_episodes]
        log(f"{agent}/{config}: {len(done)} saved, {len(work)} to run")
        streak = 0
        with path.open("a", encoding="utf-8", buffering=1) as handle:
            for i, (case, seed) in enumerate(work, 1):
                started = time.time()
                try:
                    row = run_case(case, config, llm=agent_factory(case, seed), agent_client=agent_client_factory(seed),
                                   reviewer_client=reviewer_client_factory(seed) if needs_reviewer_model(config) else None,
                                   seed=seed, evidence_root=out_dir / "evidence")
                except Exception as exc:  # noqa: BLE001 -- a harness failure is a recorded row, never a dropped episode
                    row = {"schema": "pgx-agentdojo-episode-v1", "case_id": case.case_id, "suite": case.suite,
                           "user_task_id": case.user_task_id, "injection_task_id": case.injection_task_id, "split": case.split,
                           "in_scope": case.in_scope, "under_specified": case.under_specified, "config": config, "seed": seed,
                           "utility": None, "attack_success": None, "error": f"harness {type(exc).__name__}: {exc}"}
                row.update({"agent": agent, "reviewer": reviewer, "pgx_digest": digest, "plan_sha256": plan_sha,
                            "wall_seconds": round(time.time() - started, 2), **{k: v for k, v in identities.items()}})
                handle.write(json.dumps(row, default=str) + "\n")
                streak = streak + 1 if row.get("error") else 0
                total_errors += bool(row.get("error"))
                log(f"[{i}/{len(work)}] {config} {case.case_id} seed={seed} utility={row.get('utility')} "
                    f"attack={row.get('attack_success')}{' ERROR ' + str(row.get('error'))[:80] if row.get('error') else ''}")
                if streak >= max_consecutive_errors:
                    log(f"STOPPED: {streak} consecutive errors; fix the cause and rerun the same command to resume")
                    return 4
    log(f"done ({total_errors} errored episodes recorded)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--agent-endpoint", default=os.environ.get("AGENT_ENDPOINT"))
    ap.add_argument("--reviewer", required=True, help="slug of the independent reviewer model (the same slug = self-review)")
    ap.add_argument("--reviewer-endpoint", default=os.environ.get("REVIEWER_ENDPOINT"))
    ap.add_argument("--config", action="append", dest="configs")
    ap.add_argument("--group", action="append", dest="groups", help="core | baselines | bridge | ablations | frontier")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--suite", action="append", dest="suites", choices=SUITES)
    ap.add_argument("--variance", action="store_true", help="also seeds 314 and 2718 on the variance subset")
    ap.add_argument("--smoke", type=int, help="only this many cases per suite (1 benign + the rest attacks)")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--max-consecutive-errors", type=int, default=8)
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)
    try:
        configs = resolve(args.configs, args.groups or (["core"] if not args.configs else None))
    except KeyError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if not args.agent_endpoint or not args.reviewer_endpoint:
        print("REFUSED: --agent-endpoint and --reviewer-endpoint (or AGENT_ENDPOINT / REVIEWER_ENDPOINT) are required")
        return 2
    splits = ["dev", "test"] if args.split == "all" else [args.split]
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from pg.week7_models import build_llm
    from pg.week8_protocol import ProtocolViolation
    from scripts.week11_lib import REVIEW_LIMITS
    from scripts.week13_xreviewer import load_session

    try:
        agent = load_session(args.agent, args.agent_endpoint)
        judge = agent if args.reviewer == args.agent else load_session(args.reviewer, args.reviewer_endpoint)
        if "test" in splits:
            check_registration(pgx_digest(ROOT)[0])
    except (ProtocolViolation, KeyError, Refused) as exc:
        print(f"REFUSED: {exc}")
        return 2
    identities = {"agent_model_digest": agent["identity"]["model_digest"], "reviewer_model_digest": judge["identity"]["model_digest"],
                  "agent_profile_sha256": agent["identity"]["portable_profile_sha256"]}
    print(f"agent {args.agent} ({identities['agent_model_digest'][:12]}), reviewer {args.reviewer} "
          f"({identities['reviewer_model_digest'][:12]}); configs {configs}; split {splits}; v2 code {pgx_digest(ROOT)[0][:12]}")
    if args.check_only:
        n = len(load_cases(splits, args.suites or list(SUITES)))
        print(f"check-only: identities verified; {n} cases per config at seed 42")
        return 0
    try:
        return run(agent=args.agent, reviewer=args.reviewer, configs=configs, splits=splits, suites=args.suites or list(SUITES),
                   variance=args.variance, results_root=args.results_root,
                   agent_factory=lambda case, seed: build_llm(agent["profile"], seed=seed, limits=REVIEW_LIMITS),
                   agent_client_factory=lambda seed: build_llm(agent["profile"], seed=seed, limits=REVIEW_LIMITS).model_client,
                   reviewer_client_factory=lambda seed: build_llm(judge["profile"], seed=seed, limits=REVIEW_LIMITS).model_client,
                   identities=identities, max_episodes=args.max_episodes, max_consecutive_errors=args.max_consecutive_errors,
                   smoke=args.smoke)
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
