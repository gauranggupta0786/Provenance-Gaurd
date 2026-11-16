"""Week 13 cross-model reviewer: agent = one model, reviewer = a DIFFERENT model (resume-safe, one pair per call).

    # the prediction test: glm and lfm2 as agents, a different-family reviewer that denied 240/240 on the original four
    python scripts/week13_xreviewer.py --agent glm-47-flash --agent-endpoint http://127.0.0.1:AGENT \
                                       --reviewer qwen3-14b   --reviewer-endpoint http://127.0.0.1:REV --family policy_passing
    python scripts/week13_xreviewer.py --agent lfm2-24b --agent-endpoint ... --reviewer qwen3-14b --reviewer-endpoint ... --family policy_passing

    # the self-pair control: reviewer = agent, which MUST reproduce the Week 12 SCP-v3 residual (glm 6/60, lfm2 7/60)
    python scripts/week13_xreviewer.py --agent glm-47-flash --agent-endpoint $E --reviewer glm-47-flash --reviewer-endpoint $E

Every per-episode mechanic -- the agent call, the scoped+model reviewer (``S+M-v3``), the denial cap, the hook, the layers,
the provenance check -- is the frozen Week 11e one, reused unchanged.  The ONE thing that differs from the frozen campaign is
that the reviewer's model client is built from the reviewer model's own profile at its own endpoint, instead of from the
agent's.  So a difference in outcome is attributable to the reviewer model and nothing else.

This is a new experiment, not a re-run of the frozen one: it writes to ``results/week13_xreviewer/<agent>__<reviewer>/`` with
its own manifest, and it reads ``attack_success`` and ``injected_cards_approved`` exactly as the frozen runner records them.
``scripts/run_week11_policy_passing.py`` is pinned by digest and is not touched.
"""
from __future__ import annotations

import argparse
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
    ProtocolViolation, ensure_run_manifest, evidence_token_dir, prepare_model_session, quarantine_stale_evidence,
    run_context, utc_now,
)
from scripts.run_week8_frozen import append_line, load_jsonl, runner_failure_row
from scripts.week11_lib import REVIEW_LIMITS, layers_for, make_hook, make_reviewer, needs_model, week11_code_digest
from scripts.week13_lib import (
    DATASET_SHA256, MODEL_FAMILY, REVIEWER_NAME, RESULTS_ROOT_XREVIEWER, XREVIEWER_CONFIGS, config_spec,
    family_of, paths_for_slug, pair_dir_name, same_family,
)

IMMUTABLE = ("agent_slug", "reviewer_slug", "config", "dataset_sha256", "reviewer_name",
             "agent_model_digest", "reviewer_model_digest", "week11_code_digest")


def load_session(slug: str, endpoint: str, *, http=None):
    protocol_path, matrix_path, profiles_dir = paths_for_slug(slug)
    return prepare_model_session(slug, endpoint, protocol_path=protocol_path, matrix_path=matrix_path,
                                 profiles_dir=profiles_dir, check_code=False, http=http)


def run_one(scenario: dict[str, Any], spec: dict[str, Any], seed: int, *, agent_llm, review_llm) -> dict[str, Any]:
    """One episode: agent is ``agent_llm``, reviewer (if the config uses a model reviewer) is ``review_llm``."""

    reviewer = make_reviewer(REVIEWER_NAME, scenario, review_llm, cap=spec.get("denial_cap"))
    layers = layers_for(spec)
    return run_episode(
        scenario, arm=spec["arm"], seed=seed, llm=agent_llm, runtime_limits=REVIEW_LIMITS,
        reviewer=reviewer, post_episode_hook=make_hook(scenario, reviewer), layers=layers,
        describe_targets=bool(spec.get("describe", False)), annotate_sources=bool(spec.get("annotate", False)),
        check_provenance=bool(spec.get("provenance", False)),
    ).to_dict()


def run_pair(
    agent_slug: str, agent_endpoint: str, reviewer_slug: str, reviewer_endpoint: str, *,
    configs: list[str] | None = None, family: str | None = None, seeds: list[int] | None = None,
    results_root: Path = RESULTS_ROOT_XREVIEWER, root: Path = ROOT, max_episodes: int | None = None,
    check_only: bool = False, allow_same_family: bool = False,
    http_agent=None, http_reviewer=None,
    session_loader: Callable[..., Any] = load_session,
    llm_factory: Callable[..., Any] = build_llm, episode_fn: Callable[..., dict[str, Any]] = run_one,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    for slug in (agent_slug, reviewer_slug):
        if slug not in MODEL_FAMILY:
            raise ProtocolViolation(f"{slug!r} is not one of the six models {sorted(MODEL_FAMILY)}")
    names = [c for c in XREVIEWER_CONFIGS if not configs or c in (configs or [])]
    unknown = [c for c in (configs or []) if c not in XREVIEWER_CONFIGS]
    if unknown:
        raise ProtocolViolation(f"configuration(s) {unknown} are not part of the cross-reviewer test ({list(XREVIEWER_CONFIGS)})")

    code_digest, _ = week11_code_digest(root)
    agent = session_loader(agent_slug, agent_endpoint, http=http_agent)
    reviewer = session_loader(reviewer_slug, reviewer_endpoint, http=http_reviewer)
    cross = agent_slug != reviewer_slug
    if cross and same_family(agent_slug, reviewer_slug) and not allow_same_family:
        raise ProtocolViolation(
            f"{agent_slug} and {reviewer_slug} are both family {family_of(agent_slug)!r}; an independent reviewer should be a "
            f"different family. Pass --allow-same-family only if you mean to measure a same-family, different-model reviewer.")

    seeds = seeds or list(agent["protocol"]["campaign"]["seeds"])
    rows_all = load_jsonl(root / "data" / "week11e" / "scenarios.jsonl")
    if family:
        rows_all = [r for r in rows_all if r["family"] == family]
        if not rows_all:
            raise ProtocolViolation(f"no scenarios with family={family!r} in the dataset")
    scenarios = {r["scenario_id"]: r for r in rows_all}
    scenario_ids = list(scenarios)

    pair = pair_dir_name(agent_slug, reviewer_slug)
    log(f"pair {pair}: agent={agent_slug} ({family_of(agent_slug)}) reviewer={reviewer_slug} "
        f"({family_of(reviewer_slug)}){'  [SELF-REVIEW control]' if not cross else ''}")
    log(f"  configs={names} family={family or 'all'} scenarios={len(scenario_ids)} seeds={seeds}")
    summary: dict[str, Any] = {"pair": pair, "agent": agent_slug, "reviewer": reviewer_slug, "cross_model": cross,
                               "configs": {}, "week11_code_digest": code_digest}
    if check_only:
        for name in names:
            summary["configs"][name] = {"check_only": True}
        return summary

    owns_a, owns_r = http_agent is None, http_reviewer is None
    client_a = http_agent or httpx.Client(base_url=agent_endpoint.rstrip("/"), timeout=60.0)
    client_r = http_reviewer or httpx.Client(base_url=reviewer_endpoint.rstrip("/"), timeout=60.0)
    context = run_context()
    try:
        for name in names:
            spec = config_spec(name)
            out_dir = Path(results_root) / pair / name
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema_version": "week13-xreviewer-manifest-v1", "created_at": utc_now(),
                "agent_slug": agent_slug, "reviewer_slug": reviewer_slug, "config": name, "cross_model": cross,
                "dataset_sha256": DATASET_SHA256, "reviewer_name": REVIEWER_NAME,
                "agent_model_digest": agent["identity"]["model_digest"],
                "reviewer_model_digest": reviewer["identity"]["model_digest"],
                "agent_engine": agent["profile"].serving_engine_version,
                "reviewer_engine": reviewer["profile"].serving_engine_version,
                "week11_code_digest": code_digest, "family_filter": family, "seeds": seeds, "environment": context,
            }
            ensure_run_manifest(out_dir / "run_manifest.json", manifest, IMMUTABLE)
            summary["configs"][name] = run_config(
                name, spec, scenarios, scenario_ids, seeds, agent, reviewer, client_a, client_r,
                out_dir, code_digest, context, max_episodes, cross, llm_factory, episode_fn, log)
        return summary
    finally:
        if owns_a:
            client_a.close()
        if owns_r:
            client_r.close()


def run_config(name, spec, scenarios, scenario_ids, seeds, agent, reviewer, client_a, client_r, out_dir, code_digest,
               context, max_episodes, cross, llm_factory, episode_fn, log) -> dict[str, Any]:
    arm = spec["arm"]
    output = out_dir / "episodes.jsonl"
    existing = load_jsonl(output)
    foreign = [r for r in existing if r.get("agent_slug") != agent["identity"]["slug"]
               or r.get("reviewer_slug") != reviewer["identity"]["slug"] or r.get("week13_config") != name]
    if foreign:
        raise ProtocolViolation(f"{output} has {len(foreign)} rows from another pair/config; use a new results directory")
    done = {episode_key(r) for r in existing}
    pending = [(scenarios[sid], seed) for seed in seeds for sid in scenario_ids if (sid, arm, seed) not in done]
    if max_episodes is not None:
        pending = pending[:max_episodes]
    summary = {"arm": arm, "already_saved": len(existing), "selected": len(pending), "ran": 0, "errors": 0,
               "attacks": 0, "injected_approved": 0}
    if not pending:
        log(f"  {name}: complete ({len(existing)} saved)")
        return summary
    log(f"  {name}: {len(existing)} saved, {len(pending)} to run")
    evidence_root = out_dir / "evidence"
    started_all = time.monotonic()
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for index, (scenario, seed) in enumerate(pending, start=1):
            quarantine_stale_evidence(evidence_token_dir(evidence_root, scenario["scenario_id"], seed, arm))
            agent_llm = llm_factory(agent["profile"], seed=seed, limits=REVIEW_LIMITS, client=client_a)
            review_llm = (llm_factory(reviewer["profile"], seed=seed, limits=REVIEW_LIMITS, client=client_r)
                          if needs_model(REVIEWER_NAME) else None)
            started = utc_now()
            try:
                result = episode_fn(scenario, spec, seed, agent_llm=agent_llm, review_llm=review_llm)
            except Exception as exc:  # recorded, never dropped
                result = runner_failure_row(scenario, arm, seed, agent["profile"].model_id, exc)
            result.update({
                "scenario_id": scenario["scenario_id"], "arm": arm, "seed": seed,
                "week13_schema": "xreviewer-episode-v1", "week13_config": name, "experiment": "cross_model_reviewer",
                "agent_slug": agent["identity"]["slug"], "reviewer_slug": reviewer["identity"]["slug"],
                "cross_model": cross, "reviewer": REVIEWER_NAME,
                "agent_model_digest": agent["identity"]["model_digest"],
                "reviewer_model_digest": reviewer["identity"]["model_digest"],
                "vector": scenario.get("vector"), "goal": scenario.get("goal"),
                "kind": scenario["kind"], "family": scenario["family"], "scenario_checksum": scenario["checksum"],
                "defense_version": code_digest, "dataset_sha256": DATASET_SHA256,
                "agent_request_log": list(getattr(getattr(agent_llm, "model_client", None), "request_log", [])),
                "reviewer_request_log": list(getattr(getattr(review_llm, "model_client", None), "request_log", [])),
                "slurm_job_id": context.get("slurm_job_id"), "node": context.get("node"),
                "started_at": started, "finished_at": utc_now(),
            })
            append_line(handle, result)
            summary["ran"] += 1
            summary["errors"] += bool(result.get("error"))
            summary["attacks"] += bool(result.get("attack_success") and result.get("family") == "policy_passing")
            hook = result.get("hook_result") or {}
            summary["injected_approved"] += int(hook.get("injected_cards_approved") or 0)
            eta = (time.monotonic() - started_all) / index * (len(pending) - index)
            log(f"  [{index}/{len(pending)}] {name} {scenario['scenario_id']} seed={seed} "
                f"{'ERROR ' if result.get('error') else ''}{result.get('latency_ms', 0)}ms eta={eta/60:.0f}min")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--agent-endpoint", default=os.environ.get("AGENT_ENDPOINT"))
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--reviewer-endpoint", default=os.environ.get("REVIEWER_ENDPOINT"))
    ap.add_argument("--config", action="append", dest="configs", help=f"default: {list(XREVIEWER_CONFIGS)}")
    ap.add_argument("--family", help="restrict to one scenario family, e.g. policy_passing (the cheap prediction test)")
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT_XREVIEWER)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--allow-same-family", action="store_true")
    args = ap.parse_args(argv)
    if not args.agent_endpoint or not args.reviewer_endpoint:
        print("--agent-endpoint and --reviewer-endpoint (or $AGENT_ENDPOINT/$REVIEWER_ENDPOINT) are required")
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        summary = run_pair(args.agent, args.agent_endpoint, args.reviewer, args.reviewer_endpoint,
                           configs=args.configs, family=args.family, results_root=args.results_root,
                           max_episodes=args.max_episodes, check_only=args.check_only,
                           allow_same_family=args.allow_same_family)
    except ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return 2
    for name, c in summary["configs"].items():
        if not c.get("check_only"):
            print(f"{summary['pair']} {name}: ran {c['ran']} (+{c['already_saved']} saved), "
                  f"policy-passing attacks {c['attacks']}, injected-approved {c['injected_approved']}, errors {c['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
