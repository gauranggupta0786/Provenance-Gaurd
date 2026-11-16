"""Run the ProvenanceGuard experiments against a hosted OpenAI-compatible model: probe, then campaign, in one command.

    export PROVENANCEGUARD_API_BASE=https://api.example.com/v1
    export PROVENANCEGUARD_API_KEY=sk-...
    export PROVENANCEGUARD_MODEL=the-model-name

    python scripts/run_frontier_campaign.py --list                   # every experiment, what it reproduces, how big it is
    python scripts/run_frontier_campaign.py --dry-run                # what would run, no network, no key needed
    python scripts/run_frontier_campaign.py --smoke                  # one scenario per family, first seed: the cheap end-to-end check
    python scripts/run_frontier_campaign.py                          # the 72-scenario held-out ladder: 648 episodes
    python scripts/run_frontier_campaign.py --suite baselines        # the four cheap defenses the report compares against
    python scripts/run_frontier_campaign.py --suite main --max-cost 50    # the report's six-model body (held-out, baselines, evasion)
    python scripts/run_frontier_campaign.py --suite pdf --smoke      # a taste of every experiment in the report

An experiment is a *suite* (``--list``); ``main``, ``appendix`` and ``pdf`` name groups of them.  ``--dataset heldout`` is the
original spelling of ``--suite heldout`` and still works.

It first runs the capability probe (reusing a passing report for the same endpoint and model), refuses to start if a required
probe fails, then runs each suite's frozen configurations over the selected scenarios and seeds.  Every finished episode is
saved as it completes, so an interrupted or stopped run is resumed by running the same command again.

Exit codes: 0 finished, 1 finished but more than 2% of episodes errored (below the >=98%-evaluable gate: read the errors
before reporting), 2 refused before spending anything (configuration, probe, frozen-code or dataset check), 3 stopped at
--max-cost, 4 stopped after repeated identical failures.  Errors inside individual episodes are results, not reasons to
stop: they are recorded and stay in every denominator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier.campaign import (
    RESULTS_ROOT, CampaignStop, RunSettings, build_plan, planned_episodes, reviewer_config, run_campaign, select,
)
from frontier.frontier_probes import run_probes
from frontier.openai_adapter import RemoteConfig, redact
from frontier.suites import GROUPS, LEGACY_DATASET_ALIASES, SUITES, resolve
from pg.week8_protocol import ProtocolViolation, code_digest

#: the guard code the six local models were measured under (``pg/*.py`` plus the files ``code_digest`` pins).  A tree whose
#: digest differs is running a different defense, so its results are not comparable and the runner refuses it.
MEASURED_GUARD_DIGEST = "dca1a798694935c90886af0f2a535b24251de7e6a5ceef5448f2442ae1e0648b"

MAX_ERROR_FRACTION = 0.02       # the project's >=98% evaluable gate


def probe_path(results_root: Path, config: RemoteConfig) -> Path:
    return Path(results_root) / config.model.replace("/", "_") / "probe_report.json"


def ensure_probe(config: RemoteConfig, results_root: Path, log=print) -> dict:
    """A passing probe report for this endpoint and model: reuse the saved one, or run the probes now."""

    path = probe_path(results_root, config)
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("endpoint") == config.endpoint and report.get("model") == config.model and report.get("usable"):
            log(f"probe: reusing {path} (usable)")
            return report
    log(f"probe: running the capability probes for {config.model} (a handful of calls)")
    report = run_probes(config, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact(json.dumps(report, indent=2, sort_keys=True), config.api_key) + "\n", encoding="utf-8")
    if report["required_failed"]:
        raise ProtocolViolation(
            f"{config.model} failed required probe(s) {report['required_failed']}; see {path}. A model that cannot be measured "
            f"cannot be reported -- fix the configuration (docs/FRONTIER_HANDOFF.md, 'Provider differences') or choose another model")
    if report["advisory_failed"]:
        log(f"probe: advisory probes that did not pass (recorded, they do not exclude): {report['advisory_failed']}")
    return report


def list_suites() -> int:
    """Every experiment of the report, with its size for ONE model and where the report uses it."""

    print(f"{'suite':17s} {'kind':9s} {'configs':>7s} {'work':>14s}  what it reproduces")
    for key, suite in SUITES.items():
        try:
            plan = build_plan(key)
            work = f"{plan['episodes_per_config'] * len(plan['config_order']):,} {suite.unit}"
            if suite.kind == "attacker":
                work = f"{len(plan['scenario_ids'])} obj x<={plan['max_attempts']}"
        except ProtocolViolation as exc:
            work = f"? ({str(exc)[:30]})"
        print(f"{key:17s} {suite.kind:9s} {len(suite.config_order):>7d} {work:>14s}  {suite.reports}")
        print(f"{'':17s} {suite.title}")
    print("\ngroups:  " + "   ".join(f"{name} = {' + '.join(keys)}" for name, keys in GROUPS.items()))
    print("\nEach row is one model.  A configuration is one column of the report's tables (for example C0, SC-v3, SCP-v3).")
    print("Cheapest useful first run:  --suite pdf --smoke   (one scenario per family, first seed, every experiment)")
    return 0


def estimate(selections) -> dict[str, int]:
    """Episodes, attacker objectives, model calls and a low/high agent-token planning range, summed over the chosen suites."""

    total = {"episodes": 0, "objectives": 0, "calls": 0, "low": 0, "high": 0}
    for key, selection in selections.items():
        suite = SUITES[key]
        n = planned_episodes(selection)
        total[suite.unit] += n
        total["calls"] += n * suite.calls_per_unit
        total["low"] += n * suite.tokens_per_unit[0]
        total["high"] += n * suite.tokens_per_unit[1]
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", nargs="+", metavar="SUITE",
                    help=f"what to run: any of {', '.join(SUITES)}, or a group ({', '.join(GROUPS)}) (default: heldout)")
    ap.add_argument("--dataset", choices=(*SUITES, *LEGACY_DATASET_ALIASES),
                    help="the original spelling of --suite for the ladder (heldout, tuning, adaptive_evasion; both = tuning + heldout)")
    ap.add_argument("--list", action="store_true", help="list every suite with its size and what it reproduces, then exit")
    ap.add_argument("--configs", nargs="+", metavar="CONFIG",
                    help="a subset of ONE suite's configurations (default: all of them); e.g. C0 SC-v3 SCP-v3, or none tool_allowlist")
    ap.add_argument("--smoke", action="store_true", help="one scenario per family, first seed only: the cheap end-to-end check")
    ap.add_argument("--family", action="append", dest="families", help="only this scenario family (repeatable; one suite)")
    ap.add_argument("--scenario-id", action="append", dest="scenario_ids", help="only this scenario (repeatable; one suite)")
    ap.add_argument("--seeds", nargs="+", type=int, help="a subset of the frozen seeds (default: all; --smoke: the first)")
    ap.add_argument("--max-cost", type=float,
                    help="stop cleanly once the metered cost of this whole invocation reaches this (needs the price variables)")
    ap.add_argument("--max-consecutive-errors", type=int, default=5,
                    help="stop after this many episodes in a row error (0 disables; default 5)")
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--label", help="a suffix for the results directory, to keep two runs of one model apart")
    ap.add_argument("--skip-probe", action="store_true", help="do not run or require the capability probe")
    ap.add_argument("--allow-modified-code", action="store_true",
                    help="run even though the guard code differs from what was measured (results are then NOT comparable)")
    ap.add_argument("--dry-run", action="store_true", help="print what would run and stop; needs no network and no key")
    args = ap.parse_args(argv)

    if args.list:
        return list_suites()
    if args.suite and args.dataset:
        print("REFUSED: give --suite or --dataset, not both")
        return 2
    try:
        suites = resolve(args.suite or [args.dataset or "heldout"])
        if len(suites) > 1 and (args.configs or args.families or args.scenario_ids):
            raise ProtocolViolation(
                "--configs, --family and --scenario-id name things inside one suite; with several suites they are ambiguous. "
                "Run the suite you mean on its own (see --list for each suite's configurations).")
        selection_args = {"smoke": args.smoke, "families": args.families, "scenario_ids": args.scenario_ids,
                          "seeds": args.seeds, "configs": args.configs}
        plans = {k: build_plan(k) for k in suites}
        selections = {k: select(plans[k], **selection_args) for k in suites}
        digest, _ = code_digest(ROOT)
        if digest != MEASURED_GUARD_DIGEST and not args.allow_modified_code:
            raise ProtocolViolation(
                f"the guard code hashes to {digest[:16]}..., not the measured {MEASURED_GUARD_DIGEST[:16]}...: this tree is "
                f"running a different defense from the one the local results describe.  Run  python scripts/handoff_verify.py  "
                f"to see which file differs.  (--allow-modified-code overrides, and the results are then not comparable.)")
    except (ProtocolViolation, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return 2

    total = estimate(selections)
    print("plan: " + ", ".join(f"{k} {planned_episodes(selections[k])} {SUITES[k].unit} (plan {plans[k]['plan_sha256'][:12]})"
                                for k in suites))
    work = f"{total['episodes']:,} episodes" + (f" + {total['objectives']} attacker objectives (up to "
                                                f"{total['objectives'] * 15} attempts)" if total["objectives"] else "")
    print(f"      {work}, roughly {total['calls']:,} model calls and {total['low'] / 1_000_000:.1f}-{total['high'] / 1_000_000:.1f}M "
          f"agent tokens (planning figure from the local runs; the attacker is bounded by construction; probe prompts are shorter)")
    if args.dry_run:
        for k in suites:
            sel = selections[k]
            print(f"  {k}: {len(sel['scenario_ids'])} {'objectives' if SUITES[k].unit == 'objectives' else 'scenarios'} "
                  f"x seeds {sel['seeds']} x {sel['configs']}")
        print("dry run: nothing was sent.")
        return 0

    try:
        agent = RemoteConfig.from_env()
        reviewer = reviewer_config(agent)
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if not any(SUITES[k].model_reviewer for k in suites):
        reviewer = agent        # no selected suite asks a model to review, so a different reviewer model would change nothing
    if args.max_cost is not None and not (agent.price_per_1k_input or agent.price_per_1k_output):
        print("REFUSED: --max-cost needs PROVENANCEGUARD_PRICE_PER_1K_INPUT / _OUTPUT, or there is no cost to meter")
        return 2
    print(f"agent    {agent.model} @ {agent.endpoint}\nreviewer {reviewer.model} @ {reviewer.endpoint}"
          f"{'  (same model: self-review, as in every local run)' if reviewer is agent else '  (a DIFFERENT model: cross-model reviewer)'}")

    probes: dict = {}
    try:
        if not args.skip_probe:
            probes[agent.model] = ensure_probe(agent, args.results_root)
            if reviewer is not agent:
                probes[reviewer.model] = ensure_probe(reviewer, args.results_root)
    except ProtocolViolation as exc:
        print(f"REFUSED: {exc}")
        return 2

    settings = RunSettings(max_cost=args.max_cost, max_consecutive_errors=args.max_consecutive_errors)
    summaries = []
    try:
        for k in suites:
            summaries.append(run_campaign(
                suite_key=k, agent=agent, reviewer=reviewer, selection_args=selection_args, results_root=args.results_root,
                settings=settings, label=args.label, probes=probes, log=lambda m: print(m, flush=True)))
    except CampaignStop as stop:
        print(f"\n{stop}")
        usage = getattr(stop, "usage", None)
        if usage:
            print(f"spent so far: {usage['spent_total']} {usage['currency']}")
        return stop.code
    except ProtocolViolation as exc:
        print(f"REFUSED: {exc}")
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted: finished episodes are saved; run the same command again to resume.")
        return 130

    print()
    errored = ran_total = 0
    for s in summaries:
        ran = sum(c["ran"] for c in s["configs"])
        errs = sum(c["errors"] for c in s["configs"])
        errored += errs
        ran_total += ran
        print(f"{s['suite']}: ran {ran} {s['unit']} ({errs} errored) -> {s['results_dir']}   spent {s['usage']['spent']} {s['usage']['currency']}")
    print("\nnext:  python scripts/frontier_report.py --results-root " + str(args.results_root))
    if ran_total and errored / ran_total > MAX_ERROR_FRACTION:
        print(f"\nWARNING: {errored} of {ran_total} episodes errored ({errored / ran_total:.0%}), above the {MAX_ERROR_FRACTION:.0%} the "
              f"project's >=98%-evaluable gate allows.  The rows are saved and stay in every denominator, but read the errors "
              f"(scripts/frontier_report.py lists them) and fix the cause before reporting this run.")
        return 1
    if errored:
        print(f"note: {errored} episode(s) errored; they are recorded and stay in every denominator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
