"""Project GPU-hours for the v2 runs from measured smoke (or dev) episodes, before submitting the big jobs.

    python scripts/v2_estimate.py --results-root results/v2_smoke --split test --groups core baselines bridge --variance

Uses the median measured wall time per episode for each (agent, configuration) found under --results-root (falling back to the
agent's median over all configurations), times the number of episodes the planned run would schedule.  A planning figure:
episodes vary a lot in length, so treat it as the middle of a wide range.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgx.configs import CONFIGS_V2, resolve  # noqa: E402
from scripts.run_v2_agentdojo import load_cases, variance_cases  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--split", choices=("dev", "test", "all"), default="test")
    ap.add_argument("--groups", nargs="+", default=["core"])
    ap.add_argument("--variance", action="store_true")
    args = ap.parse_args(argv)
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in glob.glob(str(args.results_root / "**" / "episodes.jsonl"), recursive=True):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("wall_seconds"):
                    per[(r.get("agent"), r.get("config"))].append(float(r["wall_seconds"]))
    if not per:
        print(f"no measured episodes under {args.results_root}")
        return 1
    splits = ["dev", "test"] if args.split == "all" else [args.split]
    cases = load_cases(splits, ["workspace", "travel", "banking", "slack"])
    vset = variance_cases(cases) if args.variance else set()
    configs = resolve(None, args.groups)
    agents = sorted({a for a, _ in per})
    total = 0.0
    for agent in agents:
        fallback = statistics.median([x for (a, _), v in per.items() if a == agent for x in v])
        hours = 0.0
        for config in configs:
            mine = [c for c in cases if CONFIGS_V2[config]["kind"] != "v1" or c.suite in CONFIGS_V2[config].get("suites", [])]
            n = len(mine) + 2 * sum(1 for c in mine if c.case_id in vset)
            sec = statistics.median(per[(agent, config)]) if per.get((agent, config)) else fallback
            hours += n * sec / 3600
        total += hours
        print(f"{agent:20s} {hours:7.1f} GPU-hours  ({len(configs)} configurations, split {args.split})")
    print(f"{'total':20s} {total:7.1f} GPU-hours (median-based planning figure; jobs are 24 h and resumable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
