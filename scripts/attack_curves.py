"""Adaptive-attack curves: the share of targets broken within k attempts, per configuration, with clustered intervals.

    python scripts/attack_curves.py --results-root results/v2_adaptive
    python scripts/attack_curves.py --results-root results/v2_adaptive --json curves.json

A target is broken at attempt k if its k-th attempt succeeded (AgentDojo's security check).  The curve at k counts targets broken
at any attempt <= k over ALL targets attacked (targets whose attacker produced no usable text still count, as not broken).
Intervals resample targets (2,000 bootstrap draws, seeded).
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path


def load(root: Path) -> dict[tuple, dict[str, int | None]]:
    """(agent, reviewer, attacker, config, feedback) -> {target case_id: first successful attempt or None}."""
    out: dict[tuple, dict[str, int | None]] = defaultdict(dict)
    for path in sorted(glob.glob(str(root / "**" / "episodes.jsonl"), recursive=True)):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r.get("agent"), r.get("reviewer"), r.get("attacker"), r.get("config"), r.get("feedback", "outcome"))
            first = out[key].get(r["case_id"])
            if r.get("attack_success") and (first is None or r["attempt"] < first):
                out[key][r["case_id"]] = r["attempt"]
            else:
                out[key].setdefault(r["case_id"], first)
    return out


def curve(broken: dict[str, int | None], budget: int) -> list[float]:
    n = len(broken)
    return [sum(1 for a in broken.values() if a is not None and a <= k) / n for k in range(1, budget + 1)] if n else []


def ci(broken: dict[str, int | None], k: int, b: int = 2000, seed: int = 11) -> tuple[float, float]:
    keys = sorted(broken)
    rng = random.Random(seed)
    vals = sorted(sum(1 for t in (rng.choice(keys) for _ in keys) if broken[t] is not None and broken[t] <= k) / len(keys)
                  for _ in range(b))
    return vals[int(0.025 * (b - 1))], vals[int(0.975 * (b - 1))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    data = load(args.results_root)
    if not data:
        print(f"no adaptive episodes under {args.results_root}")
        return 1
    out = []
    for key, broken in sorted(data.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        c = curve(broken, args.budget)
        lo, hi = ci(broken, args.budget)
        out.append({"agent": key[0], "reviewer": key[1], "attacker": key[2], "config": key[3], "feedback": key[4],
                    "targets": len(broken), "curve": c, "final_ci95": [lo, hi]})
        marks = " ".join(f"{100 * v:4.0f}" for v in c)
        print(f"{key[0]}/{key[3]}/{key[4]} ({len(broken)} targets): broken within k=1..{args.budget}: {marks}   "
              f"final 95% [{100 * lo:.0f}, {100 * hi:.0f}]")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
