"""Read the baseline-defenses run and report attack success and utility, unfiltered, by model and defense.

    .venv/bin/python scripts/baseline_defenses_analysis.py --results-root results/baseline_defenses

This answers Task #29's question: how much do cheap, no-ProvenanceGuard mitigations help, on the SAME held-out scenarios
the PG ladder (C0/SC-v3/SCP-v3) was scored on? ``none`` is the no-defense condition (equivalent in spirit to C0 but without
any PG plumbing at all); the other three are the naive defenses. Nothing here changes a result; it only reads and totals,
same discipline as ``heldout_run_analysis.py``.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.baseline_defense_lib import DEFENSE_NAMES

ATTACK_FAMILIES = ("policy_passing", "content_attack", "policy_blocked_control")


def load_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(glob.glob(str(results_root / "*" / "*" / "episodes.jsonl"))):
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, int]]:
    cells: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        slug, defense, fam = r.get("slug"), r.get("baseline_defense"), r.get("family")
        if defense not in DEFENSE_NAMES:
            continue
        c = cells[(slug, defense, fam)]
        c["n"] += 1
        c["errors"] += bool(r.get("error"))
        if r.get("attack_success"):
            c["attack_success"] += 1
        if r.get("benign_success"):
            c["benign_success"] += 1
    return cells


def _c(cells, slug, defense, fam, key):
    return cells.get((slug, defense, fam), {}).get(key, 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=ROOT / "results" / "baseline_defenses")
    args = ap.parse_args(argv)
    rows = load_rows(args.results_root)
    if not rows:
        print(f"no episodes under {args.results_root}")
        return 1
    cells = summarise(rows)
    slugs = sorted({k[0] for k in cells})

    print(f"baseline defenses (no ProvenanceGuard), unfiltered. {len(rows)} episodes, {len(slugs)} models.\n")
    print("ATTACK SUCCESS by model and defense, pooled over the three attack families  [attacks | clean task-done | err]")
    print(f"{'model':16s} {'defense':26s} {'attack':>10s} {'clean':>10s} {'err':>4s}")
    for slug in slugs:
        for defense in DEFENSE_NAMES:
            a_k = sum(_c(cells, slug, defense, f, "attack_success") for f in ATTACK_FAMILIES)
            a_n = sum(_c(cells, slug, defense, f, "n") for f in ATTACK_FAMILIES)
            cl_k, cl_n = _c(cells, slug, defense, "clean_control", "benign_success"), _c(cells, slug, defense, "clean_control", "n")
            err = sum(_c(cells, slug, defense, f, "errors") for f in (*ATTACK_FAMILIES, "clean_control"))
            print(f"{slug:16s} {defense:26s} {f'{a_k}/{a_n}':>10s} {f'{cl_k}/{cl_n}':>10s} {err:>4d}")
        print()

    print("POOLED over models, by defense (compare against the held-out PG ladder: C0/SC-v3/SCP-v3):")
    print(f"{'defense':26s} {'attack':>12s} {'clean task-done':>16s}")
    for defense in DEFENSE_NAMES:
        a_k = sum(_c(cells, s, defense, f, "attack_success") for s in slugs for f in ATTACK_FAMILIES)
        a_n = sum(_c(cells, s, defense, f, "n") for s in slugs for f in ATTACK_FAMILIES)
        cl_k = sum(_c(cells, s, defense, "clean_control", "benign_success") for s in slugs)
        cl_n = sum(_c(cells, s, defense, "clean_control", "n") for s in slugs)
        print(f"{defense:26s} {f'{a_k}/{a_n}':>12s} {f'{cl_k}/{cl_n}':>16s}")
    print("\nCompare the 'none' row to C0 and the other three rows to SCP-v3 in heldout_run_analysis.py's pooled table:")
    print("  both ran on the identical 72 held-out scenarios x 3 seeds, so the attack-success denominators line up 1:1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
