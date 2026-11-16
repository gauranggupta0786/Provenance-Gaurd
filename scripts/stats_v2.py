"""Statistics for v1 and v2 results, standard library only: rates with scenario-clustered bootstrap intervals, paired tests.

    python scripts/stats_v2.py --results-root results/heldout            # v1 held-out rows (week11_config, family, scenario_id)
    python scripts/stats_v2.py --results-root results/v2 --json out.json  # v2 rows (config, suite, user_task_id, ...)

Why clustered: episodes of one scenario (three seeds, greedy decoding) or of one AgentDojo user task (one row per injection task)
are not independent, so a Wilson interval over all episodes is too narrow.  Here the resampling unit is the scenario (v1) or the
(suite, user task) (v2), within each model, 2,000 resamples, seeded, percentile intervals.  Paired comparisons between two
configurations use the same cases in both (exact McNemar on discordant pairs) and a clustered bootstrap of the difference;
p-values across the comparisons printed together are Holm-corrected.

Errored episodes stay in every denominator (an errored attack episode counts as not succeeded only if its final state shows no
success -- the runners already score the state).  ``--capable`` restricts headline rates to models whose benign utility under
'none' (v2) or clean-task completion under C0 (v1) reaches the registered threshold; the excluded models are listed.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

B = 2000
SEED = 20261112


# --------------------------------------------------------------------------- loading
def load(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(set(glob.glob(str(Path(results_root) / "**" / "episodes.jsonl"), recursive=True))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalise(row: dict[str, Any]) -> dict[str, Any] | None:
    """One common shape: model, config, cluster, case, seed, kind ('benign'|'attack'), success, scope, error."""
    if row.get("schema") == "pgx-agentdojo-episode-v1":
        attack = row.get("injection_task_id") is not None
        return {"model": row.get("agent") or "?", "config": row["config"], "cluster": f"{row['suite']}/{row['user_task_id']}",
                "case": row["case_id"], "seed": row.get("seed"), "kind": "attack" if attack else "benign",
                "success": bool(row.get("attack_success")) if attack else bool(row.get("utility")),
                "utility": bool(row.get("utility")), "scope": ("in" if row.get("in_scope") else "text") if attack else None,  # v1 rows: the attack family
                "family": row["suite"], "under_specified": bool(row.get("under_specified")), "error": bool(row.get("error"))}
    if "week11_config" in row:
        family = row.get("family")
        attack = family not in ("clean_control",)
        success = bool(row.get("attack_success")) if attack else bool((row.get("hook_result") or {}).get("task_write_done"))
        return {"model": row.get("slug") or "?", "config": row["week11_config"], "cluster": row["scenario_id"],
                "case": row["scenario_id"], "seed": row.get("seed"), "kind": "attack" if attack else "benign",
                "success": success, "utility": None, "scope": family if attack else None, "family": family,
                "under_specified": False, "error": bool(row.get("error"))}
    return None


# --------------------------------------------------------------------------- statistics
def cluster_bootstrap(rows: Sequence[dict[str, Any]], stat: Callable[[Sequence[dict[str, Any]]], float], *,
                      b: int = B, seed: int = SEED) -> tuple[float, float]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        clusters[r["cluster"]].append(r)
    keys = sorted(clusters)
    if not keys:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    values = []
    for _ in range(b):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in clusters[k]]
        values.append(stat(sample))
    values.sort()
    return values[int(0.025 * (b - 1))], values[int(math.ceil(0.975 * (b - 1)))]


def rate(rows: Sequence[dict[str, Any]]) -> float:
    return sum(r["success"] for r in rows) / len(rows) if rows else float("nan")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues: Sequence[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired(rows_a: Sequence[dict[str, Any]], rows_b: Sequence[dict[str, Any]]) -> dict[str, Any]:
    a = {(r["case"], r["seed"]): r for r in rows_a}
    bb = {(r["case"], r["seed"]): r for r in rows_b}
    keys = sorted(set(a) & set(bb))
    only_a = sum(1 for k in keys if a[k]["success"] and not bb[k]["success"])
    only_b = sum(1 for k in keys if bb[k]["success"] and not a[k]["success"])
    joined = [{"cluster": a[k]["cluster"], "d": int(a[k]["success"]) - int(bb[k]["success"])} for k in keys]
    lo, hi = cluster_bootstrap(joined, lambda rs: sum(r["d"] for r in rs) / len(rs) if rs else float("nan"))
    return {"pairs": len(keys), "only_first": only_a, "only_second": only_b, "p": mcnemar_exact(only_a, only_b),
            "diff": (sum(r["d"] for r in joined) / len(joined)) if joined else float("nan"), "diff_ci": [lo, hi]}


# --------------------------------------------------------------------------- tables
def cells(norm: Iterable[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in norm:
        out[(r["model"], r["config"], r["kind"], r["scope"])].append(r)
    return out


def capable_models(norm: Sequence[dict[str, Any]], threshold: float, control: str) -> tuple[set[str], dict[str, float]]:
    util: dict[str, list[bool]] = defaultdict(list)
    for r in norm:
        if r["config"] == control and r["kind"] == "benign":
            util[r["model"]].append(r["success"])
    scores = {m: sum(v) / len(v) for m, v in util.items() if v}
    return {m for m, s in scores.items() if s >= threshold}, scores


def report(norm: list[dict[str, Any]], *, compare: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    table = []
    for (model, config, kind, scope), rs in sorted(cells(norm).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        lo, hi = cluster_bootstrap(rs, rate)
        table.append({"model": model, "config": config, "kind": kind, "scope": scope, "n": len(rs),
                      "k": sum(r["success"] for r in rs), "rate": rate(rs), "ci95": [lo, hi], "errors": sum(r["error"] for r in rs),
                      "clusters": len({r["cluster"] for r in rs})})
    tests = []
    by = cells(norm)
    for first, second in compare:
        for (model, config, kind, scope), rs in sorted(by.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            if config != first:
                continue
            other = by.get((model, second, kind, scope))
            if other:
                tests.append({"model": model, "kind": kind, "scope": scope, "first": first, "second": second, **paired(rs, other)})
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
    return {"cells": table, "paired": tests}


def _pct(x: float) -> str:
    return "  -  " if x != x else f"{100 * x:5.1f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--compare", nargs=2, action="append", metavar=("FIRST", "SECOND"), default=[],
                    help="paired comparison of two configurations (repeatable), e.g. --compare none PG-v4")
    ap.add_argument("--capable", type=float, help="headline only models whose benign success under --control reaches this")
    ap.add_argument("--control", default=None, help="the configuration --capable is measured on (default: none for v2, C0 for v1)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    norm = [n for n in (normalise(r) for r in load(args.results_root)) if n is not None]
    if not norm:
        print(f"no episodes under {args.results_root}")
        return 1
    excluded: dict[str, float] = {}
    if args.capable is not None:
        control = args.control or ("none" if any(n["utility"] is not None for n in norm) else "C0")
        keep, scores = capable_models(norm, args.capable, control)
        excluded = {m: s for m, s in scores.items() if m not in keep}
        norm = [n for n in norm if n["model"] in keep]
    out = report(norm, compare=[tuple(c) for c in args.compare])
    print(f"{'model':22s} {'config':18s} {'kind':7s} {'scope/family':5s} {'k/n':>11s} {'rate%':>6s} {'95% CI (clustered)':>20s} {'err':>4s}")
    for c in out["cells"]:
        print(f"{c['model'][:22]:22s} {c['config'][:18]:18s} {c['kind']:7s} {str(c['scope'] or '-'):5s} {c['k']:>5d}/{c['n']:<5d} "
              f"{_pct(c['rate']):>6s}   [{_pct(c['ci95'][0])}, {_pct(c['ci95'][1])}] {c['errors']:>4d}")
    if out["paired"]:
        print("\npaired (same cases and seeds): only-first / only-second, exact McNemar, Holm-corrected; clustered CI of the difference")
        for t in out["paired"]:
            print(f"  {t['model'][:20]:20s} {t['kind']:7s} {str(t['scope'] or '-'):5s} {t['first']} vs {t['second']}: "
                  f"{t['only_first']}/{t['only_second']} of {t['pairs']}  p={t['p']:.3g} (Holm {t['p_holm']:.3g})  "
                  f"diff {_pct(t['diff'])} [{_pct(t['diff_ci'][0])}, {_pct(t['diff_ci'][1])}]")
    if excluded:
        print("\nexcluded by --capable (benign success under the control): " + ", ".join(f"{m} {100 * s:.0f}%" for m, s in excluded.items()))
    if args.json:
        args.json.write_text(json.dumps({**out, "excluded": excluded}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
