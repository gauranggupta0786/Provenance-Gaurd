"""Compare two annotators' labels, list the disagreements, and (once adjudicated) write the label table the re-score tool reads.

    python scripts/label_agreement.py labels/heldout_A.csv labels/heldout_B.csv --disagreements labels/heldout_disagree.csv
    #   a third person fills 'verdict' in labels/heldout_disagree.csv, then:
    python scripts/label_agreement.py labels/heldout_A.csv labels/heldout_B.csv \
        --adjudicated labels/heldout_disagree.csv --write-table labels/heldout_labels.json
    python scripts/content_rescore.py --results-root results/heldout --labels labels/heldout_labels.json

Reports raw agreement and Cohen's kappa (with a bootstrap 95% interval over items).  Refuses to write the table while any body
lacks a final verdict.  The table is ``content-labels-v1``, each entry noting whether the two annotators agreed or a third
person decided.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

VERDICTS = ("leak", "clean")


def read(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {r["digest"]: r for r in csv.DictReader(handle)}
    bad = [d for d, r in rows.items() if r["verdict"].strip().lower() not in VERDICTS]
    if bad:
        raise SystemExit(f"REFUSED: {path} has {len(bad)} row(s) without a verdict of leak/clean (first: {bad[0]})")
    return rows


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == v for a, _ in pairs) / n) * (sum(b == v for _, b in pairs) / n) for v in VERDICTS)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def kappa_ci(pairs: list[tuple[str, str]], b: int = 2000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    vals = sorted(kappa([rng.choice(pairs) for _ in pairs]) for _ in range(b))
    vals = [v for v in vals if v == v]
    return (vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]) if vals else (float("nan"), float("nan"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first", type=Path)
    ap.add_argument("second", type=Path)
    ap.add_argument("--disagreements", type=Path, help="write the bodies the two disagree on, for a third read")
    ap.add_argument("--adjudicated", type=Path, help="the disagreement file with the third person's verdicts filled in")
    ap.add_argument("--write-table", type=Path, help="write the content-labels-v1 table (needs every body decided)")
    args = ap.parse_args(argv)
    a, b = read(args.first), read(args.second)
    if set(a) != set(b):
        raise SystemExit(f"REFUSED: the two files cover different bodies ({len(set(a) ^ set(b))} differ); export both from the same results")
    digests = sorted(a)
    pairs = [(a[d]["verdict"].strip().lower(), b[d]["verdict"].strip().lower()) for d in digests]
    agree = sum(x == y for x, y in pairs)
    k = kappa(pairs)
    lo, hi = kappa_ci(pairs)
    print(f"{len(pairs)} bodies: agreement {agree}/{len(pairs)} ({100 * agree / len(pairs):.1f}%), Cohen's kappa {k:.3f} "
          f"[95% {lo:.3f}, {hi:.3f}];  leak: A {sum(x == 'leak' for x, _ in pairs)}, B {sum(y == 'leak' for _, y in pairs)}")
    disagree = [d for d, (x, y) in zip(digests, pairs) if x != y]
    if args.disagreements:
        with args.disagreements.open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=("digest", "request", "attacker_goal", "body", "first", "second", "verdict", "note"))
            w.writeheader()
            for d in disagree:
                w.writerow({"digest": d, "request": a[d]["request"], "attacker_goal": a[d]["attacker_goal"], "body": a[d]["body"],
                            "first": a[d]["verdict"], "second": b[d]["verdict"], "verdict": "", "note": ""})
        print(f"wrote {args.disagreements}: {len(disagree)} bodies for a third read")
    if args.write_table:
        decided: dict[str, dict] = {}
        third = {}
        if args.adjudicated:
            with args.adjudicated.open(encoding="utf-8", newline="") as handle:
                third = {r["digest"]: r["verdict"].strip().lower() for r in csv.DictReader(handle)}
        for d, (x, y) in zip(digests, pairs):
            if x == y:
                decided[d] = {"verdict": x, "note": "two annotators agreed"}
            elif third.get(d) in VERDICTS:
                decided[d] = {"verdict": third[d], "note": f"annotators split ({x}/{y}); third reader decided"}
        missing = [d for d in digests if d not in decided]
        if missing:
            print(f"REFUSED: {len(missing)} disagreement(s) not adjudicated yet; fill 'verdict' in the disagreement file")
            return 1
        args.write_table.parent.mkdir(parents=True, exist_ok=True)
        args.write_table.write_text(json.dumps({"schema_version": "content-labels-v1", "agreement": {
            "bodies": len(pairs), "agreed": agree, "kappa": k, "kappa_ci95": [lo, hi]}, "labels": decided}, indent=2) + "\n",
            encoding="utf-8")
        print(f"wrote {args.write_table} ({len(decided)} labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
