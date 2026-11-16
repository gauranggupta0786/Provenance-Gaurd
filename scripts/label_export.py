"""Export every distinct executed e-mail body of the content/provenance attacks for two people to label, blind.

    python scripts/label_export.py --results-root results/heldout --scenarios data/heldout/scenarios.jsonl \
        --out labels/heldout_A.csv --annotator A
    python scripts/label_export.py --results-root results/heldout --scenarios data/heldout/scenarios.jsonl \
        --out labels/heldout_B.csv --annotator B
    # (the same for results/adaptive_evasion with data/adaptive_evasion/scenarios.jsonl)

One row per distinct body (keyed by the 12-hex digest ``content_rescore`` uses), in an order shuffled with the annotator's name
as seed (so A and B see different orders), with the user's request, the attacker's goal and the body -- and NOTHING about which
configuration or model produced it.  Columns ``verdict`` (leak / clean) and ``note`` are left empty for the annotator.

The rule to apply is in docs/LABELLING_GUIDE.md.  ``scripts/label_agreement.py`` then compares the two files.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.frontier_report import BODY_FAMILIES, load_rows  # noqa: E402
from scripts.provenance_rescore import body_digest, executed_bodies  # noqa: E402

COLUMNS = ("item", "digest", "request", "attacker_goal", "body", "episodes", "verdict", "note")


def collect(results_root: Path, scenarios: dict[str, dict]) -> dict[str, dict]:
    bodies: dict[str, dict] = {}
    for row in load_rows(results_root):
        if row.get("family") not in BODY_FAMILIES or row.get("error"):
            continue
        try:
            sent = executed_bodies(row)
        except (KeyError, TypeError):
            continue
        scen = scenarios.get(row.get("scenario_id"), {})
        for body in sent:
            d = body_digest(body)
            slot = bodies.setdefault(d, {"digest": d, "body": body, "request": scen.get("prompt", ""),
                                         "attacker_goal": scen.get("goal_text") or row.get("goal") or "", "episodes": 0})
            slot["episodes"] += 1
    return bodies


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--scenarios", type=Path, action="append", required=True, help="scenario file(s) the rows came from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotator", required=True, help="a short name; also seeds the row order")
    args = ap.parse_args(argv)
    scenarios: dict[str, dict] = {}
    for path in args.scenarios:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                s = json.loads(line)
                scenarios[s["scenario_id"]] = s
    bodies = collect(args.results_root, scenarios)
    if not bodies:
        print(f"no executed content/provenance bodies under {args.results_root}")
        return 1
    order = sorted(bodies)
    random.Random(f"labels-{args.annotator}").shuffle(order)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=COLUMNS)
        w.writeheader()
        for i, d in enumerate(order, 1):
            w.writerow({"item": i, **{k: bodies[d][k] for k in ("digest", "request", "attacker_goal", "body", "episodes")},
                        "verdict": "", "note": ""})
    print(f"wrote {args.out}: {len(order)} distinct bodies for annotator {args.annotator} (configuration and model hidden)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
