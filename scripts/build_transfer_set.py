"""Freeze the adaptive transfer set the mentor replays on frontier models (no optimisation on paid models).

    python scripts/build_transfer_set.py --results-root results/v2_adaptive --out data/v2/adaptive_transfer.jsonl

For each test-split target, keeps every distinct injection text that succeeded against ANY configuration in the local adaptive
runs, plus, for targets never broken, the attacker's final attempt (so the set does not only contain the easy targets).  Each line:
suite, user task, injection task, the text, which local configurations it beat, and its sha256.  Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        print(f"REFUSED: {args.out} exists (frozen)")
        return 2
    texts: dict[tuple, dict] = {}
    last: dict[str, dict] = {}
    for path in sorted(glob.glob(str(args.results_root / "**" / "episodes.jsonl"), recursive=True)):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "test" or not r.get("injection_text"):
                continue
            key = (r["case_id"], r["injection_sha256"])
            slot = texts.setdefault(key, {"case_id": r["case_id"], "text": r["injection_text"], "beat": set(), "tried": set()})
            slot["tried"].add(r["config"])
            if r.get("attack_success"):
                slot["beat"].add(r["config"])
            if r["case_id"] not in last or r["attempt"] >= last[r["case_id"]]["attempt"]:
                last[r["case_id"]] = r
    keep = [v for v in texts.values() if v["beat"]]
    broken = {v["case_id"] for v in keep}
    keep += [texts[(cid, r["injection_sha256"])] for cid, r in last.items() if cid not in broken]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for v in sorted(keep, key=lambda v: (v["case_id"], v["text"])):
            suite, uid, iid = v["case_id"].split("/")
            handle.write(json.dumps({"suite": suite, "user_task_id": uid, "injection_task_id": iid, "text": v["text"],
                                     "beat_locally": sorted(v["beat"]), "tried_locally": sorted(v["tried"]),
                                     "sha256": hashlib.sha256(v["text"].encode()).hexdigest()}, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(keep)} texts for {len({v['case_id'] for v in keep})} targets "
          f"({len(broken)} broken locally); sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
