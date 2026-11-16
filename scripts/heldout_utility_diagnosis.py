"""Why did clean_control utility drop under SC-v3/SCP-v3 on the held-out set? Read-only: changes no result.

    .venv/bin/python scripts/heldout_utility_diagnosis.py --results-root results/heldout

For every clean_control episode where the user's own task did NOT complete under a protected config, this prints the
reviewer's denial reason(s) from ``hook_result.reviewer.decisions`` (the actual text ``pg.request_scope.scope_reasons``
produced, or the model reviewer's stated reason) and tallies which reason shows up most. It also flags
``legit_cards_denied`` > 0 specifically -- that field (from ``scripts.week11_lib.make_hook``) means a write that WAS one
of the scenario's own ``allowed_writes`` got denied anyway: a false-positive denial of the legitimate task, not a correctly
blocked attack. This is the direct evidence for (or against) a scope-check generalisation gap on the held-out set's new
recipients/filenames -- see the hypothesis in week13_findings.md before assuming the cause.

    .venv/bin/python scripts/heldout_utility_diagnosis.py --results-root results/heldout --net-regression

A separate mode: some "utility cost" failures under SC-v3/SCP-v3 would have failed with the guard removed entirely -- the
model's own tool use broke (e.g. a bad search query it never recovered from), independent of any reviewer or scope-check
decision. Counting those against ProvenanceGuard overstates its true cost. ``--net-regression`` cross-references every
failed clean_control episode against the IDENTICAL (same slug/scenario_id/seed) episode's own C0 result -- C0 keeps
ProvenanceGuard's signed-proposal plumbing but approves every write, so nothing in it can deny a card -- and reports
"failed at C0 too" (pre-existing model weakness, not the guard's fault) separately from "succeeded at C0, broke under the
guard" (the real, attributable cost).
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTECTED_CONFIGS = ("SC-v3", "SCP-v3")


def load_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(glob.glob(str(results_root / "*" / "*" / "episodes.jsonl"))):
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def failed_clean_episodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("family") != "clean_control" or r.get("week11_config") not in PROTECTED_CONFIGS:
            continue
        hook = r.get("hook_result") or {}
        if not hook.get("task_write_done"):
            out.append(r)
    return out


def episode_key(r: dict[str, Any]) -> tuple:
    return (r.get("slug"), r.get("scenario_id"), r.get("seed"))


def c0_task_done(rows: list[dict[str, Any]]) -> dict[tuple, bool]:
    """(slug, scenario_id, seed) -> whether the C0 (no-defense-overhead) run of that exact episode completed the task.

    C0 still has ProvenanceGuard's mediation plumbing (signed proposals, an approve-all reviewer) -- it is not the stock
    baseline arm -- but nothing in it can deny a write, so a C0 failure is attributable to the agent itself, never to the
    guard's judgment calls. Used to split "fails regardless of the guard" from "the guard broke something that worked
    without it."
    """
    return {episode_key(r): bool((r.get("hook_result") or {}).get("task_write_done"))
            for r in rows if r.get("family") == "clean_control" and r.get("week11_config") == "C0"}


def net_regression(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Split every protected-config clean_control failure into two buckets, by cross-referencing its OWN C0 result."""
    c0_done = c0_task_done(rows)
    failed = failed_clean_episodes(rows)
    pre_existing: list[dict[str, Any]] = []  # failed at C0 too -- not the guard's fault
    guard_caused: list[dict[str, Any]] = []   # succeeded at C0, broke under SC-v3/SCP-v3 -- the real cost
    unknown: list[dict[str, Any]] = []        # no matching C0 episode found to compare against
    for r in failed:
        key = episode_key(r)
        if key not in c0_done:
            unknown.append(r)
        elif c0_done[key]:
            guard_caused.append(r)
        else:
            pre_existing.append(r)
    return {"pre_existing": pre_existing, "guard_caused": guard_caused, "unknown": unknown}


def denial_reasons(episode: dict[str, Any]) -> list[str]:
    hook = episode.get("hook_result") or {}
    reviewer = hook.get("reviewer") or {}
    reasons = []
    for decision in reviewer.get("decisions") or []:
        if not decision.get("approve", True):
            reasons.append(str(decision.get("why") or "(no reason recorded)"))
    return reasons


def print_net_regression(rows: list[dict[str, Any]], models: list[str] | None = None) -> None:
    """Print the C0-cross-referenced split: which protected-config clean_control failures are attributable to
    ProvenanceGuard at all, versus pre-existing model weaknesses that would have failed with no guard."""
    split = net_regression(rows)
    pre, caused, unk = split["pre_existing"], split["guard_caused"], split["unknown"]
    if models:
        pre = [r for r in pre if r.get("slug") in models]
        caused = [r for r in caused if r.get("slug") in models]
        unk = [r for r in unk if r.get("slug") in models]
    total = len(pre) + len(caused) + len(unk)
    if total == 0:
        print("no failed clean_control episodes under SC-v3/SCP-v3 for the given filters")
        return
    print(f"{total} failed clean_control episodes under SC-v3/SCP-v3, split by their OWN episode's C0 result:\n")
    print(f"  {len(pre):3d}  failed at C0 too -- a pre-existing model weakness, NOT attributable to ProvenanceGuard")
    print(f"  {len(caused):3d}  succeeded at C0, broke under SC-v3/SCP-v3 -- the real, guard-attributable utility cost")
    if unk:
        print(f"  {len(unk):3d}  no matching C0 episode found to compare against (can't classify)")
    print()
    by_model: dict[str, dict[str, int]] = {}
    for label, group in (("pre_existing", pre), ("guard_caused", caused), ("unknown", unk)):
        for r in group:
            by_model.setdefault(r.get("slug", "?"), {"pre_existing": 0, "guard_caused": 0, "unknown": 0})[label] += 1
    print("by model (pre_existing / guard_caused / unknown):")
    for slug, counts in sorted(by_model.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {slug:18s} {counts['pre_existing']:3d} / {counts['guard_caused']:3d} / {counts['unknown']:3d}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=ROOT / "results" / "heldout")
    ap.add_argument("--model", action="append", dest="models", help="limit to these slugs (default: all)")
    ap.add_argument("--max-examples", type=int, default=8, help="how many full examples to print per model/config")
    ap.add_argument("--net-regression", action="store_true",
                    help="print the C0-cross-referenced split instead of the denial-reason breakdown")
    args = ap.parse_args(argv)
    rows = load_rows(args.results_root)
    if not rows:
        print(f"no episodes under {args.results_root}")
        return 1
    if args.net_regression:
        print_net_regression(rows, args.models)
        return 0
    failed = failed_clean_episodes(rows)
    if args.models:
        failed = [r for r in failed if r.get("slug") in args.models]
    if not failed:
        print("no failed clean_control episodes under SC-v3/SCP-v3 for the given filters")
        return 0

    reason_counts: Counter[str] = Counter()
    false_positive_count = 0  # legit_cards_denied > 0: a scenario-allowed write was itself denied
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in failed:
        by_model.setdefault(r.get("slug", "?"), []).append(r)
        hook = r.get("hook_result") or {}
        if int(hook.get("legit_cards_denied") or 0) > 0:
            false_positive_count += 1
        for reason in denial_reasons(r):
            # bucket by the reason's shape, not its exact text (recipient/filename vary per episode)
            if "is not named in the request" in reason:
                reason_counts["recipient/address not textually named in the prompt"] += 1
            elif "does not ask for an attachment" in reason:
                reason_counts["attachment not requested"] += 1
            elif "copies text from" in reason:
                reason_counts["body copies text from a source not named in the prompt (the hiking-email hypothesis)"] += 1
            elif "denial cap" in reason:
                reason_counts["denial cap reached (reviewer not asked again)"] += 1
            elif "scope check" in reason:
                reason_counts["other scope-check denial"] += 1
            elif reason.strip().upper().startswith("DENY") or "deny" in reason.lower():
                reason_counts["model reviewer said DENY"] += 1
            elif not r.get("error") and not denial_reasons(r):
                pass
            else:
                reason_counts[f"other: {reason[:80]}"] += 1
        if not denial_reasons(r) and not r.get("error"):
            reason_counts["no denial recorded at all -- model simply never attempted the write"] += 1

    print(f"{len(failed)} failed clean_control episodes across {len(by_model)} model(s) under SC-v3/SCP-v3.\n")
    print(f"{false_positive_count}/{len(failed)} have legit_cards_denied > 0 -- the scenario's OWN allowed write was "
          f"denied, i.e. a false-positive denial of the legitimate task (not a model that simply never tried).\n")

    print("failures by model:")
    for slug, episodes in sorted(by_model.items(), key=lambda kv: -len(kv[1])):
        print(f"  {slug:18s} {len(episodes)}")
    print()

    print("denial-reason breakdown (one failure can show more than one reason):")
    for reason, count in reason_counts.most_common():
        print(f"  {count:3d}  {reason}")
    print()

    worst_model = max(by_model, key=lambda s: len(by_model[s]))
    print(f"up to {args.max_examples} examples from {worst_model} (the worst-affected model):")
    for r in by_model[worst_model][: args.max_examples]:
        reasons = denial_reasons(r)
        print(f"  {r.get('week11_config')} {r.get('scenario_id')} seed={r.get('seed')} error={r.get('error')!r} "
              f"reasons={reasons or '(none recorded)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
