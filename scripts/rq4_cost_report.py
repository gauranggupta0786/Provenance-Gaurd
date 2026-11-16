"""RQ4 (cost of ProvenanceGuard): latency, token, approval, and failure cost, aggregated from episodes ALREADY
collected by the held-out ladder (Result 4) and adaptive-evasion (Result 7) runs -- no new cluster run needed, this
reads the same ``episodes.jsonl`` files those results were built from.

Every episode already carries its own cost fields (confirmed against the real schema in
``results/week11c/*/*/episodes.jsonl``, which shares this schema with the held-out and adaptive-evasion runs):

* ``prompt_tokens`` / ``completion_tokens``  -- the AGENT model's own token usage for the whole episode.
* ``latency_ms``                              -- the episode's total wall-clock latency (agent + tool calls +
  approval, everything).
* ``hook_result.reviewer``                    -- the approval layer's own, already-aggregated-per-episode counters:
  ``cards`` (proposals raised), ``approved``/``denied``/``capped``/``scope_denials``, ``prompt_tokens`` /
  ``completion_tokens`` (the REVIEWER model's own usage, separate from the agent's), and ``decisions`` (a list with
  one entry per proposal, each carrying its own ``latency_ms`` -- the reviewer model's own call latency, which is
  smaller than and does not double-count the episode's total ``latency_ms``).
* ``error``                                    -- non-null means the episode errored outright (a failure cost, not a
  security or utility outcome).

This does NOT try to re-derive any of this from ``runtime_timings`` (which carries overlapping, differently-scoped
entries for the same approval step at two instrumentation layers -- a GRAPH_NODE entry and an EXECUTE_TOOL entry can
both cover the same ``OBTAIN_APPROVAL`` call) -- the per-episode fields above are already the harness's own, cleaner
aggregation and are what every other Week 11-13 analysis script in this repo reads.

C0 is the right zero-point for an INCREMENTAL cost claim, not a true no-guard baseline: C0 keeps ProvenanceGuard's
signed-proposal/approve-all plumbing (so ``cards`` can be > 0 even at C0) but nothing in it can deny a write (see
``week13_findings.md`` Result 6). So this script's "cost ProvenanceGuard adds" numbers are the cost of the
scope/policy/provenance LAYERS specifically (SC-v3, SCP-v3), over and above the base proposal/approval plumbing that
C0 already pays for -- the same framing Result 6 already uses for utility, applied here to cost. The separate
``arm="baseline"`` runs (Result 3, no ProvenanceGuard plumbing at all) are a different pipeline and are not read by
this script.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOTS: tuple[Path, ...] = (ROOT / "results" / "heldout", ROOT / "results" / "adaptive_evasion")


def load_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for fn in sorted(root.glob("*/*/episodes.jsonl")):
            with fn.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row["_root"] = root.name
                    rows.append(row)
    return rows


def episode_cost(row: dict[str, Any]) -> dict[str, Any]:
    """Pull out exactly the fields this report needs, with safe defaults -- never raises on a missing/older-schema
    field, since the held-out and adaptive-evasion runs span more than one week's schema version."""
    reviewer = (row.get("hook_result") or {}).get("reviewer") or {}
    decisions = reviewer.get("decisions") or []
    return {
        "model_id": row.get("model_id") or row.get("slug"),
        "config": row.get("week11_config"),
        "family": row.get("family"),
        "agent_prompt_tokens": row.get("prompt_tokens") or 0,
        "agent_completion_tokens": row.get("completion_tokens") or 0,
        "episode_latency_ms": row.get("latency_ms") or 0,
        "cards": reviewer.get("cards", 0) or 0,
        "approved": reviewer.get("approved", 0) or 0,
        "denied": reviewer.get("denied", 0) or 0,
        "capped": reviewer.get("capped", 0) or 0,
        "scope_denials": reviewer.get("scope_denials", 0) or 0,
        "reviewer_prompt_tokens": reviewer.get("prompt_tokens", 0) or 0,
        "reviewer_completion_tokens": reviewer.get("completion_tokens", 0) or 0,
        "reviewer_decision_latency_ms": sum((d.get("latency_ms") or 0) for d in decisions),
        "reviewer_errors": reviewer.get("errors", 0) or 0,
        "errored": bool(row.get("error")),
    }


def aggregate(costs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled stats over one group of episode-cost dicts (already filtered to one (config[, model]) cell)."""
    n = len(costs)
    if n == 0:
        return {"n": 0}

    def mean_of(key: str) -> float:
        return statistics.mean(c[key] for c in costs)

    cards_total = sum(c["cards"] for c in costs)
    return {
        "n": n,
        "mean_agent_tokens": mean_of("agent_prompt_tokens") + mean_of("agent_completion_tokens"),
        "mean_episode_latency_ms": mean_of("episode_latency_ms"),
        "median_episode_latency_ms": statistics.median(c["episode_latency_ms"] for c in costs),
        "mean_cards_per_episode": mean_of("cards"),
        "mean_reviewer_tokens_per_episode": mean_of("reviewer_prompt_tokens") + mean_of("reviewer_completion_tokens"),
        "mean_reviewer_latency_ms_per_episode": mean_of("reviewer_decision_latency_ms"),
        "approval_rate": (sum(c["approved"] for c in costs) / cards_total) if cards_total else None,
        "denial_rate": (sum(c["denied"] for c in costs) / cards_total) if cards_total else None,
        "capped_total": sum(c["capped"] for c in costs),
        "scope_denials_total": sum(c["scope_denials"] for c in costs),
        "error_rate": sum(c["errored"] for c in costs) / n,
        "reviewer_errors_total": sum(c["reviewer_errors"] for c in costs),
    }


def group_by_config(costs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in costs:
        groups.setdefault(c["config"] or "?", []).append(c)
    return groups


CONFIG_ORDER = ("C0", "SC-v3", "SCP-v3")


def _ordered(configs: list[str]) -> list[str]:
    known = [c for c in CONFIG_ORDER if c in configs]
    rest = sorted(c for c in configs if c not in CONFIG_ORDER)
    return known + rest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, action="append", dest="roots",
                     help="a results root to include (repeatable); default is results/heldout + results/adaptive_evasion")
    args = ap.parse_args(argv)
    roots = args.roots or list(DEFAULT_ROOTS)

    rows = load_rows(roots)
    if not rows:
        print(f"no episodes under any of: {', '.join(str(r) for r in roots)}")
        return 1

    costs = [episode_cost(r) for r in rows]
    groups = group_by_config(costs)
    order = _ordered(list(groups.keys()))

    print(f"{len(costs)} episodes pooled from: {', '.join(str(r) for r in roots)}\n")
    header = (f"{'config':<10}{'n':>6}{'tokens(agent)':>16}{'latency_ms(med)':>18}{'cards/ep':>10}"
              f"{'reviewer_tok':>14}{'reviewer_ms':>13}{'approve%':>10}{'deny%':>8}{'error%':>9}")
    print(header)
    print("-" * len(header))
    stats_by_config: dict[str, dict[str, Any]] = {}
    for cfg in order:
        s = aggregate(groups[cfg])
        stats_by_config[cfg] = s
        approve_pct = f"{s['approval_rate'] * 100:.1f}" if s["approval_rate"] is not None else "n/a"
        deny_pct = f"{s['denial_rate'] * 100:.1f}" if s["denial_rate"] is not None else "n/a"
        print(f"{cfg:<10}{s['n']:>6}{s['mean_agent_tokens']:>16.0f}{s['median_episode_latency_ms']:>18.0f}"
              f"{s['mean_cards_per_episode']:>10.2f}{s['mean_reviewer_tokens_per_episode']:>14.0f}"
              f"{s['mean_reviewer_latency_ms_per_episode']:>13.0f}{approve_pct:>10}{deny_pct:>8}"
              f"{s['error_rate'] * 100:>8.1f}")

    if "C0" in stats_by_config:
        base = stats_by_config["C0"]
        print("\n----- delta vs C0 (the incremental cost of the scope/policy/provenance layers) -----")
        for cfg in order:
            if cfg == "C0":
                continue
            s = stats_by_config[cfg]
            d_tokens = s["mean_agent_tokens"] - base["mean_agent_tokens"]
            d_latency = s["median_episode_latency_ms"] - base["median_episode_latency_ms"]
            d_rev_tokens = s["mean_reviewer_tokens_per_episode"] - base["mean_reviewer_tokens_per_episode"]
            d_rev_latency = s["mean_reviewer_latency_ms_per_episode"] - base["mean_reviewer_latency_ms_per_episode"]
            d_error = (s["error_rate"] - base["error_rate"]) * 100
            print(f"{cfg}: agent tokens {d_tokens:+.0f}, episode latency(median) {d_latency:+.0f}ms, "
                  f"reviewer tokens {d_rev_tokens:+.0f}, reviewer latency {d_rev_latency:+.0f}ms, "
                  f"error rate {d_error:+.1f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
