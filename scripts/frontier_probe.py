"""Probe a hosted OpenAI-compatible model before spending anything on it.

    export PROVENANCEGUARD_API_BASE=https://api.example.com/v1
    export PROVENANCEGUARD_API_KEY=sk-...
    export PROVENANCEGUARD_MODEL=some-model
    python scripts/frontier_probe.py --out results/frontier/<name>

Costs a handful of calls.  Writes ``probe_report.json`` and prints a verdict: a model that fails a REQUIRED probe cannot
produce a measurable result, whatever it costs.  Advisory probes are reported, never used to exclude.

With ``--price-per-1k-input`` / ``--price-per-1k-output`` it also prints what a full campaign would cost, from the probe's
own measured tokens and the campaign size -- so the spend is known before it is committed, not after.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier.frontier_probes import ADVISORY, REQUIRED, run_probes
from frontier.openai_adapter import RemoteConfig, UNVERIFIABLE_REMOTELY, redact

# the Week 11e campaign, per model: 3 configurations x 48 scenarios x 3 seeds
CAMPAIGN_EPISODES = 3 * 48 * 3
# measured on the local runs: model calls per episode, agent plus reviewer
CALLS_PER_EPISODE = 6


def estimate(usage: dict, config: RemoteConfig, episodes: int = CAMPAIGN_EPISODES) -> dict:
    calls = max(usage.get("calls", 0), 1)
    per_call_in = usage.get("prompt_tokens", 0) / calls
    per_call_out = usage.get("completion_tokens", 0) / calls
    total_calls = episodes * CALLS_PER_EPISODE
    tokens_in, tokens_out = per_call_in * total_calls, per_call_out * total_calls
    return {
        "episodes": episodes, "model_calls": total_calls,
        "note": "a rough floor: probe prompts are far shorter than campaign prompts, so the real cost is higher",
        "prompt_tokens": round(tokens_in), "completion_tokens": round(tokens_out),
        "cost": round(tokens_in / 1000 * config.price_per_1k_input + tokens_out / 1000 * config.price_per_1k_output, 2),
        "currency": config.currency,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "frontier")
    ap.add_argument("--only", action="append", help="run just these probes (repeatable)")
    ap.add_argument("--price-per-1k-input", type=float)
    ap.add_argument("--price-per-1k-output", type=float)
    ap.add_argument("--episodes", type=int, default=CAMPAIGN_EPISODES)
    args = ap.parse_args(argv)
    try:
        config = RemoteConfig.from_env()
    except ValueError as exc:
        print(exc)
        return 2
    if args.price_per_1k_input is not None or args.price_per_1k_output is not None:
        from dataclasses import replace
        config = replace(config,
                         price_per_1k_input=args.price_per_1k_input or config.price_per_1k_input,
                         price_per_1k_output=args.price_per_1k_output or config.price_per_1k_output)

    print(f"endpoint {config.endpoint}\nmodel    {config.model}\n")
    report = run_probes(config, only=args.only, log=print)
    report["spend_estimate"] = estimate(report["usage"], config, args.episodes)
    report["cannot_be_verified_remotely"] = list(UNVERIFIABLE_REMOTELY)

    out_dir = Path(args.out) / config.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_report.json").write_text(
        redact(json.dumps(report, indent=2, sort_keys=True), config.api_key) + "\n", encoding="utf-8")

    print(f"\nrequired: {', '.join(REQUIRED)}")
    print(f"advisory: {', '.join(ADVISORY)}")
    if report["required_failed"]:
        print(f"\nNOT USABLE: required probe(s) failed: {report['required_failed']}")
        print("A model that cannot be measured cannot be reported; fix the configuration or choose another model.")
    else:
        print("\nUSABLE: every required probe passed.")
        if report["advisory_failed"]:
            print(f"Advisory probes that failed (report them, they do not exclude): {report['advisory_failed']}")
    spend = report["spend_estimate"]
    print(f"\nprobe used {report['usage']['total_tokens']} tokens in {report['usage']['calls']} calls "
          f"({report['seconds']}s)")
    print(f"campaign floor: {spend['model_calls']} calls, ~{spend['prompt_tokens'] + spend['completion_tokens']} tokens, "
          f"~{spend['cost']} {spend['currency']} ({spend['note']})")
    print(f"\nwrote {out_dir / 'probe_report.json'}")
    return 0 if not report["required_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
