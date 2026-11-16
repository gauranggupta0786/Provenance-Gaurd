"""Prove the OpenAI-compatible adapter runs the same experiment as the native path -- for free, on a local model.

    .venv/bin/python scripts/week12_adapter_parity.py --slug gemma4-e4b --endpoint http://127.0.0.1:PORT

Ollama serves both APIs from one process: the native one at ``/api/chat`` and an OpenAI-compatible one at ``/v1``.  The
same model, with the same seed and the same scenarios, is run down both paths and the *actions* are compared -- which
tools were called with which arguments, whether the attack succeeded, whether the user's own task was done.  Wording may
differ; actions may not.  This is the evidence that a frontier run through the adapter measures what the local runs
measured, and it costs nothing.

Known and reported, not hidden: the OpenAI payload carries ``temperature``, ``top_p``, ``seed`` and ``max_tokens`` only.
``num_ctx``, ``top_k`` and ``min_p`` have no place in that request, so on the ``/v1`` path they are the server's defaults.
The report records both payloads' keys so the difference is visible, and ``--require-identical`` makes an action-level
difference a non-zero exit for CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from frontier.openai_adapter import RemoteConfig, UsageLog, build_remote_llm
from pg.week5_runner import run_episode
from pg.week7_models import build_llm
from pg.week8_protocol import MATRIX_PATH, PROFILES_DIR, PROTOCOL_PATH, ProtocolViolation, prepare_model_session, utc_now
from scripts.week11_lib import PLAN_PATH, REVIEW_LIMITS, layers_for, load_jsonl, make_hook, make_reviewer, needs_model
from scripts.week11e_lib import PLAN_PATH_E
from scripts.week12_lib import MATRIX_PATH_W12, MODEL_BY_SLUG, PLAN_PATH_W12, PROFILES_DIR_W12, PROTOCOL_PATH_W12


def actions(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool calls an episode actually made, in order, with their arguments: the part that must match.

    ``EpisodeResult.to_dict()`` (``pg/week5_runner.py``) names these fields ``function``/``args``, not
    ``tool_name``/``arguments`` -- a real tool_calls entry looks like
    ``{"args": {...}, "function": "get_received_emails", "id": "...", "placeholder_args": ...}``, confirmed against
    real episodes in ``results/week11c``. There is no ``executed``/``denied`` key on a tool_calls entry at all --
    that verdict lives on a SEPARATE list, ``mediated_calls`` (one entry per write action that went through
    ProvenanceGuard mediation, keyed by ``agentdojo_function``/``decision``/``result_status``), which read-only tool
    calls never appear in. The two lists are compared separately rather than zipped together by position/name --
    guessing at a pairing would risk a second silent mismatch if an episode calls the same tool more than once."""
    return [{"function": c.get("function"), "args": c.get("args")}
            for c in (result.get("tool_calls") or []) if isinstance(c, dict)]


def mediated(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Every write action ProvenanceGuard actually mediated, and what happened to it -- this, not a key on
    ``tool_calls``, is where "executed" vs "denied" actually lives (``pg/week5_runner.py``'s ``mediated_calls``)."""
    return [{"function": c.get("agentdojo_function"), "decision": c.get("decision"),
             "result_status": c.get("result_status")}
            for c in (result.get("mediated_calls") or []) if isinstance(c, dict)]


def outcome(result: dict[str, Any]) -> dict[str, Any]:
    # the real field is "hook_result" (see pg/week5_runner.py's EpisodeResult) -- "post_episode_hook" and "hook" are
    # not keys EpisodeResult.to_dict() ever produces, so looking for those always returned {} and made every
    # task_done comparison a trivial None == None, never actually checking task-completion parity.
    hook = result.get("hook_result") or {}
    return {
        "attack_success": result.get("attack_success"),
        "task_done": hook.get("task_write_done") if isinstance(hook, dict) else None,
        "error": (result.get("error") or "")[:120] or None,
        "failure_category": result.get("failure_category"),
        "actions": actions(result),
        "mediated": mediated(result),
    }


def run_one(llm, scenario, *, arm, seed, spec, evidence_root, review_llm_factory) -> dict[str, Any]:
    reviewer_name = spec["reviewer"]
    review_llm = review_llm_factory() if needs_model(reviewer_name) else None
    reviewer = make_reviewer(reviewer_name, scenario, review_llm, cap=spec.get("denial_cap"))
    kwargs: dict[str, Any] = {"post_episode_hook": make_hook(scenario, reviewer)}
    if arm == "protected":
        kwargs.update({"layers": layers_for(spec), "reviewer": reviewer,
                       "describe_targets": bool(spec.get("describe", False)),
                       "annotate_sources": bool(spec.get("annotate", False)),
                       "check_provenance": bool(spec.get("provenance", False))})
    try:
        return run_episode(scenario, arm=arm, seed=seed, evidence_dir=evidence_root, llm=llm,
                           runtime_limits=REVIEW_LIMITS, **kwargs).to_dict()
    except Exception as exc:  # recorded, never dropped -- a crash on one path is itself a parity finding
        return {"attack_success": None, "error": f"{type(exc).__name__}: {exc}", "failure_category": "RUNNER_FAILURE"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--endpoint", required=True, help="the native Ollama endpoint, e.g. http://127.0.0.1:34567")
    ap.add_argument("--scenarios", type=int, default=2)
    ap.add_argument("--config", action="append", dest="configs", help="default: the first and last configuration")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "week12_parity")
    ap.add_argument("--require-identical", action="store_true", help="exit 1 if any action-level difference is found")
    args = ap.parse_args(argv)

    week12 = args.slug in MODEL_BY_SLUG
    plan_path = PLAN_PATH_W12 if week12 else (PLAN_PATH_E if PLAN_PATH_E.exists() else PLAN_PATH)
    if not plan_path.exists():
        print(f"{plan_path} not found: freeze the plan first")
        return 2
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    try:
        session = prepare_model_session(
            args.slug, args.endpoint, check_code=False,
            protocol_path=PROTOCOL_PATH_W12 if week12 else PROTOCOL_PATH,
            matrix_path=MATRIX_PATH_W12 if week12 else MATRIX_PATH,
            profiles_dir=PROFILES_DIR_W12 if week12 else PROFILES_DIR,
        )
    except ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return 2
    profile = session["profile"]
    seed = args.seed or plan["seeds"][0]
    names = args.configs or [plan["config_order"][0], plan["config_order"][-1]]
    scenarios = {s["scenario_id"]: s for s in load_jsonl(ROOT / plan["dataset"])}
    chosen = [scenarios[sid] for sid in plan["scenario_ids"][:args.scenarios]]

    remote = RemoteConfig(
        api_base=args.endpoint.rstrip("/") + "/v1", model=profile.model_id, api_key="ollama",
        context_length=profile.selected_context_length, max_output_tokens=profile.max_output_tokens,
        temperature=profile.temperature, top_p=profile.top_p, seed=seed,
        timeout_seconds=REVIEW_LIMITS.model_seconds, max_retries=0,
        thinking_mode=profile.thinking_mode, thinking_capability=profile.thinking_capability,
    )
    usage = UsageLog()
    evidence_root = Path(args.out) / "evidence"
    rows: list[dict[str, Any]] = []
    client = httpx.Client(base_url=args.endpoint.rstrip("/"), timeout=60.0)
    try:
        for name in names:
            spec = plan["configs"][name]
            for scenario in chosen:
                native = run_one(
                    build_llm(profile, seed=seed, limits=REVIEW_LIMITS, client=client), scenario,
                    arm=spec["arm"], seed=seed, spec=spec, evidence_root=evidence_root / "native",
                    review_llm_factory=lambda: build_llm(profile, seed=seed, limits=REVIEW_LIMITS, client=client))
                through_adapter = run_one(
                    build_remote_llm(remote, limits=REVIEW_LIMITS, usage=usage), scenario,
                    arm=spec["arm"], seed=seed, spec=spec, evidence_root=evidence_root / "adapter",
                    review_llm_factory=lambda: build_remote_llm(remote, limits=REVIEW_LIMITS, usage=usage))
                a, b = outcome(native), outcome(through_adapter)
                rows.append({"config": name, "scenario_id": scenario["scenario_id"], "seed": seed,
                             "native": a, "adapter": b, "identical": a == b,
                             "same_decisions": a["attack_success"] == b["attack_success"] and a["task_done"] == b["task_done"]})
                mark = "same" if rows[-1]["identical"] else ("decisions agree" if rows[-1]["same_decisions"] else "DIFFER")
                print(f"{name:<8} {scenario['scenario_id']:<28} {mark}")
    finally:
        client.close()

    report = {
        "schema_version": "week12-adapter-parity-v1", "recorded_at": utc_now(), "slug": args.slug,
        "model": profile.model_id, "native_endpoint": args.endpoint, "adapter_endpoint": remote.endpoint,
        "plan": str(plan_path.relative_to(ROOT)), "plan_sha256": plan["plan_sha256"], "seed": seed,
        "configs": names, "episodes": len(rows), "rows": rows,
        "identical": sum(1 for r in rows if r["identical"]),
        "decisions_agree": sum(1 for r in rows if r["same_decisions"]),
        "adapter_usage": usage.totals(),
        "payload_difference": {
            "native_only": ["num_ctx", "top_k", "min_p", "num_predict", "think"],
            "note": "the OpenAI request carries temperature, top_p, seed and max_tokens only; the other decoding settings "
                    "are the server's defaults on the /v1 path. State this wherever adapter results are reported.",
        },
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{report['identical']}/{len(rows)} episodes identical, "
          f"{report['decisions_agree']}/{len(rows)} agree on attack success and task completion")
    print(f"adapter usage: {usage.totals()}")
    print(f"wrote {out_dir / 'parity_report.json'}")
    if args.require_identical and report["decisions_agree"] != len(rows):
        print("REFUSING: the two paths disagree on an outcome; the adapter is not faithful yet")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
