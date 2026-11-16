"""Week 11b offline study: one reviewer model judges every logged card (resume-safe, ONE model per call).

    python scripts/run_week11b_review_eval.py --slug qwen3-14b --endpoint http://127.0.0.1:PORT --check-only
    python scripts/run_week11b_review_eval.py --slug qwen3-14b --endpoint ...
    python scripts/run_week11b_review_eval.py --slug qwen3-14b --endpoint ... --max-cards 20 --results-root results/week11b_eval_smoke

For every card (crafted suite + distinct agent cards) x prompt (v1, v2) x rendering (plain, rich) the reviewer answers APPROVE or
DENY.  A card whose rich rendering equals the plain one is judged once.  Verdicts go to <results-root>/<slug>/verdicts.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from pg.model_api import CanonicalMessage
from pg.reviewers import PROMPTS, parse_decision
from pg.week7_models import build_llm
from pg.week8_protocol import (
    MATRIX_PATH, PROFILES_DIR, PROTOCOL_PATH, ProtocolViolation, ensure_run_manifest, prepare_model_session, utc_now,
)
from scripts.run_week8_frozen import append_line, load_jsonl
from scripts.week11_lib import REVIEW_LIMITS
from scripts.week11b_lib import EVAL_PLAN_PATH, EVAL_RESULTS_ROOT, load_plan_file, verify_eval_code

IMMUTABLE = ("slug", "plan_sha256", "model_digest", "portable_profile_sha256", "defense_version")


def load_cards(plan: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """Every card file in the plan (Week 11b: crafted + agent; Week 11c: + held-out); agent duplicates are judged once."""
    cards: list[dict[str, Any]] = []
    for part in plan["cards"].values():
        cards += [c for c in load_jsonl(root / part["path"]) if c.get("representative", True)]
    return cards


def work_items(plan: dict[str, Any], cards: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str, str]]:
    items = []
    for card in cards:
        for prompt in plan["prompts"]:
            for variant in plan["variants"]:
                if variant == "rich" and not card.get("text_rich"):
                    continue
                items.append((card, prompt, variant))
    return items


def run_review_eval(
    slug: str, endpoint: str, *, results_root: Path = EVAL_RESULTS_ROOT, plan_path: Path = EVAL_PLAN_PATH,
    protocol_path: Path = PROTOCOL_PATH, matrix_path: Path = MATRIX_PATH, profiles_dir: Path = PROFILES_DIR, root: Path = ROOT,
    max_cards: int | None = None, check_only: bool = False, http=None, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    plan = load_plan_file(plan_path)
    code_digest = verify_eval_code(plan, root)
    session = prepare_model_session(slug, endpoint, protocol_path=protocol_path, matrix_path=matrix_path, profiles_dir=profiles_dir,
                                    root=root, http=http, check_code=False)
    if plan["week8_protocol_sha256"] != session["protocol"]["protocol_sha256"]:
        raise ProtocolViolation("the Week 11b evaluation plan was frozen against a different Week 8 protocol")
    identity, profile = session["identity"], session["profile"]
    cards = load_cards(plan, root)
    if max_cards:
        step = max(1, len(cards) // max_cards)
        cards = cards[::step][:max_cards]
    items = work_items(plan, cards)
    out_dir = Path(results_root) / slug
    summary = {"slug": slug, "cards": len(cards), "items": len(items), "ran": 0, "errors": 0}
    if check_only:
        log(f"{slug}: {len(cards)} cards, {len(items)} judgements to make")
        return summary
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_run_manifest(out_dir / "run_manifest.json", {
        "schema_version": "week11b-eval-manifest-v1", "created_at": utc_now(), "slug": slug, "plan_sha256": plan["plan_sha256"],
        "model_digest": identity["model_digest"], "portable_profile_sha256": identity["portable_profile_sha256"],
        "defense_version": code_digest, "environment": session["environment"],
    }, IMMUTABLE)
    output = out_dir / "verdicts.jsonl"
    existing = load_jsonl(output)
    if any(r.get("plan_sha256") != plan["plan_sha256"] for r in existing):
        raise ProtocolViolation(f"{output} has rows from another plan; use a new results directory")
    done = {(r["card_id"], r["prompt"], r["variant"]) for r in existing}
    pending = [i for i in items if (i[0]["card_id"], i[1], i[2]) not in done]
    log(f"{slug}: {len(existing)} saved, {len(pending)} to run")
    if not pending:
        return summary
    llm = build_llm(profile, seed=42, limits=REVIEW_LIMITS)
    if not llm.healthcheck()["model_available"]:
        raise ProtocolViolation(f"server reachable but {profile.model_id} unavailable")
    llm.warmup()
    started_all = time.monotonic()
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for index, (card, prompt_version, variant) in enumerate(pending, start=1):
            shown = card["text_rich"] if variant == "rich" else card["text_plain"]
            prompt = PROMPTS[prompt_version].format(request=card["request"], card=shown)
            started = time.perf_counter()
            error, text, pt, ct = None, None, 0, 0
            try:
                completion = llm.model_client.complete([CanonicalMessage(role="user", content=prompt)], ())
                text, pt, ct = completion.content, int(completion.prompt_tokens), int(completion.completion_tokens)
            except Exception as exc:  # fail closed, recorded
                error = f"{type(exc).__name__}: {exc}"
            approve, how = parse_decision(text)
            append_line(handle, {
                "card_id": card["card_id"], "source": card["source"], "truth": card["truth"], "family": card["family"],
                "vector": card["vector"], "k": card["k"], "agent": card.get("agent"), "reviewer": slug, "prompt": prompt_version,
                "variant": variant, "approve": bool(approve), "parse": how, "answer": (text or "")[:300], "error": error,
                "latency_ms": int((time.perf_counter() - started) * 1000), "prompt_tokens": pt, "completion_tokens": ct,
                "plan_sha256": plan["plan_sha256"], "model_digest": identity["model_digest"], "defense_version": code_digest,
                "at": utc_now(),
            })
            summary["ran"] += 1
            summary["errors"] += bool(error)
            if index % 50 == 0 or index == len(pending):
                eta = (time.monotonic() - started_all) / index * (len(pending) - index)
                log(f"[{index}/{len(pending)}] eta={eta / 60:.0f}min errors={summary['errors']}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT"))
    ap.add_argument("--results-root", type=Path, default=EVAL_RESULTS_ROOT)
    ap.add_argument("--plan", type=Path, default=EVAL_PLAN_PATH)
    ap.add_argument("--max-cards", type=int)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.endpoint:
        print("--endpoint (or $ENDPOINT) is required")
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        run_review_eval(args.slug, args.endpoint, results_root=args.results_root, plan_path=args.plan, max_cards=args.max_cards,
                        check_only=args.check_only)
    except ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
