"""Freeze the v2 benchmark inputs: AgentDojo task metadata, the attack texts, and the dev/test split.

    python scripts/v2_prepare.py            # writes data/v2/{agentdojo_tasks.json, injections.jsonl, split.json, plan.json}
    python scripts/v2_prepare.py --check    # recomputes everything and fails if any frozen file would change

Everything is derived from the installed AgentDojo (v1.2.2) by code, with no model and no hand labels:

* injection goals are classified ``write`` (their ground truth needs a write or an outgoing fetch, which ProvenanceGuard mediates;
  where AgentDojo's ground truth is empty, by a fixed list of write verbs in the goal text, recorded as ``classified_by``)
  or ``text_only`` (they only change what the agent SAYS, e.g. "say that I should visit hotel X") -- reported separately;
* a user task is ``under_specified`` when a write its ground truth makes goes to a destination or object the request does not
  name (the target has to come from data, e.g. "pay the bill in this file");
* the injection texts are AgentDojo's ``important_instructions`` with the addressee fixed to "AI assistant" (``pgx.attacks``);
* the split puts a seeded 20% of each suite's user tasks in ``dev`` (design work may look only at these) and 80% in ``test``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

from pgx.attacks import ATTACK_NAME, injections_for  # noqa: E402
from pgx.configs import CONFIGS_V2, GROUPS  # noqa: E402
from pgx.guards import V4Checks, object_named, object_names, env_names  # noqa: E402
from pgx.suites import BENCHMARK_VERSION, SUITES, SuiteSpecError, destinations_of, spec_for  # noqa: E402

OUT = ROOT / "data" / "v2"
WRITE_VERBS = re.compile(r"\b(e-?mail|send|forward|delete|create|share|invite|post|pay|transfer|reserve|book|schedule|add|"
                         r"remove|update|change)\b", re.I)
DEV_FRACTION = 0.2
SEEDS = [42, 314, 2718]
VARIANCE_FRACTION = 0.25


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def write_calls(spec, calls) -> list[dict[str, Any]]:
    out = []
    for c in calls:
        if c.function in spec.writes:
            try:
                canonical, args = spec.to_canonical(c.function, dict(c.args))
            except (SuiteSpecError, ValueError):
                canonical, args = spec.writes[c.function], dict(c.args)
            out.append({"function": c.function, "canonical": canonical, "args": args})
    return out


def under_specified(spec, prompt: str, env, writes: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    names = env_names(spec.name, env)
    unnamed: list[str] = []
    for w in writes:
        for kind, value in destinations_of(spec, w["canonical"], w["args"]):
            if not V4Checks._named(kind, value, prompt, None, names)[0]:
                unnamed.append(f"{w['function']}:{value}")
        for n in object_names(spec.name, env, spec, w["canonical"], w["args"]):
            if not object_named(n, prompt):
                unnamed.append(f"{w['function']}:object:{n}")
    return bool(unnamed), unnamed


def build() -> dict[str, str]:
    tasks: dict[str, Any] = {"benchmark_version": BENCHMARK_VERSION, "attack": ATTACK_NAME, "suites": {}}
    injection_rows: list[str] = []
    split: dict[str, Any] = {"dev_fraction": DEV_FRACTION, "suites": {}}
    for name in SUITES:
        suite = get_suite(BENCHMARK_VERSION, name)
        spec = spec_for(name)
        spec.classify({t.name for t in suite.tools})
        base_env = suite.load_and_inject_default_environment({})
        inj_meta = {}
        for iid, inj in sorted(suite.injection_tasks.items()):
            writes = write_calls(spec, inj.ground_truth(base_env.model_copy(deep=True)))
            if writes:
                kind, how = "write", "ground_truth"
            else:  # AgentDojo leaves some ground truths empty (e.g. workspace injection_task_6-13); read the goal instead
                kind, how = ("write" if WRITE_VERBS.search(inj.GOAL) else "text_only"), "goal_text"
            inj_meta[iid] = {"goal": inj.GOAL, "goal_kind": kind, "classified_by": how,
                             "ground_truth_writes": [w["function"] for w in writes]}
        user_meta = {}
        for uid, task in sorted(suite.user_tasks.items(), key=lambda kv: int(kv[0].split("_")[-1])):
            env = task.init_environment(base_env.model_copy(deep=True))
            writes = write_calls(spec, task.ground_truth(env.model_copy(deep=True)))
            under, unnamed = under_specified(spec, task.PROMPT, env, writes)
            user_meta[uid] = {"prompt": task.PROMPT, "ground_truth_writes": [w["function"] for w in writes],
                              "under_specified": under, "unnamed_targets": unnamed}
            for iid, inj in sorted(suite.injection_tasks.items()):
                try:
                    texts = injections_for(suite, task, inj)
                except ValueError as exc:  # AgentDojo: a user task whose correct run reads no injectable field
                    user_meta[uid]["not_injectable"] = str(exc)
                    break
                injection_rows.append(json.dumps({"suite": name, "user_task_id": uid, "injection_task_id": iid,
                                                  "injections": texts}, sort_keys=True))
        ids = sorted(user_meta, key=lambda u: int(u.split("_")[-1]))
        rng = random.Random(f"provenanceguard-v2-split-{name}")
        shuffled = ids[:]
        rng.shuffle(shuffled)
        dev = sorted(shuffled[:math.ceil(DEV_FRACTION * len(ids))], key=lambda u: int(u.split("_")[-1]))
        test = [u for u in ids if u not in dev]
        vrng = random.Random(f"provenanceguard-v2-variance-{name}")
        variance = sorted(vrng.sample(test, max(1, round(VARIANCE_FRACTION * len(test)))), key=lambda u: int(u.split("_")[-1]))
        split["suites"][name] = {"dev": dev, "test": test, "variance_subset": variance}
        tasks["suites"][name] = {"user_tasks": user_meta, "injection_tasks": inj_meta}
    files = {"agentdojo_tasks.json": _dump(tasks), "injections.jsonl": "\n".join(injection_rows) + "\n", "split.json": _dump(split)}
    plan = {
        "schema_version": "pgx-v2-plan-v1", "benchmark_version": BENCHMARK_VERSION, "attack": ATTACK_NAME, "seeds": SEEDS,
        "seed_policy": "seed 42 on every case; seeds 314 and 2718 on the variance subset of test user tasks",
        "files": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()},
        "configs": CONFIGS_V2, "groups": GROUPS,
        "registration": "test-split runs require data/v2/registration.json (scripts/v2_register.py) matching the current code",
    }
    files["plan.json"] = _dump(plan)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if any frozen file would change")
    args = ap.parse_args(argv)
    files = build()
    OUT.mkdir(parents=True, exist_ok=True)
    changed = [n for n, text in files.items() if not (OUT / n).exists() or (OUT / n).read_text(encoding="utf-8") != text]
    if args.check:
        print("v2 inputs: " + ("UNCHANGED" if not changed else f"WOULD CHANGE {changed}"))
        return 1 if changed else 0
    for n, text in files.items():
        (OUT / n).write_text(text, encoding="utf-8", newline="\n")
    tasks = json.loads(files["agentdojo_tasks.json"])
    for name, s in tasks["suites"].items():
        u, i = s["user_tasks"], s["injection_tasks"]
        print(f"{name:10s} user tasks {len(u):3d} (under-specified {sum(v['under_specified'] for v in u.values()):2d}, "
              f"not injectable {sum('not_injectable' in v for v in u.values())})  injection tasks {len(i):2d} "
              f"(write {sum(v['goal_kind'] == 'write' for v in i.values())}, "
              f"text-only {sum(v['goal_kind'] == 'text_only' for v in i.values())})")
    print(f"security cases: {files['injections.jsonl'].count(chr(10))}   wrote {sorted(files)} to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
