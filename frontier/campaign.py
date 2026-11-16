"""The frontier campaign: every experiment of the results report, run against a hosted OpenAI-compatible model.

The defense ladder (C0 -> SC-v3 -> SCP-v3) is the core of it, and ``frontier/suites.py`` adds the rest -- the four cheap baseline
defenses, every stage of building the defense, the 100-scenario stock-against-protected comparison and the scripted adaptive
attacker -- so one runner reproduces what the local models were put through.

Everything the local campaigns measured is reused unchanged -- the scenarios, the three configurations, the seeds, the guard,
the gateway, the reviewer prompts, the evidence store and the scoring.  Only the source of tokens moves, through
``frontier.openai_adapter``.  This module is the part that was missing from the Week 12 handoff: a runner that needs no
cluster files.  The local campaigns pin the model to a Week 8 protocol manifest read from the serving process (weight digest,
chat template, engine version); a hosted model has none of that, so the plan here is pinned to what *can* be pinned -- the
dataset bytes, the configurations, the seeds and the guard-code digest -- and the run records, never hides, what cannot be.

Properties this runner keeps, because a paid run that fails late is the expensive way to find out:

* **Resume-safe.**  Every finished episode is flushed to disk as it completes; re-running the same command skips finished
  ``(scenario, arm, seed)`` keys and continues.  A run directory that holds rows from another plan is refused, not mixed.
* **Bounded spend.**  ``max_cost`` stops the campaign cleanly once the metered cost reaches it.
* **Fails fast.**  ``max_consecutive_errors`` stops the campaign when episode after episode errors, which is what a wrong
  key, a wrong model name or an unsupported parameter looks like -- instead of recording hundreds of identical failures.
* **Honest rows.**  An episode that errors is recorded as an error and stays in every denominator; it is never dropped and
  never scored as a safe outcome (a truncated or malformed answer raises, see the adapter).
* **Same schema.**  Rows carry the fields ``scripts/heldout_run_analysis.py`` and ``scripts/provenance_rescore.py`` read, so
  the existing analysis tools work on a frontier run unchanged.

This file lives outside ``pg/`` and outside both frozen file lists; a test asserts it, computed from the real digest functions.
"""
from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier.openai_adapter import (
    ENV_PREFIX, RemoteConfig, UsageLog, build_remote_llm, redact, remote_session_record,
)
from pg.ablation import FULL
from pg.reviewers import PROMPTS
from pg.week5_runner import run_episode
from pg.week6_metrics import episode_key
from pg.week8_protocol import (
    SEEDS, ProtocolViolation, canonical_sha256, code_digest, ensure_run_manifest, evidence_token_dir, file_sha256,
    quarantine_stale_evidence, utc_now,
)
from frontier.suites import DATASETS, REVIEWER_DESCRIPTIONS, Suite, suite_for  # noqa: F401 (DATASETS is re-exported)
from scripts.baseline_defense_lib import run_baseline_defended_episode
from scripts.run_week8_frozen import load_jsonl, runner_failure_row
from scripts.week11_lib import REVIEW_LIMITS, layers_for, make_hook, make_reviewer, needs_model, week11_code_digest

PLAN_SCHEMA = "frontier-run-plan-v1"
EPISODE_SCHEMA = "frontier-episode-v1"
RESULTS_ROOT = ROOT / "results" / "frontier"

IMMUTABLE = ("slug", "config", "plan_sha256", "model_digest", "portable_profile_sha256", "protocol_sha256",
             "defense_version", "layers", "reviewer", "agent_model", "reviewer_model")


class CampaignStop(RuntimeError):
    """The campaign stopped on purpose (budget reached, systematic failure).  Finished episodes are saved; rerun to resume."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------------------------ plan

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "model"


def slug_for(agent: RemoteConfig, reviewer: RemoteConfig, label: str | None = None) -> str:
    """The results directory name.  A different reviewer is a different experiment, so it gets a different directory."""

    base = f"frontier-{slugify(agent.model)}"
    if reviewer.model != agent.model or reviewer.endpoint != agent.endpoint:
        base += f"__rev-{slugify(reviewer.model)}"
    return f"{base}-{slugify(label)}" if label else base


def _layers_record(suite: Suite, spec: Mapping[str, Any]) -> dict[str, Any] | None:
    """The layer switches a configuration runs with (None for a stock-arm configuration, which has no layers)."""

    if suite.kind == "config":
        layers = layers_for(spec)
        return None if layers is None else layers.to_dict()
    if suite.kind in ("arms", "attacker"):
        return FULL.to_dict() if spec["arm"] == "protected" else None
    return None  # defense: the stock arm with a mitigation bolted on


def build_plan(dataset_key: str, *, root: Path = ROOT) -> dict[str, Any]:
    """The frozen run plan for one suite: scenarios, configurations, seeds and the guard-code digest.

    ``dataset_key`` is a suite key (``heldout``, ``tuning``, ``baselines``, ...; see ``frontier/suites.py``).  For the three
    ladder suites the key is also the dataset name, which is why the parameter keeps the name it had when the ladder was the
    only experiment.

    Deterministic: the same tree and suite always give the same ``plan_sha256``, so two people running the package get plans
    they can compare by hash.  Unlike the local plans there is no ``week8_protocol_sha256`` -- a hosted model has no protocol
    manifest to pin to.  The plans of the ladder suites are byte-identical to the first hand-off package's, so rows written
    by an earlier run of it still resume.
    """

    suite = suite_for(dataset_key)
    spec = DATASETS[suite.dataset]
    dataset = root / spec["path"]
    if not dataset.exists():
        raise ProtocolViolation(f"{spec['path']} not found")
    actual = file_sha256(dataset)
    if actual != spec["sha256"]:
        raise ProtocolViolation(
            f"{spec['path']} hashes to {actual[:16]}..., not the frozen {spec['sha256'][:16]}...: this dataset is pinned "
            f"and must not change, or the run is not comparable to the local results")
    rows = load_jsonl(dataset)
    digest, hashes = week11_code_digest(root)
    if suite.kind == "attacker":
        ids = [r["objective_id"] for r in rows]
        families = {i: "adaptive" for i in ids}
        seeds = list(suite.extra["seeds"])
    else:
        ids = [r["scenario_id"] for r in rows]
        families = {r["scenario_id"]: r["family"] for r in rows}
        seeds = list(SEEDS)
    reviewers_used: list[str] = []
    for name in suite.config_order:
        reviewer_name = suite.configs[name].get("reviewer") if suite.kind == "config" else None
        if reviewer_name and reviewer_name not in reviewers_used:
            reviewers_used.append(reviewer_name)
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "dataset_key": dataset_key,
        "dataset": spec["path"],
        "dataset_sha256": actual,
        "scenario_ids": ids,
        "scenario_families": families,
        "seeds": seeds,
        "config_order": list(suite.config_order),
        "configs": {n: {**suite.configs[n], "layers": _layers_record(suite, suite.configs[n])} for n in suite.config_order},
        "reviewers": {n: REVIEWER_DESCRIPTIONS[n] for n in reviewers_used},
        "reviewer_prompt_sha256": {k: canonical_sha256({"prompt": v}) for k, v in PROMPTS.items()},
        "approval_seconds": REVIEW_LIMITS.approval_seconds,
        "episodes_per_config": len(rows) * len(seeds),
        "code": {"digest": digest, "component_hashes": hashes},
        "rules": list(suite.rules),
    }
    if not suite.ladder_plan:
        plan.update({"suite": suite.key, "kind": suite.kind, "unit": suite.unit, "title": suite.title})
        if suite.kind == "attacker":
            plan["max_attempts"] = int(suite.extra["max_attempts"])
    plan["plan_sha256"] = canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


def reviewer_config(agent: RemoteConfig, env: Mapping[str, str] | None = None) -> RemoteConfig:
    """The reviewer's endpoint and model: the agent's own by default (self-review, as in every local run), or a different one.

    ``PROVENANCEGUARD_REVIEWER_MODEL`` (and optionally ``_API_BASE`` / ``_API_KEY``) switch the reviewer to another model.  This
    is the cross-model reviewer experiment: the Week 13 result is that the guard's strength on policy-passing actions is the
    reviewer's strength, and a frontier reviewer is the open question.  Every other setting is inherited from the agent.
    """

    import os

    env = env if env is not None else os.environ
    model = env.get(f"{ENV_PREFIX}REVIEWER_MODEL") or ""
    base = env.get(f"{ENV_PREFIX}REVIEWER_API_BASE") or ""
    key = env.get(f"{ENV_PREFIX}REVIEWER_API_KEY") or ""
    if not (model or base or key):
        return agent
    if (base or key) and not model:
        raise ValueError(f"set {ENV_PREFIX}REVIEWER_MODEL as well: a different reviewer endpoint needs its model name")
    config = replace(agent, model=model, api_base=base or agent.api_base, api_key=key or agent.api_key)
    config.validate()
    return config


# ------------------------------------------------------------------------------ selection

def select(plan: Mapping[str, Any], *, smoke: bool = False, families: Sequence[str] | None = None,
           scenario_ids: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
           configs: Sequence[str] | None = None) -> dict[str, Any]:
    """Which scenarios, seeds and configurations to run.  Only ever a SUBSET of the frozen plan, never an extension."""

    unknown = [c for c in (configs or []) if c not in plan["configs"]]
    if unknown:
        raise ProtocolViolation(f"unknown configuration(s) {unknown}; the plan has {plan['config_order']}")
    bad_seeds = [s for s in (seeds or []) if s not in plan["seeds"]]
    if bad_seeds:
        raise ProtocolViolation(f"seeds {bad_seeds} are not in the frozen set {plan['seeds']}")
    chosen = list(plan["scenario_ids"])
    families_map = plan["scenario_families"]
    if families:
        wrong = sorted(set(families) - set(families_map.values()))
        if wrong:
            raise ProtocolViolation(f"unknown family(ies) {wrong}; the dataset has {sorted(set(families_map.values()))}")
        chosen = [s for s in chosen if families_map[s] in families]
    if scenario_ids:
        missing = [s for s in scenario_ids if s not in families_map]
        if missing:
            raise ProtocolViolation(f"unknown scenario id(s) {missing}")
        chosen = [s for s in chosen if s in scenario_ids]
    if smoke:
        seen: set[str] = set()
        first_per_family = []
        for sid in chosen:
            if families_map[sid] not in seen:
                seen.add(families_map[sid])
                first_per_family.append(sid)
        chosen = first_per_family
    run_seeds = list(seeds) if seeds else ([plan["seeds"][0]] if smoke else list(plan["seeds"]))
    names = [c for c in plan["config_order"] if not configs or c in configs]
    if not chosen or not names:
        raise ProtocolViolation("the selection is empty: nothing to run")
    return {"scenario_ids": chosen, "seeds": run_seeds, "configs": names}


def planned_episodes(selection: Mapping[str, Any]) -> int:
    return len(selection["scenario_ids"]) * len(selection["seeds"]) * len(selection["configs"])


# ------------------------------------------------------------------------------ the run

def environment_record() -> dict[str, Any]:
    from importlib import metadata

    versions = {}
    for name in ("agentdojo", "openai", "pydantic", "httpx", "cryptography", "langgraph"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


@dataclass
class RunSettings:
    max_cost: float | None = None
    max_consecutive_errors: int = 5
    #: running count of episodes in a row that errored, kept across configurations and datasets: a wrong key does not stop
    #: being wrong at a configuration boundary
    consecutive_errors: int = field(default=0, repr=False)
    #: the file each row of the current error streak was appended to, so the streak can be set aside if it stops the run
    streak_files: list = field(default_factory=list, repr=False)
    #: what the suites already finished in this invocation cost, so ``max_cost`` bounds the whole invocation and not each suite
    carried_cost: float = field(default=0.0, repr=False)


@dataclass
class ConfigSummary:
    config: str
    already_saved: int = 0
    selected: int = 0
    ran: int = 0
    errors: int = 0
    stopped: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config, "already_saved": self.already_saved, "selected": self.selected, "ran": self.ran,
                "errors": self.errors, "stopped": self.stopped}


def append_row(handle, payload: Mapping[str, Any], *secrets: str | None) -> None:
    """``append_line`` (flushed and fsynced, so a crash loses nothing) with API keys redacted from the serialized row."""

    handle.write(redact(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str), *secrets) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def set_aside_streak(files: Sequence[Path]) -> int:
    """Move the trailing rows that made up an error streak out of ``episodes.jsonl`` into ``aborted_batch.jsonl`` (kept).

    A run stopped by the consecutive-error breaker did not measure those episodes -- a wrong key or model name failed them,
    not the model -- so leaving them as saved rows would make a resume skip them and leave phantom errors in the results.
    Setting them aside lets the same command retry them.  Returns how many rows were moved.
    """

    moved = 0
    for path in sorted(set(files)):
        count = sum(1 for f in files if f == path)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        keep, aside = lines[:-count], lines[-count:]
        with (Path(path).parent / "aborted_batch.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n".join(aside) + "\n")
        tmp = Path(path).with_suffix(".jsonl.tmp")
        tmp.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
        tmp.replace(path)
        moved += count
    return moved


def _delta(after: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms_total", "cost", "retried_calls")
    out = {k: after[k] - before[k] for k in keys}
    out["cost"] = round(out["cost"], 6)
    out["currency"] = after["currency"]
    return out


def _manifest_candidate(*, plan: Mapping[str, Any], name: str, spec: Mapping[str, Any], arm: str, reviewer_name: str | None,
                        agent: RemoteConfig, reviewer: RemoteConfig, slug: str, session: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "frontier-run-manifest-v1", "created_at": utc_now(), "slug": slug, "config": name,
        "plan_sha256": plan["plan_sha256"], "model_digest": session["declared_identity_digest"],
        "portable_profile_sha256": session["portable_profile_sha256"], "protocol_sha256": plan["plan_sha256"],
        "defense_version": plan["code"]["digest"], "layers": spec["layers"], "reviewer": reviewer_name,
        "arm": arm, "suite": plan["dataset_key"], "agent_model": agent.model,
        "reviewer_model": reviewer.model if needs_model(reviewer_name) else None,
        "remote_session": session, "environment": environment_record(),
        "limits": {"approval_seconds": REVIEW_LIMITS.approval_seconds, "model_seconds": REVIEW_LIMITS.model_seconds},
    }


def _identity_fields(*, plan: Mapping[str, Any], name: str, slug: str, session: Mapping[str, Any], agent: RemoteConfig,
                     reviewer: RemoteConfig, reviewer_name: str | None, layers: Any) -> dict[str, Any]:
    """The fields every frontier row carries, whatever kind of episode produced it."""

    return {
        "frontier_schema": EPISODE_SCHEMA, "frontier_suite": plan["dataset_key"], "week11_config": name,
        "reviewer": reviewer_name, "plan_sha256": plan["plan_sha256"],
        "layers": None if layers is None else layers.to_dict(),
        "slug": slug, "model_digest": session["declared_identity_digest"], "protocol_sha256": plan["plan_sha256"],
        "defense_version": plan["code"]["digest"], "portable_profile_sha256": session["portable_profile_sha256"],
        "agent_model": agent.model, "reviewer_model": reviewer.model if needs_model(reviewer_name) else None,
    }


def _stop_for_errors(*, settings: RunSettings, summary: ConfigSummary, handle, last_error: Any) -> CampaignStop:
    consecutive = settings.consecutive_errors
    summary.stopped = f"{consecutive} consecutive errors"
    handle.flush()
    moved = set_aside_streak(settings.streak_files)
    settings.consecutive_errors, settings.streak_files = 0, []
    return CampaignStop(
        f"STOPPING: {consecutive} in a row errored (last: {str(last_error)[:200]}). This is "
        f"what a wrong key, a wrong model name or an unsupported parameter looks like. Fix it and rerun the same "
        f"command. Episodes that finished before the failures are kept; the {moved} failed rows were moved to "
        f"aborted_batch.jsonl (not deleted) so the rerun retries them instead of counting them.", 4)


def _budget_stop(*, settings: RunSettings, summary: ConfigSummary, spent: Callable[[], float] | None, currency: str) -> None:
    if settings.max_cost is not None and spent is not None and spent() >= settings.max_cost:
        summary.stopped = f"budget reached ({spent():.4f} >= {settings.max_cost})"
        raise CampaignStop(f"BUDGET REACHED: spent {spent():.4f} of {settings.max_cost} {currency}. "
                           f"{summary.ran} of this configuration are saved; rerun to continue.", 3)


def run_config(*, plan: Mapping[str, Any], name: str, selection: Mapping[str, Any], agent: RemoteConfig,
               reviewer: RemoteConfig, slug: str, out_dir: Path, session: Mapping[str, Any], root: Path = ROOT,
               settings: RunSettings | None = None, agent_usage: UsageLog | None = None,
               reviewer_usage: UsageLog | None = None, log: Callable[[str], None] = print,
               build_llm: Callable[..., Any] = build_remote_llm, spent: Callable[[], float] | None = None) -> ConfigSummary:
    """Run one configuration over the selected scenarios and seeds, appending to ``out_dir/episodes.jsonl``.

    What an episode is depends on the suite's kind (``plan["kind"]``; a plan with none is the original ladder):

    * ``config``    -- ``run_episode`` with the configuration's layers, reviewer and card options (the ladder and its stages);
    * ``defense``   -- the stock arm with one baseline defense applied (``scripts/baseline_defense_lib.py``);
    * ``arms``      -- ``run_episode`` on the stock arm or the protected arm with no model reviewer (the Week 8 comparison);
    * ``attacker``  -- the scripted adaptive attacker, one objective at a time (``run_attacker_config``).
    """

    kind = plan.get("kind", "config")
    if kind == "attacker":
        return run_attacker_config(plan=plan, name=name, selection=selection, agent=agent, slug=slug, out_dir=out_dir,
                                   session=session, root=root, settings=settings, agent_usage=agent_usage, log=log,
                                   build_llm=build_llm, spent=spent)
    settings = settings or RunSettings()
    spec = plan["configs"][name]
    arm = spec["arm"]
    reviewer_name = spec.get("reviewer") if kind == "config" else None
    if kind == "config":
        layers = layers_for(spec)
    else:
        layers = FULL if (kind == "arms" and arm == "protected") else None
    scenarios = {s["scenario_id"]: s for s in load_jsonl(root / plan["dataset"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "episodes.jsonl"
    existing = load_jsonl(output)
    foreign = [r for r in existing if r.get("plan_sha256") != plan["plan_sha256"] or r.get("week11_config") != name]
    if foreign:
        raise ProtocolViolation(f"{output} has {len(foreign)} rows from another plan or configuration; use a new results directory")
    ensure_run_manifest(out_dir / "run_manifest.json", _manifest_candidate(
        plan=plan, name=name, spec=spec, arm=arm, reviewer_name=reviewer_name, agent=agent, reviewer=reviewer, slug=slug,
        session=session), IMMUTABLE)

    done = {episode_key(r) for r in existing}
    pending = [(scenarios[sid], seed) for seed in selection["seeds"] for sid in selection["scenario_ids"]
               if (sid, arm, seed) not in done]
    summary = ConfigSummary(name, already_saved=len(existing), selected=len(pending))
    if not pending:
        log(f"{slug}/{name}: complete ({len(existing)} saved)")
        return summary
    agent_usage = agent_usage if agent_usage is not None else UsageLog(currency=agent.currency)
    reviewer_usage = reviewer_usage if reviewer_usage is not None else UsageLog(currency=reviewer.currency)
    evidence_root = out_dir / "evidence"
    log(f"{slug}/{name}: {len(existing)} saved, {len(pending)} to run")
    started_all = time.monotonic()
    identity = _identity_fields(plan=plan, name=name, slug=slug, session=session, agent=agent, reviewer=reviewer,
                                reviewer_name=reviewer_name, layers=layers)
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for index, (scenario, seed) in enumerate(pending, start=1):
            _budget_stop(settings=settings, summary=summary, spent=spent, currency=agent.currency)
            quarantine_stale_evidence(evidence_token_dir(evidence_root, scenario["scenario_id"], seed, arm))
            agent_before, reviewer_before = agent_usage.totals(), reviewer_usage.totals()
            started = utc_now()
            # the episode's own campaign seed goes to the model, as in every local run (see build_remote_llm)
            llm = build_llm(agent, limits=REVIEW_LIMITS, usage=agent_usage, seed=seed)
            review_llm = (build_llm(reviewer, limits=REVIEW_LIMITS, usage=reviewer_usage, seed=seed)
                          if needs_model(reviewer_name) else None)
            try:
                if kind == "defense":
                    result = run_baseline_defended_episode(
                        scenario, defense=spec["defense"], seed=seed, llm=llm, runtime_limits=REVIEW_LIMITS,
                        evidence_dir=evidence_root).to_dict()
                    result["baseline_defense"] = spec["defense"]
                elif kind == "arms":
                    result = run_episode(scenario, arm=arm, seed=seed, evidence_dir=evidence_root, llm=llm,
                                         runtime_limits=REVIEW_LIMITS).to_dict()
                else:
                    reviewer_fn = make_reviewer(reviewer_name, scenario, review_llm, cap=spec.get("denial_cap"))
                    kwargs: dict[str, Any] = {"post_episode_hook": make_hook(scenario, reviewer_fn)}
                    if arm == "protected":
                        kwargs.update({"layers": layers, "reviewer": reviewer_fn,
                                       "describe_targets": bool(spec.get("describe", False)),
                                       "annotate_sources": bool(spec.get("annotate", False)),
                                       "check_provenance": bool(spec.get("provenance", False))})
                    result = run_episode(scenario, arm=arm, seed=seed, evidence_dir=evidence_root, llm=llm,
                                         runtime_limits=REVIEW_LIMITS, **kwargs).to_dict()
            except Exception as exc:  # noqa: BLE001 - recorded as an error row, never dropped
                result = runner_failure_row(scenario, arm, seed, agent.model, exc)
            result.update(identity)
            result.update({
                "vector": scenario.get("vector"), "goal": scenario.get("goal"), "guidance": scenario.get("guidance"),
                "kind": scenario["kind"], "family": scenario["family"], "scenario_checksum": scenario["checksum"],
                "request_log": list(getattr(getattr(llm, "model_client", None), "request_log", [])),
                "reviewer_request_log": list(getattr(getattr(review_llm, "model_client", None), "request_log", [])),
                "frontier_usage": {"agent": _delta(agent_usage.totals(), agent_before),
                                   "reviewer": _delta(reviewer_usage.totals(), reviewer_before)},
                "started_at": started, "finished_at": utc_now(),
            })
            append_row(handle, result, agent.api_key, reviewer.api_key)
            summary.ran += 1
            failed = bool(result.get("error"))
            summary.errors += failed
            settings.consecutive_errors = settings.consecutive_errors + 1 if failed else 0
            settings.streak_files = settings.streak_files + [output] if failed else []
            eta = (time.monotonic() - started_all) / index * (len(pending) - index)
            log(f"[{index}/{len(pending)}] {name} {scenario['scenario_id']} seed={seed} "
                f"{'ERROR ' + str(result.get('error'))[:110] if failed else 'ok'} {result['latency_ms']}ms eta={eta / 60:.0f}min")
            if settings.max_consecutive_errors and settings.consecutive_errors >= settings.max_consecutive_errors:
                raise _stop_for_errors(settings=settings, summary=summary, handle=handle, last_error=result.get("error"))
    return summary


def run_attacker_config(*, plan: Mapping[str, Any], name: str, selection: Mapping[str, Any], agent: RemoteConfig, slug: str,
                        out_dir: Path, session: Mapping[str, Any], root: Path = ROOT, settings: RunSettings | None = None,
                        agent_usage: UsageLog | None = None, log: Callable[[str], None] = print,
                        build_llm: Callable[..., Any] = build_remote_llm,
                        spent: Callable[[], float] | None = None) -> ConfigSummary:
    """The scripted adaptive attacker against a hosted model: each objective gets up to ``max_attempts`` attempts.

    ``scripts/run_week6_adaptive.run_objective`` is the local loop, unchanged: it re-words the request in a fixed order of
    fifteen ways, tells the attacker how the previous attempt ended, and stops at the first success.  Only the model factory
    differs.  An objective's rows are written when it finishes, so an interruption never leaves a partial curve and the
    objective simply starts again; the budget is checked between objectives, so one objective (at most fifteen attempts) is the
    most that can overshoot ``--max-cost``.
    """

    from scripts import run_week6_adaptive as w6

    settings = settings or RunSettings()
    spec = plan["configs"][name]
    max_attempts = int(plan["max_attempts"])
    objectives = {o["objective_id"]: o for o in load_jsonl(root / plan["dataset"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "episodes.jsonl"
    existing = load_jsonl(output)
    foreign = [r for r in existing if r.get("plan_sha256") != plan["plan_sha256"] or r.get("week11_config") != name]
    if foreign:
        raise ProtocolViolation(f"{output} has {len(foreign)} rows from another plan or configuration; use a new results directory")
    ensure_run_manifest(out_dir / "run_manifest.json", _manifest_candidate(
        plan=plan, name=name, spec=spec, arm=spec["arm"], reviewer_name=None, agent=agent, reviewer=agent, slug=slug,
        session=session), IMMUTABLE)
    finished = {r["objective_id"] for r in existing
                if r.get("attack_success") or int(r.get("attempt", 0)) >= max_attempts or r.get("objective_failed")}
    seed = selection["seeds"][0]
    pending = [objectives[i] for i in selection["scenario_ids"] if i not in finished]
    summary = ConfigSummary(name, already_saved=len(existing), selected=len(pending))
    if not pending:
        log(f"{slug}/{name}: complete ({len(existing)} rows saved)")
        return summary
    agent_usage = agent_usage if agent_usage is not None else UsageLog(currency=agent.currency)
    evidence_root = out_dir / "evidence"
    log(f"{slug}/{name}: {len(finished)} objectives done, {len(pending)} to run (up to {max_attempts} attempts each)")
    started_all = time.monotonic()
    identity = _identity_fields(plan=plan, name=name, slug=slug, session=session, agent=agent, reviewer=agent,
                                reviewer_name=None, layers=FULL)
    extras = {k: identity[k] for k in ("slug", "model_digest", "portable_profile_sha256", "protocol_sha256", "defense_version")}
    zero = _delta(UsageLog(currency=agent.currency).totals(), UsageLog(currency=agent.currency).totals())
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for index, objective in enumerate(pending, start=1):
            _budget_stop(settings=settings, summary=summary, spent=spent, currency=agent.currency)
            quarantine_stale_evidence(evidence_root / objective["objective_id"] / f"seed-{seed}")
            marks: list[dict[str, Any]] = []

            def factory(attempt_seed: int, _marks: list[dict[str, Any]] = marks) -> Any:
                _marks.append(agent_usage.totals())  # one factory call per attempt: the usage between marks is that attempt's
                return build_llm(agent, limits=REVIEW_LIMITS, usage=agent_usage, seed=attempt_seed)

            started = utc_now()
            try:
                rows = w6.run_objective(objective, seed, evidence_root, llm_factory=factory, model_id=agent.model,
                                        row_extras=extras)
            except Exception as exc:  # noqa: BLE001 - recorded as a failed objective, never dropped
                rows = [{"week6_schema": "adaptive-attempt-v1", "objective_id": objective["objective_id"],
                         "objective_checksum": objective["checksum"], "attempt": 1, "seed": seed, "model_id": agent.model,
                         "completed": False, "attack_success": False, "latency_ms": 0, "prompt_tokens": 0,
                         "completion_tokens": 0, "error": f"runner: {type(exc).__name__}: {exc}", "objective_failed": True,
                         **extras}]
            marks.append(agent_usage.totals())
            for position, row in enumerate(rows):
                before, after = marks[min(position, len(marks) - 1)], marks[min(position + 1, len(marks) - 1)]
                row.update(identity)
                row.update({
                    "scenario_id": objective["objective_id"], "arm": spec["arm"], "kind": "attack", "family": "adaptive",
                    "scenario_checksum": objective["checksum"], "started_at": started, "finished_at": utc_now(),
                    "frontier_usage": {"agent": _delta(after, before), "reviewer": zero},
                })
                append_row(handle, row, agent.api_key)
            failed = all(r.get("error") for r in rows)
            won = any(r.get("attack_success") for r in rows)
            summary.ran += 1
            summary.errors += failed
            settings.consecutive_errors = settings.consecutive_errors + 1 if failed else 0
            settings.streak_files = settings.streak_files + [output] * len(rows) if failed else []
            eta = (time.monotonic() - started_all) / index * (len(pending) - index)
            log(f"[{index}/{len(pending)}] {name} {objective['objective_id']} attempts={len(rows)} "
                f"{'ATTACK SUCCEEDED' if won else 'blocked'}{'  ERROR ' + str(rows[-1].get('error'))[:90] if failed else ''} eta={eta / 60:.0f}min")
            if settings.max_consecutive_errors and settings.consecutive_errors >= settings.max_consecutive_errors:
                raise _stop_for_errors(settings=settings, summary=summary, handle=handle, last_error=rows[-1].get("error"))
    return summary


def run_campaign(*, dataset_key: str | None = None, suite_key: str | None = None, agent: RemoteConfig,
                 reviewer: RemoteConfig | None = None, selection_args: Mapping[str, Any] | None = None,
                 results_root: Path = RESULTS_ROOT, root: Path = ROOT, settings: RunSettings | None = None,
                 label: str | None = None, probes: Mapping[str, Any] | None = None,
                 log: Callable[[str], None] = print, build_llm: Callable[..., Any] = build_remote_llm) -> dict[str, Any]:
    """Run the selected part of one suite's frozen plan against ``agent`` (and ``reviewer``, default the same model).

    ``dataset_key`` is the original name of the first parameter and still works; ``suite_key`` is the same thing under the name
    that is accurate now that a suite is not always a dataset.  Results go to ``results_root/<suite>/<model>/<configuration>/``.
    """

    key = suite_key or dataset_key
    if not key:
        raise ProtocolViolation("name the suite to run (suite_key)")
    suite = suite_for(key)
    # a reviewer model only matters where some configuration asks one; elsewhere a different one would only rename the results
    reviewer = (reviewer or agent) if suite.model_reviewer else agent
    digest_now, _ = code_digest(root)
    plan = build_plan(key, root=root)
    selection = select(plan, **(selection_args or {}))
    slug = slug_for(agent, reviewer, label)
    agent_usage, reviewer_usage = UsageLog(currency=agent.currency), UsageLog(currency=reviewer.currency)
    probe_llm = build_llm(agent, limits=REVIEW_LIMITS, usage=UsageLog(currency=agent.currency))
    session = remote_session_record(agent, probe_llm, probes=probes)
    session["reviewer"] = {"model": reviewer.model, "endpoint": reviewer.endpoint, "same_as_agent": reviewer is agent or (
        reviewer.model == agent.model and reviewer.endpoint == agent.endpoint)}
    session["guard_code_digest"] = digest_now

    def local_spent() -> float:
        return agent_usage.totals()["cost"] + reviewer_usage.totals()["cost"]

    def spent() -> float:
        return (settings.carried_cost if settings is not None else 0.0) + local_spent()

    unit = plan.get("unit", "episodes")
    log(f"{key}: {len(selection['scenario_ids'])} {'objectives' if unit == 'objectives' else 'scenarios'} x "
        f"{len(selection['seeds'])} seeds x {len(selection['configs'])} configurations = "
        f"{planned_episodes(selection)} {unit} -> {slug}")
    summaries: list[ConfigSummary] = []

    def totals() -> dict[str, Any]:
        return {"agent": agent_usage.totals(), "reviewer": reviewer_usage.totals(), "spent": round(local_spent(), 6),
                "spent_total": round(spent(), 6), "currency": agent.currency}

    try:
        for name in selection["configs"]:
            out_dir = Path(results_root) / key / slug / name
            summaries.append(run_config(
                plan=plan, name=name, selection=selection, agent=agent, reviewer=reviewer, slug=slug, out_dir=out_dir,
                session=session, root=root, settings=settings, agent_usage=agent_usage, reviewer_usage=reviewer_usage,
                log=log, build_llm=build_llm, spent=spent))
    except CampaignStop as stop:
        stop.usage = totals()  # what was spent before stopping, for the message the CLI prints
        raise
    finally:
        if settings is not None:
            settings.carried_cost += local_spent()
    return {"dataset": key, "suite": key, "unit": unit, "slug": slug, "plan_sha256": plan["plan_sha256"],
            "results_dir": str(Path(results_root) / key / slug), "configs": [s.to_dict() for s in summaries],
            "usage": totals()}
