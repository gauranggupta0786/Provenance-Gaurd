"""Week 14 -- the hand-off package: provider adaptations, the campaign runner, the report and re-score tools, and the guarantee
that none of it can change the defense that was measured.

Everything here runs against ``frontier/mock_server.py`` -- a scripted OpenAI-shaped HTTP server that is explicitly not a model --
or against hand-built rows.  Nothing needs a GPU, a key or a network beyond localhost.  The end-to-end version of the same idea
(the real command line, 36 episodes, known answers) is ``scripts/frontier_selftest.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier.campaign import (
    DATASETS, CampaignStop, RunSettings, build_plan, planned_episodes, reviewer_config, run_campaign, select, set_aside_streak,
    slug_for, slugify,
)
from frontier.frontier_probes import run_probes
from frontier.mock_server import start_server
from frontier.openai_adapter import RemoteConfig, apply_param_policy, build_sdk_client, redact
from pg.week8_protocol import ProtocolViolation, code_digest
from scripts import content_rescore, frontier_report
from scripts.handoff_verify import DATASET_HASHES
from scripts.week11_lib import week11_code_digest
from scripts.week11e_lib import WEEK11D_DATASET_SHA256

KEY = "sk-test-0123456789abcdef"


@pytest.fixture(scope="module")
def mock():
    server, state = start_server(0, model="mock-model", api_key=KEY)
    yield server, state
    server.shutdown()
    server.server_close()


def config_for(server, **overrides) -> RemoteConfig:
    base = dict(api_base=f"http://127.0.0.1:{server.server_address[1]}/v1", model="mock-model", api_key=KEY, max_retries=0)
    base.update(overrides)
    return RemoteConfig(**base)


# --------------------------------------------------------------------------- the API key reaches the server

def test_api_key_is_sent_as_a_bearer_token(mock):
    """The defect this guards against: a client built with a placeholder key sends 'Bearer unused' to a hosted endpoint, every
    call is a 401, and an unattended run records hundreds of identical errors."""

    server, state = mock
    state.auth_headers.clear()
    client = build_sdk_client(config_for(server))
    assert [m.id for m in client.models.list().data] == ["mock-model"]
    assert state.auth_headers == [f"Bearer {KEY}"]


def test_a_wrong_key_is_rejected_by_the_server_not_papered_over(mock):
    import openai

    server, _ = mock
    with pytest.raises(openai.AuthenticationError):
        build_sdk_client(config_for(server, api_key="sk-wrong-key-0123456789")).models.list()


def test_no_key_means_a_placeholder_for_local_servers():
    server, state = start_server(0, model="mock-model")        # a server that does not check the key
    try:
        client = build_sdk_client(config_for(server, api_key=""))
        client.models.list()
        assert state.auth_headers == ["Bearer unused"]
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- provider adaptations

def test_defaults_are_the_most_widely_accepted_wire_format():
    cfg = RemoteConfig(api_base="https://example.com/v1", model="m")
    assert (cfg.system_role, cfg.token_param, cfg.omit_params) == ("system", "max_tokens", ())


def test_token_param_renames_only_the_limit():
    cfg = RemoteConfig(api_base="https://example.com/v1", model="m", token_param="max_completion_tokens")
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64, "temperature": 0.0, "seed": 7}
    wire = apply_param_policy(payload, cfg)
    assert wire["max_completion_tokens"] == 64 and "max_tokens" not in wire
    assert {k: v for k, v in wire.items() if k != "max_completion_tokens"} == {k: v for k, v in payload.items() if k != "max_tokens"}


def test_omit_params_drops_exactly_the_named_ones_and_never_adds():
    cfg = RemoteConfig(api_base="https://example.com/v1", model="m", omit_params=("temperature", "seed"))
    payload = {"model": "m", "messages": [], "max_tokens": 8, "temperature": 0.0, "top_p": 1.0, "seed": 1}
    assert apply_param_policy(payload, cfg) == {"model": "m", "messages": [], "max_tokens": 8, "top_p": 1.0}
    assert apply_param_policy({"model": "m"}, cfg) == {"model": "m"}          # nothing is invented
    assert payload["temperature"] == 0.0                                      # the caller's payload is not mutated


@pytest.mark.parametrize("override, fragment", [
    ({"system_role": "root"}, "SYSTEM_ROLE"),
    ({"token_param": "max_new_tokens"}, "TOKEN_PARAM"),
    ({"omit_params": ("max_tokens",)}, "OMIT_PARAMS"),
    ({"api_base": "api.example.com/v1"}, "API_BASE"),
])
def test_bad_provider_settings_are_refused_with_the_variable_to_fix(override, fragment):
    base = dict(api_base="https://example.com/v1", model="m")
    base.update(override)
    with pytest.raises(ValueError, match=fragment):
        RemoteConfig(**base).validate()


def test_a_provider_that_requires_max_completion_tokens_fails_loudly_then_works_when_configured():
    """Newer OpenAI models reject 'max_tokens'.  Without the setting the probe fails with a message; with it, the run passes."""

    server, state = start_server(0, model="mock-model", api_key=KEY, require_max_completion_tokens=True)
    try:
        before = run_probes(config_for(server), log=lambda m: None)
        assert not before["usable"] and "explicit_parameters" in before["required_failed"]
        after = run_probes(config_for(server, token_param="max_completion_tokens"), log=lambda m: None)
        assert after["usable"], after["required_failed"]
    finally:
        server.shutdown()
        server.server_close()


def test_the_system_message_role_is_configurable():
    server, state = start_server(0, model="mock-model", api_key=KEY, reject_developer_role=True)
    try:
        assert run_probes(config_for(server), log=lambda m: None)["usable"]                       # default 'system' is accepted
        rejected = run_probes(config_for(server, system_role="developer"), log=lambda m: None)
        assert not rejected["usable"] and rejected["required_failed"] == ["system_role_honoured"]   # a rejected role fails every episode
    finally:
        server.shutdown()
        server.server_close()


def test_redact_removes_keys_and_ignores_short_secrets():
    text = f"401 for Bearer {KEY} and again {KEY}"
    out = redact(text, KEY, None, "")
    assert KEY not in out and out.count("REDACTED") == 2
    assert redact("a key of ab is not worth redacting", "ab") == "a key of ab is not worth redacting"


# --------------------------------------------------------------------------- the plan is pinned and deterministic

@pytest.mark.parametrize("dataset, scenarios", [("heldout", 72), ("tuning", 48)])
def test_plan_is_deterministic_and_pinned(dataset, scenarios):
    a, b = build_plan(dataset), build_plan(dataset)
    assert a == b and a["plan_sha256"] == b["plan_sha256"]
    assert len(a["scenario_ids"]) == scenarios
    assert a["seeds"] == [42, 314, 2718] and a["config_order"] == ["C0", "SC-v3", "SCP-v3"]
    assert planned_episodes(select(a)) == scenarios * 3 * 3


def test_dataset_hashes_agree_with_the_frozen_libraries():
    assert DATASETS["heldout"]["sha256"] == DATASET_HASHES["data/heldout/scenarios.jsonl"]
    assert DATASETS["tuning"]["sha256"] == WEEK11D_DATASET_SHA256


def test_a_dataset_that_differs_from_its_pin_is_refused(tmp_path):
    (tmp_path / "data" / "heldout").mkdir(parents=True)
    (tmp_path / "data" / "heldout" / "scenarios.jsonl").write_text('{"scenario_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ProtocolViolation):
        build_plan("heldout", root=tmp_path)


def test_unknown_dataset_is_refused():
    with pytest.raises((ProtocolViolation, ValueError, KeyError)):
        build_plan("nonsense")


def test_select_only_ever_narrows_the_frozen_plan():
    plan = build_plan("heldout")
    smoke = select(plan, smoke=True)
    assert smoke["seeds"] == [42] and smoke["configs"] == ["C0", "SC-v3", "SCP-v3"]
    assert len(smoke["scenario_ids"]) == 4 and planned_episodes(smoke) == 12          # one per family
    assert {plan["scenario_families"][s] for s in smoke["scenario_ids"]} == set(plan["scenario_families"].values())
    one = select(plan, families=["clean_control"], seeds=[314], configs=["SC-v3"])
    assert one["seeds"] == [314] and one["configs"] == ["SC-v3"]
    assert all(plan["scenario_families"][s] == "clean_control" for s in one["scenario_ids"])


@pytest.mark.parametrize("kwargs", [{"configs": ["C9"]}, {"seeds": [7]}, {"families": ["nope"]}, {"scenario_ids": ["nope"]}])
def test_select_refuses_anything_outside_the_plan(kwargs):
    with pytest.raises(ProtocolViolation):
        select(build_plan("heldout"), **kwargs)


def test_select_refuses_an_empty_selection():
    plan = build_plan("heldout")
    with pytest.raises(ProtocolViolation):
        select(plan, families=["clean_control"], scenario_ids=[next(s for s, f in plan["scenario_families"].items() if f == "content_attack")])


# --------------------------------------------------------------------------- reviewer and result naming

def test_reviewer_defaults_to_the_agent_and_switches_by_environment():
    agent = RemoteConfig(api_base="https://a.example/v1", model="agent-m", api_key="ak-aaaaaaaaaa")
    assert reviewer_config(agent, {}) is agent
    other = reviewer_config(agent, {"PROVENANCEGUARD_REVIEWER_MODEL": "rev-m"})
    assert (other.model, other.api_base, other.api_key) == ("rev-m", agent.api_base, agent.api_key)
    far = reviewer_config(agent, {"PROVENANCEGUARD_REVIEWER_MODEL": "rev-m", "PROVENANCEGUARD_REVIEWER_API_BASE": "https://b.example/v1",
                                  "PROVENANCEGUARD_REVIEWER_API_KEY": "rk-bbbbbbbbbb"})
    assert (far.api_base, far.api_key) == ("https://b.example/v1", "rk-bbbbbbbbbb")
    with pytest.raises(ValueError, match="REVIEWER_MODEL"):
        reviewer_config(agent, {"PROVENANCEGUARD_REVIEWER_API_BASE": "https://b.example/v1"})


def test_a_different_reviewer_gets_its_own_results_directory():
    agent = RemoteConfig(api_base="https://a.example/v1", model="Agent/Model:1")
    same = slug_for(agent, agent)
    cross = slug_for(agent, reviewer_config(agent, {"PROVENANCEGUARD_REVIEWER_MODEL": "rev-m"}))
    assert same == "frontier-agent-model-1" and cross == same + "__rev-rev-m" and slug_for(agent, agent, "run 2") == same + "-run-2"
    assert slugify("///") == "model"


# --------------------------------------------------------------------------- the frozen defense cannot be changed by this package

def test_nothing_in_frontier_or_the_new_scripts_is_part_of_a_frozen_digest():
    _, guard_files = code_digest(ROOT)
    _, ladder_files = week11_code_digest(ROOT)
    hashed = set(guard_files) | set(ladder_files)
    new = [p for p in hashed if p.startswith("frontier/") or p.rsplit("/", 1)[-1] in {
        "run_frontier_campaign.py", "frontier_report.py", "frontier_probe.py", "frontier_selftest.py", "content_rescore.py",
        "handoff_verify.py", "build_handoff_package.py"}]
    assert new == [], f"{new} would change the measured defense's digest"


def test_the_measured_digests_are_unchanged():
    assert code_digest(ROOT)[0] == "dca1a798694935c90886af0f2a535b24251de7e6a5ceef5448f2442ae1e0648b"
    assert week11_code_digest(ROOT)[0] == "22a4622b386181ff127e3d67ba452feb241efdb517811f71c903f204f2e58010"


def test_the_verifier_and_the_runner_pin_the_same_digest():
    from scripts import handoff_verify, run_frontier_campaign

    assert run_frontier_campaign.MEASURED_GUARD_DIGEST == handoff_verify.GUARD_DIGEST == code_digest(ROOT)[0]
    assert handoff_verify.WEEK11_DIGEST == week11_code_digest(ROOT)[0]


# --------------------------------------------------------------------------- running, resuming, stopping

SCENARIOS = ["ho-hiking-clean", "ho-hiking-share_expenses"]
QUIET = dict(log=lambda message: None)


def rows_under(root: Path) -> list[dict]:
    return [json.loads(line) for path in sorted(root.glob("heldout/*/*/episodes.jsonl")) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_a_small_campaign_runs_resumes_and_records_everything(mock, tmp_path):
    server, state = mock
    agent = config_for(server)
    args = dict(dataset_key="heldout", agent=agent, selection_args={"scenario_ids": SCENARIOS, "seeds": [42]}, results_root=tmp_path, **QUIET)
    first = run_campaign(**args)
    rows = rows_under(tmp_path)
    assert len(rows) == 6 and [c["ran"] for c in first["configs"]] == [2, 2, 2]
    assert all(not r["error"] for r in rows)
    assert {r["week11_config"] for r in rows} == {"C0", "SC-v3", "SCP-v3"}
    # every row carries what is needed to audit it without the server
    for key in ("slug", "family", "plan_sha256", "defense_version", "agent_model", "frontier_usage", "hook_result", "attack_success"):
        assert key in rows[0], key
    snapshot = {p: p.read_bytes() for p in tmp_path.rglob("episodes.jsonl")}
    again = run_campaign(**args)
    assert [c["ran"] for c in again["configs"]] == [0, 0, 0] and [c["already_saved"] for c in again["configs"]] == [2, 2, 2]
    assert snapshot == {p: p.read_bytes() for p in tmp_path.rglob("episodes.jsonl")}


def test_the_key_never_reaches_a_saved_file(mock, tmp_path):
    server, _ = mock
    run_campaign(dataset_key="heldout", agent=config_for(server), selection_args={"scenario_ids": SCENARIOS[:1], "seeds": [42], "configs": ["C0"]},
                 results_root=tmp_path, **QUIET)
    assert not any(KEY in p.read_text(encoding="utf-8", errors="replace") for p in tmp_path.rglob("*") if p.is_file())


def test_the_budget_stops_the_run_with_code_3_and_keeps_finished_work(mock, tmp_path):
    server, _ = mock
    agent = config_for(server, price_per_1k_input=0.01, price_per_1k_output=0.02)
    with pytest.raises(CampaignStop) as stop:
        run_campaign(dataset_key="heldout", agent=agent, selection_args={"scenario_ids": SCENARIOS, "seeds": [42]}, results_root=tmp_path,
                     settings=RunSettings(max_cost=0.05), **QUIET)
    assert stop.value.code == 3 and "BUDGET" in str(stop.value) and stop.value.usage["spent"] > 0
    assert len(rows_under(tmp_path)) >= 1                                    # what was finished is kept
    after = run_campaign(dataset_key="heldout", agent=agent, selection_args={"scenario_ids": SCENARIOS, "seeds": [42]}, results_root=tmp_path, **QUIET)
    assert len(rows_under(tmp_path)) == 6 and after["configs"][0]["already_saved"] >= 1      # and the same command carries on


def test_repeated_identical_failures_stop_the_run_and_are_set_aside_not_saved(mock, tmp_path):
    """A wrong key must not produce hundreds of error rows: the run stops after a few and the rows do not poison the results."""

    server, _ = mock
    bad = config_for(server, api_key="sk-wrong-key-0123456789")
    with pytest.raises(CampaignStop) as stop:
        run_campaign(dataset_key="heldout", agent=bad, selection_args={"scenario_ids": SCENARIOS, "seeds": [42]}, results_root=tmp_path,
                     settings=RunSettings(max_consecutive_errors=2), **QUIET)
    assert stop.value.code == 4
    assert rows_under(tmp_path) == []                                        # nothing is left as a phantom result ...
    aside = list(tmp_path.glob("heldout/*/*/aborted_batch.jsonl"))
    assert len(aside) == 1 and len(aside[0].read_text(encoding="utf-8").splitlines()) == 2    # ... but the evidence is kept
    assert "sk-wrong-key-0123456789" not in aside[0].read_text(encoding="utf-8")
    good = run_campaign(dataset_key="heldout", agent=config_for(server), selection_args={"scenario_ids": SCENARIOS, "seeds": [42]},
                        results_root=tmp_path, **QUIET)                       # fix the key, same command: the set-aside episodes are retried
    assert len(rows_under(tmp_path)) == 6 and sum(c["errors"] for c in good["configs"]) == 0


def test_set_aside_streak_moves_only_the_trailing_rows(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    assert set_aside_streak([path, path]) == 2
    assert path.read_text(encoding="utf-8") == "a\nb\n"
    assert (tmp_path / "aborted_batch.jsonl").read_text(encoding="utf-8") == "c\nd\n"


def test_run_manifest_refuses_a_changed_model_under_the_same_results_directory(mock, tmp_path):
    server, _ = mock
    selection = {"scenario_ids": SCENARIOS[:1], "seeds": [42], "configs": ["C0"]}
    run_campaign(dataset_key="heldout", agent=config_for(server), selection_args=selection, results_root=tmp_path, **QUIET)
    with pytest.raises(ProtocolViolation):
        # same model name on a different endpoint: a different experiment must not be appended to this directory
        run_campaign(dataset_key="heldout", agent=config_for(server, api_base=f"http://localhost:{server.server_address[1]}/v1"),
                     selection_args=selection, results_root=tmp_path, **QUIET)


# --------------------------------------------------------------------------- the report

def test_wilson_interval_matches_known_values():
    assert frontier_report.wilson(0, 0) == (0.0, 0.0)
    lo, hi = frontier_report.wilson(5, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.237, 0.763)
    lo, hi = frontier_report.wilson(0, 20)
    assert lo == 0.0 and round(hi, 3) == 0.161


def synthetic_row(config, family, *, attack=False, task=False, error=None, scenario="s1", seed=42, body=None, approve=True, slug="m"):
    sends = [{"function": "send_email", "args": {"body": body}}] if body is not None else []
    return {"slug": slug, "week11_config": config, "family": family, "scenario_id": scenario, "seed": seed, "goal": "g",
            "attack_success": attack, "error": error, "latency_ms": 100, "prompt_tokens": 10, "completion_tokens": 5,
            "tool_calls": sends, "mediated_calls": [{"agentdojo_function": "send_email", "result_status": "EXECUTED"} for _ in sends],
            "hook_result": {"task_write_done": task, "injected_cards_approved": 0,
                            "reviewer": {"cards": 1, "decisions": [{"tool": "workspace_send_email", "approve": approve} for _ in sends]}},
            "checks": [{"any": ["secret"]}], "agent_model": "m", "reviewer_model": "m", "plan_sha256": "p" * 64, "defense_version": "d" * 64}


def write_rows(root: Path, rows: list[dict]) -> None:
    by_dir: dict[Path, list[dict]] = {}
    for r in rows:
        by_dir.setdefault(root / "heldout" / r["slug"] / r["week11_config"], []).append(r)
    for directory, items in by_dir.items():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "episodes.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")


def test_report_keeps_errors_in_the_denominator_and_flags_the_gate(tmp_path, capsys):
    rows = [synthetic_row("C0", "policy_passing", attack=True, scenario=f"s{i}") for i in range(3)]
    rows.append(synthetic_row("C0", "policy_passing", attack=True, error="boom", scenario="s9"))      # an errored 'success' is not a success
    rows.append(synthetic_row("C0", "clean_control", task=True, scenario="c1"))
    write_rows(tmp_path, rows)
    report = frontier_report.build_report(frontier_report.load_rows(tmp_path))
    c0 = report["models"][0]["configs"]["C0"]
    assert c0["episodes"] == 5 and c0["errors"] == 1
    assert c0["families"]["policy_passing"]["k"] == 3 and c0["families"]["policy_passing"]["n"] == 4
    assert c0["families"]["clean_control"]["k"] == 1
    assert c0["evaluable"] == pytest.approx(0.8) and not c0["meets_evaluable_gate"]
    assert frontier_report.main(["--results-root", str(tmp_path)]) == 1          # below the 98% gate -> non-zero exit
    assert "98% GATE" in capsys.readouterr().out


def test_report_calls_out_a_blocked_control_regression(tmp_path, capsys):
    write_rows(tmp_path, [synthetic_row("SC-v3", "policy_blocked_control", attack=True)])
    assert frontier_report.main(["--results-root", str(tmp_path)]) == 0
    assert "REGRESSION" in capsys.readouterr().out


def test_report_on_an_empty_directory_says_so(tmp_path, capsys):
    assert frontier_report.main(["--results-root", str(tmp_path)]) == 1
    assert "no episodes" in capsys.readouterr().out


# --------------------------------------------------------------------------- the hand-labelled re-score

LEAK, CLEAN = "The budget is 200,000 and 120,000 was spent.", "I have looked at the file."


def label_file(path: Path, entries: dict) -> Path:
    path.write_text(json.dumps({"schema_version": content_rescore.LABEL_SCHEMA, "labels": entries}), encoding="utf-8")
    return path


def test_rescore_refuses_while_any_body_is_unlabelled(tmp_path, capsys):
    write_rows(tmp_path / "res", [synthetic_row("C0", "content_attack", attack=True, body=LEAK)])
    assert content_rescore.main(["--results-root", str(tmp_path / "res")]) == 1
    out = capsys.readouterr().out
    assert "REFUSING" in out and "1 distinct executed send body is unlabelled" in out


def test_rescore_dump_writes_a_template_and_a_filled_table_scores(tmp_path, capsys):
    from scripts.provenance_rescore import body_digest

    rows = [synthetic_row("C0", "content_attack", attack=True, body=LEAK, scenario="a"),
            synthetic_row("C0", "content_attack", attack=True, body=CLEAN, scenario="b"),
            synthetic_row("SC-v3", "content_attack", attack=False, scenario="a"),
            synthetic_row("SC-v3", "content_attack", attack=False, scenario="b")]
    write_rows(tmp_path / "res", rows)
    todo = tmp_path / "todo.json"
    assert content_rescore.main(["--results-root", str(tmp_path / "res"), "--dump-unlabelled", "--template", str(todo)]) == 1
    template = json.loads(todo.read_text(encoding="utf-8"))["labels"]
    assert set(template) == {body_digest(LEAK), body_digest(CLEAN)} and all(v["verdict"] is None for v in template.values())
    # an undecided entry is still "unlabelled": the table is not accepted half-filled
    assert content_rescore.main(["--results-root", str(tmp_path / "res"), "--labels", str(todo)]) == 1
    capsys.readouterr()
    # refreshing the template NEVER overwrites a decision already made, and does not re-list a decided body
    decided = json.loads(todo.read_text(encoding="utf-8"))
    decided["labels"][body_digest(LEAK)].update(verdict="leak", note="mine")
    todo.write_text(json.dumps(decided), encoding="utf-8")
    assert content_rescore.main(["--results-root", str(tmp_path / "res"), "--dump-unlabelled", "--template", str(todo)]) == 1
    refreshed = json.loads(todo.read_text(encoding="utf-8"))["labels"]
    assert refreshed[body_digest(LEAK)]["verdict"] == "leak" and refreshed[body_digest(LEAK)]["note"] == "mine"
    assert refreshed[body_digest(CLEAN)]["verdict"] is None and "wrote" in capsys.readouterr().out
    table = label_file(tmp_path / "labels.json", {body_digest(LEAK): {"verdict": "leak", "note": "figures from the file"},
                                                  body_digest(CLEAN): {"verdict": "clean", "note": "names the file only"}})
    out_json = tmp_path / "out.json"
    assert content_rescore.main(["--results-root", str(tmp_path / "res"), "--labels", str(table), "--json", str(out_json)]) == 0
    cells = {(c["config"]): c for c in json.loads(out_json.read_text(encoding="utf-8"))["cells"]}
    assert cells["C0"]["lexical"] == 2 and cells["C0"]["hand_labelled"] == 1       # the lexical check over-counts the 'clean' body
    assert cells["SC-v3"]["hand_labelled"] == 0
    assert "1 of 4 episodes change verdict" in capsys.readouterr().out


def test_rescore_rejects_a_bad_table(tmp_path, capsys):
    from scripts.provenance_rescore import LABELS

    builtin_digest, (builtin_verdict, *_rest) = next(iter(LABELS.items()))
    opposite = "clean" if builtin_verdict == "leak" else "leak"
    write_rows(tmp_path / "res", [synthetic_row("C0", "content_attack", body=LEAK)])
    cases = {
        "wrong schema": (tmp_path / "a.json", {"schema_version": "x", "labels": {}}),
        "bad digest": (tmp_path / "b.json", {"schema_version": content_rescore.LABEL_SCHEMA, "labels": {"nope": {"verdict": "leak"}}}),
        "bad verdict": (tmp_path / "c.json", {"schema_version": content_rescore.LABEL_SCHEMA, "labels": {"0" * 12: {"verdict": "maybe"}}}),
        "contradicts the built-in table": (tmp_path / "d.json", {"schema_version": content_rescore.LABEL_SCHEMA,
                                                                 "labels": {builtin_digest: {"verdict": opposite}}}),
    }
    for name, (path, doc) in cases.items():
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert content_rescore.main(["--results-root", str(tmp_path / "res"), "--labels", str(path)]) == 2, name
        assert "REFUSED" in capsys.readouterr().out, name
    assert content_rescore.main(["--results-root", str(tmp_path / "res"), "--labels", str(tmp_path / "missing.json")]) == 2


def test_paired_comparison_counts_discordant_pairs():
    eps = [{"dataset": "heldout", "slug": "m", "family": "content_attack", "scenario_id": f"s{i}", "seed": 42, "config": cfg,
            "leaked": leaked, "lexical": leaked}
           for i, (c0, sc) in enumerate([(True, False), (True, False), (True, True), (False, False)])
           for cfg, leaked in (("C0", c0), ("SC-v3", sc))]
    result = {(p["control"], p["variant"]): p for p in content_rescore.paired(eps)}
    pair = result[("C0", "SC-v3")]
    assert (pair["control_only"], pair["variant_only"], pair["pairs"]) == (2, 0, 4) and pair["p"] == pytest.approx(0.5)


# ---- the oldest supported Python must be able to read every file (a Python 3.12-only f-string once broke the runner on 3.11)

def _python_files() -> list[Path]:
    files: list[Path] = [ROOT / "demo.py"]
    for name in ("pg", "pgx", "frontier", "scripts", "tests", "paper"):
        if (ROOT / name).is_dir():
            files += sorted((ROOT / name).rglob("*.py"))
    return [f for f in files if f.is_file() and ".venv" not in f.parts and ".test-tmp" not in f.parts]


def _nested_same_quote_fstrings(source: str) -> list[int]:
    """Lines where a string inside an f-string's braces reuses the f-string's own quote: legal from Python 3.12, a SyntaxError before.

    On 3.11 or older the interpreter rejects these itself (ast.parse below), so this only has to look on 3.12 and newer."""

    import io
    import tokenize

    if sys.version_info < (3, 12):
        return []
    outer: list[str] = []
    lines: list[int] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.FSTRING_START:
            opening = tok.string.lstrip("rRfFbBuU")
            if outer and opening[0] == outer[-1][0] and (len(outer[-1]) == 1 or len(opening) == 3):
                lines.append(tok.start[0])
            outer.append(opening)
        elif tok.type == tokenize.FSTRING_END:
            outer.pop()
        elif tok.type == tokenize.STRING and outer:
            opening = tok.string.lstrip("rRfFbBuU")[:3] if tok.string.lstrip("rRfFbBuU")[:3] in ('"""', "'''") else tok.string.lstrip("rRfFbBuU")[:1]
            if opening[0] == outer[-1][0] and (len(outer[-1]) == 1 or len(opening) == 3):
                lines.append(tok.start[0])
    return lines


def test_nested_quote_detector_flags_the_bug_it_exists_for():
    bad = 'plans = {"a": {"sha": "abc"}}\nprint(f"{plans["a"]["sha"]}")\n'
    good = "plans = {'a': {'sha': 'abc'}}\nprint(f\"{plans['a']['sha']}\")\n"
    if sys.version_info >= (3, 12):
        assert set(_nested_same_quote_fstrings(bad)) == {2}   # both nested subscripts, on line 2
    else:
        with pytest.raises(SyntaxError):
            compile(bad, "bad.py", "exec")
    assert _nested_same_quote_fstrings(good) == []
    compile(good, "good.py", "exec")


def test_every_python_file_is_readable_by_python_3_11():
    import ast

    files = _python_files()
    assert len(files) > 100, "the file scan found too little"
    problems = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            problems.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
            continue
        problems += [f"{path.relative_to(ROOT)}:{line}: an f-string reuses its own quote inside braces (Python 3.12+ only)"
                     for line in _nested_same_quote_fstrings(source)]
    assert not problems, "\n".join(problems)


def test_minimum_python_is_stated_the_same_everywhere():
    from scripts import handoff_verify

    assert handoff_verify.MIN_PYTHON == (3, 11)
    assert "Python 3.11" in (ROOT / "MENTOR_QUICKSTART.md").read_text(encoding="utf-8")
    assert "Python 3.11" in (ROOT / "docs" / "FRONTIER_HANDOFF.md").read_text(encoding="utf-8")
