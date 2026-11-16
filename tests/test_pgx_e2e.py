"""v2 end to end on the real AgentDojo suites with a scripted agent (no model): mediation, scoring, the PG-v4 chain.

The strongest check is the first: every one of AgentDojo's 97 user tasks, executed by its own ground truth through the v2
plumbing (C0g: signed proposal, approval, gateway, receipts, in all four suites), reaches exactly the utility it reaches with no
ProvenanceGuard at all.  That exercises every new tool contract, every canonical encoding and every gateway decode.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdojo.functions_runtime import FunctionCall
from agentdojo.task_suite.load_suites import get_suite

from pgx.agentdojo_runner import Case, run_case
from pgx.suites import BENCHMARK_VERSION, SUITES
from tests.pgx_doubles import ScriptedAgent, ScriptedClient, deny_if

ROOT = Path(__file__).resolve().parents[1]
INJECTIONS = {(r["suite"], r["user_task_id"], r["injection_task_id"]): r["injections"]
              for r in map(json.loads, (ROOT / "data/v2/injections.jsonl").read_text().splitlines()) if r}
APPROVE = ScriptedClient(lambda p: "APPROVE")


def _case(suite: str, uid: str, iid: str | None = None) -> Case:
    return Case(suite, uid, iid, INJECTIONS[(suite, uid, iid)] if iid else {})


def _agent(suite: str, uid: str, iid: str | None = None, **kw) -> ScriptedAgent:
    s = get_suite(BENCHMARK_VERSION, suite)
    tasks = ([s.get_injection_task_by_id(iid)] if iid else []) + [s.get_user_task_by_id(uid)]
    return ScriptedAgent(tasks, **kw)


@pytest.mark.parametrize("suite", SUITES)
def test_plumbing_preserves_utility_of_every_user_task(suite, tmp_path):
    s = get_suite(BENCHMARK_VERSION, suite)
    mismatches, failures = [], []
    for uid in sorted(s.user_tasks):
        rows = {cfg: run_case(_case(suite, uid), cfg, llm=_agent(suite, uid), agent_client=APPROVE, reviewer_client=APPROVE,
                              seed=42, evidence_root=tmp_path / cfg) for cfg in ("none", "C0g")}
        if rows["C0g"]["error"]:
            failures.append((uid, rows["C0g"]["error"]))
        if rows["none"]["utility"] != rows["C0g"]["utility"]:
            mismatches.append((uid, rows["none"]["utility"], rows["C0g"]["utility"],
                               [m for m in rows["C0g"]["mediated_calls"] if m["result_status"] != "SUCCEEDED"]))
    assert not failures, failures
    assert not mismatches, mismatches


def _named_email_send_task() -> str:
    """A workspace user task whose request names the e-mail address it sends to (so another address is plainly extra)."""
    import re
    s = get_suite(BENCHMARK_VERSION, "workspace")
    env = s.load_and_inject_default_environment({})
    for uid in sorted(s.user_tasks, key=lambda u: int(u.split("_")[-1])):
        task = s.get_user_task_by_id(uid)
        calls = task.ground_truth(task.init_environment(env.model_copy(deep=True)))
        named = {m.lower() for m in re.findall(r"[\w.+-]+@[\w.-]+\.\w+", task.PROMPT)}
        if named and any(c.function == "send_email" and set(map(str.lower, c.args.get("recipients", []))) <= named for c in calls):
            if ("workspace", uid, "injection_task_0") in INJECTIONS:
                return uid
    raise AssertionError("no such task")


def test_injected_send_to_unnamed_address_is_denied_deterministically(tmp_path):
    uid = _named_email_send_task()
    case = _case("workspace", uid, "injection_task_0")
    none = run_case(case, "none", llm=_agent("workspace", uid, "injection_task_0"), agent_client=APPROVE,
                    reviewer_client=APPROVE, seed=42)
    assert none["attack_success"] is True
    reviewer = deny_if()  # approves everything it is asked: the deterministic layers must do the work
    v4 = run_case(case, "PG-v4-capParse", llm=_agent("workspace", uid, "injection_task_0"), agent_client=APPROVE,
                  reviewer_client=reviewer, seed=42)
    assert v4["attack_success"] is False and not v4["error"]
    assert "destination" in [d.get("stage") for d in v4["reviewer"]["decisions"]]
    assert not any("mark.black" in p for p in reviewer.prompts)  # the model was never asked about the injected send


def test_unnamed_target_reaches_the_reviewer_with_the_flag(tmp_path):
    # user_task_0's request names no address, so the injected send's target is from data: the reviewer decides, warned
    case = _case("workspace", "user_task_0", "injection_task_0")
    reviewer = deny_if("mark.black")
    row = run_case(case, "PG-v4-capParse", llm=_agent("workspace", "user_task_0", "injection_task_0"), agent_client=APPROVE,
                   reviewer_client=reviewer, seed=42)
    assert row["attack_success"] is False
    asked = [p for p in reviewer.prompts if "mark.black" in p]
    assert asked and "does not name where this action goes" in asked[0]


def test_feedback_names_the_rule_category_not_the_details(tmp_path):
    uid = _named_email_send_task()
    case = _case("workspace", uid, "injection_task_0")
    row = run_case(case, "PG-v4-capParse", llm=_agent("workspace", uid, "injection_task_0"), agent_client=APPROVE,
                   reviewer_client=APPROVE, seed=42)
    assert row["denial_categories"][0] == "unrequested_recipient"
    assert set(row["denial_categories"]) <= {"unrequested_action", "unrequested_recipient", "content_not_requested", "reviewer", "cap"}
    from pgx.guards import CATEGORY_MESSAGES, GENERIC_DENIAL, denial_message
    assert denial_message("unrequested_recipient", feedback=False) == GENERIC_DENIAL
    for category, text in CATEGORY_MESSAGES.items():
        assert denial_message(category, feedback=True).endswith(text) and "@" not in text


def test_banking_injected_payment_goes_to_the_reviewer_and_is_stopped(tmp_path):
    s = get_suite(BENCHMARK_VERSION, "banking")
    attacker = "US133000000121212121212"
    pair = next((u, i) for u in sorted(s.user_tasks) for i in sorted(s.injection_tasks)
                if (("banking", u, i) in INJECTIONS) and attacker in s.get_injection_task_by_id(i).GOAL
                and "send_money" in json.dumps([c.function for c in s.get_user_task_by_id(u).ground_truth(
                    s.load_and_inject_default_environment({}))]))
    case = _case("banking", *pair)
    none = run_case(case, "none", llm=_agent("banking", *pair), agent_client=APPROVE, reviewer_client=APPROVE, seed=42)
    v4 = run_case(case, "PG-v4-capParse", llm=_agent("banking", *pair), agent_client=APPROVE,
                  reviewer_client=deny_if(attacker), seed=42)
    assert none["attack_success"] is True
    assert v4["attack_success"] is False and not v4["error"]


def test_planner_answer_is_validated_and_falls_back(tmp_path):
    case = _case("workspace", "user_task_0", "injection_task_0")
    bad_planner = ScriptedClient(lambda p: "I think you need delete_everything")
    row = run_case(case, "PG-v4", llm=_agent("workspace", "user_task_0", "injection_task_0"), agent_client=bad_planner,
                   reviewer_client=APPROVE, seed=42)
    assert row["capabilities"]["source"] == "plan-fallback-parse" and row["capabilities"]["planner_error"]
    good_planner = ScriptedClient(lambda p: '{"actions": []}')
    row = run_case(case, "PG-v4", llm=_agent("workspace", "user_task_0", "injection_task_0"), agent_client=good_planner,
                   reviewer_client=APPROVE, seed=42)
    assert row["capabilities"]["source"] == "plan" and row["attack_success"] is False


def test_v1_bridge_runs_on_workspace_only(tmp_path):
    row = run_case(_case("workspace", "user_task_0", "injection_task_0"), "SCP-v3",
                   llm=_agent("workspace", "user_task_0", "injection_task_0"), agent_client=APPROVE, reviewer_client=None, seed=42)
    assert row["attack_success"] is False  # the v1 allow-list does not contain the attacker's address
    with pytest.raises(ValueError):
        run_case(_case("banking", "user_task_0"), "SCP-v3", llm=_agent("banking", "user_task_0"), agent_client=APPROVE,
                 reviewer_client=None, seed=42)


@pytest.mark.parametrize("config", ["spotlighting", "repeat_user_prompt", "tool_filter", "trace_auditor"])
def test_baselines_run_and_score(config, tmp_path):
    case = _case("slack", "user_task_1", "injection_task_1")
    tool_names = ScriptedClient(lambda p: ", ".join(t.name for t in get_suite(BENCHMARK_VERSION, "slack").tools))
    row = run_case(case, config, llm=_agent("slack", "user_task_1", "injection_task_1"), agent_client=tool_names,
                   reviewer_client=deny_if("injection", "phishing", "www.secure-systems-252.com"), seed=42)
    assert row["error"] is None and row["utility"] is not None and row["attack_success"] is not None


def test_unencodable_amount_is_a_recorded_schema_denial(tmp_path):
    bad = FunctionCall(function="send_money", args={"recipient": "GB29NWBK60161331926819", "amount": "lots",
                                                    "subject": "x", "date": "2022-01-01"})
    row = run_case(_case("banking", "user_task_0"), "C0g", llm=_agent("banking", "user_task_0", extra_calls=[bad]),
                   agent_client=APPROVE, reviewer_client=APPROVE, seed=42)
    assert any(m["decision"] == "SCHEMA_DENY" for m in row["mediated_calls"]) and row["error"] is None


@pytest.mark.parametrize("suite", SUITES)
def test_plumbing_preserves_attack_outcome_on_security_cases(suite, tmp_path):
    """Injected agent (injection ground truth, then the user task's): C0g must reproduce 'none' exactly -- same attack outcome,
    same utility, no write blocked by the plumbing itself.  First three user tasks of each suite x every injection task."""
    s = get_suite(BENCHMARK_VERSION, suite)
    uids = sorted(s.user_tasks, key=lambda u: int(u.split("_")[-1]))[:3]
    problems = []
    for uid in uids:
        for iid in sorted(s.injection_tasks):
            if (suite, uid, iid) not in INJECTIONS:
                continue
            rows = {cfg: run_case(_case(suite, uid, iid), cfg, llm=_agent(suite, uid, iid), agent_client=APPROVE,
                                  reviewer_client=APPROVE, seed=42, evidence_root=tmp_path / cfg) for cfg in ("none", "C0g")}
            blocked = [m for m in rows["C0g"]["mediated_calls"] if m["result_status"] == "BLOCKED"]
            same = all(rows["none"][k] == rows["C0g"][k] for k in ("attack_success", "utility"))
            if blocked or not same or rows["C0g"]["error"]:
                problems.append((uid, iid, rows["none"]["attack_success"], rows["C0g"]["attack_success"],
                                 rows["none"]["utility"], rows["C0g"]["utility"], blocked[:1], rows["C0g"]["error"]))
    assert not problems, problems[:5]


def test_identical_injection_texts_get_one_source_label():
    from pgx.agentdojo_runner import injection_sources
    out = injection_sources({"b": "X", "a": "X", "c": "Y"})
    assert out == {"agentdojo_injection:a+b": "X", "agentdojo_injection:c": "Y"}
