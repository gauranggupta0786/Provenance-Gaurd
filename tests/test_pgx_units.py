"""v2 unit tests: registry leaves v1 untouched, encodings round-trip, capabilities, numeric rule, guard chain, digests, frozen data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------- registry
def test_registry_only_adds_and_v1_tools_render_identically():
    import pg.approval as approval
    import pg.schemas as schemas
    from pg.policy import evaluate
    from pgx import registry
    from pgx.suites import NEW_CONTRACTS

    v1_tools = [t for t in schemas.TOOL_CONTRACTS if t not in NEW_CONTRACTS]
    sample = {"workspace_send_email": {"recipients": ["a@b.co"], "subject": "s", "body": "b", "attachments": [], "cc": [], "bcc": []},
              "workspace_delete_file": {"file_id": "3"}, "send_email": {"recipient": "a@b.co", "subject": "s", "body": "b"}}
    before = {t: approval._target_and_changes(t, a) for t, a in sample.items()}
    contracts_before = {t: schemas.TOOL_CONTRACTS[t] for t in v1_tools}
    registry.install()
    registry.install()  # idempotent
    assert {t: approval._target_and_changes(t, a) for t, a in sample.items()} == before
    assert all(schemas.TOOL_CONTRACTS[t] is contracts_before[t] for t in v1_tools)
    assert set(NEW_CONTRACTS) <= set(schemas.TOOL_CONTRACTS) and set(NEW_CONTRACTS) <= approval.ROLE_PERMISSIONS["approver"]
    # v1's static policy still fails closed on every new tool
    assert evaluate(principal_id="user_gaurang", tool_name="banking_send_money",
                    arguments={"recipient": "X", "amount": "1", "subject": "s", "date": "d"}).decision.value == "DENY"


def test_every_new_tool_has_a_card_layout():
    from pgx import registry
    from pgx.suites import NEW_CONTRACTS
    import pg.approval as approval

    registry.install()
    for tool, contract in NEW_CONTRACTS.items():
        args = {k: (["x@y.co"] if t == list[str] else 1 if t is int else [] if t == list[dict] else "v")
                for k, t in contract["required_args"].items()}
        args.update({k: d for k, (t, d) in contract.get("optional_args", {}).items()})
        target, changed = approval._target_and_changes(tool, args)
        assert isinstance(target, str) and isinstance(changed, dict)


# ---------------------------------------------------------------------------- suites and encodings
def test_suite_specs_classify_every_agentdojo_tool():
    from agentdojo.task_suite.load_suites import get_suite
    from pgx.suites import BENCHMARK_VERSION, SPECS

    for name, spec in SPECS.items():
        spec.classify({t.name for t in get_suite(BENCHMARK_VERSION, name).tools})


def test_money_and_flag_encodings_round_trip():
    from pgx.suites import BANKING, SuiteSpecError, money

    canonical, args = BANKING.to_canonical("schedule_transaction", {"recipient": "X", "amount": 1200.50, "subject": "s",
                                                                    "date": "2022-01-01", "recurring": True})
    assert canonical == "banking_schedule_transaction" and args["amount"] == "1200.5" and args["recurring"] == "true"
    function, back = BANKING.to_agentdojo(canonical, args)
    assert function == "schedule_transaction" and back["amount"] == 1200.5 and back["recurring"] is True
    assert money(100) == money(100.0) == "100"
    for bad in ("lots", float("nan"), True):
        with pytest.raises(SuiteSpecError):
            money(bad)


# ---------------------------------------------------------------------------- numeric rule (E2)
@pytest.mark.parametrize("body", ["Marketing got about 200K.", "Marketing: two hundred thousand.", "Marketing 200,000",
                                  "The code is four six three eight two zero", "first part 4638, rest 20"])
def test_numeric_rule_catches_each_week13_evasion_shape(body):
    from pg.content_provenance import ReadSource
    from pgx.content_v2 import numeric_leaks

    sources = [ReadSource("file", "1", "budget.xlsx", "Marketing 200000 Sales 120000"),
               ReadSource("email", "2", "Your security code", "Your code is 463820")]
    assert numeric_leaks(body, "Send the meeting summary to bob@x.co", sources)


def test_numeric_rule_spares_licensed_and_small_numbers():
    from pg.content_provenance import ReadSource
    from pgx.content_v2 import numeric_leaks

    sources = [ReadSource("file", "1", "budget.xlsx", "Marketing 200000"), ReadSource("file", "2", "notes.txt", "room 12")]
    assert not numeric_leaks("Marketing got 200K", "Summarise budget.xlsx for bob@x.co", sources)   # source named
    assert not numeric_leaks("Marketing got 200K", "Tell bob the marketing budget of 200000", sources)  # in the request
    assert not numeric_leaks("Meet in room 12", "Send the meeting summary to bob@x.co", sources)      # below MIN_VALUE


# ---------------------------------------------------------------------------- capabilities (E1)
def test_parser_and_planner_capabilities():
    from pgx.capabilities import parse_capabilities, plan_capabilities
    from pgx.suites import BANKING, SLACK

    caps = parse_capabilities("Pay the rent to GB29NWBK60161331926819 please", BANKING)
    assert "banking_send_money" in caps.tools and "banking_update_password" not in caps.tools
    assert caps.ibans == {"GB29NWBK60161331926819"}
    caps = parse_capabilities("Read the content on webpage www.informations.com.", SLACK)
    assert "slack_get_webpage" in caps.tools and "informations.com" in caps.urls
    plan = plan_capabilities("Pay my rent", BANKING, lambda p: '{"actions": ["send_money"]}')
    assert plan.source == "plan" and plan.tools == {"banking_send_money"}
    for bad in ("nope", '{"actions": ["wire_everything"]}', '{"actions": "send_money"}'):
        fb = plan_capabilities("Pay my rent", BANKING, lambda p, bad=bad: bad)
        assert fb.source == "plan-fallback-parse" and fb.planner_error


def test_planner_never_sees_anything_but_the_request_and_tool_list():
    from pgx.capabilities import plan_capabilities
    from pgx.suites import WORKSPACE

    seen = []
    plan_capabilities("Email bob@x.co the agenda", WORKSPACE, lambda p: seen.append(p) or '{"actions": ["send_email"]}')
    assert len(seen) == 1 and "Email bob@x.co the agenda" in seen[0] and "<INFORMATION>" not in seen[0]


# ---------------------------------------------------------------------------- guard chain
def _ctx(spec, canonical, args, request, caps=None, env=None, sources=()):
    return {"spec": spec, "canonical": canonical, "arguments": args, "request": request, "capabilities": caps,
            "env": env, "read_sources": sources, "copied_from": []}


def test_v4_checks_order_and_bypass():
    from pgx.capabilities import parse_capabilities
    from pgx.guards import V4Checks
    from pgx.suites import BANKING

    req = "Send 50 to GB29NWBK60161331926819 for the dinner"
    caps = parse_capabilities(req, BANKING)
    checks = V4Checks()
    ok = {"recipient": "GB29NWBK60161331926819", "amount": "50", "subject": "dinner", "date": "2022-03-01"}
    assert checks(_ctx(BANKING, "banking_send_money", ok, req, caps)).stage == "licensed"
    other = dict(ok, recipient="US133000000121212121212")
    v = checks(_ctx(BANKING, "banking_send_money", other, req, caps))
    assert (v.approve, v.stage, v.category) == (False, "destination", "unrequested_recipient")
    v = checks(_ctx(BANKING, "banking_update_password", {"password": "x"}, req, caps))
    assert (v.approve, v.stage) == (False, "capability")
    ctx = _ctx(BANKING, "banking_send_money", ok, "Pay the bill in bill-december-2023.txt", parse_capabilities(
        "Pay the bill in bill-december-2023.txt", BANKING))
    assert checks(ctx) is None and ctx["flags"]["target_from_data"]  # target from data: the reviewer decides, warned
    assert V4Checks(bypass=False)(_ctx(BANKING, "banking_send_money", ok, req, caps)) is None


def test_v4_content_rule_denies_unlicensed_figures():
    from pg.content_provenance import ReadSource
    from pgx.capabilities import parse_capabilities
    from pgx.guards import V4Checks
    from pgx.suites import WORKSPACE

    req = "Email bob@x.co a one-line status update"
    caps = parse_capabilities(req, WORKSPACE)
    src = [ReadSource("file", "9", "salaries.xlsx", "Alice 185000")]
    args = {"recipients": ["bob@x.co"], "subject": "status", "body": "All good. Alice earns about 185K.", "attachments": [],
            "cc": [], "bcc": []}
    env = SimpleNamespace()
    v = V4Checks()(_ctx(WORKSPACE, "workspace_send_email", args, req, caps, env, src))
    assert (v.approve, v.stage) == (False, "content")
    assert V4Checks(numeric=False)(_ctx(WORKSPACE, "workspace_send_email", args, req, caps, env, src)).stage == "licensed"


# ---------------------------------------------------------------------------- digests and frozen data
def test_v1_digest_unchanged_by_pgx():
    from pg.week8_protocol import code_digest
    from scripts.week11_lib import week11_code_digest

    assert code_digest(ROOT)[0] == "dca1a798694935c90886af0f2a535b24251de7e6a5ceef5448f2442ae1e0648b"
    assert week11_code_digest(ROOT)[0] == "22a4622b386181ff127e3d67ba452feb241efdb517811f71c903f204f2e58010"


def test_pgx_digest_covers_pg_and_pgx_and_v2_data():
    from pgx.version import pgx_digest

    _, files = pgx_digest(ROOT)
    assert any(f.startswith("pg/") for f in files) and any(f.startswith("pgx/") for f in files)
    assert "data/v2/injections.jsonl" in files and "data/v2/split.json" in files


def test_v2_data_files_match_the_plan():
    plan = json.loads((ROOT / "data/v2/plan.json").read_text())
    for name, digest in plan["files"].items():
        assert hashlib.sha256((ROOT / "data/v2" / name).read_bytes()).hexdigest() == digest, name
    split = json.loads((ROOT / "data/v2/split.json").read_text())
    for suite, s in split["suites"].items():
        assert not set(s["dev"]) & set(s["test"]) and set(s["variance_subset"]) <= set(s["test"])
    assert sum(1 for line in (ROOT / "data/v2/injections.jsonl").read_text().splitlines() if line) == 949
