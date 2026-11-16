"""Week 12: the OpenAI-compatible adapter that makes the frontier run someone else's one-command job.

What these tests hold in place:

* the adapter is configured only by environment variables, so no provider needs a code change;
* a hosted profile never claims provenance it does not have (``model_digest`` is a declared identity, and the session
  record lists what cannot be verified);
* a truncated hosted answer raises the *same* failure the local guard raises -- otherwise a cut-off episode would be
  scored as a safe one, which is exactly the bias the guard exists to prevent;
* retries cover transport failures only and never a response that arrived, so a retry cannot change a measured outcome;
* the probes classify a model as usable or not, and never raise.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from frontier.frontier_probes import ADVISORY, REQUIRED, run_probes
from pg.model_api import CanonicalMessage
from frontier.openai_adapter import (
    ENV_PREFIX, REQUIRED_ENV, UNVERIFIABLE_REMOTELY, MeteredRemoteClient, RemoteConfig, UsageLog, build_remote_llm,
    build_remote_profile, finish_reason, remote_session_record,
)


def response(*, content="ok", tool_calls=None, finish="stop", prompt=7, completion=3, response_id="r1"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        id=response_id, choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion))


def tool_call(name="get_weather", arguments='{"city":"Pilani"}', call_id="call-1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeOpenAI:
    """Enough of the OpenAI client for the adapter: models.list() and chat.completions.create()."""

    def __init__(self, replies=None, models=("some-model",), fail_times=0, error=None):
        self.replies = list(replies or [])
        self.payloads: list[dict] = []
        self.fail_times = fail_times
        self.error = error or ConnectionError("connection reset by peer")
        self.calls = 0
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=m) for m in models]))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **payload):
        self.calls += 1
        self.payloads.append(payload)
        if self.calls <= self.fail_times:
            raise self.error
        return self.replies.pop(0) if self.replies else response()


@pytest.fixture
def config():
    return RemoteConfig(api_base="https://api.example.com/v1/", model="some-model", api_key="k", max_retries=2,
                        retry_backoff_seconds=0.0)


def client_for(config, fake, usage=None):
    return MeteredRemoteClient(build_remote_profile(config), config, client=fake, usage=usage, sleep=lambda _: None)


# ------------------------------------------------------------------ configuration comes from the environment

def test_from_env_reads_every_setting(monkeypatch):
    for name, value in {"API_BASE": "https://x/v1", "MODEL": "m", "API_KEY": "k", "CONTEXT": "16384",
                        "MAX_OUTPUT_TOKENS": "512", "TEMPERATURE": "0.2", "SEED": "7", "MAX_RETRIES": "1",
                        "PRICE_PER_1K_INPUT": "0.5", "PRICE_PER_1K_OUTPUT": "1.5", "CURRENCY": "EUR"}.items():
        monkeypatch.setenv(f"{ENV_PREFIX}{name}", value)
    config = RemoteConfig.from_env()
    assert (config.api_base, config.model, config.api_key) == ("https://x/v1", "m", "k")
    assert (config.context_length, config.max_output_tokens, config.temperature, config.seed) == (16384, 512, 0.2, 7)
    assert config.cost(1000, 1000) == 2.0 and config.currency == "EUR"


def test_from_env_names_what_is_missing(monkeypatch):
    for name in REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="API_BASE"):
        RemoteConfig.from_env()


def test_from_env_rejects_a_malformed_number(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}API_BASE", "https://x/v1")
    monkeypatch.setenv(f"{ENV_PREFIX}MODEL", "m")
    monkeypatch.setenv(f"{ENV_PREFIX}CONTEXT", "big")
    with pytest.raises(ValueError, match="CONTEXT"):
        RemoteConfig.from_env()


# ------------------------------------------------------------------ a hosted profile claims only what it can

def test_remote_profile_is_valid_and_carries_the_declared_settings(config):
    profile = build_remote_profile(config)
    assert profile.provider == "openai-compatible"
    assert profile.endpoint == "https://api.example.com/v1"      # trailing slash stripped
    assert profile.model_id == "some-model"
    assert profile.model_digest == config.identity_digest
    assert profile.serving_engine_version == "unknown" and profile.quantization == "unknown"
    assert profile.temperature == config.temperature and profile.max_output_tokens == config.max_output_tokens


def test_identity_digest_is_of_the_request_not_the_weights(config):
    from dataclasses import replace

    assert config.identity_digest != replace(config, model="other").identity_digest
    assert config.identity_digest != replace(config, api_base="https://other/v1").identity_digest
    # the key itself must not change identity: the same model behind the same endpoint is the same declared model
    assert config.identity_digest == replace(config, api_key="different").identity_digest


def test_session_record_lists_what_cannot_be_verified(config):
    llm = build_remote_llm(config, client=FakeOpenAI())
    record = remote_session_record(config, llm, probes={"native_tool_capability": True})
    assert record["cannot_be_verified_remotely"] == list(UNVERIFIABLE_REMOTELY)
    assert any("weight" in line for line in record["cannot_be_verified_remotely"])
    assert "not a weight digest" in record["declared_identity_digest_note"]
    assert record["parameter_count_known"] is False
    assert record["declared_settings"]["seed"] == config.seed
    assert "NOT in model provenance" in record["comparability"]


# ------------------------------------------------------------------ the request carries the settings we pin

def test_payload_pins_the_settings_the_openai_api_accepts(config):
    fake = FakeOpenAI()
    client_for(config, fake).complete([CanonicalMessage(role="user", content="hi")], [])
    payload = fake.payloads[0]
    assert payload["model"] == "some-model"
    assert payload["temperature"] == 0.0 and payload["top_p"] == 1.0
    assert payload["seed"] == 42 and payload["max_tokens"] == 4096


def test_system_messages_use_the_configured_role(config):
    # Week 14: the default is "system" (what the local native runs sent, accepted by every OpenAI-compatible server);
    # "developer" is an explicit opt-in, because several non-OpenAI servers reject or mangle it.
    from dataclasses import replace

    fake = FakeOpenAI()
    client_for(config, fake).complete([CanonicalMessage(role="system", content="rules"),
                                       CanonicalMessage(role="user", content="hi")], [])
    assert fake.payloads[0]["messages"][0]["role"] == config.system_role == "system"
    opt_in = replace(config, system_role="developer")
    fake = FakeOpenAI()
    client_for(opt_in, fake).complete([CanonicalMessage(role="system", content="rules"),
                                       CanonicalMessage(role="user", content="hi")], [])
    assert fake.payloads[0]["messages"][0]["role"] == "developer"


# ------------------------------------------------------------------ truncation must not look like safety

def test_finish_reason_reads_objects_and_dicts(config):
    from pg.model_api import ModelCompletion

    adapter = MeteredRemoteClient(build_remote_profile(config), config, client=FakeOpenAI()).adapter
    assert finish_reason(adapter.normalize_response(response(finish="length"))) == "length"
    assert finish_reason(adapter.normalize_response(response(finish="stop"))) == "stop"
    as_dict = ModelCompletion(response_id="r", content="x", tool_intents=(), prompt_tokens=1, completion_tokens=1,
                              total_tokens=2, raw_response={"choices": [{"finish_reason": "length"}]})
    assert finish_reason(as_dict) == "length"
    no_choices = ModelCompletion(response_id="r", content="x", tool_intents=(), prompt_tokens=1, completion_tokens=1,
                                 total_tokens=2, raw_response={})
    assert finish_reason(no_choices) is None


def test_truncated_hosted_answer_raises_the_same_failure_as_the_local_guard(config):
    from pg.diagnostics import EpisodeStage, FailureCategory, classify_failure

    fake = FakeOpenAI(replies=[response(content="one two three", finish="length")])
    with pytest.raises(ValueError, match="structured output truncated") as exc:
        client_for(config, fake).complete([CanonicalMessage(role="user", content="count")], [])
    assert classify_failure(exc.value, EpisodeStage.CALL_MODEL) is FailureCategory.INVALID_STRUCTURED_OUTPUT


def test_a_complete_answer_is_not_flagged(config):
    fake = FakeOpenAI(replies=[response(content="done", finish="stop")])
    assert client_for(config, fake).complete([CanonicalMessage(role="user", content="hi")], []).content == "done"


# ------------------------------------------------------------------ retries survive the network, never a result

def test_retries_cover_a_transport_failure(config):
    fake = FakeOpenAI(fail_times=2)
    usage = UsageLog()
    client_for(config, fake, usage).complete([CanonicalMessage(role="user", content="hi")], [])
    assert fake.calls == 3 and usage.calls[0]["attempts"] == 3


def test_retries_stop_at_the_configured_limit(config):
    fake = FakeOpenAI(fail_times=99)
    with pytest.raises(ConnectionError):
        client_for(config, fake).complete([CanonicalMessage(role="user", content="hi")], [])
    assert fake.calls == config.max_retries + 1


def test_a_response_that_arrived_is_never_retried(config):
    """A truncated answer is a result, not a transport failure: retrying it would change a measured outcome."""

    fake = FakeOpenAI(replies=[response(finish="length"), response(content="second try")])
    with pytest.raises(ValueError, match="structured output truncated"):
        client_for(config, fake).complete([CanonicalMessage(role="user", content="hi")], [])
    assert fake.calls == 1


def test_a_refusal_the_provider_returns_is_not_retried(config):
    fake = FakeOpenAI(fail_times=1, error=ValueError("invalid_request_error: unknown parameter"))
    with pytest.raises(ValueError, match="invalid_request_error"):
        client_for(config, fake).complete([CanonicalMessage(role="user", content="hi")], [])
    assert fake.calls == 1


# ------------------------------------------------------------------ tokens, latency and cost

def test_usage_log_totals_tokens_latency_and_cost():
    from dataclasses import replace

    config = replace(RemoteConfig(api_base="https://x/v1", model="m"), price_per_1k_input=1.0, price_per_1k_output=2.0)
    usage = UsageLog(currency=config.currency)
    fake = FakeOpenAI(replies=[response(prompt=1000, completion=500), response(prompt=2000, completion=1000)])
    client = MeteredRemoteClient(build_remote_profile(config), config, client=fake, usage=usage, sleep=lambda _: None)
    for _ in range(2):
        client.complete([CanonicalMessage(role="user", content="hi")], [])
    totals = usage.totals()
    assert totals["calls"] == 2 and totals["prompt_tokens"] == 3000 and totals["completion_tokens"] == 1500
    assert totals["cost"] == pytest.approx(3000 / 1000 * 1.0 + 1500 / 1000 * 2.0)
    assert totals["latency_ms_total"] >= 0 and totals["retried_calls"] == 0


def test_cost_is_zero_for_a_local_endpoint(config):
    assert config.price_per_1k_input == 0.0 and config.cost(10_000, 10_000) == 0.0


# ------------------------------------------------------------------ the pipeline element is the one the local runs use

def test_build_remote_llm_returns_a_guarded_pipeline_element(config):
    llm = build_remote_llm(config, client=FakeOpenAI())
    assert llm.profile.provider == "openai-compatible"
    assert hasattr(llm.model_client, "request_log"), "the guard wrapper must still be in place"
    assert llm.healthcheck()["model_available"] is True


def test_build_remote_llm_can_run_unguarded_for_probing(config):
    llm = build_remote_llm(config, client=FakeOpenAI(), guard=False)
    assert not hasattr(llm.model_client, "request_log")


# ------------------------------------------------------------------ the probes

def probe_client(**kw):
    return FakeOpenAI(**kw)


def test_probes_pass_on_a_well_behaved_endpoint(config):
    replies = [
        response(content=None, tool_calls=[tool_call()]),            # native_tool_capability
        response(content="ok"),                                      # explicit_parameters
        response(content="BLUE"),                                    # system_role_honoured
        response(content=None, tool_calls=[tool_call()]),            # tool_call_roundtrip, first leg
        response(content="It is 31 degrees and clear."),             # tool_call_roundtrip, second leg
        response(content="one two", finish="length"),                # output_limit
        response(content="blue"), response(content="blue"), response(content="blue"),   # determinism
        response(content="51"),                                      # thinking_control
        response(content="OK"),                                      # token_accounting
    ]
    report = run_probes(config, client=probe_client(replies=replies, models=("some-model",)))
    failed = [p["probe"] for p in report["probes"] if not p["passed"] and p["applicable"]]
    assert failed == [], failed
    assert report["usable"] is True and report["required_failed"] == []
    assert report["usage"]["calls"] >= len(replies) - 1


def test_probes_mark_a_model_without_structured_tool_calls_unusable(config):
    """The failure that killed command-r7b, granite3.3 and glm4 locally: a tools badge with prose behind it."""

    report = run_probes(config, client=probe_client(replies=[response(content='I would call get_weather(city="Pilani")')]))
    by_name = {p["probe"]: p for p in report["probes"]}
    assert by_name["native_tool_capability"]["passed"] is False
    assert "native_tool_capability" in report["required_failed"] and report["usable"] is False


def test_probes_record_a_leaked_reasoning_block(config):
    only = ("thinking_control",)
    report = run_probes(config, client=probe_client(replies=[response(content="<think>hmm</think>51")]), only=only)
    assert report["probes"][0]["passed"] is False
    assert report["probes"][0]["details"]["leaked"] is True


def test_thinking_control_is_not_applicable_when_thinking_is_declared_on(config):
    from dataclasses import replace

    cfg = replace(config, thinking_capability=True, thinking_mode="high")
    report = run_probes(cfg, client=probe_client(), only=("thinking_control",))
    assert report["probes"][0]["applicable"] is False and report["probes"][0]["passed"] is True


def test_probes_never_raise_when_the_endpoint_is_broken(config):
    class Broken(FakeOpenAI):
        def _create(self, **payload):
            raise RuntimeError("endpoint on fire")

    report = run_probes(config, client=Broken(models=()))
    assert report["usable"] is False
    assert all("probe" in entry for entry in report["probes"])
    assert len(report["probes"]) == len(REQUIRED) + len(ADVISORY)


def test_an_endpoint_that_does_not_list_models_still_passes_reachable(config):
    report = run_probes(config, client=probe_client(models=()), only=("reachable",))
    assert report["probes"][0]["passed"] is True
    assert report["probes"][0]["details"]["model_listed"] is False


def test_output_limit_probe_fails_when_truncation_is_invisible(config):
    report = run_probes(config, client=probe_client(replies=[response(content="one two", finish="stop")]),
                        only=("output_limit",))
    assert report["probes"][0]["passed"] is False
    assert "truncation would be invisible" in report["probes"][0]["details"]["reason"]


def test_required_and_advisory_probes_are_disjoint_and_complete():
    names = {name for name, _ in __import__("frontier.frontier_probes", fromlist=["PROBES"]).PROBES}
    assert set(REQUIRED).isdisjoint(ADVISORY)
    assert set(REQUIRED) | set(ADVISORY) == names


# ------------------------------------------------------------------ the CLI entry points

def test_frontier_probe_refuses_without_configuration(monkeypatch, capsys):
    from scripts.frontier_probe import main

    for name in REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    assert main([]) == 2
    assert "API_BASE" in capsys.readouterr().out


def test_spend_estimate_scales_with_price():
    from scripts.frontier_probe import CALLS_PER_EPISODE, estimate

    usage = {"calls": 2, "prompt_tokens": 2000, "completion_tokens": 1000}
    config = RemoteConfig(api_base="https://x/v1", model="m", price_per_1k_input=1.0, price_per_1k_output=2.0)
    out = estimate(usage, config, episodes=10)
    assert out["model_calls"] == 10 * CALLS_PER_EPISODE
    assert out["cost"] == pytest.approx(out["prompt_tokens"] / 1000 + out["completion_tokens"] / 1000 * 2.0, rel=1e-6)
    assert "floor" in out["note"]


def test_parity_script_compares_actions_not_wording():
    """``outcome()``'s fields must be read with ``EpisodeResult.to_dict()``'s REAL key names (``pg/week5_runner.py``),
    confirmed against a real episode in ``results/week11c``: tool_calls entries use ``function``/``args`` (never
    ``tool_name``/``arguments``), the post-episode hook result is keyed ``hook_result`` (never ``post_episode_hook``
    or ``hook``), and executed/denied verdicts live on the separate ``mediated_calls`` list, not on tool_calls. An
    earlier version of this test used the wrong names on both sides and so could not have caught the real bug (the
    production code read the same wrong names) -- this fixture is built from the real schema instead."""
    from scripts.week12_adapter_parity import outcome

    base = {
        "attack_success": False,
        "tool_calls": [{"function": "send_email", "args": {"to": "a"}, "id": "call-1", "placeholder_args": None}],
        "mediated_calls": [{"agentdojo_function": "send_email", "canonical_tool": "workspace_send_email",
                             "decision": "EXECUTE", "result_status": "EXECUTED", "proposal_id": "p1"}],
        "hook_result": {"task_write_done": True},
    }
    same_words_different_action = {**base, "tool_calls": [{"function": "send_email", "args": {"to": "b"},
                                                           "id": "call-1", "placeholder_args": None}]}
    different_mediation_outcome = {**base, "mediated_calls": [{**base["mediated_calls"][0], "decision": "DENY",
                                                                "result_status": "BLOCKED"}]}
    different_task_completion = {**base, "hook_result": {"task_write_done": False}}

    assert outcome(base) == outcome({**base, "content": "worded differently"})
    assert outcome(base) != outcome(same_words_different_action)
    assert outcome(base) != outcome(different_mediation_outcome)
    assert outcome(base) != outcome(different_task_completion)
    assert outcome(base)["task_done"] is True   # not None: hook_result IS being read now
    assert outcome(base)["actions"] == [{"function": "send_email", "args": {"to": "a"}}]
    assert outcome(base)["mediated"] == [{"function": "send_email", "decision": "EXECUTE", "result_status": "EXECUTED"}]


def test_outcome_handles_a_missing_hook_result_and_empty_call_lists():
    from scripts.week12_adapter_parity import outcome

    o = outcome({"attack_success": None, "error": None})
    assert o["task_done"] is None and o["actions"] == [] and o["mediated"] == []


def test_parity_report_states_the_payload_difference():
    source = (__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "week12_adapter_parity.py") \
        .read_text(encoding="utf-8")
    assert "num_ctx" in source and "server's defaults" in source
    assert "--require-identical" in source
