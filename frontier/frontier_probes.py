"""Capability probes for a hosted OpenAI-compatible model, run before any campaign spends money.

The Week 7 probes in ``pg/week7_probes.py`` read Ollama's native API and cannot be pointed at a hosted endpoint.  These
ask the same questions through ``/chat/completions``, keep the Week 7 probe names where the question is the same, and
return the same ``{probe, passed, applicable, details}`` shape, so one table can hold local and hosted models side by side.

Why probe at all: every expensive failure in this project was a model that *looked* usable and was not -- a tools badge
with no structured call behind it, a reasoning mode that could not be switched off, an output limit that silently cut the
answer.  Finding that out costs a handful of calls here, or a whole paid campaign later.

``REQUIRED`` lists the probes a model must pass to be worth running: without them the result would not be measurable.
``ADVISORY`` probes are recorded and reported, never used to exclude -- a hosted API that ignores ``seed`` is still a
legitimate subject, it just cannot claim determinism.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

from pg.model_api import CanonicalMessage, CanonicalToolSchema, ModelCompletion
from frontier.openai_adapter import (
    MeteredRemoteClient, RemoteConfig, UsageLog, apply_param_policy, build_remote_profile, finish_reason,
)

REQUIRED = ("reachable", "native_tool_capability", "explicit_parameters", "tool_call_roundtrip", "token_accounting")
ADVISORY = ("system_role_honoured", "output_limit", "determinism", "thinking_control")

WEATHER_TOOL = CanonicalToolSchema.from_openai({
    "name": "get_weather",
    "description": "Look up the current weather for one city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]},
})


def _result(name: str, passed: bool, *, applicable: bool = True, **details: Any) -> dict[str, Any]:
    return {"probe": name, "passed": bool(passed), "applicable": applicable, "details": details}


def _client(config: RemoteConfig, *, client: Any = None, usage: UsageLog | None = None, **overrides: Any) -> MeteredRemoteClient:
    from dataclasses import replace

    cfg = replace(config, **overrides) if overrides else config
    return MeteredRemoteClient(build_remote_profile(cfg), cfg, client=client, usage=usage)


def _text(completion: ModelCompletion) -> str:
    return (completion.content or "").strip()


def probe_reachable(config: RemoteConfig, *, client: Any = None, usage: UsageLog | None = None) -> dict[str, Any]:
    """The endpoint answers, and -- if it lists models at all -- the configured one is among them."""

    remote = _client(config, client=client, usage=usage)
    try:
        models = remote.list_models()
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, it does not raise them
        return _result("reachable", False, error=f"{type(exc).__name__}: {exc}"[:300])
    listed = config.model in models
    # Some gateways do not implement /models, or list aliases only. Not listing is not a failure; not answering is.
    return _result("reachable", True, model_listed=listed, models_returned=len(models),
                   note="" if listed else "the endpoint did not list the configured model; it may use aliases")


def probe_native_tool_capability(config: RemoteConfig, *, client: Any = None, usage: UsageLog | None = None) -> dict[str, Any]:
    """A structured tool call comes back, with arguments that parse -- not a tool call written out as prose."""

    remote = _client(config, client=client, usage=usage)
    try:
        completion = remote.complete(
            [CanonicalMessage(role="user", content="What is the weather in Pilani? Use the tool.")], [WEATHER_TOOL])
    except Exception as exc:  # noqa: BLE001
        return _result("native_tool_capability", False, error=f"{type(exc).__name__}: {exc}"[:300])
    intents = completion.tool_intents
    parsed = [i for i in intents if i.arguments is not None and not i.parse_error]
    return _result("native_tool_capability", bool(parsed), tool_calls=len(intents), parsed=len(parsed),
                   parse_errors=[i.parse_error for i in intents if i.parse_error][:3],
                   content_preview=_text(completion)[:200])


def probe_explicit_parameters(config: RemoteConfig, *, client: Any = None, usage: UsageLog | None = None) -> dict[str, Any]:
    """The endpoint accepts every decoding parameter we set explicitly, rather than rejecting the request."""

    remote = _client(config, client=client, usage=usage)
    # what actually goes on the wire: the adapter's payload after the provider adaptations in the config (token-limit name,
    # omitted parameters) -- so the report never claims a parameter was sent that was not
    payload = apply_param_policy(
        remote.adapter.build_payload(remote.profile, [CanonicalMessage(role="user", content="Say OK.")], []), config)
    try:
        completion = remote.complete([CanonicalMessage(role="user", content="Say OK.")], [])
    except Exception as exc:  # noqa: BLE001
        return _result("explicit_parameters", False, sent=sorted(payload), error=f"{type(exc).__name__}: {exc}"[:300])
    return _result("explicit_parameters", True, sent=sorted(payload),
                   temperature=payload.get("temperature"), top_p=payload.get("top_p"),
                   seed=payload.get("seed"), token_limit={config.token_param: payload.get(config.token_param)},
                   omitted=list(config.omit_params), answered=bool(_text(completion)))


def probe_system_role_honoured(config: RemoteConfig, *, client: Any = None, usage: UsageLog | None = None) -> dict[str, Any]:
    remote = _client(config, client=client, usage=usage)
    try:
        completion = remote.complete([
            CanonicalMessage(role="system", content="You always answer with exactly the word BLUE and nothing else."),
            CanonicalMessage(role="user", content="What colour is the sky?"),
        ], [])
    except Exception as exc:  # noqa: BLE001
        # A server that REJECTS the configured system role would fail every episode of the campaign, so a rejection (unlike a
        # model that merely ignores the instruction) is a hard failure: run_probes counts it as required.
        return _result("system_role_honoured", False, rejected=True, error=f"{type(exc).__name__}: {exc}"[:300],
                       hint="set PROVENANCEGUARD_SYSTEM_ROLE to the other of system/developer")
    answer = _text(completion)
    return _result("system_role_honoured", answer.upper().startswith("BLUE"), answer=answer[:120],
                   system_role=remote.profile.system_role)


def probe_tool_call_roundtrip(config: RemoteConfig, *, client: Any = None, usage: UsageLog | None = None) -> dict[str, Any]:
    """A tool result is accepted back and used: the half of tool calling that silently breaks on some endpoints."""

    remote = _client(config, client=client, usage=usage)
    try:
        first = remote.complete(
            [CanonicalMessage(role="user", content="What is the weather in Pilani? Use the tool.")], [WEATHER_TOOL])
        intents = first.tool_intents
        if not intents:
            return _result("tool_call_roundtrip", False, reason="no tool call to return a result for")
        call = intents[0]
        messages = [
            CanonicalMessage(role="user", content="What is the weather in Pilani? Use the tool."),
            CanonicalMessage(role="assistant", content=first.content or "", tool_calls=[{
                "id": call.call_id, "type": "function",
                "function": {"name": call.tool_name, "arguments": call.raw_arguments
                             if isinstance(call.raw_arguments, str) else str(call.raw_arguments)}}]),
            CanonicalMessage(role="tool", content="31 degrees and clear", tool_call_id=call.call_id),
        ]
        second = remote.complete(messages, [WEATHER_TOOL])
    except Exception as exc:  # noqa: BLE001
        return _result("tool_call_roundtrip", False, error=f"{type(exc).__name__}: {exc}"[:300])
    answer = _text(second)
    return _result("tool_call_roundtrip", "31" in answer, answer=answer[:200])


def probe_output_limit(config: RemoteConfig, *, client: Any = None, usage: UsageLog | None = None) -> dict[str, Any]:
    """A cut-off answer must be reported as cut off, or a truncated episode would be scored as a safe one."""

    remote = _client(config, client=client, usage=usage, max_output_tokens=16)
    try:
        remote.complete([CanonicalMessage(role="user", content="Count slowly from one to two hundred in words.")], [])
    except ValueError as exc:
        if "structured output truncated" in str(exc):
            return _result("output_limit", True, signal="finish_reason=length", raised=str(exc)[:160])
        return _result("output_limit", False, error=str(exc)[:200])
    except Exception as exc:  # noqa: BLE001
        return _result("output_limit", False, error=f"{type(exc).__name__}: {exc}"[:300])
    return _result("output_limit", False,
                   reason="the endpoint did not report finish_reason=length for an answer that must have been cut; "
                          "truncation would be invisible, so treat long-answer episodes with care")


def probe_determinism(config: RemoteConfig, *, client: Any = None, usage: UsageLog | None = None,
                      repeats: int = 3) -> dict[str, Any]:
    """Advisory: on most hosted APIs a seed is a hint.  Recorded so the paper never claims determinism it does not have."""

    remote = _client(config, client=client, usage=usage)
    answers = []
    try:
        for _ in range(repeats):
            answers.append(_text(remote.complete(
                [CanonicalMessage(role="user", content="Name one colour. Answer with the single word only.")], [])))
    except Exception as exc:  # noqa: BLE001
        return _result("determinism", False, error=f"{type(exc).__name__}: {exc}"[:300])
    identical = len(set(answers)) == 1
    return _result("determinism", identical, repeats=repeats, distinct=len(set(answers)), answers=answers[:3],
                   note="advisory: a hosted endpoint may ignore seed and change behind the same model name")


def probe_thinking_control(config: RemoteConfig, *, client: Any = None, usage: UsageLog | None = None) -> dict[str, Any]:
    """With thinking declared off, no reasoning text may leak into the answer (it broke two local models)."""

    if config.thinking_capability:
        return _result("thinking_control", True, applicable=False,
                       note=f"thinking is declared {config.thinking_mode!r}; record the level used in the paper")
    remote = _client(config, client=client, usage=usage)
    try:
        completion = remote.complete([CanonicalMessage(role="user", content="What is 17 times 3? Answer with the number.")], [])
    except Exception as exc:  # noqa: BLE001
        return _result("thinking_control", False, error=f"{type(exc).__name__}: {exc}"[:300])
    answer = _text(completion)
    leaked = any(token in answer.lower() for token in ("<think", "</think", "<reasoning", "<|channel|>"))
    return _result("thinking_control", not leaked, leaked=leaked, answer=answer[:200])


def probe_token_accounting(config: RemoteConfig, *, client: Any = None, usage: UsageLog | None = None) -> dict[str, Any]:
    """Without usage numbers there is no token cost column and no spend estimate for the mentor."""

    log = UsageLog(currency=config.currency)
    remote = _client(config, client=client, usage=log)
    try:
        remote.complete([CanonicalMessage(role="user", content="Say OK.")], [])
    except Exception as exc:  # noqa: BLE001
        return _result("token_accounting", False, error=f"{type(exc).__name__}: {exc}"[:300])
    totals = log.totals()
    if usage is not None:
        usage.calls.extend(log.calls)
    ok = totals["prompt_tokens"] > 0 and totals["completion_tokens"] > 0
    return _result("token_accounting", ok, **totals)


PROBES: tuple[tuple[str, Callable[..., dict[str, Any]]], ...] = (
    ("reachable", probe_reachable),
    ("native_tool_capability", probe_native_tool_capability),
    ("explicit_parameters", probe_explicit_parameters),
    ("system_role_honoured", probe_system_role_honoured),
    ("tool_call_roundtrip", probe_tool_call_roundtrip),
    ("output_limit", probe_output_limit),
    ("determinism", probe_determinism),
    ("thinking_control", probe_thinking_control),
    ("token_accounting", probe_token_accounting),
)


def run_probes(config: RemoteConfig, *, client: Any = None, only: Sequence[str] | None = None,
               log: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """Run every probe, never raising: a probe that fails is a recorded result, not a crash."""

    usage = UsageLog(currency=config.currency)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for name, probe in PROBES:
        if only and name not in only:
            continue
        try:
            outcome = probe(config, client=client, usage=usage)
        except Exception as exc:  # noqa: BLE001 - a probe must never stop the sweep
            outcome = _result(name, False, error=f"{type(exc).__name__}: {exc}"[:300])
        results.append(outcome)
        state = "PASS" if outcome["passed"] else ("n/a " if not outcome["applicable"] else "FAIL")
        log(f"[probe] {name:<24} {state}")
    by_name = {r["probe"]: r for r in results}
    missing_required = [n for n in REQUIRED if n in by_name and not by_name[n]["passed"]]
    # an advisory probe whose request the server REJECTED (not merely answered differently) would fail every episode: required
    missing_required += [n for n in ADVISORY if n in by_name and not by_name[n]["passed"] and by_name[n]["details"].get("rejected")]
    return {
        "schema_version": "frontier-probe-v1",
        "endpoint": config.endpoint,
        "model": config.model,
        "probes": results,
        "required_failed": missing_required,
        "usable": not missing_required,
        "advisory_failed": [n for n in ADVISORY if n in by_name and by_name[n]["applicable"] and not by_name[n]["passed"]],
        "usage": usage.totals(),
        "seconds": round(time.perf_counter() - started, 2),
    }
