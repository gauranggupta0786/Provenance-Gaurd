"""Run ProvenanceGuard against any OpenAI-compatible endpoint, configured only by environment variables.

This is the piece that makes the frontier evaluation someone else's one-command job.  Everything above the model -- the
guard, the gateway, the reviewer, the evidence store, the scoring -- is unchanged; only where the tokens come from moves.

    export PROVENANCEGUARD_API_BASE=https://api.example.com/v1
    export PROVENANCEGUARD_API_KEY=sk-...
    export PROVENANCEGUARD_MODEL=some-model
    python scripts/frontier_probe.py

Ollama serves an OpenAI-compatible endpoint of its own at ``/v1``, so the whole path is developed and proven for free
against a local model before anyone spends anything: ``scripts/week12_adapter_parity.py`` runs the same scenarios through
the native Ollama client and through this adapter and compares them.

**What cannot be checked on a hosted model, stated plainly.**  The Week 8 identity checks pin a local model by weight
digest, chat template and engine version read from the serving process.  A hosted endpoint exposes none of those: the
provider can change the weights, the template, the tokenizer or the decoding defaults between two calls and nothing in the
response would say so.  ``remote_session_record`` therefore records what *is* observable (endpoint, model string, declared
settings, probe results, response fingerprints, timing) and lists the unverifiable parts explicitly, so a frontier run is
never reported as if it had the same provenance guarantees as the local runs.  ``model_digest`` on a remote profile is a
digest of the *declared identity*, not of any weights, and is labelled as such everywhere it appears.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from pg.diagnostics import EpisodeStage, RuntimeLimits
from pg.model_api import CanonicalMessage, CanonicalToolSchema, ModelCompletion, OpenAICompatibleModelClient, canonical_hash
from pg.ollama_model import SeededOllamaLLM
from pg.trusted_models import ModelProfile

ENV_PREFIX = "PROVENANCEGUARD_"
REQUIRED_ENV = (f"{ENV_PREFIX}API_BASE", f"{ENV_PREFIX}MODEL")
REMOTE_TEMPLATE = "<served remotely: the chat template is applied by the provider and is not retrievable>"

#: provider differences the request can be adapted to, and nothing else.  Each is recorded in the session record, because a
#: dropped decoding parameter means the provider's default was used and a run must never claim otherwise.
TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")
OMITTABLE_PARAMS = ("temperature", "top_p", "seed")
SYSTEM_ROLES = ("system", "developer")

#: what a hosted endpoint cannot tell us, and so what a frontier run cannot claim
UNVERIFIABLE_REMOTELY = (
    "model weights: no digest is exposed, so a silent model update is undetectable",
    "chat template: applied provider-side and not retrievable",
    "tokenizer and context handling: prompt truncation, if any, is not reported",
    "decoding defaults: settings not named in the request are the provider's, and may change",
    "serving engine and version: not exposed",
    "determinism: a seed is advisory on most hosted APIs even when accepted",
)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(f"{ENV_PREFIX}{name}")
    return value if value not in (None, "") else default


def _env_number(name: str, default, cast):
    raw = _env(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name}={raw!r} is not a valid {cast.__name__}") from exc


@dataclass(frozen=True)
class RemoteConfig:
    """Everything the adapter needs, all of it from the environment so no code changes per provider."""

    api_base: str
    model: str
    api_key: str = ""
    context_length: int = 32768
    max_output_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    timeout_seconds: float = 180.0
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    thinking_mode: str = "disabled"
    thinking_capability: bool = False
    # "system" is what the local native runs sent (the Ollama client maps developer -> system) and is accepted by every
    # OpenAI-compatible server; "developer" is OpenAI-specific and is rejected or mangled by several others.
    system_role: str = "system"
    # Some providers' newer models reject ``max_tokens`` and want ``max_completion_tokens``; some reject a non-default
    # ``temperature``/``top_p``/``seed`` outright.  Both are env-configurable instead of a code change per provider.
    token_param: str = "max_tokens"
    omit_params: tuple[str, ...] = ()
    parameter_count: int = 1          # hosted models rarely state one; recorded as unknown, never as fact
    parameter_count_known: bool = False
    price_per_1k_input: float = 0.0   # 0 for a local endpoint; set for a priced model to get a cost column
    price_per_1k_output: float = 0.0
    currency: str = "USD"

    @classmethod
    def from_env(cls, **overrides: Any) -> "RemoteConfig":
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise ValueError(f"set {', '.join(missing)} (see docs/FRONTIER_HANDOFF.md)")
        config = cls(
            api_base=_env("API_BASE"),
            model=_env("MODEL"),
            api_key=_env("API_KEY", "") or "",
            context_length=_env_number("CONTEXT", 32768, int),
            max_output_tokens=_env_number("MAX_OUTPUT_TOKENS", 4096, int),
            temperature=_env_number("TEMPERATURE", 0.0, float),
            top_p=_env_number("TOP_P", 1.0, float),
            seed=_env_number("SEED", 42, int),
            timeout_seconds=_env_number("TIMEOUT_SECONDS", 180.0, float),
            max_retries=_env_number("MAX_RETRIES", 2, int),
            thinking_mode=_env("THINKING", "disabled"),
            thinking_capability=(_env("THINKING", "disabled") or "disabled") != "disabled",
            system_role=_env("SYSTEM_ROLE", "system"),
            token_param=_env("TOKEN_PARAM", "max_tokens"),
            omit_params=tuple(p.strip() for p in (_env("OMIT_PARAMS", "") or "").split(",") if p.strip()),
            price_per_1k_input=_env_number("PRICE_PER_1K_INPUT", 0.0, float),
            price_per_1k_output=_env_number("PRICE_PER_1K_OUTPUT", 0.0, float),
            currency=_env("CURRENCY", "USD"),
        )
        config = replace(config, **overrides) if overrides else config
        config.validate()
        return config

    def validate(self) -> None:
        """Refuse a setting that would fail on the first call, with the variable to fix, instead of an HTTP error mid-run."""

        if self.system_role not in SYSTEM_ROLES:
            raise ValueError(f"{ENV_PREFIX}SYSTEM_ROLE={self.system_role!r} must be one of {SYSTEM_ROLES}")
        if self.token_param not in TOKEN_PARAMS:
            raise ValueError(f"{ENV_PREFIX}TOKEN_PARAM={self.token_param!r} must be one of {TOKEN_PARAMS}")
        unknown = [p for p in self.omit_params if p not in OMITTABLE_PARAMS]
        if unknown:
            raise ValueError(f"{ENV_PREFIX}OMIT_PARAMS has {unknown}; only {OMITTABLE_PARAMS} can be omitted")
        if not self.api_base.startswith(("http://", "https://")):
            raise ValueError(f"{ENV_PREFIX}API_BASE={self.api_base!r} must start with http:// or https://")

    @property
    def endpoint(self) -> str:
        return self.api_base.rstrip("/")

    @property
    def identity_digest(self) -> str:
        """A digest of the DECLARED identity -- endpoint and model string.  Not a digest of any weights."""

        return hashlib.sha256(json.dumps({"endpoint": self.endpoint, "model": self.model},
                                         sort_keys=True).encode("utf-8")).hexdigest()

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(prompt_tokens / 1000 * self.price_per_1k_input
                     + completion_tokens / 1000 * self.price_per_1k_output, 6)


def finish_reason(completion: ModelCompletion) -> str | None:
    """``choices[0].finish_reason`` from an OpenAI-shaped response, whether it is an SDK object or a plain dict."""

    raw = completion.raw_response
    choices = raw.get("choices") if isinstance(raw, Mapping) else getattr(raw, "choices", None)
    if not choices:
        return None
    first = choices[0]
    value = first.get("finish_reason") if isinstance(first, Mapping) else getattr(first, "finish_reason", None)
    return str(value) if value is not None else None


def redact(text: str, *secrets: str | None) -> str:
    """``text`` with every secret removed.  Providers sometimes echo (part of) a rejected key in their error body, and
    everything this package writes to disk -- episode rows, probe reports -- passes through here first.  Secrets shorter than
    8 characters are placeholders (``unused``, ``ollama``) and are left alone."""

    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "***REDACTED***")
    return text


def apply_param_policy(payload: Mapping[str, Any], config: RemoteConfig) -> dict[str, Any]:
    """The request as it goes on the wire: the adapter's payload with the provider adaptations in ``config`` applied.

    Only renames the token limit and drops the named decoding parameters.  It never adds a parameter and never touches the
    messages or tools, so what the model is asked is unchanged.
    """

    wire = dict(payload)
    if config.token_param != "max_tokens" and "max_tokens" in wire:
        wire[config.token_param] = wire.pop("max_tokens")
    for name in config.omit_params:
        wire.pop(name, None)
    return wire


class _PolicyCompletions:
    def __init__(self, inner: Any, config: RemoteConfig) -> None:
        self._inner, self._config = inner, config

    def create(self, **payload: Any) -> Any:
        return self._inner.create(**apply_param_policy(payload, self._config))


class _PolicyClient:
    """Wraps an SDK client so ``chat.completions.create`` sends the provider-adapted request; everything else passes through."""

    def __init__(self, inner: Any, config: RemoteConfig) -> None:
        self._inner = inner
        self.models = inner.models
        self.chat = type("Chat", (), {"completions": _PolicyCompletions(inner.chat.completions, config)})()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_sdk_client(config: RemoteConfig) -> Any:
    """The real OpenAI SDK client for ``config``, carrying the configured API key.

    ``pg.model_api.OpenAICompatibleModelClient`` builds its own client when it is not handed one, and that client sends the
    literal key ``"unused"`` to any endpoint that is not on 127.0.0.1 -- correct for a local server, a guaranteed 401 for a
    hosted one.  ``pg/`` is frozen (``code_digest`` hashes it), so the key is supplied here, by passing the client in.
    """

    import openai

    return openai.OpenAI(api_key=config.api_key or "unused", base_url=config.endpoint,
                         timeout=config.timeout_seconds, max_retries=0)


def build_remote_profile(config: RemoteConfig) -> ModelProfile:
    """A ModelProfile for a hosted model.  Fields a hosted endpoint cannot supply are declared, never invented as fact."""

    return ModelProfile.from_dict({
        "schema_version": "model-profile-v1",
        "profile_id": f"remote-{config.identity_digest[:12]}",
        "model_id": config.model,
        "model_digest": config.identity_digest,          # declared identity, not weights -- see UNVERIFIABLE_REMOTELY
        "model_family": "remote",
        "parameter_count": max(1, int(config.parameter_count)),
        "serving_engine": "openai-compatible",
        "serving_engine_version": "unknown",
        "provider": "openai-compatible",
        "endpoint": config.endpoint,
        "quantization": "unknown",
        "numerical_precision": "unknown",
        "chat_template": REMOTE_TEMPLATE,
        "chat_template_sha256": hashlib.sha256(REMOTE_TEMPLATE.encode("utf-8")).hexdigest(),
        "tool_call_format": "openai-functions-v1",
        "native_context_length": config.context_length,
        "selected_context_length": config.context_length,
        "thinking_capability": config.thinking_capability,
        "thinking_mode": config.thinking_mode,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": 0,
        "min_p": 0.0,
        "seed": config.seed,
        "max_output_tokens": config.max_output_tokens,
        "stop_sequences": [],
        "model_timeout_seconds": config.timeout_seconds,
        "retry_policy": f"{config.max_retries} retries, exponential backoff from {config.retry_backoff_seconds}s",
        "infrastructure_retries": 1 if config.max_retries else 0,
        "system_role": config.system_role,
        "extra_options": {},
    })


@dataclass
class UsageLog:
    """Per-call tokens, latency and cost.  Cost is 0 for a local endpoint and real once prices are set."""

    currency: str = "USD"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, completion: ModelCompletion, *, latency_ms: int, cost: float, attempts: int) -> None:
        self.calls.append({
            "response_id": completion.response_id, "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens, "total_tokens": completion.total_tokens,
            "latency_ms": latency_ms, "cost": cost, "currency": self.currency, "attempts": attempts,
            "tool_calls": len(completion.tool_intents),
        })

    def totals(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in self.calls),
            "completion_tokens": sum(c["completion_tokens"] for c in self.calls),
            "total_tokens": sum(c["total_tokens"] for c in self.calls),
            "latency_ms_total": sum(c["latency_ms"] for c in self.calls),
            "latency_ms_mean": round(sum(c["latency_ms"] for c in self.calls) / len(self.calls)) if self.calls else 0,
            "cost": round(sum(c["cost"] for c in self.calls), 6),
            "currency": self.currency,
            "retried_calls": sum(1 for c in self.calls if c["attempts"] > 1),
        }


class MeteredRemoteClient:
    """``OpenAICompatibleModelClient`` plus retries with backoff and a token/latency/cost log.

    Retries cover transport and 5xx/429 failures only.  A response that arrives is never retried, so a retry can never
    change a measured outcome -- it only survives a flaky network.
    """

    RETRYABLE = ("timeout", "connection", "temporarily", "rate limit", "429", "500", "502", "503", "504", "overloaded")

    def __init__(self, profile: ModelProfile, config: RemoteConfig, *, client: Any | None = None,
                 usage: UsageLog | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.profile = profile
        self.config = config
        sdk = client if client is not None else build_sdk_client(config)
        if config.token_param != "max_tokens" or config.omit_params:
            sdk = _PolicyClient(sdk, config)
        self.inner = OpenAICompatibleModelClient(profile, client=sdk)
        self.adapter = self.inner.adapter
        self.usage = usage if usage is not None else UsageLog(currency=config.currency)
        self._sleep = sleep

    def _retryable(self, exc: Exception) -> bool:
        text = f"{type(exc).__name__} {exc}".lower()
        return any(token in text for token in self.RETRYABLE)

    def _with_retries(self, operation: Callable[[], Any], stage: EpisodeStage) -> tuple[Any, int]:
        last: Exception | None = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                return operation(), attempt
            except Exception as exc:  # noqa: BLE001 - classified below, re-raised when not retryable
                last = exc
                if attempt > self.config.max_retries or not self._retryable(exc):
                    raise
                delay = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                self._sleep(delay + random.uniform(0, 0.25 * delay))
        raise last  # pragma: no cover - the loop either returns or raises

    def list_models(self) -> list[str]:
        models, _ = self._with_retries(self.inner.list_models, EpisodeStage.HEALTHCHECK)
        return models

    def server_information(self) -> Mapping[str, Any]:
        return {"provider": "openai-compatible", "endpoint": self.profile.endpoint, "model": self.profile.model_id,
                "version": "unknown (hosted endpoint)"}

    def complete(self, messages: Sequence[CanonicalMessage], tools: Sequence[CanonicalToolSchema] = ()) -> ModelCompletion:
        started = time.perf_counter()
        completion, attempts = self._with_retries(lambda: self.inner.complete(messages, tools), EpisodeStage.CALL_MODEL)
        self.usage.record(
            completion, latency_ms=round((time.perf_counter() - started) * 1000),
            cost=self.config.cost(completion.prompt_tokens, completion.completion_tokens), attempts=attempts,
        )
        # GuardedModelClient reads Ollama's ``done_reason`` out of a Mapping response; an OpenAI response carries the same
        # fact as ``choices[0].finish_reason`` on an object, so the same failure is raised here, with the same wording, or a
        # truncated hosted answer would be scored as "the model did nothing" -- the exact bias the guard exists to prevent.
        if finish_reason(completion) == "length":
            raise ValueError(
                f"structured output truncated at the max_output_tokens limit (finish_reason=length, "
                f"{completion.completion_tokens} tokens); the episode outcome is unknown, not safe: "
                f"{(completion.content or '')[:200]!r}")
        return completion


def build_remote_llm(config: RemoteConfig | None = None, *, limits: RuntimeLimits | None = None, client: Any | None = None,
                     usage: UsageLog | None = None, guard: bool = True,
                     sleep: Callable[[float], None] = time.sleep, seed: int | None = None) -> SeededOllamaLLM:
    """The same pipeline element the local runs use, talking to a hosted model instead.

    ``seed`` is the episode's campaign seed (42, 314 or 2718).  The local runs built every model with the episode's own seed;
    without this the three campaign seeds would all send ``config.seed`` and be one setting repeated three times.  A provider
    that ignores or rejects ``seed`` (``PROVENANCEGUARD_OMIT_PARAMS``) makes this advisory, as the hand-off guide says.
    """

    from pg.week7_models import GuardedModelClient

    config = config or RemoteConfig.from_env()
    if seed is not None and int(seed) != config.seed:
        config = replace(config, seed=int(seed))
    limits = limits or RuntimeLimits(model_seconds=config.timeout_seconds,
                                     infrastructure_retries=1 if config.max_retries else 0)
    profile = build_remote_profile(config)
    inner = MeteredRemoteClient(profile, config, client=client, usage=usage, sleep=sleep)
    return SeededOllamaLLM(
        model=profile.model_id, seed=profile.seed, temperature=profile.temperature,
        max_tokens=profile.max_output_tokens, runtime_limits=limits,
        model_client=GuardedModelClient(inner) if guard else inner,
    )


def remote_session_record(config: RemoteConfig, llm: SeededOllamaLLM, *, probes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What a hosted run can honestly say about its own identity, and what it cannot."""

    profile = llm.profile if hasattr(llm, "profile") else build_remote_profile(config)
    return {
        "schema_version": "remote-session-v1",
        "endpoint": config.endpoint,
        "model": config.model,
        "declared_identity_digest": config.identity_digest,
        "declared_identity_digest_note": "sha256 of {endpoint, model}: it identifies what we ASKED for, not what ran. "
                                         "It is not a weight digest and must never be reported as one.",
        "declared_settings": {
            "context_length": config.context_length, "max_output_tokens": config.max_output_tokens,
            "temperature": config.temperature, "top_p": config.top_p, "seed": config.seed,
            "thinking_mode": config.thinking_mode, "system_role": config.system_role,
            "timeout_seconds": config.timeout_seconds, "max_retries": config.max_retries,
            "token_param": config.token_param,
            "omitted_params": list(config.omit_params),
            "omitted_params_note": "a parameter listed here was NOT sent, so the provider's own default applied to it",
        },
        "parameter_count_known": config.parameter_count_known,
        "portable_profile_sha256": canonical_hash({**profile.to_dict(), "endpoint": "<runtime>"}),
        "pricing": {"per_1k_input": config.price_per_1k_input, "per_1k_output": config.price_per_1k_output,
                    "currency": config.currency},
        "probes": dict(probes or {}),
        "cannot_be_verified_remotely": list(UNVERIFIABLE_REMOTELY),
        "comparability": "A hosted run is comparable to the local runs in dataset, scenarios, configurations and seeds, "
                         "and NOT in model provenance. Report it as a separate row with this record attached.",
    }
