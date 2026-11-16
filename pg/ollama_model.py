"""Pinned local Qwen3.5-9B model element for AgentDojo.

Ollama exposes an OpenAI-compatible endpoint, but the stock AgentDojo local
wrapper chooses a fresh random seed for every call.  This element keeps the
same AgentDojo message/tool conversion while fixing the experimental seed and
capturing token usage for reproducibility.
"""
from __future__ import annotations

from collections.abc import Sequence
import time
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import (
    _function_to_openai,
    _message_to_openai,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage

from .diagnostics import (
    EpisodeStage,
    RuntimeLimits,
    run_with_infrastructure_retry,
)
from .model_api import (
    CanonicalMessage,
    CanonicalToolSchema,
    ModelClient,
    OllamaNativeModelClient,
    OpenAICompatibleModelClient,
    load_model_profile,
)


DEFAULT_MODEL_PROFILE = load_model_profile()
MODEL_ID = DEFAULT_MODEL_PROFILE.model_id
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OLLAMA_NATIVE_BASE_URL = "http://127.0.0.1:11434"


class SeededOllamaLLM(BasePipelineElement):
    name = f"ollama-{MODEL_ID}"

    def __init__(
        self,
        *,
        model: str = MODEL_ID,
        base_url: str = OLLAMA_BASE_URL,
        seed: int = 42,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        client: Any | None = None,
        model_client: ModelClient | None = None,
        runtime_limits: RuntimeLimits | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.runtime_limits = runtime_limits or RuntimeLimits()
        profile = DEFAULT_MODEL_PROFILE.revalidated(
            model_id=model,
            seed=seed,
            temperature=temperature,
            max_output_tokens=max_tokens,
            model_timeout_seconds=self.runtime_limits.model_seconds,
            infrastructure_retries=self.runtime_limits.infrastructure_retries,
        )
        if model_client is not None:
            self.model_client = model_client
            self.profile = model_client.profile
        elif client is not None:
            # Preserve dependency injection for OpenAI-compatible test clients.
            self.profile = profile.revalidated(
                provider="openai-compatible", endpoint=self.base_url,
                tool_call_format="openai-functions-v1", system_role="developer",
            )
            self.model_client = OpenAICompatibleModelClient(self.profile, client=client)
        else:
            self.profile = profile.revalidated(endpoint=OLLAMA_NATIVE_BASE_URL)
            self.model_client = OllamaNativeModelClient(self.profile)
        self.client = self.model_client
        self.usage: list[dict[str, int]] = []
        self.call_diagnostics: list[dict[str, Any]] = []
        self.service_diagnostics: list[dict[str, Any]] = []

    def healthcheck(self) -> dict[str, Any]:
        attempt = 0
        started = 0.0

        def operation():
            nonlocal attempt, started
            attempt += 1
            started = time.perf_counter()
            return self.model_client.list_models()

        def failed(exc, category, failed_attempt, will_retry):
            self.service_diagnostics.append({
                "stage": EpisodeStage.HEALTHCHECK.value,
                "attempt": failed_attempt,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "failure_category": category.value,
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "will_retry": will_retry,
            })

        models = run_with_infrastructure_retry(
            operation,
            stage=EpisodeStage.HEALTHCHECK,
            max_retries=self.runtime_limits.infrastructure_retries,
            on_failure=failed,
        )
        ids = list(models)
        result = {
            "reachable": True,
            "configured_model": self.model,
            "available_models": ids,
            "model_available": self.model in ids,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "attempt": attempt,
        }
        self.service_diagnostics.append({"stage": EpisodeStage.HEALTHCHECK.value, **result})
        return result

    def warmup(self) -> dict[str, Any]:
        """Load the configured model before measured episodes."""

        attempt = 0
        started = 0.0

        def operation():
            nonlocal attempt, started
            attempt += 1
            started = time.perf_counter()
            return self.model_client.complete(
                [CanonicalMessage(role="user", content="Reply exactly READY.")],
            )

        def failed(exc, category, failed_attempt, will_retry):
            self.service_diagnostics.append({
                "stage": EpisodeStage.WARMUP.value,
                "attempt": failed_attempt,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "failure_category": category.value,
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "will_retry": will_retry,
            })

        completion = run_with_infrastructure_retry(
            operation,
            stage=EpisodeStage.WARMUP,
            max_retries=self.runtime_limits.infrastructure_retries,
            on_failure=failed,
        )
        result = {
            "stage": EpisodeStage.WARMUP.value,
            "attempt": attempt,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "model": self.model,
            "ready": True,
            "response_id": completion.response_id,
        }
        self.service_diagnostics.append(result)
        return result

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        extra_args = dict(extra_args or {})
        canonical_messages = [
            CanonicalMessage.from_mapping(_message_to_openai(message, self.model))
            for message in messages
        ]
        canonical_tools = [
            CanonicalToolSchema.from_openai(_function_to_openai(tool))
            for tool in runtime.functions.values()
        ]
        attempt = 0
        started = 0.0

        def operation():
            nonlocal attempt, started
            attempt += 1
            started = time.perf_counter()
            return self.model_client.complete(canonical_messages, canonical_tools)

        def failed(exc, category, failed_attempt, will_retry):
            self.call_diagnostics.append({
                "stage": EpisodeStage.CALL_MODEL.value,
                "attempt": failed_attempt,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "failure_category": category.value,
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "will_retry": will_retry,
                "raw_response": None,
                "parsed_tool_calls": [],
            })

        completion = run_with_infrastructure_retry(
            operation,
            stage=EpisodeStage.CALL_MODEL,
            max_retries=self.runtime_limits.infrastructure_retries,
            on_failure=failed,
        )
        usage_record = {
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens,
        }
        self.usage.append(usage_record)
        extra_args.setdefault("model_usage", []).append(usage_record)
        malformed = [intent for intent in completion.tool_intents if not intent.valid]
        if malformed:
            raise ValueError(f"malformed tool call: {malformed[0].parse_error}")
        output = ChatAssistantMessage(
            role="assistant",
            content=(
                [{"type": "text", "content": completion.content}]
                if completion.content else None
            ),
            tool_calls=(
                [
                    FunctionCall(
                        function=intent.tool_name,
                        args=dict(intent.arguments or {}),
                        id=intent.call_id,
                    )
                    for intent in completion.tool_intents
                ] or None
            ),
        )
        self.call_diagnostics.append({
            "stage": EpisodeStage.CALL_MODEL.value,
            "attempt": attempt,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "failure_category": None,
            "exception_type": None,
            "error": None,
            "will_retry": False,
            "raw_response": completion.evidence_dict()["raw_response"],
            "parsed_tool_calls": [intent.to_dict() for intent in completion.tool_intents],
            "model_profile_sha256": self.profile.fingerprint,
            "provider_adapter": self.model_client.adapter.name,
            **usage_record,
        })
        return query, runtime, env, [*messages, output], extra_args
