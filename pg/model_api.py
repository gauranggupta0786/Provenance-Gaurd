"""Week 2 model-independent inference API and canonical provider adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

import httpx
import openai

from .trusted_models import ModelProfile, ToolIntent


Role = Literal["system", "developer", "user", "assistant", "tool"]
ThinkingMode = Literal["disabled", "enabled", "low", "medium", "high"]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CanonicalMessage:
    role: Role
    content: Any
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CanonicalMessage":
        role = str(value["role"])
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported canonical role: {role}")
        return cls(
            role=role,  # type: ignore[arg-type]
            content=value.get("content"),
            tool_calls=tuple(value.get("tool_calls") or ()),
            tool_call_id=value.get("tool_call_id"),
            name=value.get("name"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [dict(call) for call in self.tool_calls]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            result["name"] = self.name
        return result


@dataclass(frozen=True)
class CanonicalToolSchema:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.parameters, Mapping):
            raise ValueError("canonical tools require a name and parameter mapping")

    @classmethod
    def from_openai(cls, value: Mapping[str, Any]) -> "CanonicalToolSchema":
        function = value.get("function", value)
        return cls(
            name=str(function["name"]),
            description=str(function.get("description", "")),
            parameters=dict(function.get("parameters", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class ModelCompletion:
    response_id: str | None
    content: str | None
    tool_intents: tuple[ToolIntent, ...]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: Any

    def evidence_dict(self) -> dict[str, Any]:
        raw = self.raw_response
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="json")
        return {
            "response_id": self.response_id,
            "content": self.content,
            "tool_intents": [intent.to_dict() for intent in self.tool_intents],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "raw_response": raw,
        }


class ProviderAdapter(Protocol):
    name: str

    def build_payload(
        self,
        profile: ModelProfile,
        messages: Sequence[CanonicalMessage],
        tools: Sequence[CanonicalToolSchema],
    ) -> dict[str, Any]: ...

    def normalize_response(self, response: Any) -> ModelCompletion: ...


class ModelClient(Protocol):
    profile: ModelProfile
    adapter: ProviderAdapter

    def list_models(self) -> list[str]: ...

    def server_information(self) -> Mapping[str, Any]: ...

    def complete(
        self,
        messages: Sequence[CanonicalMessage],
        tools: Sequence[CanonicalToolSchema] = (),
    ) -> ModelCompletion: ...


def _stable_call_id(response_id: str | None, index: int, name: str, arguments: Any) -> str:
    material = {"response_id": response_id, "index": index, "name": name, "arguments": arguments}
    return f"call-{canonical_hash(material)[:16]}"


def _parse_arguments(raw: Any) -> tuple[Mapping[str, Any] | None, str | None]:
    if isinstance(raw, Mapping):
        return dict(raw), None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"JSONDecodeError: {exc}"
        if isinstance(parsed, Mapping):
            return dict(parsed), None
        return None, "tool arguments must decode to an object"
    return None, f"unsupported tool argument type: {type(raw).__name__}"


def _text_content(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts = []
        for block in content:
            if isinstance(block, Mapping):
                value = block.get("text", block.get("content"))
                if value is not None:
                    parts.append(str(value))
        return "\n".join(parts)
    return str(content)


class OpenAIProviderAdapter:
    name = "openai-chat-completions-v1"

    def build_payload(self, profile, messages, tools):
        provider_messages = []
        for message in messages:
            payload = message.to_dict()
            if payload["role"] in {"system", "developer"}:
                payload["role"] = profile.system_role
            provider_messages.append(payload)
        payload: dict[str, Any] = {
            "model": profile.model_id,
            "messages": provider_messages,
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "seed": profile.seed,
            "max_tokens": profile.max_output_tokens,
        }
        if profile.stop_sequences:
            payload["stop"] = list(profile.stop_sequences)
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool.to_dict()} for tool in tools
            ]
        if profile.thinking_capability:
            payload["reasoning_effort"] = "none" if profile.thinking_mode == "disabled" else profile.thinking_mode
        return payload

    def normalize_response(self, response: Any) -> ModelCompletion:
        message = response.choices[0].message
        response_id = getattr(response, "id", None)
        intents = []
        for index, call in enumerate(getattr(message, "tool_calls", None) or []):
            raw_arguments = call.function.arguments
            arguments, error = _parse_arguments(raw_arguments)
            intents.append(ToolIntent(
                schema_version="tool-intent-v1",
                call_id=getattr(call, "id", None) or _stable_call_id(
                    response_id, index, call.function.name, raw_arguments,
                ),
                tool_name=call.function.name,
                arguments=arguments,
                raw_arguments=raw_arguments,
                source_response_id=response_id,
                parse_error=error,
            ))
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", prompt + completion) or prompt + completion)
        return ModelCompletion(
            response_id=response_id,
            content=getattr(message, "content", None),
            tool_intents=tuple(intents),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            raw_response=response,
        )


class OllamaNativeProviderAdapter:
    name = "ollama-native-chat-v1"

    def build_payload(self, profile, messages, tools):
        provider_messages = []
        for message in messages:
            item = message.to_dict()
            if item["role"] == "developer":
                item["role"] = "system"
            item["content"] = _text_content(item.get("content")) or ""
            if item.get("tool_calls"):
                native_calls = []
                for call in item["tool_calls"]:
                    function = call.get("function", call)
                    arguments, error = _parse_arguments(function.get("arguments", {}))
                    if error:
                        raise ValueError(f"cannot serialize prior tool call: {error}")
                    native_calls.append({
                        "function": {"name": function["name"], "arguments": dict(arguments or {})},
                    })
                item["tool_calls"] = native_calls
            provider_messages.append(item)
        options = {
            "num_ctx": profile.selected_context_length,
            "num_predict": profile.max_output_tokens,
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "top_k": profile.top_k,
            "min_p": profile.min_p,
            "seed": profile.seed,
            **dict(profile.extra_options),
        }
        payload: dict[str, Any] = {
            "model": profile.model_id,
            "messages": provider_messages,
            "stream": False,
            "options": options,
        }
        if profile.thinking_capability:
            # Never send `think` to a model whose /api/show capabilities lack
            # "thinking": some Ollama versions reject it for those models.
            mode = profile.thinking_mode
            # Ollama accepts true/false or a level (low/medium/high/max); the string "enabled" is rejected by
            # qwen3 ("invalid think value"), so a plain "on" is sent as true.
            payload["think"] = False if mode == "disabled" else (True if mode == "enabled" else mode)
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool.to_dict()} for tool in tools
            ]
        return payload

    def normalize_response(self, response: Mapping[str, Any]) -> ModelCompletion:
        message = response.get("message", {})
        response_id = response.get("id") or response.get("created_at")
        intents = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function", {})
            name = str(function.get("name", ""))
            raw_arguments = function.get("arguments", {})
            arguments, error = _parse_arguments(raw_arguments)
            intents.append(ToolIntent(
                schema_version="tool-intent-v1",
                call_id=call.get("id") or _stable_call_id(response_id, index, name, raw_arguments),
                tool_name=name,
                arguments=arguments,
                raw_arguments=raw_arguments,
                source_response_id=response_id,
                parse_error=error,
            ))
        prompt = int(response.get("prompt_eval_count", 0) or 0)
        completion = int(response.get("eval_count", 0) or 0)
        return ModelCompletion(
            response_id=response_id,
            content=message.get("content"),
            tool_intents=tuple(intents),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            raw_response=dict(response),
        )


class OpenAICompatibleModelClient:
    def __init__(self, profile: ModelProfile, *, client: Any | None = None) -> None:
        self.profile = profile
        self.adapter = OpenAIProviderAdapter()
        self.client = client or openai.OpenAI(
            api_key="ollama" if "127.0.0.1" in profile.endpoint else "unused",
            base_url=profile.endpoint.rstrip("/"),
            timeout=profile.model_timeout_seconds,
            max_retries=0,
        )

    def list_models(self) -> list[str]:
        return [entry.id for entry in self.client.models.list().data]

    def server_information(self) -> Mapping[str, Any]:
        return {"provider": "openai-compatible", "endpoint": self.profile.endpoint}

    def complete(self, messages, tools=()) -> ModelCompletion:
        response = self.client.chat.completions.create(**self.adapter.build_payload(self.profile, messages, tools))
        return self.adapter.normalize_response(response)


class OllamaNativeModelClient:
    def __init__(self, profile: ModelProfile, *, client: httpx.Client | None = None) -> None:
        self.profile = profile
        self.adapter = OllamaNativeProviderAdapter()
        self.client = client or httpx.Client(
            base_url=profile.endpoint.rstrip("/"), timeout=profile.model_timeout_seconds,
        )

    def list_models(self) -> list[str]:
        response = self.client.get("/api/tags")
        response.raise_for_status()
        return [str(item["name"]) for item in response.json().get("models", [])]

    def server_information(self) -> Mapping[str, Any]:
        version = self.client.get("/api/version")
        version.raise_for_status()
        return {
            "provider": "ollama-native",
            "endpoint": self.profile.endpoint,
            "version": version.json().get("version"),
        }

    def complete(self, messages, tools=()) -> ModelCompletion:
        response = self.client.post(
            "/api/chat", json=self.adapter.build_payload(self.profile, messages, tools),
        )
        response.raise_for_status()
        return self.adapter.normalize_response(response.json())


def load_model_profile(path: str | Path | None = None) -> ModelProfile:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "models" / "qwen35-9b-ollama.json"
    return ModelProfile.from_json(path)


def build_inference_manifest(
    profile: ModelProfile,
    *,
    adapter: ProviderAdapter,
    messages: Sequence[CanonicalMessage],
    tools: Sequence[CanonicalToolSchema],
    server_information: Mapping[str, Any],
    hardware_information: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hardware = dict(hardware_information or {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    })
    return {
        "schema_version": "week2-inference-manifest-v1",
        "profile": profile.to_dict(),
        "profile_sha256": profile.fingerprint,
        "adapter": adapter.name,
        "server": dict(server_information),
        "hardware": hardware,
        "conversation_sha256": canonical_hash([message.to_dict() for message in messages]),
        "prompt_sha256": [canonical_hash(message.content) for message in messages if message.role == "user"],
        "tool_schema_sha256": canonical_hash([tool.to_dict() for tool in tools]),
        "chat_template_sha256": profile.chat_template_sha256,
    }
