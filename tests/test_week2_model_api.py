import json
from types import SimpleNamespace

import pytest

from pg.model_api import (
    CanonicalMessage,
    CanonicalToolSchema,
    ModelProfile,
    OllamaNativeProviderAdapter,
    OpenAIProviderAdapter,
    build_inference_manifest,
    load_model_profile,
)


def test_pinned_model_profile_is_complete_and_self_hashing():
    profile = load_model_profile()
    assert profile.model_digest == "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
    assert profile.parameter_count == 9_653_104_368
    assert profile.selected_context_length == 8192
    assert profile.thinking_mode == "disabled"
    assert len(profile.fingerprint) == 64


def test_profile_rejects_unverifiable_digest_template_and_context():
    profile = load_model_profile()
    with pytest.raises(ValueError, match="pattern"):
        type(profile).model_validate({**profile.model_dump(), "model_digest": "6488c96fa5fa"})
    with pytest.raises(ValueError, match="chat_template_sha256"):
        type(profile).model_validate({**profile.model_dump(), "chat_template_sha256": "0" * 64})
    with pytest.raises(ValueError, match="native context"):
        type(profile).model_validate({**profile.model_dump(), "selected_context_length": profile.native_context_length + 1})


def test_both_adapters_resolve_the_same_canonical_request_without_defaults():
    profile = load_model_profile()
    messages = [CanonicalMessage(role="system", content="Use tools."), CanonicalMessage(role="user", content="List files")]
    tools = [CanonicalToolSchema("list_files", "List files", {"type": "object", "properties": {}})]
    native = OllamaNativeProviderAdapter().build_payload(profile, messages, tools)
    openai_profile = profile.revalidated(
        provider="openai-compatible", endpoint="http://localhost:11434/v1",
        tool_call_format="openai-functions-v1", system_role="developer",
    )
    compatible = OpenAIProviderAdapter().build_payload(openai_profile, messages, tools)
    assert native["options"] == {
        "num_ctx": 8192, "num_predict": 1024, "temperature": 0.0,
        "top_p": 1.0, "top_k": 20, "min_p": 0.0, "seed": 42,
        "num_batch": 512, "num_keep": 0,
    }
    assert native["think"] is False
    assert compatible["temperature"] == 0.0
    assert compatible["top_p"] == 1.0
    assert compatible["seed"] == 42
    assert compatible["max_tokens"] == 1024
    assert compatible["reasoning_effort"] == "none"
    assert native["tools"][0]["function"]["name"] == compatible["tools"][0]["function"]["name"]


def test_provider_outputs_normalize_to_equivalent_tool_intents():
    native_raw = {
        "id": "native-response", "message": {"content": "", "tool_calls": [
            {"id": "call-1", "function": {"name": "send_email", "arguments": {"subject": "Hi"}}},
        ]}, "prompt_eval_count": 7, "eval_count": 3,
    }
    native = OllamaNativeProviderAdapter().normalize_response(native_raw)
    function = SimpleNamespace(name="send_email", arguments='{"subject":"Hi"}')
    call = SimpleNamespace(id="call-1", function=function)
    response = SimpleNamespace(
        id="openai-response",
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
    )
    compatible = OpenAIProviderAdapter().normalize_response(response)
    assert native.tool_intents[0].tool_name == compatible.tool_intents[0].tool_name == "send_email"
    assert native.tool_intents[0].arguments == compatible.tool_intents[0].arguments == {"subject": "Hi"}
    assert native.total_tokens == compatible.total_tokens == 10


def test_malformed_tool_arguments_remain_explicit_in_canonical_intent():
    function = SimpleNamespace(name="send_email", arguments="not-json")
    response = SimpleNamespace(
        id="bad", choices=[SimpleNamespace(message=SimpleNamespace(
            content=None, tool_calls=[SimpleNamespace(id="bad-call", function=function)],
        ))], usage=None,
    )
    intent = OpenAIProviderAdapter().normalize_response(response).tool_intents[0]
    assert not intent.valid
    assert "JSONDecodeError" in intent.parse_error
    assert intent.raw_arguments == "not-json"


def test_inference_manifest_hashes_profile_prompt_template_and_tools():
    profile = load_model_profile()
    messages = [CanonicalMessage(role="user", content="List files")]
    tools = [CanonicalToolSchema("list_files", "List files", {"type": "object"})]
    manifest = build_inference_manifest(
        profile, adapter=OllamaNativeProviderAdapter(), messages=messages, tools=tools,
        server_information={"version": "0.34.4"}, hardware_information={"cpu": "test"},
    )
    assert manifest["profile_sha256"] == profile.fingerprint
    assert manifest["chat_template_sha256"] == profile.chat_template_sha256
    assert len(manifest["conversation_sha256"]) == 64
    assert len(manifest["tool_schema_sha256"]) == 64
    assert manifest["server"]["version"] == "0.34.4"
    assert manifest["hardware"]["cpu"] == "test"
