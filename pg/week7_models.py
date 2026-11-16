"""Week 7 -- resolve fully explicit, reproducible model profiles from a live server.

Nothing here guesses a digest, template, or context length. Every value that
ends up in a ``ModelProfile`` is read from the serving engine (``/api/tags``,
``/api/show``, ``/api/version``) or is an explicit, documented experimental
setting shared by all models in the pilot.

Important reproducibility detail: for models that use a built-in Ollama
renderer/parser (for example the installed Qwen3.5 profile, whose template is
just ``{{ .Prompt }}``), the ``template`` field alone does not pin the prompt
format. The resolution record therefore also stores the SHA-256 of the full
Modelfile text and any ``RENDERER`` / ``PARSER`` lines.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import httpx

from .diagnostics import RuntimeLimits
from .model_api import ModelCompletion, OllamaNativeModelClient
from .ollama_model import SeededOllamaLLM
from .trusted_models import ModelProfile

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "data" / "week7" / "model_candidates.json"
PROFILES_DIR = ROOT / "data" / "week7" / "profiles_final"  # the four frozen profiles

# Controlled, low-variance sampling shared by every model in the Week 7 pilot.
# Vendor-recommended sampling is a Week 9 ablation, not a Week 7 variable.
PILOT_SAMPLING = {
    "temperature": 0.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "seed": 42,
    # Measured (week7_diagnostics, 12 dev TIP episodes, 8192 cap): llama max 445, mistral 84, gemma4 3154 tokens in one
    # call. 4096 covers every model that finishes; qwen3.5 under greedy decoding runs away to the cap (see qwen diagnostic).
    "max_output_tokens": 4096, "stop_sequences": (),
}
PILOT_EXTRA_OPTIONS = {"num_batch": 512, "num_keep": 0}
# The Week 6 8K bucket ignored read-tool outputs: list_files alone adds ~8K tokens (a pilot episode reached
# 11,663 prompt tokens and was silently truncated). 32K is the smallest standard bucket that covers the two
# largest read-tool outputs plus the fixed tool schema; the Week 7 probes re-derive it per model.
PILOT_CONTEXT = 32768


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_candidates(path: Path = CANDIDATES_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_by_slug(slug: str, path: Path = CANDIDATES_PATH) -> dict[str, Any]:
    for entry in [*load_candidates(path)["candidates"], *load_candidates(path)["reserves"]]:
        if entry["slug"] == slug:
            return entry
    raise KeyError(f"unknown model slug {slug!r}")


def _parse_parameter_size(text: str | None) -> int | None:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([BMK])\s*", text or "", flags=re.I)
    if not match:
        return None
    scale = {"B": 10**9, "M": 10**6, "K": 10**3}[match.group(2).upper()]
    return int(float(match.group(1)) * scale)


def _modelfile_directives(modelfile: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in modelfile.splitlines():
        match = re.match(r"^\s*(RENDERER|PARSER)\s+(\S+)", line)
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def profile_from_server_metadata(
    *,
    candidate: Mapping[str, Any],
    show: Mapping[str, Any],
    tags_entry: Mapping[str, Any],
    server_version: str,
    endpoint: str,
    selected_context_length: int = PILOT_CONTEXT,
    thinking_mode: str = "disabled",
    kv_cache_type: str = "f16",
    flash_attention: str = "unspecified",
    max_output_tokens: int = PILOT_SAMPLING["max_output_tokens"],
) -> tuple[ModelProfile, dict[str, Any]]:
    """Build a validated profile and a sidecar resolution record."""

    tag = candidate["ollama_tag"]
    details = dict(show.get("details") or {})
    info = dict(show.get("model_info") or {})
    capabilities = sorted(str(item) for item in (show.get("capabilities") or []))
    architecture = str(info.get("general.architecture") or details.get("family") or "unknown")
    native_context = info.get(f"{architecture}.context_length")
    if not isinstance(native_context, int) or native_context <= 0:
        raise ValueError(f"{tag}: server did not report a native context length ({architecture}.context_length)")
    parameter_count = info.get("general.parameter_count") or _parse_parameter_size(details.get("parameter_size"))
    if not parameter_count:
        raise ValueError(f"{tag}: server did not report a parameter count")
    digest = str(tags_entry.get("digest") or "").removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{tag}: /api/tags did not return a 64-hex digest (got {digest!r})")
    template = str(show.get("template") or "") or "{{ .Prompt }}"
    quantization = str(details.get("quantization_level") or "unknown")
    thinking_capable = "thinking" in capabilities
    if thinking_mode != "disabled" and not thinking_capable:
        raise ValueError(f"{tag} does not report the thinking capability")
    modelfile = str(show.get("modelfile") or "")
    profile = ModelProfile.from_dict({
        "schema_version": "model-profile-v1",
        "profile_id": f"{candidate['slug']}-{quantization.lower()}-ollama-native-pilot-v1"[:160],
        "model_id": tag,
        "model_digest": digest,
        "model_family": str(candidate.get("family") or details.get("family") or architecture),
        "parameter_count": int(parameter_count),
        "serving_engine": "ollama",
        "serving_engine_version": server_version,
        "provider": "ollama-native",
        "endpoint": endpoint,
        "quantization": quantization,
        "numerical_precision": f"{quantization} quantized weights; {kv_cache_type} KV cache",
        "chat_template": template,
        "chat_template_sha256": sha256_text(template),
        "tool_call_format": "ollama-native-tools-v1",
        "native_context_length": int(native_context),
        "selected_context_length": min(int(selected_context_length), int(native_context)),
        "thinking_capability": thinking_capable,
        "thinking_mode": thinking_mode,
        "temperature": PILOT_SAMPLING["temperature"],
        "top_p": PILOT_SAMPLING["top_p"],
        "top_k": PILOT_SAMPLING["top_k"],
        "min_p": PILOT_SAMPLING["min_p"],
        "seed": PILOT_SAMPLING["seed"],
        "max_output_tokens": int(max_output_tokens),
        "stop_sequences": list(PILOT_SAMPLING["stop_sequences"]),
        "model_timeout_seconds": 300.0,
        "retry_policy": "one retry only for temporary model timeout or server failure",
        "infrastructure_retries": 1,
        "system_role": "system",
        "extra_options": dict(PILOT_EXTRA_OPTIONS),
    })
    directives = _modelfile_directives(modelfile)
    record = {
        "schema_version": "week7-profile-resolution-v1",
        "slug": candidate["slug"],
        "ollama_tag": tag,
        "profile_sha256": profile.fingerprint,
        "capabilities": capabilities,
        "has_native_tools": "tools" in capabilities,
        "has_thinking": thinking_capable,
        "architecture": architecture,
        "details": details,
        "modified_at": show.get("modified_at") or tags_entry.get("modified_at"),
        "modelfile_sha256": sha256_text(modelfile) if modelfile else None,
        "template_sha256": profile.chat_template_sha256,
        "template_is_passthrough": template.strip() == "{{ .Prompt }}",
        "renderer": directives.get("renderer"),
        "parser": directives.get("parser"),
        "default_parameters_text": show.get("parameters"),
        "license_sha256": sha256_text(str(show.get("license") or "")) if show.get("license") else None,
        "license_head": str(show.get("license") or "")[:160] or None,
        "server": {
            "version": server_version, "kv_cache_type": kv_cache_type,
            "flash_attention": flash_attention,
        },
    }
    if record["template_is_passthrough"] and not (record["renderer"] or record["parser"]):
        record["warning"] = (
            "template is a pass-through and no RENDERER/PARSER directive was found; "
            "the exact prompt format is not pinned by this record"
        )
    return profile, record


def fetch_server_metadata(
    candidate: Mapping[str, Any], *, endpoint: str, timeout: float = 60.0, client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Read digest, show-record, and version for one model from a live server."""

    http = client or httpx.Client(base_url=endpoint.rstrip("/"), timeout=timeout)
    tag = candidate["ollama_tag"]
    tags = http.get("/api/tags")
    tags.raise_for_status()
    accepted = {tag, f"{tag}:latest"}  # `ollama pull command-r7b` is listed as command-r7b:latest
    entry = next((m for m in tags.json().get("models", []) if m.get("name") in accepted or m.get("model") in accepted), None)
    if entry is None:
        raise LookupError(f"{tag} is not installed on {endpoint} (run `ollama pull {tag}`)")
    show = http.post("/api/show", json={"model": tag})
    show.raise_for_status()
    version = http.get("/api/version")
    version.raise_for_status()
    return {"tags_entry": entry, "show": show.json(), "server_version": str(version.json().get("version"))}


def load_profile(slug: str, profiles_dir: Path = PROFILES_DIR) -> ModelProfile:
    return ModelProfile.from_json(profiles_dir / f"{slug}.json")


def _raw_done_reason(completion: ModelCompletion) -> str | None:
    raw = completion.raw_response
    return raw.get("done_reason") if isinstance(raw, Mapping) else None


class ContextOverflowError(RuntimeError):
    """Raised instead of continuing with a prompt the server silently cut."""


def truncation_signature(prompt_tokens: int, context_length: int) -> bool:
    """llama.cpp-style truncation keeps the last ctx/2 tokens (+keep): the reported count lands on ctx/2.

    Observed on Ollama 0.40.2: ``truncating input prompt limit=4096 prompt=11663 keep=0 new=4096`` with
    num_ctx 8192, and the response reported prompt_eval_count == 4096.
    """
    half = context_length // 2
    return prompt_tokens in (half, half + 1) or prompt_tokens >= context_length


class GuardedModelClient:
    """Wrap a model client so two silent failure modes become recorded failures.

    1. context overflow  -> ContextOverflowError  (classified CONTEXT_OVERFLOW, never retried)
    2. tool call emitted only as text -> ValueError('malformed tool call: ...') (classified MALFORMED_TOOL_CALL)
    3. response cut by max_output_tokens -> ValueError('structured output truncated ...') (INVALID_STRUCTURED_OUTPUT)
       All three used to look like "the model did nothing" and would have inflated apparent safety
       (Qwen ran out of tokens decoding Caesar/Morse attacks in the second pilot: attack_success=False by default).
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        # One entry per model call: hash of the full native request and of the tool schemas alone.
        self.request_log: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def adapter(self):
        return self._inner.adapter

    @adapter.setter
    def adapter(self, value) -> None:
        self._inner.adapter = value

    def complete(self, messages, tools=()) -> ModelCompletion:
        from .week7_pilot import detect_text_tool_call  # local import: week7_pilot imports gateway/schemas
        from .model_api import canonical_hash

        try:
            payload = self._inner.adapter.build_payload(self._inner.profile, messages, tools)
            self.request_log.append({
                "call": len(self.request_log) + 1, "request_sha256": canonical_hash(payload),
                "tools_sha256": canonical_hash(payload.get("tools") or []), "messages": len(payload.get("messages") or []),
            })
        except Exception as exc:  # never let bookkeeping break a measured call
            self.request_log.append({"call": len(self.request_log) + 1, "error": f"{type(exc).__name__}: {exc}"[:200]})
        completion = self._inner.complete(messages, tools)
        ctx = self._inner.profile.selected_context_length
        if truncation_signature(completion.prompt_tokens, ctx):
            raise ContextOverflowError(
                f"context overflow: prompt_eval_count={completion.prompt_tokens} matches the server's truncation "
                f"signature for num_ctx={ctx}; the prompt was probably cut, so the episode is not valid")
        if completion.prompt_tokens + completion.completion_tokens >= ctx:
            raise ContextOverflowError(
                f"context overflow: prompt+output {completion.prompt_tokens + completion.completion_tokens} >= num_ctx={ctx}")
        if _raw_done_reason(completion) == "length":
            raise ValueError(
                f"structured output truncated at the max_output_tokens limit (done_reason=length, "
                f"{completion.completion_tokens} tokens); the episode outcome is unknown, not safe: "
                f"{(completion.content or '')[:200]!r}")
        if not completion.tool_intents:
            reason = detect_text_tool_call(completion.content, [t.name for t in tools])
            if reason:
                raise ValueError(f"malformed tool call: tool call emitted as text ({reason}): "
                                 f"{(completion.content or '')[:300]!r}")
        return completion


def build_llm(
    profile: ModelProfile, *, seed: int, limits: RuntimeLimits, client: httpx.Client | None = None,
    guard: bool = True,
) -> SeededOllamaLLM:
    """A model element for any profile (the stock constructor is Qwen-default)."""

    run_profile = profile.revalidated(
        seed=seed,
        model_timeout_seconds=limits.model_seconds,
        infrastructure_retries=limits.infrastructure_retries,
    )
    inner = OllamaNativeModelClient(run_profile, client=client)
    return SeededOllamaLLM(
        model=run_profile.model_id,
        seed=seed,
        temperature=run_profile.temperature,
        max_tokens=run_profile.max_output_tokens,
        model_client=GuardedModelClient(inner) if guard else inner,
        runtime_limits=limits,
    )
