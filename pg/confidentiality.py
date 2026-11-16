"""Source-aware read authorization and deterministic final-output guarding."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Iterable, Mapping

from pydantic import BaseModel

from .trusted_models import (
    DataSensitivity,
    OutputGuardDecision,
    OutputGuardRecord,
    SensitivityEvidence,
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ReadAuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class ReadAuthorizationPolicy:
    principal_id: str
    allowed_tools: frozenset[str]
    allowed_read_sensitivities: frozenset[DataSensitivity]
    allowed_output_sensitivities: frozenset[DataSensitivity] = frozenset({
        DataSensitivity.PUBLIC, DataSensitivity.INTERNAL,
    })
    authorized_output_evidence_ids: frozenset[str] = frozenset()
    rule_id: str = "authenticated-workspace-owner-v1"

    def authorize_read(self, principal_id: str, tool_name: str) -> None:
        if principal_id != self.principal_id:
            raise ReadAuthorizationError("read principal does not match the authenticated policy subject")
        if tool_name not in self.allowed_tools:
            raise ReadAuthorizationError(f"read tool {tool_name!r} is not authorized")

    def disclosure_allowed(self, evidence: SensitivityEvidence) -> bool:
        return (
            evidence.sensitivity in self.allowed_output_sensitivities
            or evidence.evidence_id in self.authorized_output_evidence_ids
        )


@dataclass(frozen=True)
class GuardedOutput:
    text: str
    record: OutputGuardRecord
    disclosure_attempted: bool
    confidentiality_violation: bool


@dataclass(frozen=True)
class _ObservedSecret:
    value: str
    evidence: SensitivityEvidence


class ConfidentialityGuard:
    """Keeps raw protected values only in memory and persists digest evidence."""

    _RESTRICTED_FIELDS = re.compile(r"(password|secret|token|credential|api[_-]?key|ssn)", re.I)
    _CONFIDENTIAL_FIELDS = re.compile(r"(body|content|description|notes?|message|payload)", re.I)
    _INTERNAL_FIELDS = re.compile(r"(email|sender|recipient|participant|address|filename|title|subject)", re.I)

    def __init__(
        self,
        *,
        graph_run_id: str,
        policy: ReadAuthorizationPolicy,
        evidence_sink: Callable[[SensitivityEvidence], None] | None = None,
        enabled: bool = True,
    ) -> None:
        # `enabled=False` is the Week 10 output-guard ablation: reads are not
        # tracked and the final response is passed through unexamined.
        self.enabled = enabled
        self.graph_run_id = graph_run_id
        self.policy = policy
        self.evidence_sink = evidence_sink
        self._observed: list[_ObservedSecret] = []

    @property
    def evidence(self) -> tuple[SensitivityEvidence, ...]:
        return tuple(item.evidence for item in self._observed)

    @classmethod
    def classify(cls, field_path: str, value: str) -> DataSensitivity:
        if cls._RESTRICTED_FIELDS.search(field_path):
            return DataSensitivity.RESTRICTED
        if cls._CONFIDENTIAL_FIELDS.search(field_path):
            return DataSensitivity.CONFIDENTIAL
        if cls._INTERNAL_FIELDS.search(field_path) or "@" in value:
            return DataSensitivity.INTERNAL
        return DataSensitivity.PUBLIC

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return {str(key): ConfidentialityGuard._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [ConfidentialityGuard._jsonable(item) for item in value]
        if hasattr(value, "__dict__"):
            return {str(key): ConfidentialityGuard._jsonable(item) for key, item in vars(value).items()}
        return value

    @staticmethod
    def _flatten(value: Any, path: str = "result") -> Iterable[tuple[str, str, str]]:
        if isinstance(value, Mapping):
            object_id = str(value.get("id") or value.get("id_") or value.get("email_id") or path)
            for key, item in value.items():
                yield from ConfidentialityGuard._flatten(item, f"{path}.{key}")
            if not value:
                yield path, object_id, ""
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from ConfidentialityGuard._flatten(item, f"{path}[{index}]")
        elif isinstance(value, (str, int, float, bool)) and value is not None:
            yield path, path.rsplit(".", 1)[0], str(value)

    def observe_read(
        self,
        *,
        principal_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        result: Any,
        source_version: int = 1,
    ) -> tuple[SensitivityEvidence, ...]:
        if not self.enabled:
            return ()
        self.policy.authorize_read(principal_id, tool_name)
        source_payload = json.dumps(
            {"tool": tool_name, "arguments": dict(arguments)}, sort_keys=True, default=str,
        )
        source_id = f"read-{_digest(source_payload)[:20]}"
        created: list[SensitivityEvidence] = []
        for field_path, object_id, value in self._flatten(self._jsonable(result)):
            if not value or len(value) < 4:
                continue
            sensitivity = self.classify(field_path, value)
            authorized = sensitivity in self.policy.allowed_read_sensitivities
            evidence = SensitivityEvidence(
                evidence_id=str(uuid.uuid4()),
                graph_run_id=self.graph_run_id,
                principal_id=principal_id,
                source_id=source_id,
                source_version=source_version,
                object_id=object_id[:160],
                field_path=field_path,
                value_digest=_digest(value),
                sensitivity=sensitivity,
                read_authorized=authorized,
                authorization_rule=self.policy.rule_id,
                observed_at=_now(),
            )
            if self.evidence_sink is not None:
                self.evidence_sink(evidence)
            if not authorized:
                raise ReadAuthorizationError(
                    f"{self.policy.rule_id} denies {sensitivity.value} data from {field_path}"
                )
            if sensitivity in {DataSensitivity.CONFIDENTIAL, DataSensitivity.RESTRICTED}:
                self._observed.append(_ObservedSecret(value=value, evidence=evidence))
            created.append(evidence)
        return tuple(created)

    def guard_output(self, text: str) -> GuardedOutput:
        if not self.enabled:
            record = OutputGuardRecord(
                guard_id=str(uuid.uuid4()),
                graph_run_id=self.graph_run_id,
                principal_id=self.policy.principal_id,
                response_digest=_digest(text),
                decision=OutputGuardDecision.ALLOW,
                matched_evidence_ids=(),
                matched_value_digests=(),
                disclosure_authorized=True,
                reason="output guard disabled (ablation): response not examined",
                evaluated_at=_now(),
            )
            return GuardedOutput(
                text=text, record=record, disclosure_attempted=False, confidentiality_violation=False,
            )
        matches = [
            item for item in self._observed
            if item.value.casefold() in text.casefold() and not self.policy.disclosure_allowed(item.evidence)
        ]
        attempted = bool(matches)
        decision = OutputGuardDecision.BLOCK if attempted else OutputGuardDecision.ALLOW
        guarded_text = "[BLOCKED: protected data disclosure]" if attempted else text
        record = OutputGuardRecord(
            guard_id=str(uuid.uuid4()),
            graph_run_id=self.graph_run_id,
            principal_id=self.policy.principal_id,
            response_digest=_digest(text),
            decision=decision,
            matched_evidence_ids=tuple(item.evidence.evidence_id for item in matches),
            matched_value_digests=tuple(item.evidence.value_digest for item in matches),
            disclosure_authorized=False if matches else True,
            reason=("protected read-derived value detected in final response"
                    if matches else "no unauthorized protected value detected"),
            evaluated_at=_now(),
        )
        return GuardedOutput(
            text=guarded_text,
            record=record,
            disclosure_attempted=attempted,
            confidentiality_violation=False,
        )


def final_assistant_text(messages: Iterable[Any]) -> str:
    """Extract final assistant prose across AgentDojo message representations."""
    for message in reversed(list(messages)):
        data = message.model_dump(mode="json") if isinstance(message, BaseModel) else (
            dict(message) if isinstance(message, Mapping) else vars(message)
        )
        if data.get("role") != "assistant":
            continue
        content = data.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, Mapping):
                    parts.append(str(item.get("content") or item.get("text") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
    return ""
