"""Strict, versioned Pydantic trust-boundary records for revised Week 3."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)


Identifier = Annotated[str, StringConstraints(min_length=1, max_length=160, strip_whitespace=True)]
LongText = Annotated[str, StringConstraints(max_length=16_384)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Signature = Annotated[str, StringConstraints(min_length=16, max_length=4096)]
EvidenceRef = Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class StrictRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
        str_strip_whitespace=False,
    )

    def revalidated(self, **updates: Any):
        """Copy only by passing the complete result through validation again."""
        return type(self).model_validate({**self.model_dump(mode="python"), **updates})


class RunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    RETRYABLE = "RETRYABLE"
    REJECTED = "REJECTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class WorkflowNode(str, Enum):
    START = "START"
    LOAD_SCENARIO = "LOAD_SCENARIO"
    BUILD_CONVERSATION = "BUILD_CONVERSATION"
    CALL_MODEL = "CALL_MODEL"
    PARSE_TOOL_INTENT = "PARSE_TOOL_INTENT"
    CLASSIFY_TOOL = "CLASSIFY_TOOL"
    EXECUTE_READ = "EXECUTE_READ"
    LABEL_SOURCES = "LABEL_SOURCES"
    VALIDATE_WRITE = "VALIDATE_WRITE"
    EVALUATE_POLICY = "EVALUATE_POLICY"
    REJECT_AND_AUDIT = "REJECT_AND_AUDIT"
    BUILD_AND_SIGN_PROPOSAL = "BUILD_AND_SIGN_PROPOSAL"
    OBTAIN_APPROVAL = "OBTAIN_APPROVAL"
    GATEWAY_REVALIDATE = "GATEWAY_REVALIDATE"
    CONSUME_AND_RESERVE = "CONSUME_AND_RESERVE"
    EXECUTE_ADAPTER = "EXECUTE_ADAPTER"
    PERSIST_RECEIPT = "PERSIST_RECEIPT"
    EVALUATE_FINAL_STATE = "EVALUATE_FINAL_STATE"
    END = "END"


class ToolClass(str, Enum):
    READ = "READ"
    WRITE = "WRITE"


class CallStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class PolicyDecision(str, Enum):
    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"


class ExecutionStatus(str, Enum):
    RESERVED = "RESERVED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"


class EvaluationStatus(str, Enum):
    EVALUABLE = "EVALUABLE"
    ERROR = "ERROR"


class DataSensitivity(str, Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class OutputGuardDecision(str, Enum):
    ALLOW = "ALLOW"
    REDACT = "REDACT"
    BLOCK = "BLOCK"


class ModelProfile(StrictRecord):
    schema_version: Literal["model-profile-v1"] = "model-profile-v1"
    profile_id: Identifier
    model_id: Identifier
    model_digest: Digest
    model_family: Identifier
    parameter_count: Annotated[int, Field(strict=True, gt=0)]
    serving_engine: Identifier
    serving_engine_version: Identifier
    provider: Literal["ollama-native", "openai-compatible"]
    endpoint: Annotated[str, StringConstraints(min_length=8, max_length=2048)]
    quantization: Identifier
    numerical_precision: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    chat_template: Annotated[str, StringConstraints(min_length=1, max_length=65_536)]
    chat_template_sha256: Digest
    tool_call_format: Identifier
    native_context_length: Annotated[int, Field(strict=True, gt=0)]
    selected_context_length: Annotated[int, Field(strict=True, gt=0)]
    thinking_capability: bool
    thinking_mode: Literal["disabled", "enabled", "low", "medium", "high"]
    temperature: Annotated[float, Field(strict=True, ge=0.0, le=2.0)]
    top_p: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    top_k: Annotated[int, Field(strict=True, ge=0)]
    min_p: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    seed: Annotated[int, Field(strict=True, ge=0)]
    max_output_tokens: Annotated[int, Field(strict=True, gt=0, le=65_536)]
    stop_sequences: Annotated[tuple[Annotated[str, StringConstraints(max_length=256)], ...], Field(max_length=64)]
    model_timeout_seconds: Annotated[float, Field(strict=True, gt=0.0, le=3600.0)]
    retry_policy: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    infrastructure_retries: Literal[0, 1]
    system_role: Literal["system", "developer"] = "system"
    extra_options: dict[Identifier, JsonValue] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def validate_profile(self) -> "ModelProfile":
        expected = hashlib.sha256(self.chat_template.encode("utf-8")).hexdigest()
        if self.chat_template_sha256 != expected:
            raise ValueError("chat_template_sha256 does not match chat_template")
        if self.selected_context_length > self.native_context_length:
            raise ValueError("selected_context_length must fit the native context")
        if self.thinking_mode != "disabled" and not self.thinking_capability:
            raise ValueError("thinking mode cannot be enabled for an incapable model")
        return self

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelProfile":
        return cls.model_validate_json(json.dumps(dict(value), allow_nan=False))

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelProfile":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelCallRecord(StrictRecord):
    schema_version: Literal["model-call-v1"] = "model-call-v1"
    call_id: Identifier
    graph_run_id: Identifier
    profile_sha256: Digest
    stage: Literal["HEALTHCHECK", "WARMUP", "CALL_MODEL"]
    attempt: Annotated[int, Field(strict=True, ge=1, le=2)]
    request_evidence_ref: EvidenceRef
    response_evidence_ref: EvidenceRef | None = None
    latency_ms: Annotated[int, Field(strict=True, ge=0)]
    prompt_tokens: Annotated[int, Field(strict=True, ge=0)] = 0
    completion_tokens: Annotated[int, Field(strict=True, ge=0)] = 0
    status: CallStatus
    error_type: Identifier | None = None
    error: LongText | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "ModelCallRecord":
        if self.status is CallStatus.SUCCEEDED and (self.error or self.error_type):
            raise ValueError("successful model calls cannot carry errors")
        if self.status is CallStatus.FAILED and not self.error:
            raise ValueError("failed model calls require an error")
        return self


class ToolIntent(StrictRecord):
    schema_version: Literal["tool-intent-v1"] = "tool-intent-v1"
    call_id: Identifier
    tool_name: Identifier
    arguments: dict[Identifier, JsonValue] | None
    raw_arguments: JsonValue
    source_response_id: Identifier | None = None
    parse_error: LongText | None = None

    @model_validator(mode="after")
    def validate_parse_result(self) -> "ToolIntent":
        if (self.arguments is None) == (self.parse_error is None):
            raise ValueError("exactly one of validated arguments or parse_error is required")
        return self

    @property
    def valid(self) -> bool:
        return self.parse_error is None and self.arguments is not None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SourceEvidence(StrictRecord):
    schema_version: Literal["source-evidence-v1"] = "source-evidence-v1"
    source_id: Identifier
    tenant_id: Identifier
    source_type: Identifier
    source_identity: Identifier
    trust_label: Literal["trusted", "untrusted"]
    content_digest: Digest
    version: Annotated[int, Field(strict=True, ge=1)]
    observed_at: Identifier
    raw_evidence_ref: EvidenceRef | None = None


class SensitivityEvidence(StrictRecord):
    schema_version: Literal["sensitivity-evidence-v1"] = "sensitivity-evidence-v1"
    evidence_id: Identifier
    graph_run_id: Identifier
    principal_id: Identifier
    source_id: Identifier
    source_version: Annotated[int, Field(strict=True, ge=1)]
    object_id: Identifier
    field_path: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    value_digest: Digest
    sensitivity: DataSensitivity
    read_authorized: bool
    authorization_rule: Identifier
    observed_at: Identifier


class OutputGuardRecord(StrictRecord):
    schema_version: Literal["output-guard-record-v1"] = "output-guard-record-v1"
    guard_id: Identifier
    graph_run_id: Identifier
    principal_id: Identifier
    response_digest: Digest
    decision: OutputGuardDecision
    matched_evidence_ids: Annotated[tuple[Identifier, ...], Field(max_length=128)] = ()
    matched_value_digests: Annotated[tuple[Digest, ...], Field(max_length=128)] = ()
    disclosure_authorized: bool
    reason: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    evaluated_at: Identifier

    @model_validator(mode="after")
    def validate_guard(self) -> "OutputGuardRecord":
        if len(self.matched_evidence_ids) != len(self.matched_value_digests):
            raise ValueError("matched evidence IDs and digests must have equal lengths")
        if self.decision is OutputGuardDecision.ALLOW and self.matched_evidence_ids and not self.disclosure_authorized:
            raise ValueError("unauthorized protected matches cannot be allowed")
        return self


class PolicyOutcome(StrictRecord):
    schema_version: Literal["policy-outcome-v1"] = "policy-outcome-v1"
    proposal_id: Identifier | None = None
    phase: Literal["proposal", "execution"]
    decision: PolicyDecision
    reason: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    policy_version: Identifier
    evaluated_at: Identifier


class ProposalRecord(StrictRecord):
    schema_version: Literal["1.0"] = "1.0"
    proposal_id: Identifier
    session_id: Identifier
    principal_id: Identifier
    user_goal_hash: Digest
    model_id: Identifier
    tool_name: Identifier
    operation: Identifier
    arguments: dict[Identifier, JsonValue] = Field(max_length=16)
    source_digests: Annotated[tuple[Digest, ...], Field(min_length=1, max_length=64)]
    policy_version: Identifier
    risk_level: Literal["low", "high"]
    created_at: Identifier
    key_id: Identifier
    model_explanation: LongText | None = None
    claimed_fields: dict[Identifier, JsonValue] | None = Field(default=None, max_length=16)
    digest: Digest | None = None
    signature: Signature | None = None

    @model_validator(mode="after")
    def validate_proposal(self) -> "ProposalRecord":
        if self.tool_name != self.operation:
            raise ValueError("tool_name and operation must match")
        if len(self.source_digests) != len(set(self.source_digests)):
            raise ValueError("source_digests must be unique")
        if (self.digest is None) != (self.signature is None):
            raise ValueError("digest and signature must both be present or absent")
        return self


class ApprovalRecord(StrictRecord):
    schema_version: Literal["approval-record-v1"] = "approval-record-v1"
    approval_id: Identifier
    proposal_id: Identifier
    proposal_digest: Digest
    session_id: Identifier
    principal_id: Identifier
    approver_id: Identifier
    decision: ApprovalDecision
    issued_at: Identifier
    expires_at: Identifier
    nonce: Identifier
    audience: Literal["gateway"]
    policy_version: Identifier
    renderer_version: Identifier
    key_id: Identifier
    digest: Digest
    signature: Signature
    consumed: bool = False
    consumed_at: Identifier | None = None

    @model_validator(mode="after")
    def validate_approval(self) -> "ApprovalRecord":
        try:
            issued = datetime.fromisoformat(self.issued_at.replace("Z", "+00:00"))
            expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("approval timestamps must be ISO-8601") from exc
        if expires <= issued:
            raise ValueError("approval expiry must follow issuance")
        if self.consumed != (self.consumed_at is not None):
            raise ValueError("consumed and consumed_at must agree")
        return self


class ExecutionRecord(StrictRecord):
    schema_version: Literal["execution-record-v1"] = "execution-record-v1"
    attempt_id: Identifier
    proposal_id: Identifier
    approval_id: Identifier
    idempotency_key: Digest
    adapter: Identifier
    status: ExecutionStatus
    started_at: Identifier
    completed_at: Identifier | None = None
    error: LongText | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> "ExecutionRecord":
        terminal = self.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED}
        if terminal != (self.completed_at is not None):
            raise ValueError("terminal executions require completed_at")
        if self.status is ExecutionStatus.FAILED and not self.error:
            raise ValueError("failed executions require an error")
        return self


class ReceiptRecord(StrictRecord):
    schema_version: Literal["receipt-record-v1"] = "receipt-record-v1"
    attempt_id: Identifier
    proposal_id: Identifier
    idempotency_key: Digest
    action_digest: Digest
    result_status: Literal["SUCCEEDED", "FAILED"]
    result_digest: Digest
    detail: LongText = ""
    created_at: Identifier

    @model_validator(mode="after")
    def validate_binding(self) -> "ReceiptRecord":
        if self.idempotency_key != self.action_digest:
            raise ValueError("receipt idempotency key must equal the signed action digest")
        return self


class EvaluationRecord(StrictRecord):
    schema_version: Literal["evaluation-record-v1"] = "evaluation-record-v1"
    graph_run_id: Identifier
    scenario_id: Identifier
    arm: Literal["baseline", "protected"]
    seed: Annotated[int, Field(strict=True, ge=0)]
    status: EvaluationStatus
    completed: bool
    benign_success: bool | None = None
    attack_success: bool | None = None
    final_state_digest: Digest | None = None
    evaluator_version: Identifier
    error: LongText | None = None

    @model_validator(mode="after")
    def validate_evaluation(self) -> "EvaluationRecord":
        if self.status is EvaluationStatus.EVALUABLE:
            if not self.completed or self.error or self.final_state_digest is None:
                raise ValueError("evaluable results must be complete, error-free, and state-bound")
        elif not self.error:
            raise ValueError("evaluation errors require an error message")
        if self.benign_success is not None and self.attack_success is not None:
            raise ValueError("an episode cannot be both benign and attack-classified")
        return self


class NodeDiagnostic(StrictRecord):
    schema_version: Literal["graph-node-diagnostic-v1"] = "graph-node-diagnostic-v1"
    graph_run_id: Identifier
    node: WorkflowNode
    sequence: Annotated[int, Field(strict=True, ge=1)]
    started_at: Identifier
    latency_ms: Annotated[int, Field(strict=True, ge=0)]
    status: Literal["SUCCEEDED", "FAILED", "INTERRUPTED"]
    evidence_ref: EvidenceRef | None = None
    error_type: Identifier | None = None
    error: LongText | None = None

    @model_validator(mode="after")
    def validate_node_status(self) -> "NodeDiagnostic":
        if self.status == "SUCCEEDED" and (self.error or self.error_type):
            raise ValueError("successful nodes cannot carry errors")
        if self.status != "SUCCEEDED" and not self.error:
            raise ValueError("failed or interrupted nodes require an error")
        return self


class AgentRunState(StrictRecord):
    schema_version: Literal["agent-run-state-v1"] = "agent-run-state-v1"
    graph_run_id: Identifier
    scenario_id: Identifier
    arm: Literal["baseline", "protected"]
    seed: Annotated[int, Field(strict=True, ge=0)]
    current_node: WorkflowNode = WorkflowNode.START
    status: RunStatus = RunStatus.CREATED
    session_id: Identifier
    principal_id: Identifier
    model_profile_sha256: Digest
    scenario_evidence_ref: EvidenceRef
    conversation_evidence_ref: EvidenceRef | None = None
    model_call: ModelCallRecord | None = None
    tool_intent: ToolIntent | None = None
    tool_class: ToolClass | None = None
    source_evidence: Annotated[tuple[SourceEvidence, ...], Field(max_length=64)] = ()
    policy_outcome: PolicyOutcome | None = None
    proposal_id: Identifier | None = None
    approval_id: Identifier | None = None
    execution: ExecutionRecord | None = None
    receipt: ReceiptRecord | None = None
    evaluation: EvaluationRecord | None = None
    retry_count: Literal[0, 1] = 0
    last_error: LongText | None = None
    path: Annotated[tuple[WorkflowNode, ...], Field(max_length=256)] = ()
    node_diagnostics: Annotated[tuple[NodeDiagnostic, ...], Field(max_length=256)] = ()

    @model_validator(mode="after")
    def validate_state(self) -> "AgentRunState":
        if self.path and self.path[-1] != self.current_node:
            raise ValueError("the last path entry must equal current_node")
        if self.status is RunStatus.SUCCEEDED and self.current_node is not WorkflowNode.END:
            raise ValueError("successful graph state must be at END")
        if self.status in {RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.RETRYABLE} and not self.last_error:
            raise ValueError("non-success terminal/interrupted states require last_error")
        return self


SCHEMA_MODELS: tuple[type[StrictRecord], ...] = (
    AgentRunState,
    ModelProfile,
    ModelCallRecord,
    ToolIntent,
    SourceEvidence,
    PolicyOutcome,
    ProposalRecord,
    ApprovalRecord,
    ExecutionRecord,
    ReceiptRecord,
    EvaluationRecord,
    NodeDiagnostic,
    SensitivityEvidence,
    OutputGuardRecord,
)


def schema_manifest(root: str | Path | None = None) -> dict[str, Any]:
    """Load and verify the generated schema manifest used by experiments."""
    schema_root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "schemas" / "v1"
    manifest_path = schema_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["schemas"]:
        schema_path = schema_root / item["file"]
        actual = hashlib.sha256(schema_path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"generated schema hash mismatch: {schema_path}")
    return manifest
