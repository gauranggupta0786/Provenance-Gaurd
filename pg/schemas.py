"""Week 1 deliverable: typed record schemas, canonicalization, and strict parsing.

Implements the "Records to define in Week 1" section of the execution plan:
Proposal, Approval, AuditEvent, Receipt, and the action state machine.

Canonicalization here is a simplified JCS (RFC 8785 style): keys sorted,
no insignificant whitespace, UTF-8, no NaN/Infinity. It is documented as
simplified (not full RFC 8785 number formatting) -- sufficient for this
prototype's fixed schema of strings/ints, called out explicitly per the
plan's honesty requirements.
"""
from __future__ import annotations

import json
import time
import uuid
import hashlib
import dataclasses
import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, get_args, get_origin


SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Tool / operation contract (v1 keeps exactly three action types, per plan)
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"
    HIGH = "high"


TOOL_CONTRACTS: dict[str, dict[str, Any]] = {
    "summarize": {
        "required_args": {"message_id": str, "max_words": int},
        "risk_level": RiskLevel.LOW,
        "requires_approval": False,
        "max_str_len": 256,
    },
    "send_email": {
        "required_args": {"recipient": str, "subject": str, "body": str},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 4096,
    },
    "update_record": {
        "required_args": {"record_id": str, "field": str, "new_value": str},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 4096,
    },
    # AgentDojo Workspace v1.2.2 write/state-changing operations.  The
    # model-visible AgentDojo names are translated to these names by the
    # Week-5 adapter, which prevents a stock benchmark tool from bypassing
    # the signed proposal and approval path.
    "workspace_send_email": {
        "required_args": {"recipients": list[str], "subject": str, "body": str},
        "optional_args": {
            "attachments": (list[dict], []), "cc": (list[str], []), "bcc": (list[str], []),
        },
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 4096,
    },
    "workspace_delete_email": {
        "required_args": {"email_id": str},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 256,
    },
    "workspace_get_unread_emails": {
        "required_args": {},
        "risk_level": RiskLevel.LOW,
        "requires_approval": False,
        "max_str_len": 256,
    },
    "workspace_create_calendar_event": {
        "required_args": {
            "title": str, "start_time": str, "end_time": str,
        },
        "optional_args": {
            "description": (str, ""), "participants": (list[str], []), "location": (str | None, None),
        },
        "allow_empty_fields": {"description"},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 4096,
    },
    "workspace_cancel_calendar_event": {
        "required_args": {"event_id": str},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 256,
    },
    "workspace_reschedule_calendar_event": {
        "required_args": {"event_id": str, "new_start_time": str},
        "optional_args": {"new_end_time": (str | None, None)},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 256,
    },
    "workspace_add_calendar_event_participants": {
        "required_args": {"event_id": str, "participants": list[str]},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 256,
    },
    "workspace_append_to_file": {
        "required_args": {"file_id": str, "content": str},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 8192,
    },
    "workspace_create_file": {
        "required_args": {"filename": str, "content": str},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 8192,
    },
    "workspace_delete_file": {
        "required_args": {"file_id": str},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 256,
    },
    "workspace_share_file": {
        "required_args": {"file_id": str, "email": str, "permission": str},
        "risk_level": RiskLevel.HIGH,
        "requires_approval": True,
        "max_str_len": 256,
    },
}

MAX_ARG_COUNT = 16  # guards against oversized/ambiguous payloads


class SchemaError(ValueError):
    """Raised on any malformed, oversized, ambiguous, or invalid record."""


# ---------------------------------------------------------------------------
# Duplicate-key-safe JSON decoding (input boundary)
# ---------------------------------------------------------------------------

def strict_json_loads(raw: str) -> Any:
    """Parse JSON, rejecting duplicate object keys (Week 2 Day 2 requirement)."""

    def _no_dupes(pairs):
        seen = set()
        out = {}
        for k, v in pairs:
            if k in seen:
                raise SchemaError(f"duplicate JSON key at input boundary: {k!r}")
            seen.add(k)
            out[k] = v
        return out

    try:
        return json.loads(raw, object_pairs_hook=_no_dupes)
    except json.JSONDecodeError as e:
        raise SchemaError(f"invalid JSON: {e}") from e


def canonicalize(obj: dict[str, Any]) -> bytes:
    """Deterministic byte representation used for hashing/signing (simplified JCS)."""
    for v in obj.values():
        if isinstance(v, float):
            raise SchemaError("floats are not permitted in canonical records (use integers)")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Argument normalization / strict rejection
# ---------------------------------------------------------------------------

def normalize_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name not in TOOL_CONTRACTS:
        raise SchemaError(f"unknown tool: {tool_name!r}")
    if not isinstance(arguments, dict):
        raise SchemaError("arguments must be an object")
    if len(arguments) > MAX_ARG_COUNT:
        raise SchemaError("too many argument fields")

    contract = TOOL_CONTRACTS[tool_name]
    required = contract["required_args"]
    optional = contract.get("optional_args", {})
    max_len = contract["max_str_len"]
    allow_empty_fields = contract.get("allow_empty_fields", set())

    extra = set(arguments) - set(required) - set(optional)
    if extra:
        raise SchemaError(f"unexpected extra fields for {tool_name}: {sorted(extra)}")
    missing = set(required) - set(arguments)
    if missing:
        raise SchemaError(f"missing required fields for {tool_name}: {sorted(missing)}")

    normalized: dict[str, Any] = {}
    for k, expected_type in required.items():
        normalized[k] = _normalize_value(k, arguments[k], expected_type, max_len, k in allow_empty_fields)
    for k, (expected_type, default) in optional.items():
        normalized[k] = _normalize_value(k, arguments.get(k, default), expected_type, max_len, k in allow_empty_fields)

    if tool_name == "send_email":
        if "@" not in normalized["recipient"] or " " in normalized["recipient"]:
            raise SchemaError("recipient is not a well-formed address")
    if tool_name == "update_record":
        if normalized["field"] not in {"status", "notes", "owner"}:
            raise SchemaError(f"field {normalized['field']!r} is not editable")

    if tool_name == "workspace_send_email":
        destinations = normalized["recipients"] + normalized["cc"] + normalized["bcc"]
        if not normalized["recipients"]:
            raise SchemaError("recipients must contain at least one address")
        _validate_emails(destinations)
        if len(destinations) != len(set(address.lower() for address in destinations)):
            raise SchemaError("duplicate destination across recipients/cc/bcc")
        for attachment in normalized["attachments"]:
            if not isinstance(attachment, dict) or attachment.get("type") not in {"file", "event"}:
                raise SchemaError("each attachment must be a file or event object")
            if attachment["type"] == "file" and not isinstance(attachment.get("file_id"), str):
                raise SchemaError("file attachment requires file_id")
            if attachment["type"] == "event" and not isinstance(attachment.get("event_details"), dict):
                raise SchemaError("event attachment requires event_details")

    if tool_name in {"workspace_create_calendar_event", "workspace_reschedule_calendar_event"}:
        start_key = "start_time" if "start_time" in normalized else "new_start_time"
        end_key = "end_time" if "end_time" in normalized else "new_end_time"
        start = _parse_datetime(normalized[start_key], start_key)
        if normalized[end_key] is not None:
            end = _parse_datetime(normalized[end_key], end_key)
            if end <= start:
                raise SchemaError("calendar end time must be after start time")

    if tool_name in {"workspace_create_calendar_event", "workspace_add_calendar_event_participants"}:
        _validate_emails(normalized["participants"])

    if tool_name == "workspace_share_file":
        _validate_emails([normalized["email"]])
        if normalized["permission"] not in {"r", "rw"}:
            raise SchemaError("permission must be 'r' or 'rw'")

    return normalized


def _normalize_value(field_name: str, value: Any, expected_type: Any, max_len: int,
                     allow_empty: bool = False) -> Any:
    origin = get_origin(expected_type)
    type_args = get_args(expected_type)
    if origin is list:
        if not isinstance(value, list):
            raise SchemaError(f"field {field_name!r} must be a list")
        if len(value) > 32:
            raise SchemaError(f"field {field_name!r} has too many items")
        item_type = type_args[0]
        return [_normalize_value(f"{field_name}[{i}]", item, item_type, max_len) for i, item in enumerate(value)]
    if origin is not None and type(None) in type_args:
        if value is None:
            return None
        non_none = next(t for t in type_args if t is not type(None))
        return _normalize_value(field_name, value, non_none, max_len, allow_empty)
    if expected_type is dict:
        if not isinstance(value, dict):
            raise SchemaError(f"field {field_name!r} must be an object")
        # Round-trip through JSON to reject non-serializable model objects and
        # detach caller-owned mutable structures before signing.
        try:
            return json.loads(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"field {field_name!r} must be JSON-compatible") from exc
    if expected_type is str:
        if not isinstance(value, str) or isinstance(value, bool):
            raise SchemaError(f"field {field_name!r} must be a string")
        value = value.strip()
        if not value and not allow_empty:
            raise SchemaError(f"field {field_name!r} must not be empty")
        if len(value) > max_len:
            raise SchemaError(f"field {field_name!r} exceeds max length {max_len}")
        if any(ord(ch) < 32 and ch not in "\n\t" for ch in value):
            raise SchemaError(f"field {field_name!r} contains control characters")
        return value
    if expected_type is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaError(f"field {field_name!r} must be an integer")
        if not (0 < value <= 2000):
            raise SchemaError(f"field {field_name!r} out of allowed range")
        return value
    raise SchemaError(f"unsupported type contract for {field_name!r}")


def _validate_emails(addresses: list[str]) -> None:
    for address in addresses:
        if "@" not in address or " " in address or address.startswith("@") or address.endswith("@"):
            raise SchemaError(f"invalid email address: {address!r}")


def _parse_datetime(value: str, field_name: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaError(f"field {field_name!r} must be an ISO-8601 datetime") from exc


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------

# Fields that make up the *signed* canonical content of a proposal.
PROPOSAL_SIGNED_FIELDS = [
    "schema_version", "proposal_id", "session_id", "principal_id",
    "user_goal_hash", "model_id", "tool_name", "operation", "arguments",
    "source_digests", "policy_version", "risk_level", "created_at", "key_id",
]


@dataclass
class Proposal:
    schema_version: str
    proposal_id: str
    session_id: str
    principal_id: str
    user_goal_hash: str
    model_id: str
    tool_name: str
    operation: str
    arguments: dict[str, Any]
    source_digests: list[str]
    policy_version: str
    risk_level: str
    created_at: str
    key_id: str
    # Not signed -- untrusted, informational only (Week 3 renderer must not
    # treat this as authoritative). Kept separate from PROPOSAL_SIGNED_FIELDS.
    model_explanation: Optional[str] = None
    # Also not signed: structured claims the model *asserts* about its own
    # action (e.g. {"recipient": "office@vendor.example"}). Week 3's card
    # renderer diffs these against the authoritative signed `arguments` and
    # surfaces any mismatch -- it never trusts them on their own.
    claimed_fields: Optional[dict[str, Any]] = None
    # Populated after signing.
    digest: Optional[str] = None
    signature: Optional[str] = None

    def signed_view(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in PROPOSAL_SIGNED_FIELDS}

    def compute_digest(self) -> str:
        return sha256_digest(canonicalize(self.signed_view()))

    def to_row(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["arguments"] = json.dumps(d["arguments"], sort_keys=True)
        d["source_digests"] = json.dumps(d["source_digests"])
        return d


def build_proposal(
    *,
    session_id: str,
    principal_id: str,
    user_goal_hash: str,
    model_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    source_digests: list[str],
    policy_version: str,
    key_id: str,
    proposal_id: Optional[str] = None,
    created_at: Optional[str] = None,
    model_explanation: Optional[str] = None,
    claimed_fields: Optional[dict[str, Any]] = None,
    validate: bool = True,
) -> Proposal:
    """Construct an unsigned Proposal with normalized, validated arguments.

    ``validate=False`` is the Week 10 schema-layer ablation: the tool contract
    is not enforced and the arguments are signed exactly as the caller sent
    them.  Everything else is unchanged.

    Trusted context (session_id, principal_id, created_at, key_id) must be
    supplied by trusted code, never taken from model/agent output.
    """
    if tool_name not in TOOL_CONTRACTS:
        raise SchemaError(f"unknown tool: {tool_name!r}")
    if not session_id or not principal_id:
        raise SchemaError("session_id and principal_id are required (trusted identity)")
    if not isinstance(source_digests, list) or not all(isinstance(s, str) for s in source_digests):
        raise SchemaError("source_digests must be a list of strings")

    normalized_args = normalize_arguments(tool_name, arguments) if validate else dict(arguments)
    contract = TOOL_CONTRACTS[tool_name]

    return Proposal(
        schema_version=SCHEMA_VERSION,
        proposal_id=proposal_id or str(uuid.uuid4()),
        session_id=session_id,
        principal_id=principal_id,
        user_goal_hash=user_goal_hash,
        model_id=model_id,
        tool_name=tool_name,
        operation=tool_name,
        arguments=normalized_args,
        source_digests=list(source_digests),
        policy_version=policy_version,
        risk_level=contract["risk_level"].value,
        created_at=created_at or _now_iso(),
        key_id=key_id,
        model_explanation=model_explanation,
        claimed_fields=dict(claimed_fields) if claimed_fields else None,
    )


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------

APPROVAL_SIGNED_FIELDS = [
    "proposal_digest", "session_id", "principal_id", "approver_id",
    "decision", "issued_at", "expires_at", "nonce", "audience",
    "policy_version", "renderer_version",
]


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"


@dataclass
class Approval:
    proposal_digest: str
    session_id: str
    principal_id: str
    approver_id: str
    decision: str
    issued_at: str
    expires_at: str
    nonce: str
    audience: str
    policy_version: str
    renderer_version: str
    key_id: Optional[str] = None
    digest: Optional[str] = None
    signature: Optional[str] = None

    def signed_view(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in APPROVAL_SIGNED_FIELDS}

    def compute_digest(self) -> str:
        return sha256_digest(canonicalize(self.signed_view()))

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        return now >= _parse_iso(self.expires_at)


# ---------------------------------------------------------------------------
# Audit event
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    seq: int
    prev_hash: str
    event_type: str
    timestamp: str
    record_ids: dict[str, str]
    digest: str
    signer_key_id: str
    signature: Optional[str] = None

    def signed_view(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "prev_hash": self.prev_hash,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "record_ids": self.record_ids,
            "digest": self.digest,
            "signer_key_id": self.signer_key_id,
        }

    def event_hash(self) -> str:
        return sha256_digest(canonicalize(self.signed_view()))


# ---------------------------------------------------------------------------
# Execution receipt + state machine
# ---------------------------------------------------------------------------

class ActionState(str, Enum):
    PROPOSED = "PROPOSED"
    VERIFIED = "VERIFIED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


TERMINAL_NON_EXECUTABLE = {
    ActionState.BLOCKED, ActionState.DENIED, ActionState.EXPIRED, ActionState.REVOKED,
}

_ALLOWED_TRANSITIONS = {
    ActionState.PROPOSED: {ActionState.VERIFIED, ActionState.BLOCKED},
    ActionState.VERIFIED: {ActionState.PENDING_APPROVAL, ActionState.BLOCKED},
    ActionState.PENDING_APPROVAL: {ActionState.APPROVED, ActionState.DENIED, ActionState.EXPIRED},
    ActionState.APPROVED: {ActionState.EXECUTING, ActionState.REVOKED, ActionState.EXPIRED},
    ActionState.EXECUTING: {ActionState.SUCCEEDED, ActionState.FAILED},
}


def check_transition(current: ActionState, target: ActionState) -> None:
    if target not in _ALLOWED_TRANSITIONS.get(current, set()):
        raise SchemaError(f"illegal state transition {current} -> {target}")


@dataclass
class Receipt:
    action_digest: str
    attempt_id: str
    result_status: str
    result_digest: str
    idempotency_key: str


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_iso(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
