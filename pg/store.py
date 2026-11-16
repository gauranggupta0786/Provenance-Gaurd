"""Week 2 Day 4-5: transactional, tamper-evident, append-only event store.

- SQLite is the transactional store (one controlled writer: this module).
- `events` and `proposals` tables have DB-level triggers that reject
  UPDATE/DELETE, so even a bug or a compromised second writer cannot
  silently mutate history through normal SQL.
- A hash chain links every audit event to the previous one.
- A checkpoint is exported to a SEPARATE file (not another row in the
  same table) and is treated as read-only once written: verification
  compares the live chain against the last trusted checkpoint before
  accepting new writes, so log-tail deletion or whole-log replacement
  is detectable even though a local hash chain alone cannot reveal it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import glob
import threading
import uuid
from dataclasses import asdict
from typing import Optional

from .ablation import FULL, Layers
from .schemas import (
    Proposal, Approval, AuditEvent, Receipt, ActionState,
    canonicalize, sha256_digest, SchemaError, check_transition,
)
from .crypto import Signer, Verifier
from .schema_adapters import (
    approval_record_from_legacy,
    approval_record_from_sqlite,
    proposal_legacy_from_record,
    proposal_record_from_legacy,
    proposal_record_from_sqlite,
    receipt_record_from_legacy,
    receipt_record_from_sqlite,
    source_evidence_from_legacy,
)
from .trusted_models import (
    DataSensitivity, OutputGuardRecord, PolicyDecision, PolicyOutcome, SensitivityEvidence,
)


class DuplicateExecutionError(Exception):
    """Raised when a second execution attempt is made for a proposal that
    already has a reservation/receipt -- the idempotency/replay guard."""


class ReplayError(Exception):
    """Raised when an approval's nonce/consumption state is reused."""

GENESIS_HASH = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    digest TEXT NOT NULL,
    signature TEXT NOT NULL,
    key_id TEXT NOT NULL,
    extra TEXT,
    inserted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    prev_hash TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    record_ids TEXT NOT NULL,
    digest TEXT NOT NULL,
    signer_key_id TEXT NOT NULL,
    signature TEXT NOT NULL,
    event_hash TEXT NOT NULL
);

-- Week 3/4: append-only action-state history. "Current state" of a
-- proposal is the row with the highest seq for that proposal_id. Never
-- updated in place -- every transition is a new row -- so it inherits the
-- same tamper-evidence story as `events` via the same trigger pattern.
CREATE TABLE IF NOT EXISTS action_state_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT NOT NULL,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Week 3: minted approvals ("short-lived tokens"). `consumed` is the one
-- field allowed to change, and only 0->1, enforced by trigger below --
-- this is what makes an approval single-use (invariant I4).
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    proposal_digest TEXT NOT NULL,
    payload TEXT NOT NULL,
    nonce TEXT NOT NULL UNIQUE,
    signature TEXT NOT NULL,
    key_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed INTEGER NOT NULL DEFAULT 0,
    consumed_at TEXT,
    inserted_at TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT 'approval-record-v1'
);

-- Week 4: durable execution attempts / receipts. `idempotency_key` is
-- UNIQUE (we use the proposal digest) so a second execution attempt for
-- the same approved proposal fails at the DB layer even under a race.
CREATE TABLE IF NOT EXISTS receipts (
    attempt_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    action_digest TEXT NOT NULL,
    result_status TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT 'receipt-record-v1'
);

CREATE TABLE IF NOT EXISTS execution_reservations (
    idempotency_key TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL UNIQUE,
    approval_id TEXT NOT NULL UNIQUE,
    attempt_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    schema_version TEXT NOT NULL DEFAULT 'execution-record-v1'
);

CREATE TABLE IF NOT EXISTS policy_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT 'policy-outcome-v1'
);

CREATE TABLE IF NOT EXISTS source_records (
    source_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_identity TEXT NOT NULL,
    trust_label TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    version INTEGER NOT NULL,
    observed_at TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT 'source-evidence-v1'
);

CREATE TABLE IF NOT EXISTS store_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    database_id TEXT NOT NULL,
    schema_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sensitivity_records (
    evidence_id TEXT PRIMARY KEY,
    graph_run_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_version INTEGER NOT NULL,
    object_id TEXT NOT NULL,
    field_path TEXT NOT NULL,
    value_digest TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    read_authorized INTEGER NOT NULL,
    authorization_rule TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT 'sensitivity-evidence-v1'
);

CREATE TABLE IF NOT EXISTS output_guard_records (
    guard_id TEXT PRIMARY KEY,
    graph_run_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    record_json TEXT NOT NULL,
    response_digest TEXT NOT NULL,
    decision TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT 'output-guard-record-v1'
);

-- Week 4 Day 3: per-principal/tool counters, keyed on trusted identity and
-- a fixed time window -- never on model-supplied wording, so rephrasing a
-- request cannot reset the count.
CREATE TABLE IF NOT EXISTS budgets (
    principal_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    counter_kind TEXT NOT NULL,
    window_start INTEGER NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (principal_id, tool_name, counter_kind, window_start)
);

CREATE TRIGGER IF NOT EXISTS proposals_no_update
BEFORE UPDATE ON proposals
BEGIN
    SELECT RAISE(ABORT, 'proposals table is append-only: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS proposals_no_delete
BEFORE DELETE ON proposals
BEGIN
    SELECT RAISE(ABORT, 'proposals table is append-only: DELETE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS events_no_update
BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events table is append-only: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS events_no_delete
BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events table is append-only: DELETE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS action_state_log_no_update
BEFORE UPDATE ON action_state_log
BEGIN
    SELECT RAISE(ABORT, 'action_state_log is append-only: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS action_state_log_no_delete
BEFORE DELETE ON action_state_log
BEGIN
    SELECT RAISE(ABORT, 'action_state_log is append-only: DELETE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS approvals_no_delete
BEFORE DELETE ON approvals
BEGIN
    SELECT RAISE(ABORT, 'approvals table is append-only: DELETE forbidden');
END;

-- Only the single-use consumption transition (0->1 of `consumed`, and its
-- timestamp) may ever change on an approval row; every other field --
-- including the signed payload, decision, nonce, and signature -- is
-- immutable once written.
CREATE TRIGGER IF NOT EXISTS approvals_consume_only
BEFORE UPDATE ON approvals
WHEN NOT (
    OLD.consumed = 0 AND NEW.consumed = 1
    AND NEW.approval_id = OLD.approval_id AND NEW.proposal_id = OLD.proposal_id
    AND NEW.proposal_digest = OLD.proposal_digest AND NEW.payload = OLD.payload
    AND NEW.nonce = OLD.nonce AND NEW.signature = OLD.signature
    AND NEW.key_id = OLD.key_id AND NEW.decision = OLD.decision
    AND NEW.expires_at = OLD.expires_at AND NEW.inserted_at = OLD.inserted_at
)
BEGIN
    SELECT RAISE(ABORT, 'approvals table only allows the single-use consumed 0->1 transition');
END;

CREATE TRIGGER IF NOT EXISTS receipts_no_update
BEFORE UPDATE ON receipts
BEGIN
    SELECT RAISE(ABORT, 'receipts table is append-only: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS receipts_no_delete
BEFORE DELETE ON receipts
BEGIN
    SELECT RAISE(ABORT, 'receipts table is append-only: DELETE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS policy_decisions_no_update
BEFORE UPDATE ON policy_decisions
BEGIN
    SELECT RAISE(ABORT, 'policy_decisions is append-only: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS policy_decisions_no_delete
BEFORE DELETE ON policy_decisions
BEGIN
    SELECT RAISE(ABORT, 'policy_decisions is append-only: DELETE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS source_records_no_update
BEFORE UPDATE ON source_records
BEGIN
    SELECT RAISE(ABORT, 'source_records is append-only: UPDATE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS source_records_no_delete
BEFORE DELETE ON source_records
BEGIN
    SELECT RAISE(ABORT, 'source_records is append-only: DELETE forbidden');
END;

CREATE TRIGGER IF NOT EXISTS sensitivity_records_no_update
BEFORE UPDATE ON sensitivity_records BEGIN SELECT RAISE(ABORT, 'sensitivity_records is append-only'); END;
CREATE TRIGGER IF NOT EXISTS sensitivity_records_no_delete
BEFORE DELETE ON sensitivity_records BEGIN SELECT RAISE(ABORT, 'sensitivity_records is append-only'); END;
CREATE TRIGGER IF NOT EXISTS output_guard_records_no_update
BEFORE UPDATE ON output_guard_records BEGIN SELECT RAISE(ABORT, 'output_guard_records is append-only'); END;
CREATE TRIGGER IF NOT EXISTS output_guard_records_no_delete
BEFORE DELETE ON output_guard_records BEGIN SELECT RAISE(ABORT, 'output_guard_records is append-only'); END;
"""


class IntegrityError(Exception):
    pass


UNSIGNED_PLACEHOLDER = "unsigned-ablation-placeholder"
UNCHAINED_HASH = "0" * 64


class EventStore:
    def __init__(self, db_path: str, checkpoint_dir: str, layers: Layers = FULL):
        self.db_path = db_path
        self.checkpoint_dir = checkpoint_dir
        # Week 10: which protection layers this store enforces.  FULL is the
        # historical behaviour; every other value is an ablation.
        self.layers = layers
        # Only read when layers.atomic_consume is False: an artificial delay
        # between the single-use check and the side effect so a threaded
        # double-execution test is deterministic instead of timing luck.
        self.nonatomic_window_seconds = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)
        # check_same_thread=False: safe here because every multi-step write
        # path takes the SQLite write lock itself (BEGIN IMMEDIATE) rather
        # than relying on Python-level locking, and single-statement writes
        # are already serialized by SQLite. This is what lets the Week 4
        # concurrent-execution test exercise two real threads against one
        # store, matching the plan's "concurrent submissions"/"concurrent
        # attempts" exit checks.
        self._conn = sqlite3.connect(db_path, timeout=5.0, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._migrate_schema_version_columns()
        row = self._conn.execute("SELECT database_id FROM store_metadata WHERE singleton=1").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO store_metadata (singleton,database_id,schema_version) VALUES (1,?,?)",
                (str(uuid.uuid4()), "event-store-metadata-v1"),
            )
        self._write_lock = threading.RLock()
        self._integrity_verifier: Optional[Verifier] = None

    def _migrate_schema_version_columns(self) -> None:
        """Add explicit record versions to databases created before revised Week 3."""
        versions = {
            "approvals": "approval-record-v1",
            "receipts": "receipt-record-v1",
            "execution_reservations": "execution-record-v1",
            "policy_decisions": "policy-outcome-v1",
            "source_records": "source-evidence-v1",
        }
        for table, version in versions.items():
            columns = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if "schema_version" not in columns:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN schema_version TEXT NOT NULL DEFAULT '{version}'"
                )

    def register_integrity_verifier(self, verifier: Verifier) -> None:
        self._integrity_verifier = verifier

    def _pre_write_integrity_check(self) -> None:
        if self.last_trusted_checkpoint() is None:
            return
        if self._integrity_verifier is None:
            raise IntegrityError("a trusted checkpoint exists but no verifier is registered")
        self.verify_chain(self._integrity_verifier)
        self.verify_against_checkpoint(self._integrity_verifier)

    def close(self):
        self._conn.close()

    # -- writes ------------------------------------------------------------

    def append_proposal(self, proposal: Proposal) -> None:
        # Pydantic is the executable contract at the persistence boundary;
        # the legacy signed dataclass remains the cryptographic wire format.
        proposal_record_from_legacy(proposal)
        if proposal.digest is None or proposal.signature is None:
            raise SchemaError("cannot store an unsigned proposal")
        payload = json.dumps(proposal.signed_view(), sort_keys=True)
        extra = None
        if proposal.model_explanation is not None or proposal.claimed_fields is not None:
            extra = json.dumps({
                "model_explanation": proposal.model_explanation,
                "claimed_fields": proposal.claimed_fields,
            }, sort_keys=True)
        self._pre_write_integrity_check()
        try:
            with self._write_lock, self._conn:
                self._conn.execute(
                    "INSERT INTO proposals (proposal_id, payload, digest, signature, key_id, extra, inserted_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (proposal.proposal_id, payload, proposal.digest, proposal.signature,
                     proposal.key_id, extra, _now_iso()),
                )
        except sqlite3.IntegrityError as e:
            raise SchemaError(f"duplicate or invalid proposal_id: {e}") from e

    def append_event(self, event_type: str, record_ids: dict[str, str],
                      content_digest: str, signer: Signer) -> AuditEvent:
        # BEGIN IMMEDIATE grabs the write lock before we read the current
        # tail, so concurrent writers (from this or other connections/
        # processes) serialize instead of racing on the next seq number.
        # The `timeout=` set at connect() makes SQLite retry internally
        # while a competing writer holds the lock.
        self._pre_write_integrity_check()
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                event = self._append_event_in_transaction(
                    event_type, record_ids, content_digest, signer,
                )
                self._conn.execute("COMMIT")
                return event
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _append_event_in_transaction(
        self,
        event_type: str,
        record_ids: dict[str, str],
        content_digest: str,
        signer: Signer,
    ) -> AuditEvent:
        row = self._conn.execute(
            "SELECT seq, event_hash FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq = (row[0] + 1) if row else 1
        if not self.layers.event_chain:
            # Ablation: a plain log row with no hash link and no signature.
            event = AuditEvent(
                seq=seq, prev_hash=GENESIS_HASH, event_type=event_type, timestamp=_now_iso(),
                record_ids=record_ids, digest=content_digest, signer_key_id=signer.key_id,
                signature=UNSIGNED_PLACEHOLDER,
            )
            self._conn.execute(
                "INSERT INTO events (seq, prev_hash, event_type, timestamp, record_ids, "
                "digest, signer_key_id, signature, event_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    event.seq, event.prev_hash, event.event_type, event.timestamp,
                    json.dumps(record_ids, sort_keys=True), event.digest,
                    event.signer_key_id, event.signature, UNCHAINED_HASH,
                ),
            )
            return event
        prev_hash = row[1] if row else GENESIS_HASH
        event = AuditEvent(
            seq=seq,
            prev_hash=prev_hash,
            event_type=event_type,
            timestamp=_now_iso(),
            record_ids=record_ids,
            digest=content_digest,
            signer_key_id=signer.key_id,
        )
        event_hash = event.event_hash()
        event.signature = signer.sign(canonicalize(event.signed_view()))
        self._conn.execute(
            "INSERT INTO events (seq, prev_hash, event_type, timestamp, record_ids, "
            "digest, signer_key_id, signature, event_hash) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                event.seq, event.prev_hash, event.event_type, event.timestamp,
                json.dumps(record_ids, sort_keys=True), event.digest,
                event.signer_key_id, event.signature, event_hash,
            ),
        )
        return event

    # -- reads ---------------------------------------------------------------

    def get_proposal(self, proposal_id: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT payload, digest, signature, key_id, extra FROM proposals WHERE proposal_id=?",
            (proposal_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        payload, digest, signature, key_id, extra = row
        extra_obj = json.loads(extra) if extra else {"model_explanation": None, "claimed_fields": None}
        return {"payload": json.loads(payload), "digest": digest,
                "signature": signature, "key_id": key_id,
                "model_explanation": extra_obj.get("model_explanation"),
                "claimed_fields": extra_obj.get("claimed_fields")}

    def load_verified_proposal(self, proposal_id: str, verifier: Verifier) -> Proposal:
        """Reload a proposal *from the store* (never from caller-held state)
        and verify its signature before returning it. This is what every
        Week 3/4 consumer (renderer, approval service, gateway) must call --
        never trust an in-memory `Proposal` object handed to it directly."""
        row = self.get_proposal(proposal_id)
        if row is None:
            raise SchemaError(f"no such proposal: {proposal_id!r}")
        payload = row["payload"]
        if self.layers.signed_proposal and not verifier.verify(
            row["key_id"], canonicalize(payload), row["signature"],
        ):
            raise SchemaError(f"proposal {proposal_id!r} fails signature verification")
        recomputed = sha256_digest(canonicalize(payload))
        if recomputed != row["digest"]:
            raise SchemaError(f"proposal {proposal_id!r} digest does not match stored payload")
        return proposal_legacy_from_record(proposal_record_from_sqlite(row))

    def all_events(self) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute("SELECT * FROM events ORDER BY seq ASC")
        rows = cur.fetchall()
        self._conn.row_factory = None
        return rows

    def rejected_attempts(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT seq, event_type, timestamp, record_ids, digest "
            "FROM events WHERE event_type LIKE 'REJECTED_%' ORDER BY seq ASC"
        ).fetchall()
        return [
            {
                "seq": row[0],
                "event_type": row[1],
                "timestamp": row[2],
                "record_ids": json.loads(row[3]),
                "digest": row[4],
            }
            for row in rows
        ]

    def append_source_record(self, record) -> None:
        source_evidence_from_legacy(record)
        self._pre_write_integrity_check()
        with self._write_lock, self._conn:
            self._conn.execute(
                "INSERT INTO source_records "
                "(source_id, tenant_id, source_type, source_identity, trust_label, "
                "content_digest, version, observed_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    record.source_id, record.tenant_id, record.source_type,
                    record.source_identity, record.trust_label, record.content_digest,
                    record.version, record.observed_at,
                ),
            )

    def sources_by_digest(self, digests: list[str]) -> list[dict]:
        if not digests:
            return []
        placeholders = ",".join("?" for _ in digests)
        rows = self._conn.execute(
            f"SELECT source_id, tenant_id, source_type, source_identity, trust_label, "
            f"content_digest, version, observed_at FROM source_records "
            f"WHERE content_digest IN ({placeholders}) ORDER BY observed_at ASC",
            tuple(digests),
        ).fetchall()
        keys = [
            "source_id", "tenant_id", "source_type", "source_identity",
            "trust_label", "content_digest", "version", "observed_at",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def assert_source_versions_current(self, digests: list[str]) -> None:
        records = self.sources_by_digest(digests)
        found = {record["content_digest"] for record in records}
        missing = set(digests) - found
        if missing:
            raise IntegrityError(f"proposal source evidence is missing: {sorted(missing)[:3]}")
        for record in records:
            latest = self._conn.execute(
                "SELECT MAX(version) FROM source_records WHERE tenant_id=? AND source_type=? "
                "AND source_identity=?",
                (record["tenant_id"], record["source_type"], record["source_identity"]),
            ).fetchone()[0]
            if latest is not None and int(latest) > int(record["version"]):
                raise IntegrityError(
                    f"source version drift for {record['source_identity']!r}: "
                    f"proposal={record['version']} latest={latest}"
                )

    def append_sensitivity_evidence(self, record: SensitivityEvidence) -> None:
        checked = SensitivityEvidence.model_validate_json(record.model_dump_json())
        self._pre_write_integrity_check()
        with self._write_lock, self._conn:
            self._conn.execute(
                "INSERT INTO sensitivity_records "
                "(evidence_id,graph_run_id,principal_id,source_id,source_version,object_id,field_path,"
                "value_digest,sensitivity,read_authorized,authorization_rule,observed_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    checked.evidence_id, checked.graph_run_id, checked.principal_id,
                    checked.source_id, checked.source_version, checked.object_id,
                    checked.field_path, checked.value_digest, checked.sensitivity.value,
                    int(checked.read_authorized), checked.authorization_rule, checked.observed_at,
                ),
            )

    def append_output_guard_record(self, record: OutputGuardRecord) -> None:
        checked = OutputGuardRecord.model_validate_json(record.model_dump_json())
        self._pre_write_integrity_check()
        with self._write_lock, self._conn:
            self._conn.execute(
                "INSERT INTO output_guard_records "
                "(guard_id,graph_run_id,principal_id,record_json,response_digest,decision,evaluated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    checked.guard_id, checked.graph_run_id, checked.principal_id,
                    checked.model_dump_json(), checked.response_digest,
                    checked.decision.value, checked.evaluated_at,
                ),
            )

    def sensitivity_evidence_for_run(self, graph_run_id: str) -> list[SensitivityEvidence]:
        rows = self._conn.execute(
            "SELECT evidence_id,graph_run_id,principal_id,source_id,source_version,object_id,"
            "field_path,value_digest,sensitivity,read_authorized,authorization_rule,observed_at "
            "FROM sensitivity_records WHERE graph_run_id=? ORDER BY observed_at,evidence_id",
            (graph_run_id,),
        ).fetchall()
        keys = ("evidence_id", "graph_run_id", "principal_id", "source_id", "source_version",
                "object_id", "field_path", "value_digest", "sensitivity", "read_authorized",
                "authorization_rule", "observed_at")
        return [SensitivityEvidence.model_validate({
            **dict(zip(keys, row)), "sensitivity": DataSensitivity(row[8]),
            "read_authorized": bool(row[9]),
        }) for row in rows]

    def latest_seq(self) -> int:
        cur = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM events")
        return cur.fetchone()[0]

    # -- integrity -------------------------------------------------------------

    def verify_chain(self, verifier: Verifier) -> None:
        """Raise IntegrityError on any broken link, bad signature, or gap."""
        if not self.layers.event_chain:
            return
        rows = self.all_events()
        prev_hash = GENESIS_HASH
        expected_seq = 1
        for r in rows:
            if r["seq"] != expected_seq:
                raise IntegrityError(f"sequence gap or reordering at seq={r['seq']} (expected {expected_seq})")
            if r["prev_hash"] != prev_hash:
                raise IntegrityError(f"broken hash link at seq={r['seq']}")
            ev = AuditEvent(
                seq=r["seq"], prev_hash=r["prev_hash"], event_type=r["event_type"],
                timestamp=r["timestamp"], record_ids=json.loads(r["record_ids"]),
                digest=r["digest"], signer_key_id=r["signer_key_id"],
            )
            recomputed = ev.event_hash()
            if recomputed != r["event_hash"]:
                raise IntegrityError(f"event content altered at seq={r['seq']}")
            if not verifier.verify(r["signer_key_id"], canonicalize(ev.signed_view()), r["signature"]):
                raise IntegrityError(f"invalid or unverifiable signature at seq={r['seq']}")
            prev_hash = r["event_hash"]
            expected_seq += 1

    def export_checkpoint(self, signer: Signer) -> str:
        """Write a new, timestamped checkpoint file (append-only history;
        never rewritten in place). Returns the checkpoint file path."""
        if not self.layers.checkpoints:
            return ""
        seq = self.latest_seq()
        if seq == 0:
            head_hash = GENESIS_HASH
        else:
            rows = self.all_events()
            head_hash = rows[-1]["event_hash"]
        previous_files = self._checkpoint_files()
        previous = None
        if previous_files:
            with open(previous_files[-1], encoding="utf-8") as handle:
                previous = json.load(handle)
        checkpoint = {
            "schema_version": "signed-checkpoint-v2",
            "checkpoint_index": len(previous_files) + 1,
            "database_id": self.database_id(),
            "seq": seq,
            "head_hash": head_hash,
            "exported_at": _now_iso(),
            "previous_checkpoint_hash": (
                sha256_digest(canonicalize(previous)) if previous is not None else GENESIS_HASH
            ),
        }
        body = canonicalize(checkpoint)
        checkpoint["signature"] = signer.sign(body)
        checkpoint["signer_key_id"] = signer.key_id
        fname = os.path.join(self.checkpoint_dir, f"cp_{len(previous_files) + 1:06d}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, sort_keys=True, indent=2)
        checkpoint_hash = sha256_digest(canonicalize(checkpoint))
        anchor = {
            "schema_version": "checkpoint-anchor-v1",
            "database_id": self.database_id(),
            "checkpoint_count": len(previous_files) + 1,
            "latest_filename": os.path.basename(fname),
            "latest_checkpoint_hash": checkpoint_hash,
            "updated_at": _now_iso(),
        }
        anchor["signature"] = signer.sign(canonicalize(anchor))
        anchor["signer_key_id"] = signer.key_id
        anchor_path = os.path.join(self.checkpoint_dir, "anchor.json")
        temporary = anchor_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(anchor, handle, sort_keys=True, indent=2)
        os.replace(temporary, anchor_path)
        return fname

    def database_id(self) -> str:
        row = self._conn.execute("SELECT database_id FROM store_metadata WHERE singleton=1").fetchone()
        if row is None:
            raise IntegrityError("event store is missing its database identity")
        return str(row[0])

    def _checkpoint_files(self) -> list[str]:
        return sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.json"))
            + glob.glob(os.path.join(self.checkpoint_dir, "cp_*.json"))
        )

    def last_trusted_checkpoint(self) -> Optional[dict]:
        files = self._checkpoint_files()
        if not files:
            return None
        with open(files[-1]) as f:
            return json.load(f)

    def verify_checkpoint_history(self, verifier: Verifier) -> None:
        if not self.layers.checkpoints:
            return
        files = self._checkpoint_files()
        if not files:
            return
        previous_hash = GENESIS_HASH
        for index, path in enumerate(files, start=1):
            with open(path, encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            if checkpoint.get("schema_version") == "signed-checkpoint-v2":
                body = {k: v for k, v in checkpoint.items() if k not in {"signature", "signer_key_id"}}
                if not verifier.verify(
                    checkpoint["signer_key_id"], canonicalize(body), checkpoint["signature"],
                ):
                    raise IntegrityError("checkpoint file itself fails signature verification")
                if checkpoint["checkpoint_index"] != index:
                    raise IntegrityError("checkpoint history has a gap or reordering")
                if checkpoint["previous_checkpoint_hash"] != previous_hash:
                    raise IntegrityError("checkpoint history has a middle deletion or replacement")
                if checkpoint["database_id"] != self.database_id():
                    raise IntegrityError("database diverges: checkpoint belongs to a different database identity")
            else:
                body = {k: checkpoint[k] for k in ("seq", "head_hash", "exported_at")}
                if not verifier.verify(
                    checkpoint["signer_key_id"], canonicalize(body), checkpoint["signature"],
                ):
                    raise IntegrityError("checkpoint file itself fails signature verification")
            previous_hash = sha256_digest(canonicalize(checkpoint))

        anchor_path = os.path.join(self.checkpoint_dir, "anchor.json")
        if not os.path.exists(anchor_path):
            # Legacy histories predate anchors; the signed checkpoint still
            # pins the database prefix, but checkpoint-file tail deletion is
            # not retrospectively detectable.
            if any(json.load(open(path, encoding="utf-8")).get("schema_version") == "signed-checkpoint-v2"
                   for path in files):
                raise IntegrityError("checkpoint anchor is missing")
            return
        with open(anchor_path, encoding="utf-8") as handle:
            anchor = json.load(handle)
        anchor_body = {k: v for k, v in anchor.items() if k not in {"signature", "signer_key_id"}}
        if not verifier.verify(anchor["signer_key_id"], canonicalize(anchor_body), anchor["signature"]):
            raise IntegrityError("checkpoint anchor signature is invalid")
        if anchor["database_id"] != self.database_id():
            raise IntegrityError("database diverges: checkpoint anchor belongs to a different database")
        if anchor["checkpoint_count"] != len(files):
            raise IntegrityError("checkpoint tail deletion detected by anchor count")
        if anchor["latest_filename"] != os.path.basename(files[-1]) or anchor["latest_checkpoint_hash"] != previous_hash:
            raise IntegrityError("checkpoint anchor does not match checkpoint history")

    def verify_against_checkpoint(self, verifier: Verifier) -> None:
        """Verify the live chain still contains the last trusted checkpoint's
        (seq, head_hash) -- catches tail deletion and full-log replacement,
        which a local hash chain alone cannot reveal."""
        if not self.layers.checkpoints:
            return
        cp = self.last_trusted_checkpoint()
        if cp is None:
            return  # no checkpoint yet -- nothing to check against
        self.verify_checkpoint_history(verifier)
        body = ({k: v for k, v in cp.items() if k not in {"signature", "signer_key_id"}}
                if cp.get("schema_version") == "signed-checkpoint-v2"
                else {k: cp[k] for k in ("seq", "head_hash", "exported_at")})
        if not verifier.verify(cp["signer_key_id"], canonicalize(body), cp["signature"]):
            raise IntegrityError("checkpoint file itself fails signature verification")
        if cp["seq"] == 0:
            return
        rows = self.all_events()
        if len(rows) < cp["seq"]:
            raise IntegrityError(
                f"log truncated: checkpoint expects seq={cp['seq']} but only {len(rows)} events present"
            )
        row_at_seq = rows[cp["seq"] - 1]
        if row_at_seq["seq"] != cp["seq"] or row_at_seq["event_hash"] != cp["head_hash"]:
            raise IntegrityError("log diverges from last trusted checkpoint (possible replacement)")


    # -- action state (Week 3/4) ---------------------------------------------

    def current_state(self, proposal_id: str) -> Optional[ActionState]:
        cur = self._conn.execute(
            "SELECT state FROM action_state_log WHERE proposal_id=? ORDER BY seq DESC LIMIT 1",
            (proposal_id,),
        )
        row = cur.fetchone()
        return ActionState(row[0]) if row else None

    def set_state(self, proposal_id: str, target: ActionState, *, signer: Signer,
                   event_extra: Optional[dict] = None) -> None:
        """Append a new action-state row, enforcing the Week 1 state machine
        (`check_transition`), and log a signed audit event for it. Fails
        closed: an illegal transition raises and nothing is written."""
        self._pre_write_integrity_check()
        with self._write_lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                current = self._set_state_in_transaction(proposal_id, target)
                self._append_state_event_in_transaction(
                    proposal_id, current, target, signer, event_extra,
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _set_state_in_transaction(
        self, proposal_id: str, target: ActionState,
    ) -> Optional[ActionState]:
        row = self._conn.execute(
            "SELECT state FROM action_state_log WHERE proposal_id=? ORDER BY seq DESC LIMIT 1",
            (proposal_id,),
        ).fetchone()
        current = ActionState(row[0]) if row else None
        if current is None:
            if target != ActionState.PROPOSED:
                raise SchemaError(f"first state for a proposal must be PROPOSED, not {target}")
        else:
            check_transition(current, target)
        self._conn.execute(
            "INSERT INTO action_state_log (proposal_id, state, updated_at) VALUES (?,?,?)",
            (proposal_id, target.value, _now_iso()),
        )
        return current

    def _append_state_event_in_transaction(
        self,
        proposal_id: str,
        current: Optional[ActionState],
        target: ActionState,
        signer: Signer,
        event_extra: Optional[dict] = None,
    ) -> None:
        record_ids = {
            "proposal_id": proposal_id,
            "from": current.value if current else "NONE",
            "to": target.value,
        }
        if event_extra:
            record_ids.update(event_extra)
        self._append_event_in_transaction(
            event_type="STATE_TRANSITION",
            record_ids=record_ids,
            content_digest=sha256_digest(f"{proposal_id}|{target.value}".encode("utf-8")),
            signer=signer,
        )

    def persist_verified_proposal(
        self,
        proposal: Proposal,
        *,
        attempt_id: str,
        policy_decision: str,
        policy_reason: str,
        signer: Signer,
        source_records: Optional[list] = None,
    ) -> None:
        """Persist proposal, policy evidence, states, and events atomically."""
        if proposal.digest is None or proposal.signature is None:
            raise SchemaError("cannot store an unsigned proposal")
        self._pre_write_integrity_check()
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for record in source_records or []:
                    self._conn.execute(
                        "INSERT INTO source_records "
                        "(source_id, tenant_id, source_type, source_identity, trust_label, "
                        "content_digest, version, observed_at) VALUES (?,?,?,?,?,?,?,?)",
                        (
                            record.source_id, record.tenant_id, record.source_type,
                            record.source_identity, record.trust_label, record.content_digest,
                            record.version, record.observed_at,
                        ),
                    )
                extra = None
                if proposal.model_explanation is not None or proposal.claimed_fields is not None:
                    extra = json.dumps({
                        "model_explanation": proposal.model_explanation,
                        "claimed_fields": proposal.claimed_fields,
                    }, sort_keys=True)
                self._conn.execute(
                    "INSERT INTO proposals (proposal_id, payload, digest, signature, key_id, extra, inserted_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        proposal.proposal_id,
                        json.dumps(proposal.signed_view(), sort_keys=True),
                        proposal.digest,
                        proposal.signature,
                        proposal.key_id,
                        extra,
                        _now_iso(),
                    ),
                )
                self._append_event_in_transaction(
                    "PROPOSAL_STORED",
                    {"proposal_id": proposal.proposal_id, "attempt_id": attempt_id},
                    proposal.digest,
                    signer,
                )
                self._insert_policy_decision_in_transaction(
                    proposal.proposal_id, "proposal", policy_decision,
                    policy_reason, proposal.policy_version, signer,
                )
                current = self._set_state_in_transaction(proposal.proposal_id, ActionState.PROPOSED)
                self._append_state_event_in_transaction(
                    proposal.proposal_id, current, ActionState.PROPOSED, signer,
                )
                current = self._set_state_in_transaction(proposal.proposal_id, ActionState.VERIFIED)
                self._append_state_event_in_transaction(
                    proposal.proposal_id, current, ActionState.VERIFIED, signer,
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def proposals_in_state(self, state: ActionState) -> list[str]:
        """All proposal_ids whose *current* (latest-seq) state matches --
        used by the audit console and by reconciliation (e.g. finding
        proposals stuck in EXECUTING with no receipt after a crash)."""
        cur = self._conn.execute(
            "SELECT t.proposal_id FROM action_state_log t "
            "INNER JOIN (SELECT proposal_id, MAX(seq) AS max_seq FROM action_state_log GROUP BY proposal_id) m "
            "ON t.proposal_id = m.proposal_id AND t.seq = m.max_seq "
            "WHERE t.state = ? ORDER BY t.proposal_id",
            (state.value,),
        )
        return [r[0] for r in cur.fetchall()]

    def state_history(self, proposal_id: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT seq, state, updated_at FROM action_state_log WHERE proposal_id=? ORDER BY seq ASC",
            (proposal_id,),
        )
        return [{"seq": r[0], "state": r[1], "updated_at": r[2]} for r in cur.fetchall()]

    # -- approvals (Week 3) ---------------------------------------------------

    def append_approval(self, approval: Approval, approval_id: str, proposal_id: str) -> None:
        """`proposal_id` is passed explicitly by the caller (it is not one
        of APPROVAL_SIGNED_FIELDS -- only the proposal's *digest* is signed
        over) purely so this row can be looked up/joined by proposal_id;
        the binding that actually matters for security is `proposal_digest`,
        which the gateway re-checks against the freshly reloaded proposal."""
        approval_record_from_legacy(
            approval, approval_id=approval_id, proposal_id=proposal_id,
        )
        if approval.digest is None or approval.signature is None:
            raise SchemaError("cannot store an unsigned approval")
        payload = json.dumps(approval.signed_view(), sort_keys=True)
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO approvals (approval_id, proposal_id, proposal_digest, payload, nonce, "
                    "signature, key_id, decision, expires_at, consumed, inserted_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,0,?)",
                    (approval_id, proposal_id, approval.proposal_digest, payload, approval.nonce,
                     approval.signature, approval.key_id, approval.decision, approval.expires_at,
                     _now_iso()),
                )
        except sqlite3.IntegrityError as e:
            raise SchemaError(f"duplicate approval_id or reused nonce: {e}") from e

    def get_approval(self, approval_id: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT approval_id, proposal_id, proposal_digest, payload, nonce, signature, key_id, "
            "decision, expires_at, consumed, consumed_at FROM approvals WHERE approval_id=?",
            (approval_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = ["approval_id", "proposal_id", "proposal_digest", "payload", "nonce", "signature",
                "key_id", "decision", "expires_at", "consumed", "consumed_at"]
        d = dict(zip(keys, row))
        d["payload"] = json.loads(d["payload"])
        d["consumed"] = bool(d["consumed"])
        approval_record_from_sqlite(d)
        return d

    def approvals_for_proposal(self, proposal_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT approval_id FROM approvals WHERE proposal_id=? ORDER BY inserted_at ASC",
            (proposal_id,),
        ).fetchall()
        return [self.get_approval(row[0]) for row in rows]

    def _insert_policy_decision_in_transaction(
        self,
        proposal_id: str,
        phase: str,
        decision: str,
        reason: str,
        policy_version: str,
        signer: Signer,
    ) -> None:
        decided_at = _now_iso()
        PolicyOutcome(
            proposal_id=proposal_id,
            phase=phase,
            decision=PolicyDecision(decision),
            reason=reason,
            policy_version=policy_version,
            evaluated_at=decided_at,
        )
        self._conn.execute(
            "INSERT INTO policy_decisions (proposal_id, phase, decision, reason, policy_version, decided_at) "
            "VALUES (?,?,?,?,?,?)",
            (proposal_id, phase, decision, reason, policy_version, decided_at),
        )
        digest = sha256_digest(canonicalize({
            "proposal_id": proposal_id,
            "phase": phase,
            "decision": decision,
            "reason": reason,
            "policy_version": policy_version,
            "decided_at": decided_at,
        }))
        self._append_event_in_transaction(
            "POLICY_DECISION",
            {"proposal_id": proposal_id, "phase": phase, "decision": decision},
            digest,
            signer,
        )

    def append_policy_decision(
        self,
        proposal_id: str,
        phase: str,
        decision: str,
        reason: str,
        policy_version: str,
        signer: Signer,
    ) -> None:
        self._pre_write_integrity_check()
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._insert_policy_decision_in_transaction(
                    proposal_id, phase, decision, reason, policy_version, signer,
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def policy_decisions_for_proposal(self, proposal_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT phase, decision, reason, policy_version, decided_at "
            "FROM policy_decisions WHERE proposal_id=? ORDER BY decision_id ASC",
            (proposal_id,),
        ).fetchall()
        keys = ["phase", "decision", "reason", "policy_version", "decided_at"]
        return [dict(zip(keys, row)) for row in rows]

    def consume_approval(self, approval_id: str) -> bool:
        """Atomically mark an approval as consumed. Returns False (does NOT
        raise) if it was already consumed -- the caller decides whether
        that means 'replay attempt' or 'benign double-submit'."""
        for _ in range(50):
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                cur = self._conn.execute("SELECT consumed FROM approvals WHERE approval_id=?", (approval_id,))
                row = cur.fetchone()
                if row is None:
                    self._conn.execute("ROLLBACK")
                    raise SchemaError(f"no such approval: {approval_id!r}")
                if row[0]:
                    self._conn.execute("ROLLBACK")
                    return False
                self._conn.execute(
                    "UPDATE approvals SET consumed=1, consumed_at=? WHERE approval_id=?",
                    (_now_iso(), approval_id),
                )
                self._conn.execute("COMMIT")
                return True
            except sqlite3.OperationalError:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                time.sleep(0.01)
        raise IntegrityError("consume_approval: exhausted retries under contention")

    def consume_approval_and_reserve_execution(
        self,
        *,
        approval_id: str,
        proposal_id: str,
        idempotency_key: str,
        attempt_id: str,
        signer: Signer,
    ) -> bool:
        """Atomically consume approval, reserve idempotency, and enter EXECUTING."""
        self._pre_write_integrity_check()
        if not self.layers.atomic_consume:
            return self._consume_nonatomic(approval_id=approval_id, proposal_id=proposal_id)
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT consumed FROM approvals WHERE approval_id=?",
                    (approval_id,),
                ).fetchone()
                if row is None:
                    raise SchemaError(f"no such approval: {approval_id!r}")
                if row[0]:
                    self._conn.execute("ROLLBACK")
                    return False
                current = self.current_state(proposal_id)
                if current != ActionState.APPROVED:
                    raise SchemaError(
                        f"proposal {proposal_id!r} is in state {current}, not APPROVED"
                    )
                self._conn.execute(
                    "INSERT INTO execution_reservations "
                    "(idempotency_key, proposal_id, approval_id, attempt_id, status, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (idempotency_key, proposal_id, approval_id, attempt_id, "RESERVED", _now_iso()),
                )
                self._conn.execute(
                    "UPDATE approvals SET consumed=1, consumed_at=? WHERE approval_id=?",
                    (_now_iso(), approval_id),
                )
                previous = self._set_state_in_transaction(proposal_id, ActionState.EXECUTING)
                self._append_state_event_in_transaction(
                    proposal_id,
                    previous,
                    ActionState.EXECUTING,
                    signer,
                    {"approval_id": approval_id, "attempt_id": attempt_id},
                )
                self._conn.execute("COMMIT")
                return True
            except sqlite3.IntegrityError as error:
                self._conn.execute("ROLLBACK")
                raise DuplicateExecutionError(
                    f"execution already reserved for proposal {proposal_id!r}: {error}"
                ) from error
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _consume_nonatomic(self, *, approval_id: str, proposal_id: str) -> bool:
        """Ablation of the atomic consume-and-reserve step.

        A plain check-then-act: read the single-use flag and the state, wait
        out the (optional) race window, and say "go".  Nothing is written
        here -- the flag, the EXECUTING state and the receipt are recorded
        only after the side effect has happened (see `_complete_nonatomic`).
        Two callers that both pass this check both execute.
        """
        row = self._conn.execute(
            "SELECT consumed FROM approvals WHERE approval_id=?", (approval_id,),
        ).fetchone()
        if row is None:
            raise SchemaError(f"no such approval: {approval_id!r}")
        if row[0]:
            return False
        current = self.current_state(proposal_id)
        if current != ActionState.APPROVED:
            raise SchemaError(f"proposal {proposal_id!r} is in state {current}, not APPROVED")
        if self.nonatomic_window_seconds > 0:
            time.sleep(self.nonatomic_window_seconds)
        return True

    def _complete_nonatomic(
        self, *, receipt: Receipt, proposal_id: str, detail: str, signer: Signer,
        approval_id: Optional[str], created_at: str,
    ) -> None:
        """Bookkeeping written after the side effect, with no reservation row
        and no enclosing transaction.  A second execution of the same
        approval therefore fails here, after it has already acted."""
        terminal = ActionState.SUCCEEDED if receipt.result_status == "SUCCEEDED" else ActionState.FAILED
        self.set_state(proposal_id, ActionState.EXECUTING, signer=signer,
                       event_extra={"attempt_id": receipt.attempt_id})
        with self._write_lock, self._conn:
            self._conn.execute(
                "INSERT INTO receipts (attempt_id, proposal_id, idempotency_key, action_digest, "
                "result_status, result_digest, detail, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    receipt.attempt_id, proposal_id, receipt.idempotency_key,
                    receipt.action_digest, receipt.result_status, receipt.result_digest,
                    detail, created_at,
                ),
            )
            if approval_id is not None:
                self._conn.execute(
                    "UPDATE approvals SET consumed=1, consumed_at=? WHERE approval_id=?",
                    (_now_iso(), approval_id),
                )
        self.set_state(proposal_id, terminal, signer=signer,
                       event_extra={"attempt_id": receipt.attempt_id,
                                    "result_status": receipt.result_status})
        self.append_event(
            "EXECUTION_RESULT",
            {"proposal_id": proposal_id, "attempt_id": receipt.attempt_id, "status": receipt.result_status},
            receipt.result_digest, signer,
        )

    def complete_reserved_execution(
        self,
        *,
        receipt: Receipt,
        proposal_id: str,
        detail: str,
        signer: Signer,
        approval_id: Optional[str] = None,
    ) -> None:
        """Atomically persist receipt, terminal state, and signed result events."""
        self._pre_write_integrity_check()
        created_at = _now_iso()
        receipt_record_from_legacy(
            receipt, proposal_id=proposal_id, detail=detail, created_at=created_at,
        )
        if not self.layers.atomic_consume:
            self._complete_nonatomic(
                receipt=receipt, proposal_id=proposal_id, detail=detail, signer=signer,
                approval_id=approval_id, created_at=created_at,
            )
            return
        terminal = ActionState.SUCCEEDED if receipt.result_status == "SUCCEEDED" else ActionState.FAILED
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                reservation = self._conn.execute(
                    "SELECT status FROM execution_reservations WHERE idempotency_key=? AND proposal_id=?",
                    (receipt.idempotency_key, proposal_id),
                ).fetchone()
                if reservation is None or reservation[0] != "RESERVED":
                    raise DuplicateExecutionError("missing or completed execution reservation")
                self._conn.execute(
                    "INSERT INTO receipts (attempt_id, proposal_id, idempotency_key, action_digest, "
                    "result_status, result_digest, detail, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        receipt.attempt_id, proposal_id, receipt.idempotency_key,
                        receipt.action_digest, receipt.result_status, receipt.result_digest,
                        detail, created_at,
                    ),
                )
                self._conn.execute(
                    "UPDATE execution_reservations SET status='COMPLETED', completed_at=? "
                    "WHERE idempotency_key=?",
                    (_now_iso(), receipt.idempotency_key),
                )
                previous = self._set_state_in_transaction(proposal_id, terminal)
                self._append_state_event_in_transaction(
                    proposal_id,
                    previous,
                    terminal,
                    signer,
                    {"attempt_id": receipt.attempt_id, "result_status": receipt.result_status},
                )
                self._append_event_in_transaction(
                    "EXECUTION_RESULT",
                    {
                        "proposal_id": proposal_id,
                        "attempt_id": receipt.attempt_id,
                        "status": receipt.result_status,
                    },
                    receipt.result_digest,
                    signer,
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # -- receipts (Week 4) -----------------------------------------------------

    def reserve_and_append_receipt(self, receipt: Receipt, proposal_id: str, detail: str = "") -> None:
        """INSERT with a UNIQUE idempotency_key -- the DB itself is the
        atomic reservation: a second attempt for the same proposal fails
        here even under a race between two threads/processes."""
        created_at = _now_iso()
        receipt_record_from_legacy(
            receipt, proposal_id=proposal_id, detail=detail, created_at=created_at,
        )
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO receipts (attempt_id, proposal_id, idempotency_key, action_digest, "
                    "result_status, result_digest, detail, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (receipt.attempt_id, proposal_id, receipt.idempotency_key, receipt.action_digest,
                     receipt.result_status, receipt.result_digest, detail, created_at),
                )
        except sqlite3.IntegrityError as e:
            raise DuplicateExecutionError(
                f"execution already attempted for proposal {proposal_id!r} "
                f"(idempotency_key={receipt.idempotency_key!r}): {e}"
            ) from e

    def get_receipt_by_proposal(self, proposal_id: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT attempt_id, idempotency_key, action_digest, result_status, result_digest, detail, "
            "created_at FROM receipts WHERE proposal_id=?",
            (proposal_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = ["attempt_id", "idempotency_key", "action_digest", "result_status", "result_digest",
                "detail", "created_at"]
        result = dict(zip(keys, row))
        result["proposal_id"] = proposal_id
        receipt_record_from_sqlite(result)
        result.pop("proposal_id")
        return result

    def all_receipts(self) -> list[dict]:
        cur = self._conn.execute(
            "SELECT attempt_id, proposal_id, idempotency_key, action_digest, result_status, "
            "result_digest, detail, created_at FROM receipts ORDER BY created_at ASC"
        )
        keys = ["attempt_id", "proposal_id", "idempotency_key", "action_digest", "result_status",
                "result_digest", "detail", "created_at"]
        return [dict(zip(keys, r)) for r in cur.fetchall()]

    # -- budgets / probe counters (Week 4 Day 3) --------------------------------

    def increment_budget(self, principal_id: str, tool_name: str, counter_kind: str,
                          window_seconds: int = 600) -> int:
        """Increment (or create) the counter for the *current* fixed window
        and return the new count. Keyed only on trusted identity + tool +
        wall-clock window -- never on any agent/model-supplied text, so
        rewording a request cannot reset it."""
        window_start = int(time.time() // window_seconds) * window_seconds
        with self._conn:
            self._conn.execute(
                "INSERT INTO budgets (principal_id, tool_name, counter_kind, window_start, count) "
                "VALUES (?,?,?,?,1) "
                "ON CONFLICT(principal_id, tool_name, counter_kind, window_start) "
                "DO UPDATE SET count = count + 1",
                (principal_id, tool_name, counter_kind, window_start),
            )
            cur = self._conn.execute(
                "SELECT count FROM budgets WHERE principal_id=? AND tool_name=? AND counter_kind=? AND window_start=?",
                (principal_id, tool_name, counter_kind, window_start),
            )
            return cur.fetchone()[0]

    def get_budget_count(self, principal_id: str, tool_name: str, counter_kind: str,
                          window_seconds: int = 600) -> int:
        """Read-only version of increment_budget's window logic -- does not
        create a row or increment anything."""
        window_start = int(time.time() // window_seconds) * window_seconds
        cur = self._conn.execute(
            "SELECT count FROM budgets WHERE principal_id=? AND tool_name=? AND counter_kind=? AND window_start=?",
            (principal_id, tool_name, counter_kind, window_start),
        )
        row = cur.fetchone()
        return row[0] if row else 0


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
