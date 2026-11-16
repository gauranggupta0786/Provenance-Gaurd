"""Trusted source labeling for the Weeks 1-4 ingestion boundary."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from .schemas import sha256_digest, _now_iso


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    tenant_id: str
    source_type: str
    source_identity: str
    trust_label: str
    content_digest: str
    version: int
    observed_at: str


class SourceLabeler:
    """Assigns provenance from application context, never model claims."""

    TRUSTED_TYPES = {"authenticated_user", "internal_record", "system_generated"}

    def label(
        self,
        *,
        tenant_id: str,
        source_type: str,
        source_identity: str,
        content: str,
        version: int = 1,
    ) -> SourceRecord:
        if not tenant_id or not source_identity or not content:
            raise ValueError("tenant, source identity, and content are required")
        return SourceRecord(
            source_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            source_type=source_type,
            source_identity=source_identity,
            trust_label="trusted" if source_type in self.TRUSTED_TYPES else "untrusted",
            content_digest=sha256_digest(content.encode("utf-8")),
            version=version,
            observed_at=_now_iso(),
        )
