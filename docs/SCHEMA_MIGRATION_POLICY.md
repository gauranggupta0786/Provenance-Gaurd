# ProvenanceGuard schema migration policy

All trusted records use an explicit, immutable version string. JSON Schemas are generated into
`schemas/v1/` and their SHA-256 hashes are recorded in `manifest.json` and experiment manifests.

## Compatibility rules

1. A version is accepted only by the matching Pydantic model; unknown or incompatible versions fail closed.
2. Additive changes require a new schema version because `extra="forbid"` is part of the security contract.
3. Breaking changes require a new model and adapter. Existing models are never silently widened or made coercive.
4. Migrations deserialize with the old model, transform explicitly, and validate with the new model before writing.
5. Signed proposal and approval payloads are never rewritten. A migrated action receives a new signed record and identity.
6. SQLite migrations may add version columns with fixed defaults for historical rows, but may not update append-only evidence.
7. Generated schemas are deterministic. Any hash change requires review, regenerated manifests, and negative tests.
8. Readers must reject missing versions after migration and must not guess a version from payload shape.

The current relational migration adds explicit versions for approval, policy, source, execution-reservation, and receipt rows.
Proposal payloads already carry `schema_version` inside their signed content.
