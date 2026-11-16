"""ProvenanceGuard prototype package (Weeks 1-4).

Modules:
    schemas   -- typed records: Proposal, Approval, AuditEvent, Receipt + strict normalization
    crypto    -- Ed25519 signer/verifier with key IDs (separate signing/verification access)
    store     -- append-only SQLite event store (+ action-state log, approvals, receipts,
                 budgets), a separate read-only checkpoint file
    policy    -- Week-1 initial policy configuration (tool allow-list, permissions, argument limits)
    tools     -- three simulated tools (summarize, send_email, update_record) + simulated world state
    builder   -- propose_action(): the only way an agent can turn a request into a proposal
    agent     -- deterministic agent stub (no direct tool credentials, no signing key)
    approval  -- Week 3: canonical approval-card renderer, reviewer auth, approval/token service
    gateway   -- Week 4: the sole write-tool caller; verifies + consumes approvals, dispatches
                 to adapters, writes durable receipts
    audit     -- Week 4: audit-console queries (pending approvals, alerts, full trace)
"""
