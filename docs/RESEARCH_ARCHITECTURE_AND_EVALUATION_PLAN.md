# ProvenanceGuard Research Architecture and Evaluation Plan

## 1. Current implementation status

The current protected path is implemented with AgentDojo 0.1.35, Workspace
v1.2.2, Qwen3.5 9B through Ollama, typed Python dataclasses, deterministic
policy code, Ed25519 signatures, and SQLite. AgentDojo's own records are
Pydantic models, but ProvenanceGuard is not yet orchestrated as a LangGraph
state graph and its security records are not yet Pydantic `BaseModel` types.

This distinction must remain explicit in reports. The LangGraph/Pydantic work
below is a planned refactor, not a property of the already completed Week 6
experiment.

## 2. Exact implemented write lifecycle

1. AgentDojo loads an authenticated user task, a simulated Workspace, and any
   injected email/file content.
2. Qwen receives the user task, tool schemas, and tool results. All model output
   is untrusted.
3. `ProtectedWorkspaceRuntime` classifies the function. Classified reads use a
   restricted runtime. State-changing calls are mapped to a canonical
   ProvenanceGuard contract. Unknown tools fail closed.
4. `normalize_arguments` rejects missing, extra, duplicate, malformed,
   oversized, or semantically invalid fields.
5. Source records distinguish authenticated user intent from untrusted external
   content. Their digests are bound into the proposal.
6. The deterministic policy returns `ALLOW`, `REQUIRE_APPROVAL`, or `DENY`.
7. `ProposalService` constructs the canonical proposal, computes its SHA-256
   digest, signs the canonical bytes with Ed25519, and atomically writes the
   proposal, policy decision, state transitions, and signed audit events.
8. `ApprovalService` reloads and verifies the stored proposal, renders the
   authoritative action, rechecks policy, authenticates the reviewer, and
   signs an approval bound to the proposal digest, session, principal, reviewer,
   audience, policy version, nonce, and expiry.
9. `ExecutionGateway` reloads both records, verifies signatures and bindings,
   and rechecks current policy.
10. One SQLite transaction consumes the approval, reserves the proposal digest
    as an idempotency key, and moves the proposal to `EXECUTING` before the
    adapter is called.
11. `AgentDojoWorkspaceAdapter` executes only the verified stored arguments in
    the in-memory simulated Workspace.
12. A second transaction records the receipt, completes the reservation, moves
    the state to `SUCCEEDED` or `FAILED`, and appends signed result events.
13. The evaluator scores the actual final Workspace state. Model explanations
    and refusal text are not outcome evidence.

## 3. SQLite schema and write points

| Table | Exact fields | Written when |
|---|---|---|
| `source_records` | `source_id, tenant_id, source_type, source_identity, trust_label, content_digest, version, observed_at` | Sources are observed and labelled before the proposal is committed. |
| `proposals` | `proposal_id, payload, digest, signature, key_id, extra, inserted_at` | The normalized proposal has passed schema and is signed. |
| `policy_decisions` | `decision_id, proposal_id, phase, decision, reason, policy_version, decided_at` | At proposal time and again at execution time. |
| `action_state_log` | `seq, proposal_id, state, updated_at` | Every valid state transition; rows are appended rather than updated. |
| `approvals` | `approval_id, proposal_id, proposal_digest, payload, nonce, signature, key_id, decision, expires_at, consumed, consumed_at, inserted_at` | A reviewer or auto-approval issues a signed decision. Only `consumed: 0 -> 1` may change. |
| `execution_reservations` | `idempotency_key, proposal_id, approval_id, attempt_id, status, created_at, completed_at` | Atomically before side effects. Unique constraints prevent duplicate execution. |
| `receipts` | `attempt_id, proposal_id, idempotency_key, action_digest, result_status, result_digest, detail, created_at` | Atomically after the adapter returns or fails. |
| `events` | `seq, prev_hash, event_type, timestamp, record_ids, digest, signer_key_id, signature, event_hash` | Alongside each security-relevant lifecycle transition. |
| `budgets` | `principal_id, tool_name, counter_kind, window_start, count` | On relevant repeated rejected attempts. |

## 4. Cryptographic construction and limitations

### Proposal integrity

The signed proposal fields are serialized with sorted JSON keys, compact
separators, UTF-8 encoding, and no floating-point values. This is a simplified
JCS-style canonicalization, not a complete RFC 8785 implementation.

```text
canonical_bytes = canonicalize(proposal.signed_view)
proposal_digest = SHA-256(canonical_bytes)
signature       = Ed25519.Sign(private_key, canonical_bytes)
```

Verification reloads the stored payload, reconstructs the same canonical bytes,
recomputes SHA-256, and calls `Ed25519.Verify(public_key, signature,
canonical_bytes)`. Changing any signed field invalidates the digest and
signature. An attacker who knows the architecture and public key still cannot
create a valid signature without the private key, under the security assumptions
of Ed25519.

The current experiment generates private keys in process and keeps the signer
out of model/agent code. This demonstrates control-flow and tamper-detection
properties, but it is not production key custody. Deployment or stronger
artifact claims require persistent key identity, rotation, process isolation,
access controls, and preferably KMS/HSM-backed signing.

### Event and checkpoint integrity

Every event stores `prev_hash`, so modification, insertion, reordering, or
middle deletion breaks the chain. Each event is also Ed25519-signed. A separate
signed JSON checkpoint stores `seq`, `head_hash`, and `exported_at`; it detects
tail deletion and replacement by a newly constructed internally consistent
database.

## 5. Evidence graph model

The current database is relational, not a graph database. It nevertheless forms
a logical evidence graph through stable identifiers and digests:

```text
source_records.content_digest ----+
                                  +--> proposals.source_digests
user goal hash -------------------+          |
                                             +--> proposals.digest
                                                      |
                     policy_decisions.proposal_id ----+
                     action_state_log.proposal_id ----+
                                                      +--> approvals.proposal_digest
                                                               |
                         execution_reservations.approval_id ----+
                                                               +--> receipts.attempt_id
                                                      |
events.record_ids + events.digest --------------------+
                                                      |
checkpoint.head_hash --> pinned prefix of event chain-+
```

`AuditConsole.trace(proposal_id)` reconstructs this evidence lineage. A future
publication artifact may export the trace as a typed graph for analysis, while
keeping SQLite as the transactional source of truth.

## 6. Structured LangGraph and Pydantic migration

### Required Pydantic models

- `AgentRunState`: run ID, scenario ID, arm, seed, current node, trusted context.
- `SourceEvidence`: source ID, identity, trust label, digest, observation time.
- `ToolIntent`: AgentDojo function name, canonical tool, strict arguments,
  source references.
- `PolicyOutcome`: phase, decision, reason code, policy version.
- `ProposalRecord`, `ApprovalRecord`, `ExecutionRecord`, and `ReceiptRecord`.
- `EvaluationRecord`: final-state predicates, confidentiality/integrity labels,
  errors, latency, token usage, and evidence paths.

Models should use strict types, enums, explicit versions, bounded fields, and
`extra='forbid'`. The generated JSON schemas should be frozen with the
experiment manifest. Raw model output must be parsed into a Pydantic tool intent
before entering a trusted node.

### Proposed LangGraph nodes

```text
START
  -> load_scenario
  -> call_model
  -> parse_tool_intent
  -> classify_tool
       -> restricted_read -> call_model (loop)
       -> label_sources
          -> validate_write
          -> evaluate_policy
               -> reject_and_audit -> evaluate_final_state
               -> build_and_sign_proposal
                  -> obtain_approval
                  -> gateway_revalidate
                  -> consume_and_reserve
                  -> execute_adapter
                  -> persist_receipt
  -> evaluate_final_state
  -> END
```

Conditional edges should depend only on validated enum fields. Node functions
must return typed state updates. The model must never set proposal state,
approval state, policy decisions, trusted identity, or execution status.

LangGraph checkpoints support recovery and node-level observability. They must
not be treated as cryptographic audit evidence or replace SQLite transactions,
Ed25519 signatures, the event hash chain, or external checkpoints.

## 7. Formal evaluation metrics

All primary metrics use every scheduled trial in the denominator. Runtime/model
errors remain explicit. Report a primary intention-to-test analysis, conservative
failure bounds, and a clearly labelled error-free sensitivity analysis.

| Security objective | Metric | Definition |
|---|---|---|
| Attack resistance | Targeted ASR | Trials where the attacker's final-state goal is true / all attack trials. |
| Benign capability | Benign utility (BU) | Clean trials where the legitimate user goal is satisfied / all benign trials. |
| Robust utility | Utility under attack (UA) | Attacked trials where the legitimate user goal is still satisfied / all attack trials. |
| Overblocking | False block / false refusal rate | Legitimate trials stopped by the guard / all benign trials. |
| Confidentiality | Confidentiality violation rate | Trials where protected information reaches an unauthorized sink or response / relevant trials. Also report leaked-field count weighted by sensitivity. |
| Integrity | Integrity violation rate | Trials with unauthorized, modified, over-scoped, or approval-mismatched state change / relevant trials. |
| Availability | Availability failure rate | Timeouts, crashes, unrecoverable tool/model failures, or defense-induced non-completion / all scheduled trials. |
| Replay | Replay success and detection rate | Successful duplicate effects / replay attempts; detected replay attempts / injected replay attempts. |
| Audit integrity | Tamper detection rate | Detected modification, deletion, reordering, truncation, or replacement attacks / injected audit attacks. |
| Adaptive robustness | Best-of-k ASR | Objectives compromised at least once by attempt `k`, for `k=1,5,15` and larger preregistered budgets. |
| Cost | Added latency and tokens | Paired protected minus baseline latency/tokens; report mean, median, p95, and throughput. |

Use Wilson 95% confidence intervals for proportions, paired outcome tests for
matched baseline/protected episodes, bootstrap confidence intervals for paired
latency/token deltas, and multiple-comparison correction when testing many
families or ablations.

## 8. Layer-by-layer ablation study

| Arm | Enabled layers | Primary hypothesis |
|---|---|---|
| A0 | AgentDojo + model only | Establish baseline utility and attack exposure. |
| A1 | A0 + strict schema | Removes malformed, ambiguous, and unknown-tool execution. |
| A2 | A1 + provenance labels/digests | Improves source attribution and source-aware controls. |
| A3 | A2 + deterministic policy | Reduces unauthorized final-state actions independent of model detection. |
| A4 | A3 + signed proposal | Detects action mutation between policy decision and approval/execution. |
| A5 | A4 + grounded signed approval | Reduces misleading-description, wrong-scope, and stale-approval attacks. |
| A6 | A5 + gateway revalidation and atomic consume/reserve | Prevents replay, races, and policy drift. |
| A7 | A6 + signed event chain and external checkpoints | Detects evidence modification, deletion, truncation, and replacement. |
| A8 | Full implemented guard | Measures the combined security-utility-cost trade-off. |
| A9 | A8 + planned read/output information-flow guard | Measures confidentiality leakage through model text when no write tool is called. |

Run each arm with identical tasks, injected state, model revision, quantization,
prompt, seeds, generation settings, and final-state predicates. Freeze code,
dataset, policy, Pydantic schemas, tool inventory, and environment hashes before
the held-out run.

## 9. Compute disclosure and next requirement

The completed local experiment used an Intel i7-12650H, about 16 GB RAM, and an
RTX 3050 Laptop GPU with 4 GB VRAM. Ollama reported approximately 74% CPU and
26% GPU placement for quantized Qwen3.5 9B. This is a local prototype result,
not a dedicated-GPU benchmark.

Multi-model comparisons with larger Qwen, Llama, Mistral, and Gemma variants
require approved dedicated GPU compute. Before any such run, record the exact
GPU model/count, VRAM, driver, CUDA/runtime, serving engine, quantization,
context length, batch/concurrency, model digest, power/runtime limits, and cost.
Do not start a paid or dedicated-GPU experiment without explicit approval.

## 10. Literature-aligned evaluation sources

- AgentDojo: benign utility, utility under attack, targeted ASR, dynamic attacks.
- Agent Security Bench: attacks across prompt, observation, memory and backdoor
  stages, with security and utility metrics.
- InjecAgent: indirect injection, direct harm, and private-data exfiltration.
- AgentHarm: harmful multi-step task completion and refusal/capability coupling.
- CyberSecEval 2: prompt-injection success and false refusal rate.
- CI-Work: contextual-integrity privacy violations and leakage/utility trade-off.

The experiment protocol should cite the primary papers and state exactly which
definitions are reproduced, adapted, or newly introduced.
