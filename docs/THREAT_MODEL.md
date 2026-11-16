# Threat model (Week 1 deliverable, extended in Weeks 3-4)

Scope: the v1 workflow only -- read a simulated message, `summarize` it,
and optionally `send_email` or `update_record`. All writes require
approval (Week 3+); reads/outputs have their own explicit policy too
(`pg/policy.py`).

## Actors

| Actor | Controls | Cannot do (assumption/exclusion) |
|---|---|---|
| Hostile user | The `user_goal` text and request timing | Cannot forge signatures; has no signing key |
| Hostile content (email body, etc.) | Text the agent reads (`untrusted_context`) | Cannot become trusted identity; carried only as a source digest, never as session/principal |
| Compromised/adversarial model | The proposed `tool_name`/`arguments` before validation | Cannot bypass `normalize_arguments`/policy; has no store or signing access |
| Repeated prober | Query volume / retries | Rate limiting is **out of scope for Weeks 1-2** (scheduled Week 4 budgets/counters) |
| Insider with raw DB file access | Could attempt direct SQL/file mutation | Detected by hash chain + checkpoint (not prevented at the OS level -- key/host custody is excluded, see below) |

**Excluded from Weeks 1-2 (explicitly, not silently):** key theft, a fully
compromised host, and colluding approvers. These are named exclusions
per the plan, not gaps we pretend don't exist.

## Threat cases and acceptance matrix

| ID | Threat | Control (this codebase) | Observable outcome | Status |
|---|---|---|---|---|
| TC-01 | Unknown/forbidden tool name proposed | `normalize_arguments` schema check | `SchemaError` / `ProposalRejected` | ✅ tested |
| TC-02 | Missing required argument | `normalize_arguments` | `ProposalRejected` (schema) | ✅ tested |
| TC-03 | Extra/unexpected argument, incl. model-supplied `trusted=true` | `normalize_arguments` extra-field rejection | `ProposalRejected` (schema); trusted flag never read | ✅ tested |
| TC-04 | Oversized string argument | length cap per field | `ProposalRejected` (schema) | ✅ tested |
| TC-05 | Out-of-range integer argument | range check | `ProposalRejected` (schema) | ✅ tested |
| TC-06 | Duplicate JSON keys at the input boundary | `strict_json_loads` | `SchemaError` | ✅ tested |
| TC-07 | Unauthorized recipient (`send_email`) | policy allow-list | `ProposalRejected` (policy) | ✅ tested |
| TC-08 | Unauthorized/unowned record (`update_record`) | policy ownership check | `ProposalRejected` (policy) | ✅ tested |
| TC-09 | Unknown principal | policy deny-by-default | `ProposalRejected` (policy) | ✅ tested |
| TC-10 | Direct SQL `UPDATE`/`DELETE` on `proposals`/`events` | append-only DB triggers | `sqlite3.IntegrityError` | ✅ tested |
| TC-11 | Event content mutated after a trigger bypass (simulated raw DB access) | hash-chained event digests | `IntegrityError` on `verify_chain` | ✅ tested |
| TC-12 | Middle event deleted | sequence-gap detection | `IntegrityError` on `verify_chain` | ✅ tested |
| TC-13 | Event reordered / hash link altered | `prev_hash` re-derivation check | `IntegrityError` on `verify_chain` | ✅ tested |
| TC-14 | Forged event signature | Ed25519 verification against registered key | `IntegrityError` on `verify_chain` | ✅ tested |
| TC-15 | Log-tail deletion below a checkpoint | separate signed checkpoint file, seq/hash compared to live chain | `IntegrityError` on `verify_against_checkpoint` | ✅ tested |
| TC-16 | Whole-log replacement with a different, internally-consistent chain | checkpoint head-hash mismatch at the checkpointed seq | `IntegrityError` on `verify_against_checkpoint` | ✅ tested |
| TC-17 | Event store unavailable/corrupt | fail-closed: no catch-and-continue around store calls | Exception propagates; `propose_action` raises, no proposal is produced | ✅ tested |
| TC-18 | Concurrent proposal submissions | `BEGIN IMMEDIATE` + busy-timeout serialization | Gap-free, duplicate-free sequence under 8 concurrent writers | ✅ tested |
| TC-19 | Agent attempts to reach the write adapter or signing key directly | no import path from `pg.agent` to `pg.tools`/`Signer`; agent only holds a closure | Structural/AST negative-access test | ✅ tested (Week 4/5 isolation is the hardened version) |
| TC-20 | TIP-style encoded objective smuggled through untrusted context | AgentDojo Workspace plus 10 TIP forms at 3 guidance levels; final structured action still crosses schema/policy/approval/gateway | Week-5 fixtures and runner complete; full comparative measurement is Week 6 | harness complete, empirical result pending |

TC-20 is listed to keep the plan's own honesty requirement visible: this
prototype does not yet claim general TIP resistance -- Week 5 provides the
frozen harness and smoke evidence; the full empirical evidence is Week 6
work. What Weeks 1-2 *do* establish is that no matter what text produced
a tool call, the call still has to pass the same structural and policy
gates as a benign one.

## Threat cases added in Weeks 3-4

| ID | Threat | Control (this codebase) | Observable outcome | Status |
|---|---|---|---|---|
| TC-21 | Model writes a misleading approval summary/explanation | `render_card` reads only the verified, stored proposal; `model_explanation`/`claimed_fields` are shown separately, escaped, and diffed against the real arguments | Card shows the true target/fields; `claim_mismatches` non-empty | ✅ tested |
| TC-22 | Markup/badge injection via model-supplied text | `escape_display` (HTML-escape) before any untrusted text is rendered | Escaped, inert text; no raw tags in the rendered card | ✅ tested |
| TC-23 | Unauthenticated or wrong-session reviewer | `authenticate_reviewer` against `REVIEWER_SESSIONS` | `ReviewerAuthError` | ✅ tested |
| TC-24 | Reviewer decides on a tool outside their role's permission | `ROLE_PERMISSIONS` check before any decision is issued | `ReviewerNotPermittedError` | ✅ tested |
| TC-25 | Approval requested/attempted for a policy-DENY action | `ApprovalService.decide` checks the live policy result before issuing | `ApprovalError`; DENY can never become an approval | ✅ tested |
| TC-26 | Forged approval signature (tampered DB row) | Ed25519 verification of the approval payload in `ExecutionGateway.execute` | `ForgedTokenError` | ✅ tested |
| TC-27 | Sensitive value returned by an authorized read is copied to final model text | field/object sensitivity evidence plus deterministic output guard | final output blocked; digest-only guard record retained | ✅ tested |
| TC-28 | Read-only prompt injection requests a write | read path has no adapter capability; any later write is independently mediated | write must cross proposal/approval/gateway | ✅ tested by architecture and runtime tests |
| TC-29 | Source changes after approval | source version revalidated immediately before execution | `SourceDriftError`; approval remains unconsumed | ✅ tested |
| TC-30 | Middle/tail checkpoint deletion | signed checkpoint chain plus signed latest anchor | `IntegrityError` | ✅ tested |
| TC-31 | Development process restarts or rotates keys | persistent Ed25519 development key store and retained public-key metadata | old signatures remain verifiable; new active key recorded | ✅ tested |
| TC-27 | Expired approval presented at execution time | `expires_at` checked against wall clock (controllable in tests) | `TokenExpiredError` | ✅ tested |
| TC-28 | Approval replayed after being consumed | `consume_approval`'s atomic 0->1 guard | `ReplayRejected` | ✅ tested, including under concurrent threads |
| TC-29 | Approval for proposal A used to execute proposal B (argument-swap / stale-approval reuse) | exact `proposal_digest` match check | `TokenScopeError` | ✅ tested |
| TC-30 | Policy tightened between approval and execution ("policy drift") | policy re-evaluated live inside the gateway, not trusted from approval time | `PolicyDriftError`; state moves `APPROVED -> REVOKED` | ✅ tested |
| TC-31 | Repeated rejected proposals from one principal (probing) | per-principal, trusted-identity-keyed rejection counter (`budgets` table), independent of request wording | `FrozenPrincipalError` once the threshold is crossed | ✅ tested |
| TC-32 | Process interrupted between "reserved" and "receipt written" | `find_uncertain_executions` flags proposals stuck in `EXECUTING` with no receipt; the consumed approval blocks any silent retry | Surfaced for manual reconciliation, never auto-retried | ✅ tested |

## Out of scope for the primary claim (Weeks 1-4)

Full host compromise, HSM/KMS key theft, a malicious external tool that
lies about its own execution, and collusion of the full approval quorum.
These match the plan's own "Outside the primary guarantee" list. Also
still out of scope, and scheduled for later weeks: a general TIP/adaptive
attacker (Week 5-6) and multi-approver quorum (post-v1 backlog).
