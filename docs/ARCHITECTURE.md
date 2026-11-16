# Architecture (Weeks 1-5)

## Week 5 model and benchmark flow

```
AgentDojo Workspace task + injected state
                    |
                    v
       Qwen3.5-9B (Ollama, seed 42)
                    |
          AgentDojo FunctionCall
                    |
          +---------+----------+
          |                    |
     baseline arm         protected arm
 stock FunctionsRuntime   ProtectedWorkspaceRuntime
 direct simulator call    read-only calls direct
                          writes -> canonical mapping
                                    |
                                    v
 ProposalService -> Policy -> ApprovalService -> ExecutionGateway
                                    |
                                    v
                     AgentDojoWorkspaceAdapter
                                    |
                                    v
                   AgentDojo simulated Workspace
          +-------------------------+----------------+
                                    v
                same final-state evaluator in both arms
```

The protected runtime classifies every one of the 24 tools exposed by
AgentDojo Workspace v1.2.2. Thirteen state-preserving reads use the stock
runtime. Eleven state-changing operations, including `get_unread_emails`
(which marks messages read), have explicit signed ProvenanceGuard contracts.
An unclassified tool fails construction, so an AgentDojo upgrade cannot
silently add a new write path.

```
 User goal ----> DeterministicAgent ----> propose_channel (closure) ----> ProposalService.propose_action
 (trusted)        (pg/agent.py)           (no store/signer attrs)          (pg/builder.py)
                                                                                   |
                                                                    normalize_arguments (pg/schemas.py)
                                                                                   |
                                                                       policy.evaluate (pg/policy.py)
                                                                                   |
                                                              sign(proposal)  (Signer, pg/crypto.py -- private key)
                                                                                   |
                                                          EventStore.append_proposal + append_event
                                                              (pg/store.py -- append-only SQLite +
                                                               separate signed checkpoint file)
                                                                                   |
                                                     store.set_state: PROPOSED -> VERIFIED
                                                                                   |
                              +--------------------------------------------------+
                              |
                              v
                    ApprovalService (pg/approval.py)
                    - render_card(): reloads + re-verifies the proposal from the
                      store (never a caller-held object); re-evaluates policy live;
                      diffs any model "claimed_fields" against the real arguments
                    - authenticate_reviewer(): local-session stand-in -> role
                    - decide()/auto_approve(): VERIFIED -> PENDING_APPROVAL ->
                      APPROVED/DENIED; issues a signed, single-use Approval
                      ("token") bound to proposal digest + session + audience +
                      policy version + nonce + expiry
                              |
                              v
                    ExecutionGateway.execute() (pg/gateway.py)
                    - reload + re-verify proposal AND approval
                    - recheck signature, decision, expiry, exact digest/session/
                      audience scope, and policy LIVE (policy drift -> REVOKED)
                    - atomically consume the approval (single-use, I4) and
                      reserve the execution slot (UNIQUE idempotency key)
                    - APPROVED -> EXECUTING -> SUCCEEDED/FAILED
                    - dispatch to exactly one adapter in pg/tools.py
                    - durable, linked Receipt written to the store
                              |
                              v
                    pg/audit.py -- pending approvals, blocked/denied, missing
                    receipts (reconciliation), integrity check, repeated-probe
                    alerts, and a full propose-to-receipt trace() for one proposal
```

`pg/tools.py` (the three simulated tool adapters + `WorldState`) is reachable
**only** from `pg/gateway.py`'s `ADAPTERS` table, and only after a proposal has
reached `APPROVED` with a valid, unconsumed, correctly-scoped approval. No
other module calls into it.

## Trust boundaries

| Boundary | What crosses it | Enforcement |
|---|---|---|
| Agent → proposal channel | `session_id`/`principal_id`/`model_id` are supplied by trusted construction code, not the agent's own choice; `tool_name`/`arguments` are the agent's (untrusted) proposal | `DeterministicAgent` never receives a `Signer`, `EventStore`, or gateway; only a closure |
| Proposal channel → schema | Raw arguments | `normalize_arguments` -- strict allow-list per tool, type/length/range checks, no extra fields |
| Schema → policy | Normalized arguments only (never raw/untrusted text) | `policy.evaluate` -- deny-by-default, explicit allow-lists |
| Policy → signer | Only proposals that passed schema + were not policy-DENY | `ProposalService` is the only caller of `Signer.sign` for proposals |
| Signer → store | Signed, digested proposal/event | `EventStore` refuses to store an unsigned proposal (`SchemaError`); proposals/events/receipts are append-only via SQLite triggers |
| Proposal → card | Only the stored, re-verified proposal + a live policy re-evaluation | `render_card()` never accepts a proposal object from the caller; any `model_explanation`/`claimed_fields` are HTML-escaped and diffed, never trusted |
| Reviewer → approval | An authenticated `(reviewer_id, session_token)` pair mapped to a role, and only for tools that role may decide on | `authenticate_reviewer` + `ROLE_PERMISSIONS`; a policy-DENY can never be turned into an approval |
| Approval → gateway | A signed `Approval` row, re-verified, checked for decision/expiry/exact proposal-digest/session/audience match | `ExecutionGateway.execute` -- any mismatch raises before the adapter runs (fail closed) |
| Gateway → tools | Only the proposal's exact, normalized arguments; only after atomic single-use consumption of the approval | `ADAPTERS` dispatch table in `pg/gateway.py`; `consume_approval` + `reserve_and_append_receipt` (both DB-atomic) |

## Tool-permission matrix (v1, `pg/policy.py` + `pg/approval.py`)

| Tool | Risk | Policy gate | Approval path |
|---|---|---|---|
| `summarize` | low | principal must be in `allowed_tools` | policy `ALLOW` -> `ApprovalService.auto_approve` (system reviewer) |
| `send_email` | high | recipient must be in `allowed_recipients` | policy `REQUIRE_APPROVAL` -> human reviewer with the `approver` role |
| `update_record` | high | `record_id` must be in `owned_record_ids`; `field` must be one of `status`/`notes`/`owner` | policy `REQUIRE_APPROVAL` -> human reviewer with the `approver` role |

## Action state machine (as implemented in `pg/store.py`)

`PROPOSED -> VERIFIED -> PENDING_APPROVAL -> APPROVED -> EXECUTING ->
SUCCEEDED | FAILED`, with `BLOCKED`, `DENIED`, `EXPIRED`, and `REVOKED` as
terminal non-executable states. Every transition is validated against the
Week 1 `check_transition` table and appended (never updated in place) to
`action_state_log`, with a signed `STATE_TRANSITION` audit event alongside
it. A policy re-check that fails at execution time moves `APPROVED ->
REVOKED` rather than silently blocking -- the state itself records *why*
the action stopped.

## What is NOT yet built (by design, per the 7-week plan)

- The comparative baseline-vs-protected (A0-A4-style) measurement, ASR/
  utility/latency reporting, and adaptive-campaign curves (Week 6).
- Multi-approver quorum, organization-backed reviewer identity, and
  session-scoped (rather than principal-scoped) freeze counters -- all
  explicitly out of v1 scope per the plan; see README "Honest status".

This is stated explicitly so results are not overclaimed. Weeks 1-5 establish
the tested structural controls and a functioning model/benchmark harness.
The four live smoke episodes validate integration only. They do not show that
the system is safe to deploy or establish a general TIP-attack success rate;
the frozen comparative and adaptive evaluation is Week 6.
