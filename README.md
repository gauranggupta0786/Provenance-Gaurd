# ProvenanceGuard

A guard for LLM agents that take actions. The agent can only *propose* a write (send an e-mail, update a record, make a payment,
post a message, book a trip). The proposal is checked against what the user actually asked for, signed, approved or denied on a
card rendered from the stored proposal (never from the model's own words), and only then executed by a gateway that nothing else
can bypass. Every step is written to a tamper-evident log.

This repository is the current implementation and the scripts that evaluate it. It holds only code that the evaluation pipeline
imports or runs, plus the frozen inputs those scripts read. Earlier week-by-week analysis scripts, slide decks and the paper
source are not included.

## How the code is layered

Read this top to bottom; each layer uses the ones above it.

| layer | where | what it does |
|---|---|---|
| typed records | `pg/schemas.py`, `pg/trusted_models.py`, `pg/schema_adapters.py` | strict Pydantic records for proposals, approvals, receipts, audit events; canonical bytes for signing |
| evidence | `pg/crypto.py`, `pg/store.py`, `pg/provenance.py` | Ed25519 signatures, append-only SQLite log with hash chain and signed checkpoints, trusted source labels |
| proposal | `pg/builder.py`, `pg/policy.py`, `pg/tools.py` | the only way an agent turns a request into a proposal; allow-list and argument limits; the simulated tools |
| approval | `pg/approval.py`, `pg/card_context.py`, `pg/reviewers.py` | canonical approval card, scoped single-use approval tokens, model or human reviewer |
| execution | `pg/gateway.py`, `pg/lifecycle_graph.py` | the sole write-tool caller; re-verifies proposal and approval, consumes the approval once, writes the receipt (LangGraph lifecycle) |
| audit | `pg/audit.py` | pending approvals, denials, missing receipts, integrity check |
| content checks (v1) | `pg/request_scope.py`, `pg/content_provenance.py`, `pg/confidentiality.py` | does the e-mail go where the request said; does its body carry text the user never wrote; which reads may feed which output |
| benchmark glue (v1) | `pg/agentdojo_adapter.py`, `pg/week5_runner.py`, `pg/model_api.py`, `pg/week7_models.py`, `pg/ollama_model.py`, `pg/diagnostics.py` | AgentDojo Workspace wrapper, model clients and explicit model profiles, failure taxonomy |
| protocol and metrics | `pg/week8_protocol.py`, `pg/ablation.py`, `pg/formal_evaluation.py`, `pg/week6_metrics.py` | the frozen protocol and code digest, per-layer switches, metrics |
| **v2 extension** | `pgx/` | all four AgentDojo suites (workspace, banking, slack, travel) and the PG-v4 chain below |
| frontier tooling | `frontier/` | runs the frozen ladder against any OpenAI-compatible model; mock server for testing without a model |

`pgx/` is additive: it registers the new suites' tools into v1's tables at runtime (`pgx/registry.py`) and never edits `pg/`.

PG-v4 approval chain (`pgx/guards.py`), in order, each stage switchable for ablations:

1. **capability**: what the request allows, derived by a planner model that sees only the request (or a keyword parser)
2. **destination rule**: recipients, accounts, channels and URLs must come from the request
3. **attachment rule**
4. **content rule**: words and numbers in the write must come from the request; numbers are compared by value
5. **deterministic approve** if every field is accounted for
6. **independent model reviewer** (a different model family) only for what the rules cannot decide
7. **denial feedback**: the agent is told the rule category, never the rule

## Evaluation scripts

Everything below runs from the repository root. The step-by-step order, with exact commands, is `docs/V2_RUNBOOK.md`.

| purpose | script |
|---|---|
| freeze the v2 inputs (task metadata, attack texts, dev/test split) | `scripts/v2_prepare.py` |
| lock the preregistration and code before any test-split run | `scripts/v2_register.py` |
| run one agent on the four AgentDojo suites under any configuration | `scripts/run_v2_agentdojo.py` |
| adaptive attacker, up to 10 attempts per target | `scripts/run_v2_adaptive.py` |
| cluster jobs (Slurm, one Ollama server per model) | `scripts/v2_submit.sh`, `v2_job.sh`, `v2_adaptive_submit.sh`, `v2_adaptive_job.sh`, `v2_serve_lib.sh` |
| GPU-hour estimate from a smoke run | `scripts/v2_estimate.py` |
| statistics: clustered bootstrap, paired exact McNemar, Holm | `scripts/stats_v2.py` |
| attack curves; the transfer set for frontier models | `scripts/attack_curves.py`, `scripts/build_transfer_set.py` |
| two-annotator labels, agreement, re-scoring, failure analysis | `scripts/label_export.py`, `label_agreement.py`, `content_rescore.py`, `provenance_rescore.py`, `week14_failure_analysis.py` |
| independent recomputation of the headline numbers from raw logs | `scripts/verify_results.py` |
| frontier campaign (hosted models; every experiment in the report is a `--suite`) and its checks | `scripts/run_frontier_campaign.py`, `frontier_probe.py`, `frontier_report.py`, `frontier_selftest.py`, `week12_adapter_parity.py` |
| analysis of any suite's results (local or hosted rows) | `scripts/heldout_run_analysis.py`, `heldout_utility_diagnosis.py`, `heldout_content_triage.py`, `baseline_defenses_analysis.py`, `adaptive_evasion_analysis.py`, `adaptive_evasion_triage.py`, `rq4_cost_report.py` |
| package check and builder for the mentor hand-off | `scripts/handoff_verify.py`, `scripts/build_handoff_package.py` |

## Frozen v1 identity: do not edit these files

The v1 defense that the Week 8 to 13 results measured is identified by two digests. Both are recomputed from file bytes, so
editing any hashed file changes them and the runners refuse to start:

| digest | value | covers |
|---|---|---|
| v1 guard (`pg.week8_protocol.code_digest`) | `dca1a798...` | every `pg/*.py`, `requirements.lock`, three frozen manifests, `scripts/run_week8_frozen.py`, `run_week8_adaptive.py`, `run_week6_adaptive.py` |
| Week 11 ladder (`scripts.week11_lib.week11_code_digest`) | `22a4622b...` | the above plus the 14 files in `WEEK11_CODE_FILES` |

That is why some `scripts/week*` files are still here although nothing runs them now: they are part of what the digest
hashes. The v2 code has its own digest (`pgx.version.pgx_digest`), recorded by `scripts/v2_register.py`.

```bash
python -c "from pg.week8_protocol import code_digest; print(code_digest()[0][:12])"          # dca1a7986949
python -c "from scripts.week11_lib import week11_code_digest as d; print(d()[0][:12])"      # 22a4622b3861
```

## Install and test

Python 3.11, 3.12 or 3.13 (tested on Linux with all three).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
python -m pytest tests -q          # no model, no GPU, no network
python demo.py                     # the propose -> approve -> execute trace, with forged and replayed approvals refused
```

## What is not in this repository

- `results/` (episode logs, hundreds of MB) and `labels/`.
- Frozen manifests that were generated on the cluster in the earlier weeks: `data/week8/protocol_manifest.json`, `data/week12/*`,
  `data/week7/gpu_approval.json`, every `data/week*/plan.json`, `data/heldout/run_plan*.json`, `data/week9/ablation_plan.json`,
  and the `agent_cards.jsonl` / `eval_plan.json` files of weeks 11b and 11c. Local-model runs read them to verify each model's
  identity; `docs/V2_RUNBOOK.md` step 0 copies them from an existing checkout. Tests and the frontier path do not need them.
- Any API key. The frontier runner reads its endpoint, key and model name from environment variables (`MENTOR_QUICKSTART.md`).

## Documents

`docs/V2_RUNBOOK.md` (commands) · `docs/PREREGISTRATION_V2.md` · `docs/LABELLING_GUIDE.md` · `docs/ARCHITECTURE.md` ·
`docs/THREAT_MODEL.md` · `docs/SECURITY_GUARANTEES_AND_ASSUMPTIONS.md` · `docs/FORMAL_METRIC_SPECIFICATION.md` ·
`docs/STATISTICAL_ANALYSIS_PLAN.md` · `docs/FRONTIER_HANDOFF.md` · `docs/PROVENANCEGUARD_RESULTS_SUMMARY.md` (v1 results) ·
`docs/FULL_PROJECT_EXPLANATION.md` (plain-language walkthrough)

## Origin

Pruned from the development history at commit `7c799c3` (v1 frozen at tag `v1.0-week14`, commit `c38e950`). The pruning removed
files only, plus a few unreferenced helpers in scripts that are not hashed into either digest; nothing in `pg/` was edited.
