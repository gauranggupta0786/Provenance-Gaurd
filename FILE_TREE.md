# ProvenanceGuard repository: file tree (238 files)

Upload batches: batch_01 (95 files), batch_02 (95 files), batch_03 (48 files)

```
data/  (51 files)
  adaptive_evasion/  (2 files)
    manifest.json
    scenarios.jsonl
  heldout/  (2 files)
    manifest.json
    scenarios.jsonl
  models/  (1 files)
    qwen35-9b-ollama.json
  v2/  (5 files)
    adaptive_targets.json
    agentdojo_tasks.json
    injections.jsonl
    plan.json
    split.json
  week11/  (1 files)
    policy_passing.jsonl
  week11b/  (1 files)
    crafted_cards.jsonl
  week11c/  (2 files)
    heldout_cards.jsonl
    scenarios.jsonl
  week11d/  (2 files)
    provenance_validation.jsonl
    scenarios.jsonl
  week11e/  (1 files)
    scenarios.jsonl
  week5/  (4 files)
    agentdojo_workspace_inventory.json
    development.jsonl
    heldout.jsonl
    manifest.json
  week6/  (7 files)
    frozen_protocol/  (5 files)
      confidentiality_cases.jsonl
      context_profile.json
      formal_metric_spec.json
      manifest.json
      semantic_judge_protocol.json
    adaptive_manifest.json
    adaptive_objectives.jsonl
  week7/  (22 files)
    profiles_final/  (20 files)
      command-r7b.json
      command-r7b.resolution.json
      gemma4-e4b.json
      gemma4-e4b.resolution.json
      granite33-8b.json
      granite33-8b.resolution.json
      granite4-small-h.json
      granite4-small-h.resolution.json
      granite4-tiny-h.json
      granite4-tiny-h.resolution.json
      hermes3-8b.json
      hermes3-8b.resolution.json
      llama31-8b.json
      llama31-8b.resolution.json
      mistral-nemo-12b.json
      mistral-nemo-12b.resolution.json
      qwen3-14b.json
      qwen3-14b.resolution.json
      qwen35-9b.json
      qwen35-9b.resolution.json
    final_model_matrix.json
    model_candidates.json
  week9/  (1 files)
    vendor_sampling_overrides.json
docs/  (15 files)
  ARCHITECTURE.md
  ENVIRONMENT_MANIFEST.md
  FORMAL_METRIC_SPECIFICATION.md
  FRONTIER_HANDOFF.md
  FULL_PROJECT_EXPLANATION.md
  LABELLING_GUIDE.md
  PREREGISTRATION_V2.md
  PROVENANCEGUARD_RESULTS_SUMMARY.md
  RESEARCH_ARCHITECTURE_AND_EVALUATION_PLAN.md
  SCHEMA_MIGRATION_POLICY.md
  SECURITY_GUARANTEES_AND_ASSUMPTIONS.md
  STATISTICAL_ANALYSIS_PLAN.md
  THREAT_MODEL.md
  V2_RUNBOOK.md
  week4_lifecycle_graph.mmd
frontier/  (6 files)
  __init__.py
  campaign.py
  frontier_probes.py
  mock_server.py
  openai_adapter.py
  suites.py
pg/  (32 files)
  __init__.py
  ablation.py
  agent.py
  agentdojo_adapter.py
  approval.py
  audit.py
  builder.py
  card_context.py
  confidentiality.py
  content_provenance.py
  crypto.py
  diagnostics.py
  formal_evaluation.py
  gateway.py
  lifecycle_graph.py
  model_api.py
  ollama_model.py
  policy.py
  policy_ablation.py
  provenance.py
  request_scope.py
  reviewers.py
  schema_adapters.py
  schemas.py
  store.py
  tools.py
  trusted_models.py
  week5_runner.py
  week6_metrics.py
  week7_models.py
  week7_pilot.py
  week8_protocol.py
pgx/  (13 files)
  __init__.py
  adaptive.py
  agentdojo_runner.py
  attacks.py
  baselines.py
  capabilities.py
  configs.py
  content_v2.py
  guards.py
  registry.py
  runtime.py
  suites.py
  version.py
schemas/  (15 files)
  v1/  (15 files)
    AgentRunState.schema.json
    ApprovalRecord.schema.json
    EvaluationRecord.schema.json
    ExecutionRecord.schema.json
    ModelCallRecord.schema.json
    ModelProfile.schema.json
    NodeDiagnostic.schema.json
    OutputGuardRecord.schema.json
    PolicyOutcome.schema.json
    ProposalRecord.schema.json
    ReceiptRecord.schema.json
    SensitivityEvidence.schema.json
    SourceEvidence.schema.json
    ToolIntent.schema.json
    manifest.json
scripts/  (58 files)
  adaptive_evasion_analysis.py
  adaptive_evasion_triage.py
  attack_curves.py
  baseline_defense_lib.py
  baseline_defenses_analysis.py
  build_handoff_package.py
  build_transfer_set.py
  content_rescore.py
  frontier_probe.py
  frontier_report.py
  frontier_selftest.py
  generate_week11_dataset.py
  generate_week11c_dataset.py
  generate_week11d_dataset.py
  handoff_verify.py
  heldout_content_triage.py
  heldout_run_analysis.py
  heldout_utility_diagnosis.py
  label_agreement.py
  label_export.py
  provenance_rescore.py
  rq4_cost_report.py
  run_frontier_campaign.py
  run_v2_adaptive.py
  run_v2_agentdojo.py
  run_week11_policy_passing.py
  run_week11b_review_eval.py
  run_week6_adaptive.py
  run_week8_adaptive.py
  run_week8_frozen.py
  run_week9_ablation.py
  stats_v2.py
  v2_adaptive_job.sh
  v2_adaptive_submit.sh
  v2_estimate.py
  v2_job.sh
  v2_prepare.py
  v2_register.py
  v2_serve_lib.sh
  v2_submit.sh
  verify_results.py
  week10_attacks.py
  week10_lib.py
  week11_lib.py
  week11b_craft_cards.py
  week11b_lib.py
  week11c_craft_cards.py
  week11c_lib.py
  week11d_lib.py
  week11d_make_validation.py
  week11e_lib.py
  week11e_replay.py
  week12_adapter_parity.py
  week12_lib.py
  week13_lib.py
  week13_xreviewer.py
  week14_failure_analysis.py
  week9_lib.py
tests/  (41 files)
  conftest.py
  fake_ollama.py
  pgx_doubles.py
  test_adaptive_evasion_analysis.py
  test_adaptive_evasion_triage.py
  test_baseline_defenses.py
  test_frontier_suites.py
  test_heldout_content_triage.py
  test_heldout_utility_diagnosis.py
  test_pgx_e2e.py
  test_pgx_tools.py
  test_pgx_units.py
  test_rq4_cost_report.py
  test_week11.py
  test_week11b.py
  test_week11c.py
  test_week11d.py
  test_week11e.py
  test_week12.py
  test_week12_adapter.py
  test_week13_xreviewer.py
  test_week14_failure_analysis.py
  test_week14_handoff.py
  test_week1_exit_checks.py
  test_week1_runtime_diagnostics.py
  test_week2_input_validation.py
  test_week2_integrity.py
  test_week2_interception.py
  test_week2_model_api.py
  test_week2_restart_and_availability.py
  test_week3_approval.py
  test_week3_pydantic_schemas.py
  test_week4_gateway.py
  test_week4_hardening.py
  test_week4_lifecycle_graph.py
  test_week5_agentdojo_workspace.py
  test_week5_dataset.py
  test_week5_security_completion.py
  test_week6_evaluation.py
  test_week6_formal_protocol.py
  test_week8.py
.gitignore
MENTOR_QUICKSTART.md
README.md
demo.py
pytest.ini
requirements.lock
requirements.txt
```
