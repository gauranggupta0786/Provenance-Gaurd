# Statistical analysis plan

1. Report exact numerators and scheduled denominators by arm, family, guidance, model, and seed.
2. Report Wilson 95% intervals for every proportion; do not use normal/Wald intervals.
3. Pair baseline and protected episodes by scenario ID and seed. Report exact discordant-pair tests for binary outcomes.
4. Bootstrap paired latency, token, storage, and approval differences with 10,000 resamples and seed `20261009`; report mean difference and percentile 95% interval.
5. Keep failures in primary denominators. Report error-free results only as a sensitivity analysis with excluded counts.
6. Do not infer a security success from prose refusal. Use final-state checks, signed records, gateway outcomes, replay evidence, checkpoints, and output-guard records.
7. Freeze dataset, prompt, tool-schema, policy, evaluator, model-profile, schema-bundle, context, and expected-outcome hashes before the campaign.
8. Label exploratory subgroup and adaptive analyses; do not substitute them for frozen primary comparisons.
9. Apply multiplicity correction when making families of confirmatory cross-model/subgroup claims; retain unadjusted estimates and intervals for transparency.
