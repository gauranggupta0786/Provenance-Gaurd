# ProvenanceGuard v2: exact commands, from cloning the repository to the runs the mentor package needs

Everything before step 3 runs without a GPU. Each GPU step is one `bash scripts/...submit.sh` line that submits one resumable
Slurm job per model. If a job stops (time limit, preemption, exit 4), submit the same line again: finished episodes are skipped.

Every command below runs from the repository root (step 0 sets `PG` to it; set it again in each new shell with
`export PG=~/ProvenanceGuard-repo; cd "$PG"`).

## 0. Get the code from GitHub (CPU, about 5 minutes)

The repository holds the code and the frozen inputs that git tracks. It does **not** hold `results/`, the labels, or the frozen
manifests that were generated on the cluster during the earlier weeks (`data/week8/protocol_manifest.json`,
`data/week12/*`, `data/week7/gpu_approval.json`, every `data/week*/plan.json`, `data/heldout/run_plan*.json`,
`data/week9/ablation_plan.json`, `data/week11b/agent_cards.jsonl`, `data/week11c/agent_cards.jsonl`, `data/week11b/eval_plan.json`,
`data/week11c/eval_plan.json`). The v2 runners read them to verify each model's identity, so copy them from your existing
checkout. Set `OLD` to that checkout and `NEW` to where the repository goes:

```bash
export OLD=~/ProvenanceGuard            # your existing checkout (has .venv/, results/, data/week8, data/week12)
export NEW=~/ProvenanceGuard-repo
git clone https://github.com/<owner>/<repo>.git "$NEW"      # a private repo needs a token or an SSH key on the cluster
cd "$NEW"
rsync -a --ignore-existing "$OLD/data/" data/               # copies only the files the repository does not have
ln -s "$OLD/results" results                                # keep the finished episodes where they are
cp -a "$OLD/.venv" .venv || { python3 -m venv .venv && .venv/bin/pip install -r requirements.lock; }
export PG="$NEW"
```

Check that nothing the code reads is missing and that the frozen identities are unchanged:

```bash
.venv/bin/python scripts/handoff_verify.py | tail -1                                  # must print: READY
.venv/bin/python scripts/v2_prepare.py --check                                        # must print: v2 inputs: UNCHANGED
.venv/bin/python -c "from pg.week8_protocol import code_digest; print(code_digest()[0][:12])"                 # dca1a7986949
.venv/bin/python -c "from scripts.week11_lib import week11_code_digest as d; print(d()[0][:12])"             # 22a4622b3861
.venv/bin/python -m pytest tests -q                                                   # must end with: passed (no failures)
```

`git status` should now show the copied manifests as untracked. Leave them untracked, or `git add -f` them if you want
the repository to be self-contained. If `v2_prepare.py --check` says "WOULD CHANGE", the cluster has a different AgentDojo than
0.1.35. Stop and send me the output.

## 1. Phase 1 on existing results: labels and statistics (CPU, no new runs)

**1a. Export the bodies for two annotators.** The configuration and model are hidden in the export:

```bash
mkdir -p labels
for who in A B; do
  .venv/bin/python scripts/label_export.py --results-root results/heldout --scenarios data/heldout/scenarios.jsonl \
      --out labels/heldout_$who.csv --annotator $who
  .venv/bin/python scripts/label_export.py --results-root results/adaptive_evasion --scenarios data/adaptive_evasion/scenarios.jsonl \
      --out labels/adaptive_$who.csv --annotator $who
done
```

**1b. Label.** Two people each fill `verdict` (`leak` or `clean`) in their own file, following `docs/LABELLING_GUIDE.md`, without
talking to each other.

**1c. Measure agreement, adjudicate, and write the label tables:**

```bash
.venv/bin/python scripts/label_agreement.py labels/heldout_A.csv labels/heldout_B.csv --disagreements labels/heldout_dis.csv
#   a third person fills 'verdict' in labels/heldout_dis.csv, then:
.venv/bin/python scripts/label_agreement.py labels/heldout_A.csv labels/heldout_B.csv \
    --adjudicated labels/heldout_dis.csv --write-table labels/heldout_labels.json
#   the same three steps with adaptive_ instead of heldout_
```

**1d. Re-score and analyse with the human labels:**

```bash
.venv/bin/python scripts/content_rescore.py --results-root results/heldout --labels labels/heldout_labels.json --json labels/heldout_rescore.json
.venv/bin/python scripts/week14_failure_analysis.py --results-root results/heldout --labels labels/heldout_labels.json --csv labels/heldout_residuals.csv
.venv/bin/python scripts/stats_v2.py --results-root results/heldout --compare C0 SC-v3 --compare SC-v3 SCP-v3 --compare C0 SCP-v3 \
    --capable 0.3 --json labels/heldout_stats.json
```

Send me the printed output of these commands, plus the three JSON files and the CSV.

## 2. Freeze-check before using the GPU (CPU)

```bash
.venv/bin/python scripts/run_v2_adaptive.py --make-targets 2>&1 | tail -1   # must REFUSE: the targets are already frozen
.venv/bin/python -c "from pgx.version import pgx_digest; print('v2 code', pgx_digest()[0][:12])"
```

## 3. GPU smoke: 4 cases per suite, every agent (about 1 GPU-hour in total)

```bash
bash scripts/v2_submit.sh --smoke
squeue -u $USER -o "%.8i %.20j %.8T %.10M %R"
#   when all v2s-* jobs are gone:
grep -h "DONE\|STOPPING" v2-*.out
.venv/bin/python scripts/stats_v2.py --results-root results/v2_smoke
.venv/bin/python scripts/v2_estimate.py --results-root results/v2_smoke --split dev --groups core baselines bridge ablations
.venv/bin/python scripts/v2_estimate.py --results-root results/v2_smoke --split test --groups core baselines bridge --variance
```

**Pass condition:** every job printed `DONE`, the `err` column is 0 or close to it, and nothing shows `harness`. Send me the
output of the last three commands before step 4: they size the rest of the plan.

## 4. GPU development runs: dev split only (design may look at these)

```bash
bash scripts/v2_submit.sh --split dev --groups "core baselines bridge ablations"
bash scripts/v2_adaptive_submit.sh --split dev --configs "none PG-v4" --feedback outcome
#   when done:
.venv/bin/python scripts/stats_v2.py --results-root results/v2 --compare none PG-v4 --compare C0g PG-v4 \
    --compare PG-v4-self PG-v4 --compare PG-v4-noNum PG-v4 --compare PG-v4-noFeedback PG-v4 --json results/v2_dev_stats.json
.venv/bin/python scripts/attack_curves.py --results-root results/v2_adaptive
```

Send me both outputs. Any change to the design happens here, on dev data only, and is followed by re-running step 0's tests.

## 5. Register (CPU, before any test-split episode)

Decide every `[CONFIRM ...]` line in `docs/PREREGISTRATION_V2.md` (with the mentor if possible) and delete the markers. Then:

```bash
git add -A && git commit -m "v2 preregistration"           # if the cluster repo uses git; otherwise keep a dated copy
.venv/bin/python scripts/v2_register.py                     # writes data/v2/registration.json; refuses while markers remain
.venv/bin/python scripts/v2_register.py --show
```

From now on, any change to `pg/`, `pgx/`, `data/v2/` or the v2 runners makes the test-split runner refuse to run. That is
intended.

## 6. GPU test runs: the reported numbers

```bash
bash scripts/v2_submit.sh --split test --groups "core baselines bridge" --variance
bash scripts/v2_submit.sh --split test --groups ablations <capable-agent-1> <capable-agent-2> <capable-agent-3>
bash scripts/v2_adaptive_submit.sh --split test --configs "none PG-v4 SCP-v3" --feedback outcome
bash scripts/v2_adaptive_submit.sh --split test --configs "PG-v4" --feedback whitebox <capable-agent-1> <capable-agent-2>
```

Choose the capable agents from step 4: those at or above the registered utility gate under `none`.

## 7. Analysis and the transfer set (CPU)

```bash
.venv/bin/python scripts/stats_v2.py --results-root results/v2 --capable 0.3 --compare none PG-v4 --compare spotlighting PG-v4 \
    --compare repeat_user_prompt PG-v4 --compare tool_filter PG-v4 --compare trace_auditor PG-v4 --compare C0g PG-v4 \
    --compare PG-v4-self PG-v4 --compare PG-v4-capParse PG-v4 --compare PG-v4-noBypass PG-v4 --compare PG-v4-noNum PG-v4 \
    --compare PG-v4-noFeedback PG-v4 --json results/v2_test_stats.json
.venv/bin/python scripts/attack_curves.py --results-root results/v2_adaptive --json results/v2_attack_curves.json
.venv/bin/python scripts/build_transfer_set.py --results-root results/v2_adaptive --out data/v2/adaptive_transfer.jsonl
```

Send me the two JSON files, `data/v2/adaptive_transfer.jsonl`, and the printed output. With those I build the mentor package v2:
the frontier runner gains the `agentdojo` and `transfer` datasets, cost estimates come from your measured token counts, and the
same three-Python clean-extract test is run.

## Reference

| what | where |
|---|---|
| configurations | `pgx/configs.py` (`--group core` = none, C0g, PG-v4; `baselines`; `bridge` = SCP-v3 on workspace; `ablations`) |
| results | `results/v2[_smoke]/<agent>__<reviewer>/<config>/episodes.jsonl`, `results/v2_adaptive/...` |
| exit codes | 0 done; 2 refused (identity, registration, or a results folder from another run); 4 too many consecutive errors (fix the cause, resubmit); 143 terminated (resubmit) |
| reviewer assignment | qwen3-14b reviews every agent except itself; gemma4-e4b reviews qwen3-14b |
