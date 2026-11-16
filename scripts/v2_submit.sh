#!/usr/bin/env bash
# Submit the v2 evaluation for the agents (one resumable job each), with the registered reviewer assignment.
#   bash scripts/v2_submit.sh --smoke                                  # 4 cases per suite, core+baselines+bridge, results/v2_smoke
#   bash scripts/v2_submit.sh --split dev --groups "core baselines bridge ablations"
#   bash scripts/v2_submit.sh --split test --groups "core baselines bridge" --variance        # after scripts/v2_register.py
#   bash scripts/v2_submit.sh --split test --groups ablations qwen3-14b glm-47-flash          # just these agents
# Reviewer: qwen3-14b reviews every agent except itself; gemma4-e4b reviews qwen3-14b (docs/PREREGISTRATION_V2.md section 1).
set -u
export PG_EXCLUDE="${PG_EXCLUDE-csis.mn1}"
cd "$(dirname "$0")/.." || exit 1
SPLIT=dev; GROUPS_V="core baselines bridge"; SMOKE=""; EXTRA=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=4; shift ;;
    --split) SPLIT="$2"; shift 2 ;;
    --groups) GROUPS_V="$2"; shift 2 ;;
    --variance) EXTRA="--variance"; shift ;;
    *) break ;;
  esac
done
if [ "$#" -gt 0 ]; then AGENTS=("$@"); else AGENTS=(gemma4-e4b granite4-small-h mistral-nemo-12b qwen3-14b lfm2-24b glm-47-flash); fi
for a in "${AGENTS[@]}"; do
  r=qwen3-14b; [ "$a" = qwen3-14b ] && r=gemma4-e4b
  tag="v2${SPLIT:0:1}"; [ -n "$SMOKE" ] && tag="v2s"
  sbatch ${PG_EXCLUDE:+--exclude=$PG_EXCLUDE} --job-name="${tag}-${a%%-*}" \
    --export=ALL,V2_SPLIT="$SPLIT",V2_GROUPS="$GROUPS_V",V2_SMOKE="$SMOKE",V2_EXTRA="$EXTRA" scripts/v2_job.sh "$a" "$r"
done
echo "watch:  squeue -u \$USER -o \"%.8i %.20j %.8T %.10M %R\""
