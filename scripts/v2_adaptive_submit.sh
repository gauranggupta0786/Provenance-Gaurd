#!/usr/bin/env bash
# Submit the adaptive attacker for the agents (attacker qwen3-14b; reviewer as in v2_submit.sh).
#   bash scripts/v2_adaptive_submit.sh --split dev --configs "none PG-v4" --max-targets 10     # a small dev trial
#   bash scripts/v2_adaptive_submit.sh --split test --configs "none PG-v4 SCP-v3" --feedback outcome
#   bash scripts/v2_adaptive_submit.sh --split test --configs "PG-v4" --feedback whitebox glm-47-flash qwen3-14b
set -u
export PG_EXCLUDE="${PG_EXCLUDE-csis.mn1}"
cd "$(dirname "$0")/.." || exit 1
SPLIT=dev; CONFIGS_V="none PG-v4"; FEEDBACK=outcome
while [ "$#" -gt 0 ]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --configs) CONFIGS_V="$2"; shift 2 ;;
    --feedback) FEEDBACK="$2"; shift 2 ;;
    *) break ;;
  esac
done
if [ "$#" -gt 0 ]; then AGENTS=("$@"); else AGENTS=(gemma4-e4b granite4-small-h qwen3-14b lfm2-24b glm-47-flash); fi
for a in "${AGENTS[@]}"; do
  r=qwen3-14b; [ "$a" = qwen3-14b ] && r=gemma4-e4b
  sbatch ${PG_EXCLUDE:+--exclude=$PG_EXCLUDE} --job-name="v2a${SPLIT:0:1}${FEEDBACK:0:1}-${a%%-*}" \
    --export=ALL,V2A_SPLIT="$SPLIT",V2A_CONFIGS="$CONFIGS_V",V2A_FEEDBACK="$FEEDBACK" scripts/v2_adaptive_job.sh "$a" "$r" qwen3-14b
done
echo "watch:  squeue -u \$USER -o \"%.8i %.20j %.8T %.10M %R\""
