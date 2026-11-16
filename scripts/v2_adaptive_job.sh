#!/usr/bin/env bash
#SBATCH --partition=gpu-3day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=v2a-%j.out
#SBATCH --error=v2a-%j.err
# One job = the adaptive attacker against ONE agent (with its reviewer), for the configurations in V2A_CONFIGS.  Resumable.
#   sbatch --job-name=v2a-glm --export=ALL,V2A_SPLIT=test,V2A_CONFIGS="none PG-v4 SCP-v3",V2A_FEEDBACK=outcome \
#          scripts/v2_adaptive_job.sh glm-47-flash qwen3-14b qwen3-14b
# The attacker shares a server with the agent or the reviewer when it is the same model (keep it so: three models on one GPU
# may not fit).
set -u
AGENT="${1:-}"; REVIEWER="${2:-}"; ATTACKER="${3:-}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
die() { echo "!!! STOPPING: $*"; exit 1; }
[ -n "$AGENT" ] && [ -n "$REVIEWER" ] && [ -n "$ATTACKER" ] || die "usage: sbatch scripts/v2_adaptive_job.sh <agent> <reviewer> <attacker>"
PY=".venv/bin/python"; [ -x "$PY" ] || die ".venv/bin/python missing"
[ -f data/v2/adaptive_targets.json ] || die "data/v2/adaptive_targets.json missing: run from the repository root (see docs/V2_RUNBOOK.md, step 0)"
[ "${WEEK8_TEST_MODE:-0}" = 1 ] || nvidia-smi -L | grep -q "^GPU [0-9]" || die "no GPU visible on this node"
SPLIT="${V2A_SPLIT:-dev}"; FEEDBACK="${V2A_FEEDBACK:-outcome}"; BUDGET="${V2A_BUDGET:-10}"
read -r -a CONFIGS <<<"${V2A_CONFIGS:-none PG-v4}"
OUT="results/v2_adaptive/${AGENT}__$([ "$AGENT" = "$REVIEWER" ] && echo self || echo "$REVIEWER")__atk-${ATTACKER}"; mkdir -p "$OUT/server"
"$PY" -m pytest tests -q > "$OUT/tests_at_run_$(date +%Y%m%dT%H%M%S).txt" 2>&1 || die "tests failed on this node"
source scripts/v2_serve_lib.sh
serve "$AGENT" agent; AGENT_EP="$LAST_EP"
if [ "$REVIEWER" = "$AGENT" ]; then REVIEWER_EP="$AGENT_EP"; else serve "$REVIEWER" reviewer; REVIEWER_EP="$LAST_EP"; fi
if [ "$ATTACKER" = "$AGENT" ]; then ATTACKER_EP="$AGENT_EP"; elif [ "$ATTACKER" = "$REVIEWER" ]; then ATTACKER_EP="$REVIEWER_EP";
else serve "$ATTACKER" attacker; ATTACKER_EP="$LAST_EP"; fi
CFG_ARGS=(); for c in "${CONFIGS[@]}"; do CFG_ARGS+=(--config "$c"); done
"$PY" scripts/run_v2_adaptive.py --agent "$AGENT" --agent-endpoint "$AGENT_EP" --reviewer "$REVIEWER" --reviewer-endpoint "$REVIEWER_EP" \
  --attacker "$ATTACKER" --attacker-endpoint "$ATTACKER_EP" "${CFG_ARGS[@]}" --budget "$BUDGET" --feedback "$FEEDBACK" \
  --split "$SPLIT" || die "adaptive run stopped; resubmit to resume"
echo "=== V2 ADAPTIVE $AGENT ($SPLIT, $FEEDBACK: ${CONFIGS[*]}) DONE ($OUT) ==="
