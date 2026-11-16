#!/usr/bin/env bash
#SBATCH --partition=gpu-3day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=v2-%j.out
#SBATCH --error=v2-%j.err
# One job = ONE agent model with ONE reviewer model, the v2 AgentDojo evaluation (four suites).  Resumable: submit again.
#   sbatch --job-name=v2s-qwen3 --export=ALL,V2_SMOKE=4 scripts/v2_job.sh qwen3-14b gemma4-e4b
#   sbatch --job-name=v2d-glm --export=ALL,V2_SPLIT=dev,V2_GROUPS="core baselines bridge ablations" scripts/v2_job.sh glm-47-flash qwen3-14b
#   sbatch --job-name=v2t-glm --export=ALL,V2_SPLIT=test,V2_GROUPS="core baselines bridge",V2_EXTRA=--variance scripts/v2_job.sh glm-47-flash qwen3-14b
# Agent and reviewer are served as SEPARATE Ollama servers, each on the engine its profile was frozen with (glm-4.7-flash on
# 0.40.12, the others 0.40.2).  If agent == reviewer the server is shared (self-review).  The planner and tool filter use the
# agent's server; the reviewer and the trace auditor use the reviewer's.
set -u
AGENT="${1:-${AGENT:-}}"; REVIEWER="${2:-${REVIEWER:-}}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
die() { echo "!!! STOPPING: $*"; exit 1; }
[ -n "$AGENT" ] && [ -n "$REVIEWER" ] || die "usage: sbatch scripts/v2_job.sh <agent-slug> <reviewer-slug>"
if [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue >/dev/null 2>&1; then
  OTHER="$(squeue -u "$USER" -h -n "${SLURM_JOB_NAME:-none}" -t RUNNING,PENDING -o %i 2>/dev/null | awk -v me="$SLURM_JOB_ID" '$1 != me && $1+0 < me+0')"
  [ -z "$OTHER" ] || die "another job named ${SLURM_JOB_NAME} is already queued/running ($(echo $OTHER|tr '\n' ' ')); refusing to double-write"
fi
PY=".venv/bin/python"; [ -x "$PY" ] || die ".venv/bin/python missing"
[ -f data/v2/plan.json ] || die "data/v2/plan.json missing: run from the repository root (see docs/V2_RUNBOOK.md, step 0)"
[ "${WEEK8_TEST_MODE:-0}" = 1 ] || nvidia-smi -L | grep -q "^GPU [0-9]" || die "no GPU visible on this node"
SPLIT="${V2_SPLIT:-dev}"; read -r -a GROUPS_ARR <<<"${V2_GROUPS:-core baselines bridge}"
ROOT_OUT="results/v2"; EXTRA=()
[ -n "${V2_SMOKE:-}" ] && { ROOT_OUT="results/v2_smoke"; EXTRA+=(--smoke "$V2_SMOKE"); }
[ -n "${V2_EXTRA:-}" ] && read -r -a MORE <<<"$V2_EXTRA" && EXTRA+=("${MORE[@]}")
GROUP_ARGS=(); for g in "${GROUPS_ARR[@]}"; do GROUP_ARGS+=(--group "$g"); done
OUT="$ROOT_OUT/${AGENT}__$([ "$AGENT" = "$REVIEWER" ] && echo self || echo "$REVIEWER")"; mkdir -p "$OUT/server"

"$PY" scripts/v2_prepare.py --check >/dev/null || die "data/v2 does not match the installed AgentDojo (run scripts/v2_prepare.py --check)"
"$PY" -m pytest tests -q > "$OUT/tests_at_run_$(date +%Y%m%dT%H%M%S).txt" 2>&1 || { tail -n 5 "$OUT"/tests_at_run_*.txt; die "tests failed on this node"; }

source scripts/v2_serve_lib.sh
if [ "${WEEK8_TEST_MODE:-0}" = 1 ]; then
  AGENT_EP="${AGENT_ENDPOINT:?}"; REVIEWER_EP="${REVIEWER_ENDPOINT:?}"
else
  serve "$AGENT" agent; AGENT_EP="$LAST_EP"
  if [ "$AGENT" = "$REVIEWER" ]; then REVIEWER_EP="$AGENT_EP"; else serve "$REVIEWER" reviewer; REVIEWER_EP="$LAST_EP"; fi
fi
ARGS=(--agent "$AGENT" --agent-endpoint "$AGENT_EP" --reviewer "$REVIEWER" --reviewer-endpoint "$REVIEWER_EP"
      --split "$SPLIT" "${GROUP_ARGS[@]}" --results-root "$ROOT_OUT")
"$PY" scripts/run_v2_agentdojo.py "${ARGS[@]}" --check-only || die "identity/registration check failed"
"$PY" scripts/run_v2_agentdojo.py "${ARGS[@]}" "${EXTRA[@]}"; rc=$?
[ "$rc" = 0 ] || die "v2 run stopped (exit $rc); resubmit the same command to resume"
echo "=== V2 $AGENT / $REVIEWER ($SPLIT: ${GROUPS_ARR[*]}) DONE ($OUT) ==="
