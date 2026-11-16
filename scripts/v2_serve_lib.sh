# Sourced by scripts/v2_job.sh and scripts/v2_adaptive_job.sh: start one Ollama server per model on the engine its
# profile was frozen with, stop them all on exit.  Needs PY and OUT set, and die() defined.
declare -a S_PIDS=() S_TAGS=() S_HOSTS=() S_BINS=()
cleanup() { for i in "${!S_PIDS[@]}"; do OLLAMA_HOST="${S_HOSTS[$i]}" "${S_BINS[$i]}" stop "${S_TAGS[$i]}" >/dev/null 2>&1 || true; kill "${S_PIDS[$i]}" 2>/dev/null || true; done; }
trap cleanup EXIT
LAST_EP=""
serve() {  # serve <slug> <label>; sets LAST_EP
  local slug="$1" label="$2" tag binrel want bin have pid host ep
  read -r tag binrel want <<<"$("$PY" -c "from scripts.week13_lib import resolve_serving;print(*resolve_serving('$slug'))")" || die "cannot resolve serving for $slug"
  bin="$HOME/$binrel"; [ -x "$bin" ] || die "$bin not found: $slug needs Ollama $want"
  have="$("$bin" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  [ "$have" = "$want" ] || die "$bin is $have but $slug was profiled on $want"
  for attempt in 1 2 3; do
    host="127.0.0.1:$((20000 + (${SLURM_JOB_ID:-0} + attempt*7919 + ${#S_PIDS[@]}*131) % 20000))"; ep="http://$host"
    OLLAMA_HOST="$host" OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m \
      OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}" OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-f16}" \
      "$bin" serve >> "$OUT/server/${label}.log" 2>&1 &
    pid=$!
    for i in $(seq 1 60); do curl -sf "$ep/api/version" >/dev/null && break; kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    if curl -sf "$ep/api/version" >/dev/null; then
      S_PIDS+=("$pid"); S_TAGS+=("$tag"); S_HOSTS+=("$host"); S_BINS+=("$bin")
      OLLAMA_HOST="$host" "$bin" list | awk 'NR>1{print $1}' | grep -Fxq "$tag" || OLLAMA_HOST="$host" "$bin" pull "$tag" || die "could not pull $tag"
      echo "=== $label: $slug ($tag) on Ollama $have, $ep"; LAST_EP="$ep"; return 0
    fi
    echo "server attempt $attempt for $label failed"; tail -n 6 "$OUT/server/${label}.log"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  done
  die "ollama server for $label did not start (see $OUT/server/${label}.log)"
}
