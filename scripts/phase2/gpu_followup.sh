#!/usr/bin/env bash
# Phase 2 follow-up on the GPU (no AI agent): clean speed numbers and tool calling without thinking.
#
#   tmux new -d -s followup 'bash scripts/phase2/gpu_followup.sh'
#   tail -f ~/.cache/hybrid-agents/gpu-followup.log
#
# Set Windows power mode to Balanced and leave the machine alone while it runs (about 2 hours).
# Safe to re-run: finished parts are skipped.
set -uo pipefail
REPO="${REPO:-$HOME/projects/hybrid-ai-system}"
BIN="${LLAMA_BIN_DIR:-$HOME/src/llama.cpp/build/bin}"
MODELS="$HOME/models"; OUT="research/results/phase2"; PORT="${PORT:-8089}"
STATUS="$OUT/followup-status.md"
export PATH="$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH" LLAMA_BIN_DIR="$BIN"
mkdir -p "$HOME/.cache/hybrid-agents"; exec >>"$HOME/.cache/hybrid-agents/gpu-followup.log" 2>&1
cd "$REPO" || exit 1
note() { printf -- '- %s  **%s**  %s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> "$STATUS"; echo ">>> $1: $2"; }
publish() {
  local b; b=$(git rev-parse --abbrev-ref HEAD); [ "$b" = main ] && return 1
  for _ in 1 2 3 4 5; do
    git add "$OUT" && { git diff --cached --quiet || git commit -q -m "gpu-followup: $1"; } \
      && git pull -q --rebase origin "$b" && git push -q origin "$b" && return 0
    git rebase --abort 2>/dev/null; sleep 20
  done; echo "PUSH FAILED"
}
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }
cool() { local n=0; while [ "$(temp)" -gt 62 ] && [ $n -lt 90 ]; do sleep 5; n=$((n+1)); done; }
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null; then note preflight "FAILED: GPU busy"; exit 1; fi
note preflight "$(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader); load $(cut -d' ' -f1 /proc/loadavg)"

# ---- 1. speed, 5 repetitions, three offload levels ------------------------------------------
SPEED="$OUT/bench-rerun.jsonl"
for m in "$MODELS"/*-own-*.gguf; do
  name="$(basename "$m")"
  if [ -f "$SPEED" ] && [ "$(grep -c "\"model\": \"$name\"" "$SPEED")" -ge 3 ]; then continue; fi
  n_layer=$(uv run python -c "import sys; from inference.bench.gguf import read_header; print(read_header(sys.argv[1]).n_layer)" "$m")
  cool
  if uv run python -m inference.bench.run_bench "$m" --ngl "0,$((n_layer/2)),$((n_layer+1))" --reps 5 --out "$SPEED"; then
    note "speed-$name" ok; else note "speed-$name" "FAILED"; fi
done
[ -f "$SPEED" ] && uv run python -m inference.bench.report "$SPEED" >/dev/null
publish "speed re-run"

# ---- 2. tool calling with thinking switched off ----------------------------------------------
TC="$OUT/toolcall-nothink.jsonl"
for m in "$MODELS"/*-own-*.gguf; do
  label="$(basename "$m" .gguf)-nothink"
  if [ -f "$TC" ] && [ "$(grep -c "\"label\": \"$label\"" "$TC")" -ge 86 ]; then continue; fi
  ctx=8192; case "$m" in *Q8_0*) ctx=4096 ;; esac   # Q8 only fits with a shorter context
  cool
  "$BIN/llama-server" -m "$m" -ngl 99 -c "$ctx" --host 127.0.0.1 --port "$PORT" --jinja --reasoning-budget 0 \
    > "$HOME/.cache/hybrid-agents/llama-server-$label.log" 2>&1 &
  srv=$!; ok=0
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
    kill -0 $srv 2>/dev/null || break; sleep 2
  done
  if [ $ok = 1 ]; then
    if uv run python -m inference.toolcall.run_eval --base-url "http://127.0.0.1:$PORT" --label "$label" --seeds 1 --out "$TC"
    then note "toolcall-$label" "ok (ctx $ctx)"; else note "toolcall-$label" FAILED; fi
  else
    note "toolcall-$label" "FAILED: server did not start"; tail -3 "$HOME/.cache/hybrid-agents/llama-server-$label.log"
  fi
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
done
[ -f "$TC" ] && uv run python -m inference.toolcall.report "$TC" > "$OUT/toolcall-nothink.md" 2>&1
note done "follow-up finished"
publish "tool calling without thinking"
