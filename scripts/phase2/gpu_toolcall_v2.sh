#!/usr/bin/env bash
# Tool-calling eval v2 (114 tasks, 40 multi-step) on the GPU. No AI agent.
#
#   tmux new -d -s tcv2 'bash scripts/phase2/gpu_toolcall_v2.sh'
#   cat research/results/phase2/toolcall-v2-status.md
#
# Thinking off for all five quants (~15 min each), thinking on for Q4_K_M and Q6_K only (~1.5 h each).
# About 4-5 hours in total. Power mode Balanced, machine otherwise idle. Safe to re-run.
set -uo pipefail
REPO="${REPO:-$HOME/projects/hybrid-ai-system}"
BIN="${LLAMA_BIN_DIR:-$HOME/src/llama.cpp/build/bin}"
MODELS="$HOME/models"; OUT="research/results/phase2"; PORT="${PORT:-8089}"
TASKS="${TASKS:-inference/toolcall/tasks/finance_v1.jsonl}"; RES="${RES:-$OUT/toolcall-v2.jsonl}"
STATUS="${RES%.jsonl}-status.md"; SUFFIX="${LABEL_SUFFIX:-}"
NOTHINK_QUANTS="${NOTHINK_QUANTS-Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0}"; THINK_QUANTS="${THINK_QUANTS-Q4_K_M Q6_K}"
N_TASKS=$(wc -l < "$REPO/$TASKS")
export PATH="$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH"
mkdir -p "$HOME/.cache/hybrid-agents"; exec >>"$HOME/.cache/hybrid-agents/gpu-toolcall-v2.log" 2>&1
cd "$REPO" || exit 1
note() { printf -- '- %s  **%s**  %s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> "$STATUS"; echo ">>> $1: $2"; }
publish() {
  local b; b=$(git rev-parse --abbrev-ref HEAD); [ "$b" = main ] && return 1
  for _ in 1 2 3 4 5; do
    git add "$OUT" && { git diff --cached --quiet || git commit -q -m "gpu-toolcall-v2: $1"; } \
      && git pull -q --rebase origin "$b" && git push -q origin "$b" && return 0
    git rebase --abort 2>/dev/null; sleep 20
  done; echo "PUSH FAILED"
}
cool() { local n=0; while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)" -gt 62 ] && [ $n -lt 90 ]; do sleep 5; n=$((n+1)); done; }
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null; then note preflight "FAILED: GPU busy"; exit 1; fi
uv sync -q
note preflight "$N_TASKS tasks; $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader)"

run_one() {  # run_one <model file> <think|nothink>
  local m="$1" mode="$2" label ctx=8192 extra=() timeout=120
  label="$(basename "$m" .gguf | sed 's/Qwen_Qwen3-8B-own-//')-$mode$SUFFIX"
  if [ -f "$RES" ] && [ "$(grep -c "\"label\": \"$label\"" "$RES")" -ge "$N_TASKS" ]; then return 0; fi
  case "$m" in *Q8_0*) ctx=4096 ;; esac
  if [ "$mode" = nothink ]; then extra=(--reasoning-budget 0); else timeout=300; fi
  cool
  "$BIN/llama-server" -m "$m" -ngl 99 -c "$ctx" --host 127.0.0.1 --port "$PORT" --jinja "${extra[@]}" \
    > "$HOME/.cache/hybrid-agents/llama-server-v2-$label.log" 2>&1 &
  local srv=$! ok=0
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
    kill -0 $srv 2>/dev/null || break; sleep 2
  done
  if [ $ok = 1 ]; then
    if uv run python -m inference.toolcall.run_eval --base-url "http://127.0.0.1:$PORT" --label "$label" \
         --tasks "$TASKS" --seeds 1 --timeout "$timeout" --out "$RES"
    then note "$label" "ok (ctx $ctx)"; else note "$label" FAILED; fi
  else note "$label" "FAILED: server did not start"; fi
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  uv run python -m inference.toolcall.report "$RES" > "${RES%.jsonl}.md" 2>&1
  publish "$label"
}
for q in $NOTHINK_QUANTS; do run_one "$MODELS/Qwen_Qwen3-8B-own-$q.gguf" nothink; done
for q in $THINK_QUANTS; do run_one "$MODELS/Qwen_Qwen3-8B-own-$q.gguf" think; done
note done "finished"; publish "status"
