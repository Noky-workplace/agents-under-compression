#!/usr/bin/env bash
# Tool-calling v4 enforcement; owner runs on ASUS only.
# tmux new -d -s tcv4 'bash scripts/phase2/gpu_toolcall_v4.sh'
# Planning allowance: 3-6 hours; eight-turn retries may extend v3 timings.
set -uo pipefail
REPO="${REPO:-$HOME/projects/hybrid-ai-system}"
BIN="${LLAMA_BIN_DIR:-$HOME/src/llama.cpp/build/bin}"
MODELS="$HOME/models"; OUT="research/results/phase2"; PORT="${PORT:-8089}"
TASKS="${TASKS:-inference/toolcall/tasks/finance_v1_wait.jsonl}"; RES="${RES:-$OUT/toolcall-v4-enforce.jsonl}"
STATUS="${RES%.jsonl}-status.md"; SUFFIX="-enforce"
NOTHINK_QUANTS="${NOTHINK_QUANTS-Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0}"; THINK_QUANTS="${THINK_QUANTS-Q4_K_M}"
N_TASKS=$(wc -l < "$REPO/$TASKS")
export PATH="$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH"
mkdir -p "$HOME/.cache/hybrid-agents"; exec >>"$HOME/.cache/hybrid-agents/gpu-toolcall-v4-enforce.log" 2>&1
cd "$REPO" || exit 1
note() { printf -- '- %s  **%s**  %s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> "$STATUS"; echo ">>> $1: $2"; }
publish() {
  local b; b=$(git rev-parse --abbrev-ref HEAD); [ "$b" = main ] && return 1
  for _ in 1 2 3 4 5; do
    git add "$OUT" && { git diff --cached --quiet || git commit -q -m "gpu-toolcall-v4-enforce: $1"; } \
      && git pull -q --rebase origin "$b" && git push -q origin "$b" && return 0
    git rebase --abort 2>/dev/null; sleep 20
  done; echo "PUSH FAILED"
}
gpu_temp() {
  local value
  value=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) || return 1
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s' "$value"
}
cool() {
  local n temp
  for n in $(seq 1 90); do
    temp=$(gpu_temp) || { note cooldown "FAILED: unavailable telemetry"; return 1; }
    [ "$temp" -le 62 ] && return 0
    sleep 5
  done
  note cooldown "FAILED: still hot after cooldown"; return 1
}
SRV=""; MONITOR=""; EVALUATOR=""
PARENT=$$
stop_pid() {  # TERM, bounded wait without the wait builtin (it can block inside a trap), then KILL
  local pid="$1" n
  [ -n "$pid" ] || return 0
  kill "$pid" 2>/dev/null || return 0
  for n in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.1; done
  kill -9 "$pid" 2>/dev/null || true
}
cleanup() {
  stop_pid "$EVALUATOR"; EVALUATOR=""
  stop_pid "$MONITOR"; MONITOR=""
  stop_pid "$SRV"; SRV=""
}
trap cleanup EXIT
trap 'note thermal "FAILED: thermal or telemetry abort; evaluation interrupted, resume label"; exit 1' USR1
trap 'exit 130' INT
trap 'exit 143' TERM
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null; then note preflight "FAILED: GPU busy"; exit 1; fi
# Verify flags against the owner's installed binary before starting a GPU job.
SERVER_HELP=$("$BIN/llama-server" --help 2>&1) || exit 1
for flag in --reasoning-budget --jinja --host --port; do
  [[ "$SERVER_HELP" == *"$flag"* ]] || { note preflight "FAILED: missing $flag"; exit 1; }
done
uv sync -q || exit 1
note preflight "$N_TASKS tasks; $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader)"

run_one() {  # run_one <model file> <think|nothink>
  local m="$1" mode="$2" label ctx=8192 extra=() timeout=120 rules=(--one-call --reject-placeholders)
  if [ "$mode" = prov ]; then
    mode=nothink; SUFFIX=-prov; rules+=(--provenance)
  else SUFFIX=-enforce; fi
  label="$(basename "$m" .gguf | sed 's/Qwen_Qwen3-8B-own-//')-$mode$SUFFIX"
  # Count complete, distinct task records, never a killed writer's partial tail.
  if [ -f "$RES" ] && python3 - "$RES" "$label" "$N_TASKS" <<'PYCOUNT'
import json
import sys
from pathlib import Path

records = set()
lines = Path(sys.argv[1]).read_bytes().splitlines(keepends=True)
for i, line in enumerate(lines):
    try:
        row = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        if i == len(lines) - 1 and not line.endswith(b"\n"):
            break
        raise
    if row["label"] == sys.argv[2]:
        records.add((row["task"], row.get("seed", 0)))
sys.exit(0 if len(records) >= int(sys.argv[3]) else 1)
PYCOUNT
  then return 0; fi
  case "$m" in *Q8_0*) ctx=4096 ;; esac
  if [ "$mode" = nothink ]; then extra=(--reasoning-budget 0); else timeout=300; fi
  cool || exit 1
  "$BIN/llama-server" -m "$m" -ngl 99 -c "$ctx" --host 127.0.0.1 --port "$PORT" --jinja "${extra[@]}" \
    > "$HOME/.cache/hybrid-agents/llama-server-v4-$label.log" 2>&1 &
  local srv=$! ok=0
  SRV=$srv
  (
    while kill -0 "$srv" 2>/dev/null; do
      temp=$(gpu_temp) || temp=999
      if [ "$temp" -ge 85 ]; then
        # Parent kills evaluator first: active task never becomes a transport-failure row.
        kill -USR1 "$PARENT"; exit 1
      fi
      sleep 2
    done
  ) &
  MONITOR=$!
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
    kill -0 $srv 2>/dev/null || break; sleep 2
  done
  if [ $ok = 1 ]; then
    uv run python -m inference.toolcall.run_eval --base-url "http://127.0.0.1:$PORT" --label "$label" \
         --tasks "$TASKS" --seeds 1 --timeout "$timeout" --out "$RES" "${rules[@]}" &
    EVALUATOR=$!
    local eval_status=0
    wait "$EVALUATOR" || eval_status=$?
    EVALUATOR=""
    # Catch a monitor failure racing with evaluation completion.
    kill "$MONITOR" 2>/dev/null || true
    local monitor_status=0
    wait "$MONITOR" || monitor_status=$?
    MONITOR=""
    if [ "$monitor_status" != 0 ] && [ "$monitor_status" != 143 ]; then
      note "$label" "FAILED: telemetry monitor exited $monitor_status"; exit 1
    fi
    if [ "$eval_status" != 0 ]; then note "$label" FAILED; exit 1; fi
    note "$label" "ok (ctx $ctx)"
  else note "$label" "FAILED: server did not start"; exit 1; fi
  cleanup
  uv run python -m inference.toolcall.report "$RES" > "${RES%.jsonl}.md" 2>&1
  publish "$label"
}
for q in $NOTHINK_QUANTS; do run_one "$MODELS/Qwen_Qwen3-8B-own-$q.gguf" nothink; done
for q in $THINK_QUANTS; do run_one "$MODELS/Qwen_Qwen3-8B-own-$q.gguf" think; done
run_one "$MODELS/Qwen_Qwen3-8B-own-Q4_K_M.gguf" prov
uv run python -m inference.toolcall.compare_runs --v4 || { note analysis FAILED; exit 1; }
note done "finished"; publish "status"
