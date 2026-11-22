#!/usr/bin/env bash
# Owner runs on ASUS: bash scripts/phase2/gpu_margin.sh (~15 minutes GPU).
# MARGIN_PROBE=1 bash scripts/phase2/gpu_margin.sh  -> one-minute format probe, never a measurement.
# One run only. Aborted measurements remain .partial.jsonl, never complete.
set -euo pipefail
REPO="${REPO:-$HOME/projects/hybrid-ai-system}"
SOURCE="${LLAMA_SOURCE:-$HOME/src/llama.cpp}"
BIN="${LLAMA_BIN_DIR:-$SOURCE/build/bin}"
MODEL="${MARGIN_MODEL:-$HOME/models/Qwen_Qwen3-8B-own-Q8_0.gguf}"
PORT="${PORT:-8089}"
cd "$REPO"
OUT=research/results/phase2
mkdir -p "$OUT"
STATUS="$OUT/margin-status.md"
ABORT="$OUT/margin-abort.txt"
SRV=""; MONITOR=""; EVALUATOR=""
note() { printf -- '- %s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$STATUS"; }
stop_pid() {
  local pid="$1" n
  [ -n "$pid" ] || return 0
  kill "$pid" 2>/dev/null || return 0
  for n in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.1; done
  kill -9 "$pid" 2>/dev/null || true
}
cleanup() { stop_pid "$EVALUATOR"; stop_pid "$MONITOR"; stop_pid "$SRV"; }
trap cleanup EXIT
trap 'note "FAILED: thermal or telemetry abort; partial output is not complete"; exit 1' USR1
trap 'exit 130' INT
trap 'exit 143' TERM
if [ -e "$OUT/margin-q8.jsonl" ] || [ -e "$OUT/margin-q8.partial.jsonl" ] || [ -e "$ABORT" ]; then
  note 'STOP: existing run or abort; one-run rule requires inspection'; exit 1
fi
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null; then
  note 'FAILED: GPU busy'; exit 1
fi
HELP=$("$BIN/llama-server" --help 2>&1)
printf '%s\n' "$HELP" > "$OUT/margin-server-help.txt"
for flag in --reasoning-budget --jinja --host --port; do
  if [[ "$HELP" != *"$flag"* ]]; then note "FAILED: installed binary lacks $flag"; exit 1; fi
done
# Check endpoint and field names against this installed source, not online docs/memory.
README="$SOURCE/tools/server/README.md"
[ -f "$README" ] || README="$SOURCE/examples/server/README.md"
[ -f "$README" ] || { note 'FAILED: installed server README unavailable'; exit 1; }
for name in /v1/chat/completions /apply-template /completion n_probs; do
  grep -F -- "$name" "$README" >> "$OUT/margin-endpoint-evidence.txt" || {
    note "FAILED: installed README does not document $name"; exit 1;
  }
done
gpu_temp() {
  local value
  value=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits) || return 1
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s' "$value"
}
check_temp() {
  local temp
  temp=$(gpu_temp) || { printf '%s\n' 'unavailable telemetry' > "$ABORT"; return 1; }
  if [ "$temp" -ge 85 ]; then printf 'thermal abort: %s C\n' "$temp" > "$ABORT"; return 1; fi
}
ready=0
for _ in $(seq 1 90); do
  check_temp || { note 'FAILED: thermal or telemetry during cooldown'; exit 1; }
  temp=$(gpu_temp) || { note 'FAILED: telemetry during cooldown'; exit 1; }
  if [ "$temp" -le 62 ]; then ready=1; break; fi
  sleep 5
done
[ "$ready" = 1 ] || { note 'FAILED: cooldown timeout'; exit 1; }
"$BIN/llama-server" -m "$MODEL" -ngl 99 -c 4096 --host 127.0.0.1 --port "$PORT" \
  --jinja --reasoning-budget 0 > "$OUT/margin-server.log" 2>&1 &
SRV=$!
PARENT=$$
(
  while kill -0 "$SRV" 2>/dev/null; do
    if ! check_temp; then kill -USR1 "$PARENT"; exit 1; fi
    sleep 2
  done
  printf '%s\n' 'server exited during measurement' > "$ABORT"
  kill -USR1 "$PARENT"; exit 1
) &
MONITOR=$!
ready=0
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then ready=1; break; fi
  kill -0 "$SRV" 2>/dev/null || break
  sleep 2
done
[ "$ready" = 1 ] || { note 'FAILED: server startup'; exit 1; }
if [ "${MARGIN_PROBE:-0}" = 1 ]; then
  note 'Probe: recording server probability format for task 0 (no measurement file)'
  rc=0
  uv run python -m inference.toolcall.margin --probe --base-url "http://127.0.0.1:$PORT" || rc=$?
  note "Probe finished with exit $rc; see margin-probe.json"
  exit "$rc"
fi
note 'Starting preflight and fixed Q8 replay'
uv run python -m inference.toolcall.margin --measure --base-url "http://127.0.0.1:$PORT" \
  2>> "$OUT/margin-measure.log" &
EVALUATOR=$!
rc=0
wait "$EVALUATOR" || rc=$?
EVALUATOR=""
[ "$rc" = 0 ] || { note "FAILED: measurement exit $rc; inspect preflight/partial files"; exit "$rc"; }
# Keep monitoring until after evaluation; marker and final sample close completion races.
check_temp || { note 'FAILED: final thermal/telemetry check'; exit 1; }
kill -0 "$MONITOR" || { note 'FAILED: monitor exited'; exit 1; }
kill "$MONITOR" 2>/dev/null || true
rc=0
wait "$MONITOR" || rc=$?
MONITOR=""
[ "$rc" = 0 ] || [ "$rc" = 143 ] || { note 'FAILED: monitor error'; exit 1; }
[ ! -e "$ABORT" ] || { note 'FAILED: abort marker present'; exit 1; }
mv "$OUT/margin-q8.partial.jsonl" "$OUT/margin-q8.jsonl"
uv run python -m inference.toolcall.margin
note 'Complete: validated measurement and pre-registered analysis; no automatic commit/push'
