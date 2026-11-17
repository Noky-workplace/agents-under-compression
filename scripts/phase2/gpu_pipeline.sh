#!/usr/bin/env bash
# The whole Phase 2 GPU run, as a plain script: no AI agent involved.
#
# Why: coding agents run in a sandbox that cannot see the GPU. Everything that
# needs the GPU is deterministic anyway, so it runs here, in a normal shell.
#
#   tmux new -d -s gpu 'bash scripts/phase2/gpu_pipeline.sh'      # start
#   tail -f ~/.cache/hybrid-agents/gpu-pipeline.log               # watch
#
# Safe to re-run: finished stages are skipped. A failed stage is recorded in
# research/results/phase2/pipeline-status.md and the next independent stage runs.
# Results are committed and pushed to the current branch after each group.
set -uo pipefail

REPO="${REPO:-$HOME/projects/hybrid-ai-system}"
SRC="${LLAMA_SRC_DIR:-$HOME/src/llama.cpp}"
BIN="${LLAMA_BIN_DIR:-$SRC/build/bin}"
MODELS="$HOME/models"
HF_REPO="${HF_REPO:-bartowski/Qwen_Qwen3-8B-GGUF}"
OUT="research/results/phase2"
STATUS="$OUT/pipeline-status.md"
QUANTS="${QUANTS:-Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M}"
PORT="${PORT:-8089}"
TEMP_KILL_C="${TEMP_KILL_C:-87}"
export PATH="$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH"
export LLAMA_BIN_DIR="$BIN" LLAMA_SRC_DIR="$SRC" QUANTS

mkdir -p "$HOME/.cache/hybrid-agents"
exec >>"$HOME/.cache/hybrid-agents/gpu-pipeline.log" 2>&1
cd "$REPO" || exit 1
mkdir -p "$OUT" "$MODELS"
echo; echo "######## pipeline start $(date -u +%FT%TZ) on $(git rev-parse --abbrev-ref HEAD)"

note() { printf -- '- %s  **%s**  %s\n' "$(date -u +%FT%TZ)" "$1" "$2" >> "$STATUS"; echo ">>> $1: $2"; }
stage() {  # stage <name> <command...>   runs, records ok/FAILED, never aborts the pipeline
  local name="$1"; shift
  echo; echo "=== $name: $*"
  local t0; t0=$(date +%s)
  if "$@"; then note "$name" "ok ($(( $(date +%s) - t0 ))s)"; return 0
  else local rc=$?; note "$name" "FAILED exit $rc ($(( $(date +%s) - t0 ))s)"; return $rc; fi
}
publish() {  # commit and push whatever results exist; retries around other git users
  local branch; branch=$(git rev-parse --abbrev-ref HEAD)
  [ "$branch" = main ] && { echo "refusing to push main"; return 1; }
  for _ in 1 2 3 4 5; do
    git add "$OUT" docs 2>/dev/null && { git diff --cached --quiet || git commit -q -m "gpu-pipeline: $1"; } \
      && git pull -q --rebase origin "$branch" && git push -q origin "$branch" && { echo "pushed: $1"; return 0; }
    git rebase --abort 2>/dev/null; sleep 20
  done
  echo "PUSH FAILED: $1"
}
gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }
cool_down() { local n=0; while [ "$(gpu_temp)" -gt 65 ] && [ $n -lt 60 ]; do sleep 5; n=$((n+1)); done; }
guarded() {  # run a GPU command; kill it if the GPU reaches TEMP_KILL_C
  cool_down
  setsid "$@" & local pid=$!   # own process group, so children are stopped too
  ( while kill -0 $pid 2>/dev/null; do
      t=$(gpu_temp); if [ "${t:-0}" -ge "$TEMP_KILL_C" ]; then echo "GPU ${t}C: stopping $1"; kill -- -$pid; fi
      sleep 10
    done ) &
  local watch=$!
  wait $pid; local rc=$?
  kill $watch 2>/dev/null; wait $watch 2>/dev/null
  return $rc
}
have_rows() { [ -f "$OUT/bench.jsonl" ] && grep -q "\"model\": \"$1\"" "$OUT/bench.jsonl"; }

# ---- 0. preflight -----------------------------------------------------------
if ! nvidia-smi >/dev/null 2>&1; then note preflight "FAILED: nvidia-smi does not work in this shell"; exit 1; fi
if pgrep -x llama-server >/dev/null || pgrep -x llama-bench >/dev/null; then
  note preflight "FAILED: another llama process is using the GPU"; exit 1; fi
note preflight "GPU ok, $(nvidia-smi --query-gpu=name,memory.used,temperature.gpu --format=csv,noheader)"
stage sync uv sync -q

# ---- 1. build ---------------------------------------------------------------
if [ ! -x "$BIN/llama-bench" ] || ! "$BIN/llama-bench" --help >/dev/null 2>&1; then
  stage build bash scripts/phase2/01_build_llamacpp.sh || exit 1
fi
note build "llama.cpp $(git -C "$SRC" rev-parse --short HEAD)"

# ---- 2. first model: download + ngl sweep -----------------------------------
Q4="$(ls "$MODELS"/*Qwen3-8B*Q4_K_M.gguf 2>/dev/null | grep -v -- '-own-' | head -1)"
if [ -z "$Q4" ]; then
  stage fetch-q4 python3 scripts/phase2/02_fetch_model.py --repo "$HF_REPO" --pattern Q4_K_M --dest "$MODELS"
  Q4="$(ls "$MODELS"/*Qwen3-8B*Q4_K_M.gguf 2>/dev/null | grep -v -- '-own-' | head -1)"
fi
if [ -n "$Q4" ]; then
  have_rows "$(basename "$Q4")" || stage bench-q4 uv run python -m inference.bench.run_bench "$Q4" --ngl auto
  stage report uv run python -m inference.bench.report "$OUT/bench.jsonl"
  publish "first model sweep"
fi

# ---- 3. our own quants ------------------------------------------------------
if mountpoint -q /mnt/d; then BASE_DIR=/mnt/d/hybrid/models; else BASE_DIR="$MODELS"; fi
mkdir -p "$BASE_DIR"
BASE="$(ls "$BASE_DIR"/*Qwen3-8B*[bf]16.gguf "$MODELS"/*Qwen3-8B*[bf]16.gguf 2>/dev/null | head -1)"
if [ -z "$BASE" ]; then
  stage fetch-base python3 scripts/phase2/02_fetch_model.py --repo "$HF_REPO" --pattern 'bf16|f16' --dest "$BASE_DIR"
  BASE="$(ls "$BASE_DIR"/*Qwen3-8B*[bf]16.gguf 2>/dev/null | head -1)"
fi
if [ -n "$BASE" ]; then
  note base "full-precision model at $BASE"
  stage quantize bash scripts/phase2/03_quantize.sh "$BASE"
  for m in "$MODELS"/*-own-*.gguf; do
    [ -f "$m" ] || continue
    have_rows "$(basename "$m")" || stage "bench-$(basename "$m" .gguf | sed 's/.*-own-//')" \
      uv run python -m inference.bench.run_bench "$m" --ngl auto
  done
  stage report uv run python -m inference.bench.report "$OUT/bench.jsonl"
  publish "own quants sweep"
  if [ ! -d "$OUT/quality" ] || [ -z "$(ls "$OUT/quality" 2>/dev/null)" ]; then
    WORK_DIR="$HOME/quality-work" stage quality guarded bash scripts/phase2/04_quality.sh "$BASE" "$MODELS"/*-own-*.gguf
    publish "quality run"
  fi
else
  note base "no full-precision file found in $HF_REPO: own quants skipped"
fi

# ---- 4. tool-calling reliability per quant ----------------------------------
if [ -f inference/toolcall/run_eval.py ]; then
  TC="$OUT/toolcall.jsonl"
  for m in "$MODELS"/*-own-*.gguf $Q4; do
    [ -f "$m" ] || continue
    label="$(basename "$m" .gguf)"
    echo; echo "=== toolcall $label"
    cool_down
    "$BIN/llama-server" -m "$m" -ngl 99 -c 8192 --host 127.0.0.1 --port "$PORT" --jinja \
      > "$HOME/.cache/hybrid-agents/llama-server-$label.log" 2>&1 &
    srv=$!
    ok=0
    for _ in $(seq 1 90); do
      if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then ok=1; break; fi
      kill -0 $srv 2>/dev/null || break
      sleep 2
    done
    if [ $ok = 1 ]; then
      stage "toolcall-$label" guarded uv run python -m inference.toolcall.run_eval \
        --base-url "http://127.0.0.1:$PORT" --label "$label" --seeds 3 --out "$TC"
    else
      note "toolcall-$label" "FAILED: server did not start (does not fit in VRAM?), see llama-server-$label.log"
      tail -5 "$HOME/.cache/hybrid-agents/llama-server-$label.log"
    fi
    kill $srv 2>/dev/null; wait $srv 2>/dev/null
  done
  [ -f "$TC" ] && uv run python -m inference.toolcall.report "$TC" > "$OUT/toolcall.md" 2>&1
  publish "tool-calling eval per quant"
fi

note done "pipeline finished"
publish "pipeline status"
