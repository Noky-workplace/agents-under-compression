#!/usr/bin/env bash
# Quality of each quant against the full-precision model, on wikitext-2:
# perplexity and KL divergence from the base model's own token probabilities.
#   bash scripts/phase2/04_quality.sh ~/archive/models/<model>-bf16.gguf ~/models/<model>-own-*.gguf
#
# NOTE for whoever runs this: confirm the flags with `llama-perplexity --help`
# for the commit you built; adjust here if upstream renamed anything.
set -euo pipefail

BASE="${1:?path to the f16/bf16 GGUF}"; shift
SRC="${LLAMA_SRC_DIR:-$HOME/src/llama.cpp}"
BIN="${LLAMA_BIN_DIR:-$SRC/build/bin}"
WORK="${WORK_DIR:-$HOME/quality-work}"   # Linux disk: unzip cannot set permissions on the D: drive
OUT="${OUT_DIR:-research/results/phase2/quality}"
CHUNKS="${CHUNKS:-40}"
BASE_NGL="${BASE_NGL:-12}"   # the 16 GB base model only partly fits in 8 GB

mkdir -p "$WORK" "$OUT"
if [ ! -s "$WORK/wikitext-2-raw/wiki.test.raw" ]; then
  # unzip exits 1 on harmless warnings (e.g. permissions on Windows drives): judge by the file, not the exit code
  (cd "$WORK" && bash "$SRC/scripts/get-wikitext-2.sh") || true
fi
[ -s "$WORK/wikitext-2-raw/wiki.test.raw" ] || { echo "wikitext-2 test file missing after download" >&2; exit 1; }
DATA="$WORK/wikitext-2-raw/wiki.test.raw"
LOGITS="$WORK/base-logits.bin"

if [ ! -s "$LOGITS" ]; then
  echo "=== base logits (slow: partial offload) ==="
  "$BIN/llama-perplexity" -m "$BASE" -f "$DATA" --chunks "$CHUNKS" -ngl "$BASE_NGL" \
    --kl-divergence-base "$LOGITS" 2>&1 | tee "$OUT/base.log"
fi

for model in "$@"; do
  name="$(basename "$model" .gguf)"
  echo "=== $name ==="
  "$BIN/llama-perplexity" -m "$model" -ngl 99 \
    --kl-divergence-base "$LOGITS" --kl-divergence 2>&1 | tee "$OUT/$name.log"
done
echo "logs in $OUT"
