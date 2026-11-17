#!/usr/bin/env bash
# Quantize a full-precision GGUF into several sizes ourselves (the project goal:
# do the compression by hand and understand what it costs).
#   bash scripts/phase2/03_quantize.sh ~/archive/models/<model>-bf16.gguf
set -euo pipefail

BASE="${1:?path to the f16/bf16 GGUF}"
OUT_DIR="${OUT_DIR:-$HOME/models}"
BIN="${LLAMA_BIN_DIR:-$HOME/src/llama.cpp/build/bin}"
QUANTS="${QUANTS:-Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M}"

stem="$(basename "$BASE" .gguf)"
stem="${stem%-bf16}"; stem="${stem%-f16}"; stem="${stem%-BF16}"; stem="${stem%-F16}"
mkdir -p "$OUT_DIR"

for q in $QUANTS; do
  out="$OUT_DIR/${stem}-own-${q}.gguf"
  if [ -s "$out" ]; then
    echo "skip $q (exists)"
    continue
  fi
  echo "=== $q ==="
  "$BIN/llama-quantize" "$BASE" "$out.tmp" "$q"
  mv "$out.tmp" "$out"
  ls -lh "$out" | awk '{print $5, $9}'
done
