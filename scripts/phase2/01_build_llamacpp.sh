#!/usr/bin/env bash
# Build llama.cpp with CUDA for the RTX 4060 (no sudo needed). Safe to re-run.
#   bash scripts/phase2/01_build_llamacpp.sh
set -euo pipefail

SRC_DIR="${LLAMA_SRC_DIR:-$HOME/src/llama.cpp}"
JOBS="${JOBS:-8}"
CUDA_ARCH="${CUDA_ARCH:-89}"   # compute capability 8.9 = RTX 40 series
export PATH="/usr/local/cuda-12.8/bin:$PATH"

command -v nvcc >/dev/null || { echo "nvcc not found - run 00_system_deps.sh first" >&2; exit 1; }

mkdir -p "$(dirname "$SRC_DIR")"
if [ -d "$SRC_DIR/.git" ]; then
  git -C "$SRC_DIR" pull --ff-only
else
  git clone https://github.com/ggml-org/llama.cpp "$SRC_DIR"
fi

cd "$SRC_DIR"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$JOBS"

echo "--- check ---"
./build/bin/llama-cli --version
for bin in llama-bench llama-server llama-quantize llama-perplexity; do
  [ -x "build/bin/$bin" ] && echo "ok  $bin" || echo "MISSING $bin"
done
echo "llama.cpp commit: $(git rev-parse --short HEAD)"
