#!/usr/bin/env bash
# System packages for Phase 2. Needs sudo, so run this yourself (not Claude Code):
#   bash scripts/phase2/00_system_deps.sh
#
# Installs the CUDA *toolkit* only. Never install `cuda` or `cuda-drivers`
# inside WSL: the GPU driver lives on the Windows side and those packages break it.
set -euo pipefail

# Package sets to try, smallest last. The full toolkit also pulls in GUI profilers
# (nsight-systems) whose dependencies are not always installable on WSL; the
# compiler + development libraries are all llama.cpp needs.
CUDA_SETS=(
  "cuda-toolkit-12-8"
  "cuda-compiler-12-8 cuda-libraries-dev-12-8"
  "cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8"
)
CUDA_DIR="/usr/local/cuda-12.8"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"

if ! grep -qi microsoft /proc/version; then
  echo "This script is written for WSL2 Ubuntu. Stopping." >&2
  exit 1
fi

# The CUDA toolkit pulls in libxcb-cursor0, which lives in Ubuntu's "universe"
# component. Without it apt stops with "nsight-systems ... not installable".
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y universe

sudo apt-get update
sudo apt-get install -y build-essential cmake git curl wget unzip tmux libcurl4-openssl-dev python3-venv

if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  tmp="$(mktemp -d)"
  wget -q -O "$tmp/cuda-keyring.deb" "$KEYRING_URL"
  sudo dpkg -i "$tmp/cuda-keyring.deb"
  rm -rf "$tmp"
  sudo apt-get update
fi

if dpkg -l | grep -qE '^ii\s+(cuda-drivers|nvidia-driver)'; then
  echo "WARNING: an NVIDIA driver package is installed inside WSL. Remove it before continuing." >&2
  exit 1
fi

CUDA_PKGS=""
for set in "${CUDA_SETS[@]}"; do
  # shellcheck disable=SC2086
  if apt-get install -s $set >/dev/null 2>&1; then
    CUDA_PKGS="$set"
    break
  fi
  echo "not installable here, trying a smaller set: $set"
done
if [ -z "$CUDA_PKGS" ]; then
  echo "No CUDA package set can be installed. Diagnostics:" >&2
  apt-cache policy libxcb-cursor0 cuda-nvcc-12-8 >&2
  exit 1
fi
echo "installing: $CUDA_PKGS"
# shellcheck disable=SC2086
sudo apt-get install -y $CUDA_PKGS

if ! grep -q "$CUDA_DIR/bin" "$HOME/.bashrc"; then
  {
    echo "export PATH=$CUDA_DIR/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=$CUDA_DIR/lib64:\${LD_LIBRARY_PATH:-}"
  } >> "$HOME/.bashrc"
fi

export PATH="$CUDA_DIR/bin:$PATH"
echo "--- check ---"
nvcc --version | tail -2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Done. Open a new shell (or 'source ~/.bashrc') before building."
