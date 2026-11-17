#!/usr/bin/env bash
# Tool-calling eval v3: same 114 tasks with one added system-prompt sentence telling the model to wait for a
# tool result before making a call that depends on it. Tests whether the main chained-call failure
# (guessing a dependent argument in a batched call) is fixable by instruction, at each precision.
#
#   tmux new -d -s tcv3 'bash scripts/phase2/gpu_toolcall_v3.sh'
#   cat research/results/phase2/toolcall-v3-wait-status.md
#
# Thinking off for all five quants, thinking on for Q4_K_M. About 1.5 hours. Safe to re-run.
export TASKS="inference/toolcall/tasks/finance_v1_wait.jsonl"
export RES="research/results/phase2/toolcall-v3-wait.jsonl"
export LABEL_SUFFIX="-wait"
export THINK_QUANTS="Q4_K_M"
exec bash "$(dirname "$0")/gpu_toolcall_v2.sh"
