# Phase 2 results: Qwen3-8B on an 8 GB laptop GPU

_12 Nov 2026. Hardware: RTX 4060 Laptop (8,188 MiB), i7-13650HX, 32 GB RAM, WSL2 Ubuntu 22.04, llama.cpp `cb44fdb18`.
Model: Qwen3-8B. Quants Q3_K_M–Q8_0 made by us from the bf16 file with `llama-quantize` (no importance matrix);
one Q4_K_M downloaded from `bartowski/Qwen_Qwen3-8B-GGUF` for comparison. Every number below comes from a file in
`research/results/phase2/`._

## Summary

1. **Best setting on 8 GB: Q5_K_M.** It fits fully on the GPU with room for a ~12k context and stays close to the
   full model (top token agrees 95% of the time). Q4_K_M is the fallback when more context is needed.
2. **Q3_K_M is where tool calling breaks.** With thinking on, correctness rises steadily with precision
   (0.79 at Q3 to 0.90 at Q8, timeouts excluded); with thinking off, Q3 scores 0.71 and Q4 to Q8 score 0.80–0.85,
   indistinguishable on this test. Thinking adds a few points and costs about 8 times the time per task.
3. **Each quant step down roughly triples the drift from the full model** (KL divergence), with a cliff from Q4 to Q3.
4. **The VRAM model predicts full-offload memory to within ~50 MiB** for Q3–Q6. It under-predicts partial offload by
   ~300 MiB because the CUDA context alone takes ~970 MiB; that constant will be added to the model.

## Quality vs the full bf16 model (wikitext-2, 40 chunks of 512 tokens)

| Quant | Perplexity | vs bf16 (9.008) | Mean KL divergence | Top token same as bf16 | 99th pct KLD |
|---|---:|---:|---:|---:|---:|
| Q3_K_M | 9.885 | +9.7% | 0.1643 | 84.0% | 1.769 |
| Q4_K_M | 9.337 | +3.7% | 0.0483 | 91.4% | 0.540 |
| Q5_K_M | 9.206 | +2.2% | 0.0160 | 94.9% | 0.153 |
| Q6_K | 9.158 | +1.7% | 0.0061 | 96.4% | 0.058 |
| Q8_0 | 9.132 | +1.4% | 0.0015 | 98.1% | 0.014 |

Source: `quality/*.log`. Intervals are in the logs; all differences between adjacent quants are well outside them.

## Tool-calling reliability (86 hand-written finance tasks × 3 runs, llama-server, 8k context, thinking on)

| Quant | Arguments fully correct, all tasks | Same, excluding timed-out requests | Timeouts | Multi-step correct (36 runs) |
|---|---:|---:|---:|---:|
| Q3_K_M | 0.771 | 0.793 | 7 | 0.17 |
| Q4_K_M (ours) | 0.853 | 0.853 | 0 | 0.36 |
| Q4_K_M (downloaded) | 0.857 | 0.857 | 0 | 0.36 |
| Q5_K_M | 0.864 | 0.871 | 2 | 0.56 |
| Q6_K | 0.876 | 0.883 | 2 | 0.56 |
| Q8_0 | 0.841 | 0.900 | 17 | 0.33 (0.48 excluding timeouts) |

Source: `toolcall.jsonl` (recomputed), `toolcall.md`. Parallel calls and abstaining were 1.000 for every quant.

**What holds up**
- Excluding requests that timed out, correctness rises with every step in precision: 0.79 → 0.85 → 0.87 → 0.88 → 0.90.
  The drop from Q4 to Q3 is the largest.
- Q8_0's low raw score is a measurement artefact: 17 of its 258 requests timed out (it barely fits at 8k context
  and ran at half speed). It is not worse than Q6.
- The downloaded Q4_K_M matches ours, so its calibration made no difference here.

**What does not hold up yet**
- The multi-step numbers rest on 12 tasks, and 5 of them fail at *every* precision (and a sixth passes at every
  precision), so only 6 tasks separate the quants. "Q5 is the threshold for chained calls" is a hypothesis, not a result.
- In the failures inspected, the model skipped the first tool and did the conversion itself, wrongly, in one batch
  of calls. That is a model-behaviour failure, but the harness also never returns the model's reasoning with the
  tool result, which Qwen's documentation says degrades multi-turn accuracy. The harness may be depressing every
  quant's multi-step score.
- The three runs per task agree 93–97% of the time (temperature 0), so they are one sample, not three; the
  intervals in `toolcall.md` are too narrow.
- Thinking mode was on (the server returned `reasoning_content`), which is why a run took an hour. A run with
  thinking off is queued (`scripts/phase2/gpu_followup.sh`).

## Tool calling with thinking switched off (same 86 tasks, 1 run, `--reasoning-budget 0`; Q8_0 at 4k context)

| Quant | Arguments correct, all tasks | Multi-step correct | Median seconds per task |
|---|---:|---:|---:|
| Q3_K_M | 0.71 | 0.33 | 1.4 |
| Q4_K_M | 0.81 | 0.58 | 1.5 |
| Q5_K_M | 0.85 | 0.58 | 1.7 |
| Q6_K | 0.80 | 0.58 | 1.9 |
| Q8_0 | 0.83 | 0.58 | 3.3 |

Source: `toolcall-nothink.md`, `toolcall-nothink.jsonl`. No timeouts.
- About 8 times faster than with thinking on, at a cost of a few points of accuracy.
- **Q3 is clearly worse; Q4 to Q8 cannot be told apart on this test** (each score is uncertain by roughly ±0.08).
  The earlier suggestion that Q5 is a threshold for chained calls is not supported.
- A larger test (`finance_v1.jsonl`, 114 tasks, 40 multi-step) is queued.

## VRAM and speed (llama-bench, 512-token prompt, 128 generated tokens)

| Quant | File (MiB) | VRAM at full offload (MiB) | Predicted (MiB) | Error |
|---|---:|---:|---:|---:|
| Q3_K_M | 3,933 | 4,194 | 4,244 | +50 |
| Q4_K_M | 4,795 | 4,976 | 5,027 | +51 |
| Q5_K_M | 5,580 | 5,688 | 5,738 | +50 |
| Q6_K | 6,414 | 6,442 | 6,494 | +52 |
| Q8_0 | 8,306 | 7,870 | 8,242 | +372 |

Source: `bench.jsonl`, `bench.md`. With zero layers on the GPU the card still holds ~970 MiB (CUDA context and
buffers), which the model does not yet include, so partial-offload predictions run ~300 MiB low.

**Speed** (re-run 12 Nov, 5 repetitions, machine idle, all layers on the GPU; source `bench-rerun.jsonl`):

| Quant | Generation (tokens/s) | Prompt processing (tokens/s) |
|---|---:|---:|
| Q3_K_M | 50.6 ± 0.4 | 1,778 |
| Q4_K_M | 43.5 ± 1.5 | 1,850 |
| Q5_K_M | 37.7 ± 1.4 | 1,806 |
| Q6_K | 35.8 ± 0.6 | 1,689 |
| Q8_0 | 28.8 ± 0.3 | 929 |

Generation speed falls with file size, as expected for a memory-bandwidth-bound workload. The first sweep's
erratic numbers (43 vs 33 tokens/s for the same file) were caused by the machine not being idle.

## What fits on 8 GB (from the measurements above, KV cache f16 = 144 MiB per 1k tokens)

| Quant | Free VRAM left at 512 ctx (~7,900 MiB usable) | Context that fits |
|---|---:|---|
| Q4_K_M | ~2.9 GB | ~16k+ |
| Q5_K_M | ~2.2 GB | ~12k |
| Q6_K | ~1.5 GB | ~8k |
| Q8_0 | ~0 | short prompts only |

## Still to do
- Re-run Q8_0 tool calling at 4k context; add multi-step tasks (target 40) and use non-zero temperature for honest intervals.
- Add the ~970 MiB fixed cost to the VRAM model.
- vLLM comparison; 30B MoE with experts in system RAM; KV cache at 8 bits.
