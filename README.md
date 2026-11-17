# Agents under compression

How does quantization change the tool-calling decisions of small language models, why, and which fixes still work on
the compressed model that people actually run?

This repository holds the code, tasks and result files for that question, measured on one consumer machine:
an RTX 4060 Laptop GPU with 8 GB of VRAM running llama.cpp.

## Start here
**[Article v1: Qwen3-8B on an 8 GB GPU](docs/writeup/qwen3-8b-on-8gb.md)**

- From Q8 down to Q4, tool-calling scores on 114 tasks do not separate. Q3 is measurably worse.
- The main failure on chained tasks is the same at every precision: a dependent call is sent before the result it
  needs exists, with a guessed argument.
- One prompt instruction removes it with thinking off. A harness rule matters only where the instruction fails, and a
  blanket version of that rule breaks parallel calls.

## What is here
| Path | Content |
|---|---|
| `inference/toolcall/` | Evaluation harness for any OpenAI-compatible local server, scorer, paired analysis, enforcement rules, margin measurement |
| `inference/toolcall/tasks/` | 114 hand-written finance tool-calling tasks (single, parallel, abstain, clarify, multi-step), with and without the wait instruction |
| `inference/bench/` | GGUF header reader, VRAM model, GPU telemetry, benchmark runner |
| `scripts/phase2/` | Build, quantize, quality and GPU run scripts as used on the test machine |
| `research/results/phase2/` | Every result file the article cites, including raw model responses |
| `docs/phase2-results.md` | Speed, memory and quality of the five quants |
| `docs/prereg/` | Pre-registered tests, committed before the measurement |

## Reproduce the analysis (no GPU needed)
```bash
uv sync
uv run pytest
uv run python -m inference.toolcall.compare_runs        # instruction analysis from the checked-in responses
uv run python -m inference.toolcall.compare_runs --v4   # enforcement analysis
```

## Run it on your own model
Start `llama-server` (or any OpenAI-compatible server) with tool calling enabled, then see
`inference/toolcall/README.md` for the evaluation command. The `scripts/phase2/gpu_*.sh` files show the exact server
flags, context sizes and thermal guards used here; read them and adjust paths before running.

## Method rules
- Every number in a write-up comes from a file in `research/results/`.
- Comparisons pair the same task across configurations and use exact sign tests; sample sizes are stated.
- Hypothesis tests are pre-registered with a stop rule. Negative results are reported.

## Limits
One model family so far, a self-written task set, 40 multi-step tasks, one run per task at temperature 0.
Result files are copies from the test machine with home-directory paths replaced by `~`.

## Planned
A much larger chained-call task set, more model families, and whether probes and steering directions found at high
precision still work on quantized weights.

## Licence
Code: MIT (`LICENSE`). Tasks, result files and text: CC BY 4.0 (`LICENSE-docs`). Model weights are not included;
Qwen3 is published by its authors under its own licence.
