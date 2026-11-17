# Qwen3-8B on an 8 GB GPU: the main tool-calling failure is acting too early, not low precision

*Article v1, November 2026. Every number links to a results file in this repository.*

**Short version**
- Quantized from Q8 down to Q4, Qwen3-8B's tool-calling score on our 114 tasks does not separate. Q3 is measurably worse.
- The dominant failure on chained tasks is the same at every precision: the model sends a dependent call before the result it depends on exists, and fills the gap with a guessed value.
- One prompt instruction removes that batching with thinking off (9–14 of 31 cases to 0) and lifts chained-task success from 0.47–0.62 to 0.78–0.88.
- Enforcing the same thing in the harness adds nothing when thinking is off, recovers the gap when thinking is on, and breaks legitimately parallel calls. A strict "where did this value come from" check was unusable.

## What we tested

The machine was an RTX 4060 Laptop with 8,188 MiB VRAM, an i7-13650HX and 32 GB RAM, running WSL2 Ubuntu 22.04. Inference used llama.cpp commit `cb44fdb18`. We made the GGUF quants ourselves from bf16 with `llama-quantize`, without an importance matrix. These setup details come from [the Phase 2 results](../phase2-results.md).

The tool evaluation used 114 hand-written finance tasks: 36 single calls, 12 parallel calls, 12 abstentions, 14 clarifications and 40 multi-step tasks. Each configuration supplied one observation per task at temperature 0. The revised harness returned reasoning with tool results. Q8 used a 4k context; the other quants used 8k. Those differences matter when interpreting latency and precision together. See [the baseline analysis](../../research/results/phase2/toolcall-v2-analysis.md) and [raw baseline responses](../../research/results/phase2/toolcall-v2.jsonl).

Success means fully correct arguments for call tasks, or correct abstention/clarification for no-call tasks. A later correct call does not erase an earlier wrong call. Comparisons pair the same task across configurations and use exact, two-sided binomial sign tests on disagreements, excluding ties. Tests are exploratory and unadjusted for multiple comparisons. This is a controlled comparison on our examples, not an estimate of reliability across all financial requests. The definitions are documented in [the instruction analysis](../../research/results/phase2/toolcall-v3-analysis.md).

## Memory and speed

The compact quants leave useful headroom. The largest almost fills the card before a substantial conversation begins.

| Quant | File MiB | Measured VRAM MiB | Predicted MiB | Prediction minus measured MiB |
|---|---:|---:|---:|---:|
| Q3_K_M | 3,933 | 4,194 | 4,244 | +50 |
| Q4_K_M | 4,795 | 4,976 | 5,027 | +51 |
| Q5_K_M | 5,580 | 5,688 | 5,738 | +50 |
| Q6_K | 6,414 | 6,442 | 6,494 | +52 |
| Q8_0 | 8,306 | 7,870 | 8,242 | +372 |

Source: [memory measurements and predictions](../../research/results/phase2/bench-rerun.md), also summarized in [Phase 2 results](../phase2-results.md). These are the maximum-offload benchmark rows, not server memory guarantees at arbitrary context lengths.

The model was within roughly 50 MiB for Q3–Q6. It missed the largest quant more substantially. Partial offload was also less accurate: with no layers offloaded, Q3–Q6 still used 972 MiB, while the model predicted none. For Q4 at partial offload it predicted 2,534 MiB against 2,870 MiB measured. CUDA context and buffers need explicit treatment; extrapolating only from weight size is insufficient. These figures are in [the benchmark report](../../research/results/phase2/bench-rerun.md).

| Quant | Generation tokens/s | Prompt processing tokens/s |
|---|---:|---:|
| Q3_K_M | 50.6 | 1,778 |
| Q4_K_M | 43.5 | 1,850 |
| Q5_K_M | 37.7 | 1,806 |
| Q6_K | 35.8 | 1,689 |
| Q8_0 | 28.8 | 929 |

Source: [idle-machine rerun](../../research/results/phase2/bench-rerun.jsonl), [readable table](../../research/results/phase2/bench-rerun.md). The benchmark used a 512-token prompt, 128 generated tokens and five repetitions, as recorded in [Phase 2 results](../phase2-results.md).

Generation became slower with larger files. Prompt processing did not follow exactly the same pattern. We use the idle rerun because the initial speed sweep was affected by competing machine activity; the earlier speed numbers should not be treated as equally clean evidence.

## What quantization changes

Text prediction provides a separate view of fidelity to the full model. The quality run used wikitext-2, with 40 chunks of 512 tokens. The bf16 reference perplexity was 9.008, according to [Phase 2 results](../phase2-results.md).

| Quant | Perplexity | Mean KL divergence from bf16 | Top-token agreement |
|---|---:|---:|---:|
| Q3_K_M | 9.885 | 0.1643 | 84.0% |
| Q4_K_M | 9.337 | 0.0483 | 91.4% |
| Q5_K_M | 9.206 | 0.0160 | 94.9% |
| Q6_K | 9.158 | 0.0061 | 96.4% |
| Q8_0 | 9.132 | 0.0015 | 98.1% |

Source: [quality table in Phase 2 results](../phase2-results.md), transcribed there from [quality logs](../../research/results/phase2/quality/).

Lower perplexity and KL, and higher agreement, indicate closer predictions. Precision clearly matters here, especially between Q3 and Q4. But closer token distributions do not necessarily fix a repeated mistake in deciding when to call a tool.

## Tool calling before the instruction

| Quant and mode | Overall success | Multi-step success | Median seconds/task |
|---|---:|---:|---:|
| Q3_K_M, thinking off | 0.67 | 0.47 | 2.2 |
| Q4_K_M, thinking off | 0.75 | 0.57 | 2.5 |
| Q5_K_M, thinking off | 0.78 | 0.57 | 2.8 |
| Q6_K, thinking off | 0.76 | 0.62 | 3.2 |
| Q8_0, thinking off | 0.78 | 0.62 | 5.4 |
| Q4_K_M, thinking on | 0.75 | 0.42 | 11.4 |
| Q6_K, thinking on | 0.81 | 0.57 | 15.2 |

Source: [baseline analysis, scores](../../research/results/phase2/toolcall-v2-analysis.md), computed from [baseline JSONL](../../research/results/phase2/toolcall-v2.jsonl).

The paired comparisons put Q3 below Q4: only Q3 succeeded on 3 tasks, only Q4 on 13, with p = 0.02. Against Q8 the counts were 2 versus 15, p = 0.002. Q4 versus Q5, Q6 and Q8 gave p = 0.45, 1.00 and 0.55 respectively. These results do not establish equivalence, but they give little reason to buy more precision specifically for these tool calls. Source: [baseline paired comparisons](../../research/results/phase2/toolcall-v2-analysis.md).

Thinking was no dependable rescue. For Q4, overall disagreements were 11 in each direction, p = 1.00. Multi-step disagreements favoured thinking off, 4 versus 10, but p = 0.18. Q6’s overall comparison had p = 0.38. The slower thinking runs therefore did not establish a reliability improvement. Source: [the same paired table](../../research/results/phase2/toolcall-v2-analysis.md).

## The failure: acting before the result exists

In `finance-074`, Q4 with thinking off sent `detect_regime` and this dependent call together:

```json
{"name":"realized_volatility","arguments":{"symbol":"ETHUSDT","window_minutes":0,"as_of":"2026-11-01T12:00:00Z"}}
```

The expected dependent call was:

```json
{"name":"realized_volatility","arguments":{"symbol":"ETHUSDT","window_minutes":180,"as_of":"2026-11-01T12:00:00Z"}}
```

The observed call is quoted from [baseline responses](../../research/results/phase2/toolcall-v2.jsonl), selecting `Q4_K_M-nothink` and `finance-074`; the expected arguments are defined in [the task fixture](../../inference/toolcall/tasks/finance_v1.jsonl). The raw response’s final text also mentions the returned 180-minute recommendation. Expected calls are not stored in the response JSONL itself.

The problem is causal ordering. The user requested the window returned by the regime tool. The model guessed before receiving it. Higher precision alone cannot be assumed to correct this pattern, because the same kind of premature batching appeared across precisions.

## The instruction and its effect

The added waiting sentence was:

> When an argument for a tool call depends on the result of another tool call, call the first tool alone and wait for its result before making the dependent call.

The prompt variant also added:

> Use values from tool results exactly as returned.

Both additions are visible in [the wait-task fixture](../../inference/toolcall/tasks/finance_v1_wait.jsonl). Thus the experiment tested the combined instruction, not the waiting sentence isolated from the exact-copy reminder.

| Quant and mode | Overall, before → after | Multi-step, before → after | Multi-step sign-test p |
|---|---:|---:|---:|
| Q3_K_M, thinking off | 0.67 → 0.75 | 0.47 → 0.78 | 0.00183 |
| Q4_K_M, thinking off | 0.75 → 0.82 | 0.57 → 0.82 | 0.00635 |
| Q5_K_M, thinking off | 0.78 → 0.87 | 0.57 → 0.88 | 0.00049 |
| Q6_K, thinking off | 0.76 → 0.84 | 0.62 → 0.85 | 0.01172 |
| Q8_0, thinking off | 0.78 → 0.83 | 0.62 → 0.82 | 0.02148 |
| Q4_K_M, thinking on | 0.75 → 0.81 | 0.42 → 0.68 | 0.00635 |

Sources: [baseline analysis](../../research/results/phase2/toolcall-v2-analysis.md) and [wait-instruction analysis](../../research/results/phase2/toolcall-v3-analysis.md). Rounded shares can obscure the exact task-count changes.

Among the 31 multi-step tasks with multiple expected calls, thinking-off first-response batching fell from 9–14 cases, or 29.0–45.2%, to none at every precision. This metric includes independent calls, so it is a useful symptom measure rather than a perfect dependency detector. Q4 thinking retained 7/31 batched responses, down from 18/31. Source: [batching table](../../research/results/phase2/toolcall-v3-analysis.md).

Q3 gained a net 12/40 multi-step successes; higher precisions gained 8–12/40. Paired interaction p values ranged from 0.3984375 to 1.0. Q3 did not show a detectably smaller benefit; this does not prove identical effects. Some single, parallel and clarification scores dropped, but none of those category comparisons was significant. Overall gains were not significant for Q4 thinking off, Q8 thinking off or Q4 thinking on. Source: [paired, side-effect and precision tables](../../research/results/phase2/toolcall-v3-analysis.md).

## Making it a harness rule

A prompt is a request. We then enforced the behaviour in the harness and reran all 114 tasks with the same prompt: only the first tool call of a reply is executed (the others get an error asking the model to resend), and calls whose string arguments look like placeholders are rejected. A stricter variant also rejected any argument value that appears nowhere in the user message, earlier tool results or the tool schema. Each run is paired with the instruction-only run above.

| Run | Chained tasks, instruction only → with rule | What changed |
|---|---:|---|
| Q3–Q8, thinking off, one-call + placeholder rules | identical at every precision | No task changed outcome. No placeholder was ever rejected |
| Q4, thinking on, same rules | 0.675 → 0.825 (executed calls) | 6 tasks gained, 0 lost, p = 0.03. Median 32 s → 39 s per task |
| Q4, thinking off, plus provenance check | 0.825 → 0.175 | 17 of 40 tasks hit the 8-turn cap |

Source: [enforcement analysis](../../research/results/phase2/toolcall-v4-analysis.md), from [raw responses](../../research/results/phase2/toolcall-v4-enforce.jsonl). "Executed calls" scores only the calls the harness let through; the strict score, which also counts rejected calls, is unchanged for the thinking-on run.

Three things follow.

1. **The instruction and the rule are substitutes.** With thinking off the instruction already produces one call per reply on chained tasks, so the rule has nothing to catch. The rule earns its place where the instruction fails: with thinking on, it brings chained-task success to the thinking-off level.
2. **A blanket one-call rule has a cost.** On the 12 tasks where parallel calls are correct, the rule rejected 12 calls per run, 9–11 of which matched an expected call. Those tasks end after one reply in this harness, so the model could not resend and executed-call success fell to 0. In a real loop the cost would be extra turns; we did not measure that.
3. **Verbatim provenance is far too strict.** Models legitimately compute, reformat and choose values. On single-call tasks executed-call success fell from 0.861 to 0.083.

The rule that is actually wanted is narrower: block a call only when one of its arguments depends on a result that does not exist yet. A harness cannot know that in general, which is why this remains open.

## Related work

The failure itself is known. Practitioner write-ups describe models that send dependent calls in parallel with invented or placeholder values. [NESTFUL](https://arxiv.org/abs/2409.03797) benchmarks nested call sequences, where one call's output feeds the next, and reports low full-sequence accuracy even for the best model tested.

Quantization and tool calling has also been measured, mostly on smaller models: [Same Quantization, Different Failures](https://neurips.cc/virtual/2026/171271) covers six small models at four GGUF precisions on multi-turn BFCL, and [Compression Breaks Discipline](https://neurips.cc/virtual/2026/171153) audits error types for Qwen3-1.7B and 4B at 4 bits. [Flat Score, Amplified Failures](https://arxiv.org/abs/2607.27275) argues that aggregate scores hide tool-calling damage in quantized agents.

What this article adds is narrow: one 8B model at five precisions of our own making, on one consumer GPU, with the chained-call failure isolated, shown to be independent of precision from Q4 up, and two fixes compared on the same tasks with paired tests. We have read the abstracts of the papers above, not all of them in full; corrections are welcome.

## What we would run, and how to reproduce

For this tool workload, start with Q4_K_M, thinking off, and both instruction sentences. If thinking must stay on, add the one-call rule and allow parallel calls only for tools you know are independent. It balances speed, memory headroom and measured task success. Q5 remains reasonable when token fidelity matters more: the earlier [Phase 2 recommendation](../phase2-results.md) favoured it on that basis. Neither choice makes tool execution safe without validation.

The limits are substantial: one model, our own finance-only task set, only 40 multi-step tasks (a 95% interval of roughly ±0.15 on that category), one run per task at temperature 0, unadjusted p values and context differences between configurations. Thinking-on coverage was limited, and batching persisted there. Failures and recoveries remain; no live trading or general agent benchmark was tested. The task counts and conditions are recorded in [the baseline](../../research/results/phase2/toolcall-v2-analysis.md) and [instruction analyses](../../research/results/phase2/toolcall-v3-analysis.md).

On a prepared GPU host, follow `scripts/phase2/01_build_llamacpp.sh`, `02_fetch_model.py`, `03_quantize.sh` and `04_quality.sh` for build, weights, quantization and quality. Inspect their paths and prerequisites before running them. The benchmark pipeline and idle follow-up are `gpu_pipeline.sh` and `gpu_followup.sh`; thermal protection belongs to the benchmark runner. To reproduce tool evaluations:

```bash
bash scripts/phase2/gpu_toolcall_v2.sh
bash scripts/phase2/gpu_toolcall_v3.sh
bash scripts/phase2/gpu_toolcall_v4.sh
```

Those GPU scripts start a local `llama-server`, stop it afterwards and try to commit and push results; read them and adjust paths before running them on your own machine. To regenerate the paired analysis offline from checked-in data:

```bash
uv run python -m inference.toolcall.compare_runs
uv run python -m inference.toolcall.compare_runs --v4
uv run pytest
uv run ruff check .
```

The useful result is a concrete execution rule: obtain a tool result before using it as an argument. Measure that behaviour directly before spending memory on more precision.

Next: a much larger set of chained tasks, more model families, and whether fixes that read or steer the model's internal state still work on quantized weights.
