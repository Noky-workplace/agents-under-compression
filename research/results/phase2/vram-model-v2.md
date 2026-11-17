# VRAM model v2 — 13 November 2026

## Data and decision

Fit every successful row of `bench.jsonl` (36 rows, including both Q4 variants);
`bench-rerun.jsonl` supplies 15 held-out records. No deduplication or quant-specific
correction. All baselines are 10 MiB; `used_mib = peak_mib - baseline_mib`, so this
models incremental GPU allocation rather than absolute card occupancy. Add the
current baseline separately when choosing a whole-card budget. All records use
512 prompt and 128 generation tokens. Recorded KV is 2 MiB per offloaded block;
the report uses that recorded term, without substituting the CLI context default.

Decision: use the requested two-parameter OLS model, retaining a physical raw
weight count and applying an empirical scale only in the total. Do not introduce
a Q8-only tensor-placement correction without tensor tables or allocator logs.
The simple model has systematic residuals and worsens full-offload accuracy;
it is an empirical budget estimate, not an OOM guarantee.

## Residuals before fitting

Signed residual = measured minus prediction, in MiB. Partial includes ngl=36:
all blocks are on GPU but the output head is not requested until ngl=37.

| model | ngl | old residual | fitted residual |
|---|---:|---:|---:|
| Qwen_Qwen3-8B-Q4_K_M | 0 | 972.0 | -12.4 |
| Qwen_Qwen3-8B-Q4_K_M | 9 | 346.4 | -67.6 |
| Qwen_Qwen3-8B-Q4_K_M | 18 | 336.4 | -10.9 |
| Qwen_Qwen3-8B-Q4_K_M | 27 | 328.3 | 47.7 |
| Qwen_Qwen3-8B-Q4_K_M | 36 | 317.5 | 105.4 |
| Qwen_Qwen3-8B-Q4_K_M | 37 | -51.3 | -230.0 |
| Qwen_Qwen3-8B-own-Q3_K_M | 0 | 972.0 | -12.4 |
| Qwen_Qwen3-8B-own-Q3_K_M | 9 | 390.9 | -39.1 |
| Qwen_Qwen3-8B-own-Q3_K_M | 18 | 381.7 | 6.1 |
| Qwen_Qwen3-8B-own-Q3_K_M | 27 | 376.6 | 55.5 |
| Qwen_Qwen3-8B-own-Q3_K_M | 36 | 354.5 | 88.7 |
| Qwen_Qwen3-8B-own-Q3_K_M | 37 | -50.4 | -282.8 |
| Qwen_Qwen3-8B-own-Q4_K_M | 0 | 972.0 | -12.4 |
| Qwen_Qwen3-8B-own-Q4_K_M | 9 | 346.4 | -67.6 |
| Qwen_Qwen3-8B-own-Q4_K_M | 18 | 336.4 | -10.9 |
| Qwen_Qwen3-8B-own-Q4_K_M | 27 | 328.3 | 47.7 |
| Qwen_Qwen3-8B-own-Q4_K_M | 36 | 291.5 | 79.4 |
| Qwen_Qwen3-8B-own-Q4_K_M | 37 | -51.3 | -230.0 |
| Qwen_Qwen3-8B-own-Q5_K_M | 0 | 972.0 | -12.4 |
| Qwen_Qwen3-8B-own-Q5_K_M | 9 | 330.9 | -72.0 |
| Qwen_Qwen3-8B-own-Q5_K_M | 18 | 321.4 | -2.0 |
| Qwen_Qwen3-8B-own-Q5_K_M | 27 | 297.8 | 54.0 |
| Qwen_Qwen3-8B-own-Q5_K_M | 36 | 300.5 | 137.2 |
| Qwen_Qwen3-8B-own-Q5_K_M | 37 | -50.3 | -180.2 |
| Qwen_Qwen3-8B-own-Q6_K | 0 | 972.0 | -12.4 |
| Qwen_Qwen3-8B-own-Q6_K | 9 | 313.3 | -77.9 |
| Qwen_Qwen3-8B-own-Q6_K | 18 | 302.5 | 4.6 |
| Qwen_Qwen3-8B-own-Q6_K | 27 | 295.8 | 91.1 |
| Qwen_Qwen3-8B-own-Q6_K | 36 | 283.1 | 171.7 |
| Qwen_Qwen3-8B-own-Q6_K | 37 | -51.8 | -129.8 |
| Qwen_Qwen3-8B-own-Q8_0 | 0 | 1116.0 | 131.6 |
| Qwen_Qwen3-8B-own-Q8_0 | 9 | 412.2 | 48.6 |
| Qwen_Qwen3-8B-own-Q8_0 | 18 | 402.4 | 159.5 |
| Qwen_Qwen3-8B-own-Q8_0 | 27 | 394.6 | 272.5 |
| Qwen_Qwen3-8B-own-Q8_0 | 36 | 292.8 | 291.5 |
| Qwen_Qwen3-8B-own-Q8_0 | 37 | -371.8 | -329.9 |

CPU-only residuals are 972 MiB except Q8 at 1116 MiB. At partial offload the
old model underpredicts by 283.1–412.2 MiB. At full offload it overpredicts
Q3–Q6 by 50.3–51.8 MiB and Q8 by 371.8 MiB. This discontinuity at ngl=37
suggests that constant overhead and raw weight bytes alone cannot explain all
allocation changes. After fitting, partial residuals still trend upward with ngl
and full-offload residuals remain negative. A single weight scale is a compromise.

## Fit and validation

Ordinary least squares minimizes squared error of y = used_mib - predicted_kv_mib
against x = predicted_weights_mib, including zero-offload rows.

- fixed = 984.4145632964137 MiB
- a = 0.9313736266040277
- prediction = fixed + a * raw offloaded weights MiB + recorded KV MiB

Reproduce with `report.weight_fit(report.load(Path("research/results/phase2/bench.jsonl")))`
from `inference.bench.report`; pass the returned constants and held-out rows to
`report.model_errors`. No fitting or tuning uses the rerun file.

| dataset | offload | n | old mean abs | old worst abs | new mean abs | new worst abs |
|---|---|---:|---:|---:|---:|---:|
| train | zero | 6 | 996.0 | 1116.0 | 32.3 | 131.6 |
| train | partial | 24 | 336.8 | 412.2 | 83.7 | 291.5 |
| train | full | 6 | 104.5 | 371.8 | 230.4 | 329.9 |
| held out | zero | 5 | 1000.8 | 1116.0 | 36.2 | 131.6 |
| held out | partial | 5 | 348.9 | 402.4 | 36.6 | 159.5 |
| held out | full | 5 | 115.1 | 371.8 | 230.5 | 329.9 |

The held-out worst absolute error is **329.9 MiB**, at Q8 ngl=37 (overprediction).
This is not an improvement in every regime: full-offload mean absolute error
increases. The rerun records repeat the corresponding original memory numbers
exactly, so they validate a separate run but supply little new memory variation.

## Q8 output head / embedding hypothesis

`gguf.py` derives tensor bytes from adjacent offsets. `non_block_bytes()` assigns
`token_embd`, position and token-type tensors to input; all other non-block tensors
to output. `offloaded_weight_bytes()` excludes input at every ngl, and adds output
only for ngl > n_layer. Thus the current calculation does not accidentally offload
the input embedding. The output classification is broader than an output matrix.

| quant/model | file minus full predicted weights | head-side increment (36→37) | measured increment (36→37) |
|---|---:|---:|---:|
| Qwen_Qwen3-8B-Q4_K_M | 339.6 | 486.8 | 118.0 |
| Qwen_Qwen3-8B-own-Q3_K_M | 260.7 | 486.9 | 82.0 |
| Qwen_Qwen3-8B-own-Q4_K_M | 339.6 | 486.8 | 144.0 |
| Qwen_Qwen3-8B-own-Q5_K_M | 413.8 | 486.8 | 136.0 |
| Qwen_Qwen3-8B-own-Q6_K | 492.5 | 486.9 | 152.0 |
| Qwen_Qwen3-8B-own-Q8_0 | 636.2 | 630.6 | -34.0 |

All table values are MiB. File minus weights includes CPU input tensors AND
header/metadata/alignment, so it is not an exact embedding size. Q8 leaves 636.2
MiB outside the full predicted weight count, consistent with a large input
embedding remaining on CPU. Q8 adds 630.6 MiB of predicted output-side tensors
at ngl=37, yet measured allocation falls 34 MiB. Omitting that entire output side
would lower the old prediction by 630.6 MiB, changing its 371.8 MiB overprediction
to a 258.8 MiB underprediction. It therefore does not fully explain the mismatch.

**Hypothesis:** CPU residency of some output-side tensors, changing compute buffers,
allocator reuse or sampled peak differences at full offload explain the discrepancy.
Even Q3–Q6 measured head increments (82–152 MiB) are much smaller than predicted
486.8–486.9 MiB. The Q8 anomaly is not proof that its entire head stays on CPU.
No local model tensor table, llama.cpp placement log or source checkout was used;
the files alone cannot identify the actual placements. No byte accounting changed.

## Operational limits

One Qwen3-8B model family, one RTX 4060 Laptop GPU, one short prompt/generation
configuration, rounded measurements and peak sampling. The scale does not mean
quantized tensors physically shrink; it absorbs unmodelled allocations/residency.
Do not extrapolate it as a universal correction across devices or long contexts.
`weights_mib` remains raw; `Estimate.total_mib` applies `weight_scale`. Explicit
`weight_scale=1` restores physical weight accounting; calibration refits the fixed
intercept after removing scaled weights and KV, including ngl=0 samples.
CPU-only estimates now retain fixed overhead. `max_ngl` raises ValueError if the
budget cannot cover even that overhead, rather than claiming a feasible setting.
No GPU, Docker, network, model download or sudo needed for this offline analysis.
