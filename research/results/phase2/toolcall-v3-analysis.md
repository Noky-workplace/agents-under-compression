# Tool-calling v3: wait instruction

Sources: `toolcall-v2.jsonl`, `toolcall-v3-wait.jsonl`, and `inference/toolcall/tasks/finance_v1_wait.jsonl`. Regenerate with `python -m inference.toolcall.compare_runs`. Strict success uses argument accuracy for call tasks and correct abstention/nonempty clarification for no-call tasks. One observation per task; paired tests exclude ties. Tests are exploratory, two-sided and unadjusted for multiple comparisons. Q8 uses 4k context; others 8k. Q6 thinking has no v3 counterpart. Seconds per task is the median, matching v2.

## 1. Scores

| Run | All 114 | Single (36) | Parallel (12) | Abstain (12) | Clarify (14) | Multi-step (40) | Seconds per task |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q3_K_M-nothink-wait | 0.75 | 0.86 | 0.67 | 1.00 | 0.29 | 0.78 | 2.2 |
| Q4_K_M-nothink-wait | 0.82 | 0.86 | 0.83 | 1.00 | 0.57 | 0.82 | 2.5 |
| Q5_K_M-nothink-wait | 0.87 | 0.83 | 0.92 | 1.00 | 0.79 | 0.88 | 2.8 |
| Q6_K-nothink-wait | 0.84 | 0.81 | 0.92 | 1.00 | 0.71 | 0.85 | 3.1 |
| Q8_0-nothink-wait | 0.83 | 0.83 | 0.92 | 1.00 | 0.64 | 0.82 | 5.8 |
| Q4_K_M-think-wait | 0.81 | 0.83 | 1.00 | 1.00 | 0.79 | 0.68 | 10.9 |

## 2. Paired changes

Exact binomial sign-test p values; matching uses task IDs.

| v2 → v3 label | Subset | Only v2 right | Only v3 right | p |
| --- | --- | --- | --- | --- |
| Q3_K_M-nothink-wait | all | 4 | 14 | 0.0308837890625 |
| Q3_K_M-nothink-wait | multi_turn | 1 | 13 | 0.0018310546875 |
| Q4_K_M-nothink-wait | all | 3 | 11 | 0.057373046875 (not significant) |
| Q4_K_M-nothink-wait | multi_turn | 1 | 11 | 0.00634765625 |
| Q5_K_M-nothink-wait | all | 3 | 13 | 0.021270751953125 |
| Q5_K_M-nothink-wait | multi_turn | 0 | 12 | 0.00048828125 |
| Q6_K-nothink-wait | all | 2 | 11 | 0.0224609375 |
| Q6_K-nothink-wait | multi_turn | 1 | 10 | 0.01171875 |
| Q8_0-nothink-wait | all | 3 | 9 | 0.14599609375 (not significant) |
| Q8_0-nothink-wait | multi_turn | 1 | 9 | 0.021484375 |
| Q4_K_M-think-wait | all | 5 | 11 | 0.210113525390625 (not significant) |
| Q4_K_M-think-wait | multi_turn | 1 | 11 | 0.00634765625 |

## 3. Batching rate

Denominator: multi-step tasks with at least two expected calls. Numerator: first response has at least two calls (including independent calls).

| Label | v2 | v3 |
| --- | --- | --- |
| Q3_K_M-nothink-wait | 14/31 (45.2%) | 0/31 (0.0%) |
| Q4_K_M-nothink-wait | 10/31 (32.3%) | 0/31 (0.0%) |
| Q5_K_M-nothink-wait | 11/31 (35.5%) | 0/31 (0.0%) |
| Q6_K-nothink-wait | 10/31 (32.3%) | 0/31 (0.0%) |
| Q8_0-nothink-wait | 9/31 (29.0%) | 0/31 (0.0%) |
| Q4_K_M-think-wait | 18/31 (58.1%) | 7/31 (22.6%) |

## 4. Recovered failures

Counts / failed multi-step tasks. Expected calls must match arguments exactly (using task matchers), in order, as a subsequence of all response calls; extra calls remain strict failures. This does not prove correct timing.

| Label | v2 | v3 |
| --- | --- | --- |
| Q3_K_M-nothink-wait | 5/21 | 3/9 |
| Q4_K_M-nothink-wait | 2/17 | 2/7 |
| Q5_K_M-nothink-wait | 4/17 | 2/5 |
| Q6_K-nothink-wait | 3/15 | 2/6 |
| Q8_0-nothink-wait | 3/15 | 3/7 |
| Q4_K_M-think-wait | 2/23 | 2/13 |

## 5. Side effects

Losses and gains are task counts, not percentage points. A drop is supported only when losses exceed gains and p < 0.05.

| Label | Category | v2 correct | v3 correct | Losses | Gains | p |
| --- | --- | --- | --- | --- | --- | --- |
| Q3_K_M-nothink-wait | single | 31 | 31 | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-wait | parallel | 8 | 8 | 1 | 1 | 1.0 (not significant) |
| Q3_K_M-nothink-wait | abstain | 12 | 12 | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-wait | clarify | 6 | 4 | 2 | 0 | 0.5 (not significant) |
| Q4_K_M-nothink-wait | single | 32 | 31 | 1 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-wait | parallel | 10 | 10 | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-wait | abstain | 12 | 12 | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-wait | clarify | 9 | 8 | 1 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-wait | single | 31 | 30 | 2 | 1 | 1.0 (not significant) |
| Q5_K_M-nothink-wait | parallel | 12 | 11 | 1 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-wait | abstain | 12 | 12 | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-wait | clarify | 11 | 11 | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-wait | single | 29 | 29 | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-wait | parallel | 11 | 11 | 1 | 1 | 1.0 (not significant) |
| Q6_K-nothink-wait | abstain | 12 | 12 | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-wait | clarify | 10 | 10 | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-wait | single | 30 | 30 | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-wait | parallel | 12 | 11 | 1 | 0 | 1.0 (not significant) |
| Q8_0-nothink-wait | abstain | 12 | 12 | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-wait | clarify | 10 | 9 | 1 | 0 | 1.0 (not significant) |
| Q4_K_M-think-wait | single | 33 | 30 | 3 | 0 | 0.25 (not significant) |
| Q4_K_M-think-wait | parallel | 12 | 12 | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-wait | abstain | 12 | 12 | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-wait | clarify | 12 | 11 | 1 | 0 | 1.0 (not significant) |

## 6. Precision and improvement

Compare Q3 thinking-off with each higher precision thinking-off on the same multi-step tasks. Delta = v3 success − v2 success. The exact paired sign-flip test swaps the two precision deltas independently per task, conditional on their absolute differences; it assumes exchangeability under the null. No pooling of correlated precision runs or causal precision claim.

| Higher precision | Q3 net gain | Higher net gain | Q3 minus higher (share) | p |
| --- | --- | --- | --- | --- |
| Q4_K_M-nothink | 12/40 | 10/40 | +0.050 | 0.765625 (not significant) |
| Q5_K_M-nothink | 12/40 | 12/40 | +0.000 | 1.0 (not significant) |
| Q6_K-nothink | 12/40 | 9/40 | +0.075 | 0.5625 (not significant) |
| Q8_0-nothink | 12/40 | 8/40 | +0.100 | 0.3984375 (not significant) |

## 7. Findings

The wait instruction improves strict multi-step success in every matched run (all multi-step sign-test p < 0.05), but does not eliminate failures. First-response batching falls to 0/31 at every thinking-off precision, while Q4 thinking retains 7/31 batched responses. Recovered failures remain present in every run and are excluded from headline scores. Single, parallel and clarification scores sometimes drop, but every category comparison is not significant; abstention scores are unchanged. Q3 gains 12/40 multi-step tasks versus 8/40–12/40 at higher precisions, and every paired precision interaction is not significant. Overall gains for Q4 thinking-off, Q8 thinking-off and Q4 thinking-on are not significant.
