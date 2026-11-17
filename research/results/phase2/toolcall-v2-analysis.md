# Tool-calling eval v2: what the 114-task run shows

Source: `toolcall-v2.jsonl` (one run per task, temperature 0, reasoning returned with tool results). Analysis by
Claude from the raw responses, 13 Nov 2026. Q8_0 ran at 4k context, the others at 8k.

## Scores (share of tasks with fully correct calls)

| Run | All 114 | Single (36) | Parallel (12) | Abstain (12) | Clarify (14) | Multi-step (40) | Seconds per task |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q3_K_M, thinking off | 0.67 | 0.86 | 0.67 | 1.00 | 0.43 | 0.47 | 2.2 |
| Q4_K_M, thinking off | 0.75 | 0.89 | 0.83 | 1.00 | 0.64 | 0.57 | 2.5 |
| Q5_K_M, thinking off | 0.78 | 0.86 | 1.00 | 1.00 | 0.79 | 0.57 | 2.8 |
| Q6_K, thinking off | 0.76 | 0.81 | 0.92 | 1.00 | 0.71 | 0.62 | 3.2 |
| Q8_0, thinking off | 0.78 | 0.83 | 1.00 | 1.00 | 0.71 | 0.62 | 5.4 |
| Q4_K_M, thinking on | 0.75 | 0.92 | 1.00 | 1.00 | 0.86 | 0.42 | 11.4 |
| Q6_K, thinking on | 0.81 | 0.94 | 1.00 | 1.00 | 0.79 | 0.57 | 15.2 |

Multi-step scores carry a 95% interval of about ±0.15 (40 tasks). No request timed out.

## Paired comparisons (same tasks, sign test)

| Comparison | Tasks only the first got right | Only the second | p |
|---|---:|---:|---:|
| Q3 vs Q4, thinking off, all tasks | 3 | 13 | 0.02 |
| Q3 vs Q8, thinking off, all tasks | 2 | 15 | 0.002 |
| Q4 vs Q5, thinking off, all tasks | 2 | 5 | 0.45 |
| Q4 vs Q6, thinking off, all tasks | 4 | 5 | 1.00 |
| Q4 vs Q8, thinking off, all tasks | 4 | 7 | 0.55 |
| Q4: thinking on vs off, all tasks | 11 | 11 | 1.00 |
| Q4: thinking on vs off, multi-step | 4 | 10 | 0.18 |
| Q6: thinking on vs off, all tasks | 13 | 8 | 0.38 |

## Findings
1. **Q3_K_M is significantly worse than Q4_K_M and above. Q4 to Q8 cannot be separated** on 114 tasks. This now
   holds on two task sets and in both thinking modes.
2. **The dominant chained-call failure is not a precision effect.** In 60–80% of failed multi-step tasks, at every
   precision including Q8, the model sends the dependent call in the same turn as the first call and guesses the
   argument it could not yet know (for example `lookback_days: 10` when the first tool later returns 14, or the
   literal string `<result_of_previous_conversion>`). Of the 31 tasks with a real dependency, the first turn was
   batched in 9 (Q8) to 14 (Q3) cases with thinking off, and in 13 (Q6) to 18 (Q4) with thinking on.
3. **Thinking does not help chained calls and may hurt** (Q4 multi-step 0.42 with thinking vs 0.57 without; not
   significant). It helps single calls and clarifying questions slightly, at 4–5 times the latency.
4. **The scorer is strict about recovery.** In 2–5 failed tasks per run the model made the right call after seeing
   the result, having first made a wrong one. Counting those as passes moves multi-step scores up by 0.05–0.13 for
   every run and does not change the ranking. For read-only tools the wasted call is harmless; for tools with side
   effects it would not be, so the strict score stays the headline.
5. Seven multi-step tasks fail in every run. Reading the responses, all seven are genuine model errors of the kind
   in finding 2 or failures to copy a returned value exactly; none is a task or scorer fault.

## Next
`gpu_toolcall_v3.sh` repeats the run with one added system-prompt sentence telling the model to wait for a tool
result before making a call that depends on it (`finance_v1_wait.jsonl`). If the batching failure is a prompt
problem, multi-step scores should rise at every precision; if it interacts with precision, Q3 should gain least.

Note: Codex's audit task could not run because its work branch did not contain the result files;
the wrapper now merges the coordination branch before each task. This file replaces that audit.
