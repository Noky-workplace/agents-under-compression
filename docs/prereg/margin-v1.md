# Pre-registration: does the decision margin at Q8 predict which tasks break at Q3?

Written 16 Nov 2026, before any token probability has been recorded for these tasks.
The commit that adds this file is the timestamp.

## Background (already known)
- Tool-calling scores at Q3_K_M are significantly below Q4_K_M and above; Q4–Q8 cannot be separated
  (`research/results/phase2/toolcall-v2-analysis.md`, `toolcall-v3-analysis.md`). The per-task outcomes of every
  precision in v3 are therefore **seen**.
- Perplexity grows smoothly with compression (ratio to the unquantized model: Q8 1.014, Q6 1.017, Q4 1.037,
  Q3 1.097; `research/results/phase2/quality/`), while task success shows a step at Q3.

## Idea
Compression adds small errors to the model's scores for each candidate token. A decision changes only when the
error exceeds the gap between the best and the second-best candidate. Tasks decided by a small gap should fail first.

## Hypothesis
Among tasks that Q8_0 solves, those that Q3_K_M fails have a smaller minimum margin at Q8_0 than those Q3_K_M solves.

## Fixed design
- **Tasks:** the 114 tasks of `inference/toolcall/tasks/finance_v1_wait.jsonl`, thinking off, temperature 0.
- **Margin of a token position:** log-probability of the generated token minus log-probability of the best other
  candidate, at Q8_0. **Margin of a task:** the minimum over all generated positions of all model replies in the
  Q8_0 trajectory (tool-call text and free text; template and end-of-turn tokens included).
- **Outcome:** strict success in `toolcall-v3-wait.jsonl` for `...Q8_0-nothink-wait` and `...Q3_K_M-nothink-wait`.
- **Population:** tasks with strict success at Q8_0. Positives: those failed at Q3_K_M.
- **Statistic:** AUC of (negative margin) for predicting Q3 failure. One-sided permutation test, 10,000 label
  permutations, seed 0.
- **Decision:** *supported* if AUC > 0.5 with p < 0.05. Anything else is *not supported*.
- If fewer than 8 positives exist, report the numbers and call the test *not informative*.

## Reported but not used for the decision
- The same test with Q4_K_M failures as positives (expected: too few to say anything).
- Margin at the first generated position only; mean margin; by category.
- Share of Q3 failures captured by the lowest-margin quarter of tasks.
- Among Q8 failures, whether margins are lower than among Q8 successes (margin as a run-time confidence signal).

## Stop rule
One run. No alternative margin definitions are promoted to the headline after seeing results. If supported, the
follow-up (a separately registered test) is prediction on a new task set.

## Expectation, stated in advance
About 95 tasks pass at Q8 and roughly 15–20 of those fail at Q3, so power is modest. Expected AUC 0.60–0.70.
Batching-type failures are habits rather than close decisions and will dilute the effect.
