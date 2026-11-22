# Addendum to margin-v1: measurement details fixed after the format probe

Written 18 Nov 2026, after a one-task format probe and before the measured run. No margin of the 114 tasks
has been computed; the hypothesis, population, statistic, decision rule and stop rule are unchanged.

## What the probe showed (`research/results/phase2/margin-probe.json`, task 0 only)
- The installed llama-server returns one probability entry per generated token, including `<|im_end|>`
  (id 151645), but reports the end-of-turn token with empty text. The coverage check now identifies it by id.
- With `--reasoning-budget 0` (the setting of every baseline run), the chat endpoint starts each reply with
  `<think>` and the server then forces `</think>`: the model gave `</think>` log-probability -24.9 against
  -0.00001 for a newline.

## Decision
- Positions of that leading `<think>...</think>` block are recorded but left out of every margin. They are
  imposed by the server, not chosen by the model; including them would make almost every task's minimum
  margin about -25 and the test meaningless.
- Every other generated position counts, as registered: tool-call text, free text, template tokens after the
  block, and the end-of-turn token.
- The chat endpoint stays the first choice because the Q8 and Q3 outcomes in `toolcall-v3-wait.jsonl` were
  produced through it with the same setting. The trajectory-mismatch stop rule (above 5%) still applies.

## Seen before this addendum
Two log-probabilities of task 0's first positions and its last three positions. Nothing that bears on which
tasks fail at Q3.

## Aborted attempt, 17 Nov 2026 20:14–20:16 UTC
The first measured run stopped at task 37 (`finance-037`, the first task with two parallel calls). The
reply-agreement check compared the text between the calls (`"\n"`) with the server's trimmed message
content (`""`) and raised. 36 single-call rows had been written to the partial file.

Decision: the check now ignores surrounding whitespace when comparing free text (every whitespace token is
still covered by the position count and end-of-turn checks). The partial file is kept unchanged as
`margin-q8.aborted-1.partial.jsonl`, never analysed, and the complete run starts again from task 1. This is a
code fault that stopped the run before any analysis, not a second look at the data, so the one-run rule is
read as "one completed run".
