# Tool-calling evaluation

The default suite is `tasks/finance_v1.jsonl`. The original v0 remains unchanged;
select it with `--tasks inference/toolcall/tasks/finance_v0.jsonl` for historical
comparisons. V1 retains its cases and adds independently written dependent
trajectories, three-call chains, conditional stops, and exact result copying.
The misleading “midnight-boundary” wording in finance-078 is clarified in v1.
Scripted tool outputs are synthetic fixtures, not live market measurements.

Run on the inference host (the Mac agent must not contact it):

```sh
python -m inference.toolcall.run_eval --base-url http://localhost:8080 \
  --label Q4-v1 --out research/results/phase2/toolcall-v1.jsonl
python -m inference.toolcall.report research/results/phase2/toolcall-v1.jsonl \
  --compare Q4-v1 Q6-v1
```

The default request timeout is 300 seconds. `--temperature` defaults to 0 and
`--seeds` defaults to 1. Seeds only affect sampling above temperature 0; three
runs at temperature 0 are repeated observations of one deterministic sample,
not three independent samples. Use a positive temperature and multiple seeds
when sampling variability is the question.

Assistant history is echoed exactly, including content, reasoning_content and
tool_calls, before tool results. `--no-echo-reasoning` removes only
reasoning_content for an ablation. The checked-in runner already echoed the
whole message before this change; omission is not established as the cause of
historical failures. No network documentation lookup was performed.

Use a fresh output file or distinct label for each suite, temperature, timeout
or reasoning setting. Resume keys remain (label, task, seed); do not change
settings while resuming a label. New rows record temperature and echo_reasoning.
Dependent tool calls must occur in separate turns. Extra calls after a stop
condition fail, and argument strings copied from results must match exactly.
The scorer checks trajectories, not the financial correctness of final prose.
No matchers were relaxed without historical evidence.

Reports give every metric with all requests and again excluding timed-out
trajectories. Timeout errors remain failures and timeout_rate is reported
separately (zero on the conditional subset, undefined for an empty subset).
Historical TimeoutError and “timed out” transport errors are recognized.
Each task's repetitions are averaged before task-level bootstrap resampling;
intervals describe variation over tasks, not independent seed rows.

`--compare A B` performs an exact two-sided paired sign test of per-task
args_correct means on shared task IDs. It reports A wins, A losses, ties and
p-value; ties are excluded from the binomial test. The conditional comparison
removes a task/seed from both labels if either timed out. Only compare labels
using identical task definitions, protocol and sampling settings. An empty
intersection has no p-value. This test addresses directional consistency over
tasks, not equivalence or the size of an accuracy difference.

The task's historical `docs/phase2-results.md` and
`research/results/phase2/toolcall.jsonl` were absent from this checkout.
Consequently, response-based attribution of the original twelve cases remains
unverified; see the agent log rather than treating static task inspection as
model failure evidence.
