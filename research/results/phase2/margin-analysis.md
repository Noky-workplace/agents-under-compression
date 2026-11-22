# Pre-registered Q8 margin experiment

Decision: **not supported**. Q8-success population: 95; Q3 failures: 11. AUC (negative minimum margin): 0.5995670995670995; one-sided permutation p: 0.15408459154084592 (10,000 permutations, seed 0).

Trajectory call mismatch share: 0.0.

Secondary tables (excluded from the primary decision):

| Analysis | Tasks | Positives | AUC | One-sided p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Q4 failure, minimum margin | 95 | 4 | 0.5879120879120879 | 0.28987101289871015 | not informative |
| Q3 failure, first generated position | 95 | 11 | 0.45346320346320346 | 0.692030796920308 | not supported |
| Q3 failure, mean margin | 95 | 11 | 0.6233766233766234 | 0.09729027097290271 | not supported |
| Q8 failure, minimum margin | 114 | 19 | 0.621606648199446 | 0.04459554044595541 | supported |

Q3 failure by category, among Q8 successes:

| Category | Tasks | Positives | AUC | One-sided p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| abstain | 12 | 0 | None | None | not informative |
| clarify | 9 | 5 | 0.4 | 0.7244275572442755 | not informative |
| multi_turn | 33 | 3 | 0.5222222222222223 | 0.46295370462953706 | not informative |
| parallel | 11 | 3 | 0.6666666666666666 | 0.24317568243175683 | not informative |
| single | 30 | 0 | None | None | not informative |

Lowest-margin quarter: 24 tasks; cutoff 0.19323354959487915; Q3 failures captured 3, share 0.2727272727272727. Tie rule: fractional inclusion at boundary; quarter size rounded up.

Mean task-minimum margin among Q8 successes: 5.146694341398692; among Q8 failures: 1.4449856091033373.

AUC uses half credit for ties. Permutation p uses the add-one correction. Undefined AUC/p values are null in JSON and None in these tables. Fewer than eight positives is not informative, as pre-registered.

One run; no alternative margin definitions promoted to the headline. Any follow-up requires a new task set and separate registration.
