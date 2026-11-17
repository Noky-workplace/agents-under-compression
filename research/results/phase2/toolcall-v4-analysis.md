# Tool-calling v4: harness enforcement

Sources: `toolcall-v3-wait.jsonl`, `toolcall-v4-enforce.jsonl`, and `finance_v1_wait.jsonl`. One observation per task. Exact two-sided sign tests exclude ties; exploratory, unadjusted p values. Strict scoring includes every attempted call. Executed scoring excludes rejected calls and measures only whether the executed trajectory matches; cap hits, completion, and transport failures are recorded separately. False rejections match a complete expected call (any expected position); counts are rejection events, so multiple bad arguments may count separately. Model turns count completion requests in enforced runs (including failed requests), and returned responses in v3; medians include failed tasks.

| Run | Subset | N | v3 strict | v4 strict | Success after enforcement | v3 Median seconds | v4 Median seconds | v3 Median turns | v4 Median turns | Cap hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q3_K_M-nothink-enforce | all | 114 | 0.754 | 0.754 | 0.684 | 2.20 | 2.22 | 1.0 | 1.0 | 0 |
| Q3_K_M-nothink-enforce | single | 36 | 0.861 | 0.861 | 0.861 | 1.20 | 1.21 | 1.0 | 1.0 | 0 |
| Q3_K_M-nothink-enforce | parallel | 12 | 0.667 | 0.667 | 0.000 | 2.40 | 2.42 | 1.0 | 1.0 | 0 |
| Q3_K_M-nothink-enforce | abstain | 12 | 1.000 | 1.000 | 1.000 | 2.50 | 2.51 | 1.0 | 1.0 | 0 |
| Q3_K_M-nothink-enforce | clarify | 14 | 0.286 | 0.286 | 0.286 | 1.26 | 1.27 | 1.0 | 1.0 | 0 |
| Q3_K_M-nothink-enforce | multi_turn | 40 | 0.775 | 0.775 | 0.775 | 4.02 | 4.04 | 3.0 | 3.0 | 0 |
| Q4_K_M-nothink-enforce | all | 114 | 0.825 | 0.825 | 0.737 | 2.50 | 2.51 | 1.0 | 1.0 | 0 |
| Q4_K_M-nothink-enforce | single | 36 | 0.861 | 0.861 | 0.861 | 1.40 | 1.42 | 1.0 | 1.0 | 0 |
| Q4_K_M-nothink-enforce | parallel | 12 | 0.833 | 0.833 | 0.000 | 2.77 | 2.78 | 1.0 | 1.0 | 0 |
| Q4_K_M-nothink-enforce | abstain | 12 | 1.000 | 1.000 | 1.000 | 1.68 | 1.65 | 1.0 | 1.0 | 0 |
| Q4_K_M-nothink-enforce | clarify | 14 | 0.571 | 0.571 | 0.571 | 0.99 | 0.99 | 1.0 | 1.0 | 0 |
| Q4_K_M-nothink-enforce | multi_turn | 40 | 0.825 | 0.825 | 0.825 | 4.77 | 4.73 | 3.0 | 3.0 | 0 |
| Q5_K_M-nothink-enforce | all | 114 | 0.868 | 0.868 | 0.772 | 2.82 | 2.85 | 1.0 | 1.0 | 0 |
| Q5_K_M-nothink-enforce | single | 36 | 0.833 | 0.833 | 0.833 | 1.61 | 1.59 | 1.0 | 1.0 | 0 |
| Q5_K_M-nothink-enforce | parallel | 12 | 0.917 | 0.917 | 0.000 | 3.12 | 3.14 | 1.0 | 1.0 | 0 |
| Q5_K_M-nothink-enforce | abstain | 12 | 1.000 | 1.000 | 1.000 | 2.59 | 2.60 | 1.0 | 1.0 | 0 |
| Q5_K_M-nothink-enforce | clarify | 14 | 0.786 | 0.786 | 0.786 | 0.88 | 0.89 | 1.0 | 1.0 | 0 |
| Q5_K_M-nothink-enforce | multi_turn | 40 | 0.875 | 0.875 | 0.875 | 5.17 | 5.19 | 3.0 | 3.0 | 0 |
| Q6_K-nothink-enforce | all | 114 | 0.842 | 0.842 | 0.746 | 3.14 | 3.17 | 1.0 | 1.0 | 0 |
| Q6_K-nothink-enforce | single | 36 | 0.806 | 0.806 | 0.806 | 1.79 | 1.79 | 1.0 | 1.0 | 0 |
| Q6_K-nothink-enforce | parallel | 12 | 0.917 | 0.917 | 0.000 | 3.50 | 3.49 | 1.0 | 1.0 | 0 |
| Q6_K-nothink-enforce | abstain | 12 | 1.000 | 1.000 | 1.000 | 3.06 | 3.10 | 1.0 | 1.0 | 0 |
| Q6_K-nothink-enforce | clarify | 14 | 0.714 | 0.714 | 0.714 | 1.06 | 1.04 | 1.0 | 1.0 | 0 |
| Q6_K-nothink-enforce | multi_turn | 40 | 0.850 | 0.850 | 0.850 | 5.92 | 5.98 | 3.0 | 3.0 | 0 |
| Q8_0-nothink-enforce | all | 114 | 0.833 | 0.833 | 0.737 | 5.75 | 5.85 | 1.0 | 1.0 | 0 |
| Q8_0-nothink-enforce | single | 36 | 0.833 | 0.833 | 0.833 | 3.13 | 3.20 | 1.0 | 1.0 | 0 |
| Q8_0-nothink-enforce | parallel | 12 | 0.917 | 0.917 | 0.000 | 6.02 | 6.16 | 1.0 | 1.0 | 0 |
| Q8_0-nothink-enforce | abstain | 12 | 1.000 | 1.000 | 1.000 | 5.54 | 5.73 | 1.0 | 1.0 | 0 |
| Q8_0-nothink-enforce | clarify | 14 | 0.643 | 0.643 | 0.643 | 2.00 | 2.04 | 1.0 | 1.0 | 0 |
| Q8_0-nothink-enforce | multi_turn | 40 | 0.825 | 0.825 | 0.825 | 11.40 | 11.66 | 3.0 | 3.0 | 0 |
| Q4_K_M-think-enforce | all | 114 | 0.807 | 0.807 | 0.754 | 10.95 | 11.19 | 1.0 | 1.0 | 0 |
| Q4_K_M-think-enforce | single | 36 | 0.833 | 0.833 | 0.833 | 7.25 | 7.23 | 1.0 | 1.0 | 0 |
| Q4_K_M-think-enforce | parallel | 12 | 1.000 | 1.000 | 0.000 | 11.45 | 11.57 | 1.0 | 1.0 | 0 |
| Q4_K_M-think-enforce | abstain | 12 | 1.000 | 1.000 | 1.000 | 9.36 | 9.48 | 1.0 | 1.0 | 0 |
| Q4_K_M-think-enforce | clarify | 14 | 0.786 | 0.786 | 0.786 | 5.00 | 5.03 | 1.0 | 1.0 | 0 |
| Q4_K_M-think-enforce | multi_turn | 40 | 0.675 | 0.675 | 0.825 | 32.06 | 39.38 | 3.0 | 3.0 | 0 |
| Q4_K_M-nothink-prov | all | 114 | 0.825 | 0.596 | 0.211 | 2.50 | 2.54 | 1.0 | 1.0 | 17 |
| Q4_K_M-nothink-prov | single | 36 | 0.861 | 0.861 | 0.083 | 1.40 | 1.42 | 1.0 | 1.0 | 0 |
| Q4_K_M-nothink-prov | parallel | 12 | 0.833 | 0.833 | 0.000 | 2.77 | 2.78 | 1.0 | 1.0 | 0 |
| Q4_K_M-nothink-prov | abstain | 12 | 1.000 | 1.000 | 1.000 | 1.68 | 1.65 | 1.0 | 1.0 | 0 |
| Q4_K_M-nothink-prov | clarify | 14 | 0.571 | 0.571 | 0.571 | 0.99 | 1.00 | 1.0 | 1.0 | 0 |
| Q4_K_M-nothink-prov | multi_turn | 40 | 0.825 | 0.175 | 0.025 | 4.77 | 11.58 | 3.0 | 5.5 | 17 |

| Run | Subset | Metric | Only v3 right | Only v4 right | p |
| --- | --- | --- | --- | --- | --- |
| Q3_K_M-nothink-enforce | all | strict | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-enforce | all | executed | 8 | 0 | 0.0078125 |
| Q3_K_M-nothink-enforce | single | strict | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-enforce | single | executed | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-enforce | parallel | strict | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-enforce | parallel | executed | 8 | 0 | 0.0078125 |
| Q3_K_M-nothink-enforce | abstain | strict | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-enforce | abstain | executed | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-enforce | clarify | strict | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-enforce | clarify | executed | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-enforce | multi_turn | strict | 0 | 0 | 1.0 (not significant) |
| Q3_K_M-nothink-enforce | multi_turn | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-enforce | all | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-enforce | all | executed | 10 | 0 | 0.001953125 |
| Q4_K_M-nothink-enforce | single | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-enforce | single | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-enforce | parallel | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-enforce | parallel | executed | 10 | 0 | 0.001953125 |
| Q4_K_M-nothink-enforce | abstain | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-enforce | abstain | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-enforce | clarify | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-enforce | clarify | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-enforce | multi_turn | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-enforce | multi_turn | executed | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-enforce | all | strict | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-enforce | all | executed | 11 | 0 | 0.0009765625 |
| Q5_K_M-nothink-enforce | single | strict | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-enforce | single | executed | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-enforce | parallel | strict | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-enforce | parallel | executed | 11 | 0 | 0.0009765625 |
| Q5_K_M-nothink-enforce | abstain | strict | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-enforce | abstain | executed | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-enforce | clarify | strict | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-enforce | clarify | executed | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-enforce | multi_turn | strict | 0 | 0 | 1.0 (not significant) |
| Q5_K_M-nothink-enforce | multi_turn | executed | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-enforce | all | strict | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-enforce | all | executed | 11 | 0 | 0.0009765625 |
| Q6_K-nothink-enforce | single | strict | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-enforce | single | executed | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-enforce | parallel | strict | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-enforce | parallel | executed | 11 | 0 | 0.0009765625 |
| Q6_K-nothink-enforce | abstain | strict | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-enforce | abstain | executed | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-enforce | clarify | strict | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-enforce | clarify | executed | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-enforce | multi_turn | strict | 0 | 0 | 1.0 (not significant) |
| Q6_K-nothink-enforce | multi_turn | executed | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-enforce | all | strict | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-enforce | all | executed | 11 | 0 | 0.0009765625 |
| Q8_0-nothink-enforce | single | strict | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-enforce | single | executed | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-enforce | parallel | strict | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-enforce | parallel | executed | 11 | 0 | 0.0009765625 |
| Q8_0-nothink-enforce | abstain | strict | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-enforce | abstain | executed | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-enforce | clarify | strict | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-enforce | clarify | executed | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-enforce | multi_turn | strict | 0 | 0 | 1.0 (not significant) |
| Q8_0-nothink-enforce | multi_turn | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-enforce | all | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-enforce | all | executed | 12 | 6 | 0.237884521484375 (not significant) |
| Q4_K_M-think-enforce | single | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-enforce | single | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-enforce | parallel | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-enforce | parallel | executed | 12 | 0 | 0.00048828125 |
| Q4_K_M-think-enforce | abstain | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-enforce | abstain | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-enforce | clarify | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-enforce | clarify | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-enforce | multi_turn | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-think-enforce | multi_turn | executed | 0 | 6 | 0.03125 |
| Q4_K_M-nothink-prov | all | strict | 27 | 1 | 2.1606683731079102e-07 |
| Q4_K_M-nothink-prov | all | executed | 70 | 0 | 1.6940658945086007e-21 |
| Q4_K_M-nothink-prov | single | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov | single | executed | 28 | 0 | 7.450580596923828e-09 |
| Q4_K_M-nothink-prov | parallel | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov | parallel | executed | 10 | 0 | 0.001953125 |
| Q4_K_M-nothink-prov | abstain | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov | abstain | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov | clarify | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov | clarify | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov | multi_turn | strict | 27 | 1 | 2.1606683731079102e-07 |
| Q4_K_M-nothink-prov | multi_turn | executed | 32 | 0 | 4.656612873077393e-10 |
| Q4_K_M-nothink-prov vs enforce | single | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov vs enforce | single | executed | 28 | 0 | 7.450580596923828e-09 |
| Q4_K_M-nothink-prov vs enforce | parallel | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov vs enforce | parallel | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov vs enforce | abstain | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov vs enforce | abstain | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov vs enforce | clarify | strict | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov vs enforce | clarify | executed | 0 | 0 | 1.0 (not significant) |
| Q4_K_M-nothink-prov vs enforce | multi_turn | strict | 27 | 1 | 2.1606683731079102e-07 |
| Q4_K_M-nothink-prov vs enforce | multi_turn | executed | 32 | 0 | 4.656612873077393e-10 |

| Run | Rule | Category | Rejections | False rejections |
| --- | --- | --- | --- | --- |
| Q3_K_M-nothink-enforce | one_call | single | 0 | 0 |
| Q3_K_M-nothink-enforce | one_call | parallel | 12 | 9 |
| Q3_K_M-nothink-enforce | one_call | abstain | 0 | 0 |
| Q3_K_M-nothink-enforce | one_call | clarify | 0 | 0 |
| Q3_K_M-nothink-enforce | one_call | multi_turn | 0 | 0 |
| Q3_K_M-nothink-enforce | one_call | all | 12 | 9 |
| Q3_K_M-nothink-enforce | reject_placeholders | single | 0 | 0 |
| Q3_K_M-nothink-enforce | reject_placeholders | parallel | 0 | 0 |
| Q3_K_M-nothink-enforce | reject_placeholders | abstain | 0 | 0 |
| Q3_K_M-nothink-enforce | reject_placeholders | clarify | 0 | 0 |
| Q3_K_M-nothink-enforce | reject_placeholders | multi_turn | 0 | 0 |
| Q3_K_M-nothink-enforce | reject_placeholders | all | 0 | 0 |
| Q3_K_M-nothink-enforce | provenance | single | 0 | 0 |
| Q3_K_M-nothink-enforce | provenance | parallel | 0 | 0 |
| Q3_K_M-nothink-enforce | provenance | abstain | 0 | 0 |
| Q3_K_M-nothink-enforce | provenance | clarify | 0 | 0 |
| Q3_K_M-nothink-enforce | provenance | multi_turn | 0 | 0 |
| Q3_K_M-nothink-enforce | provenance | all | 0 | 0 |
| Q4_K_M-nothink-enforce | one_call | single | 0 | 0 |
| Q4_K_M-nothink-enforce | one_call | parallel | 12 | 10 |
| Q4_K_M-nothink-enforce | one_call | abstain | 0 | 0 |
| Q4_K_M-nothink-enforce | one_call | clarify | 2 | 0 |
| Q4_K_M-nothink-enforce | one_call | multi_turn | 0 | 0 |
| Q4_K_M-nothink-enforce | one_call | all | 14 | 10 |
| Q4_K_M-nothink-enforce | reject_placeholders | single | 0 | 0 |
| Q4_K_M-nothink-enforce | reject_placeholders | parallel | 0 | 0 |
| Q4_K_M-nothink-enforce | reject_placeholders | abstain | 0 | 0 |
| Q4_K_M-nothink-enforce | reject_placeholders | clarify | 0 | 0 |
| Q4_K_M-nothink-enforce | reject_placeholders | multi_turn | 0 | 0 |
| Q4_K_M-nothink-enforce | reject_placeholders | all | 0 | 0 |
| Q4_K_M-nothink-enforce | provenance | single | 0 | 0 |
| Q4_K_M-nothink-enforce | provenance | parallel | 0 | 0 |
| Q4_K_M-nothink-enforce | provenance | abstain | 0 | 0 |
| Q4_K_M-nothink-enforce | provenance | clarify | 0 | 0 |
| Q4_K_M-nothink-enforce | provenance | multi_turn | 0 | 0 |
| Q4_K_M-nothink-enforce | provenance | all | 0 | 0 |
| Q5_K_M-nothink-enforce | one_call | single | 0 | 0 |
| Q5_K_M-nothink-enforce | one_call | parallel | 12 | 11 |
| Q5_K_M-nothink-enforce | one_call | abstain | 0 | 0 |
| Q5_K_M-nothink-enforce | one_call | clarify | 0 | 0 |
| Q5_K_M-nothink-enforce | one_call | multi_turn | 0 | 0 |
| Q5_K_M-nothink-enforce | one_call | all | 12 | 11 |
| Q5_K_M-nothink-enforce | reject_placeholders | single | 0 | 0 |
| Q5_K_M-nothink-enforce | reject_placeholders | parallel | 0 | 0 |
| Q5_K_M-nothink-enforce | reject_placeholders | abstain | 0 | 0 |
| Q5_K_M-nothink-enforce | reject_placeholders | clarify | 0 | 0 |
| Q5_K_M-nothink-enforce | reject_placeholders | multi_turn | 0 | 0 |
| Q5_K_M-nothink-enforce | reject_placeholders | all | 0 | 0 |
| Q5_K_M-nothink-enforce | provenance | single | 0 | 0 |
| Q5_K_M-nothink-enforce | provenance | parallel | 0 | 0 |
| Q5_K_M-nothink-enforce | provenance | abstain | 0 | 0 |
| Q5_K_M-nothink-enforce | provenance | clarify | 0 | 0 |
| Q5_K_M-nothink-enforce | provenance | multi_turn | 0 | 0 |
| Q5_K_M-nothink-enforce | provenance | all | 0 | 0 |
| Q6_K-nothink-enforce | one_call | single | 0 | 0 |
| Q6_K-nothink-enforce | one_call | parallel | 12 | 11 |
| Q6_K-nothink-enforce | one_call | abstain | 0 | 0 |
| Q6_K-nothink-enforce | one_call | clarify | 0 | 0 |
| Q6_K-nothink-enforce | one_call | multi_turn | 0 | 0 |
| Q6_K-nothink-enforce | one_call | all | 12 | 11 |
| Q6_K-nothink-enforce | reject_placeholders | single | 0 | 0 |
| Q6_K-nothink-enforce | reject_placeholders | parallel | 0 | 0 |
| Q6_K-nothink-enforce | reject_placeholders | abstain | 0 | 0 |
| Q6_K-nothink-enforce | reject_placeholders | clarify | 0 | 0 |
| Q6_K-nothink-enforce | reject_placeholders | multi_turn | 0 | 0 |
| Q6_K-nothink-enforce | reject_placeholders | all | 0 | 0 |
| Q6_K-nothink-enforce | provenance | single | 0 | 0 |
| Q6_K-nothink-enforce | provenance | parallel | 0 | 0 |
| Q6_K-nothink-enforce | provenance | abstain | 0 | 0 |
| Q6_K-nothink-enforce | provenance | clarify | 0 | 0 |
| Q6_K-nothink-enforce | provenance | multi_turn | 0 | 0 |
| Q6_K-nothink-enforce | provenance | all | 0 | 0 |
| Q8_0-nothink-enforce | one_call | single | 0 | 0 |
| Q8_0-nothink-enforce | one_call | parallel | 12 | 11 |
| Q8_0-nothink-enforce | one_call | abstain | 0 | 0 |
| Q8_0-nothink-enforce | one_call | clarify | 2 | 0 |
| Q8_0-nothink-enforce | one_call | multi_turn | 0 | 0 |
| Q8_0-nothink-enforce | one_call | all | 14 | 11 |
| Q8_0-nothink-enforce | reject_placeholders | single | 0 | 0 |
| Q8_0-nothink-enforce | reject_placeholders | parallel | 0 | 0 |
| Q8_0-nothink-enforce | reject_placeholders | abstain | 0 | 0 |
| Q8_0-nothink-enforce | reject_placeholders | clarify | 0 | 0 |
| Q8_0-nothink-enforce | reject_placeholders | multi_turn | 0 | 0 |
| Q8_0-nothink-enforce | reject_placeholders | all | 0 | 0 |
| Q8_0-nothink-enforce | provenance | single | 0 | 0 |
| Q8_0-nothink-enforce | provenance | parallel | 0 | 0 |
| Q8_0-nothink-enforce | provenance | abstain | 0 | 0 |
| Q8_0-nothink-enforce | provenance | clarify | 0 | 0 |
| Q8_0-nothink-enforce | provenance | multi_turn | 0 | 0 |
| Q8_0-nothink-enforce | provenance | all | 0 | 0 |
| Q4_K_M-think-enforce | one_call | single | 1 | 0 |
| Q4_K_M-think-enforce | one_call | parallel | 12 | 12 |
| Q4_K_M-think-enforce | one_call | abstain | 0 | 0 |
| Q4_K_M-think-enforce | one_call | clarify | 1 | 0 |
| Q4_K_M-think-enforce | one_call | multi_turn | 12 | 3 |
| Q4_K_M-think-enforce | one_call | all | 26 | 15 |
| Q4_K_M-think-enforce | reject_placeholders | single | 0 | 0 |
| Q4_K_M-think-enforce | reject_placeholders | parallel | 0 | 0 |
| Q4_K_M-think-enforce | reject_placeholders | abstain | 0 | 0 |
| Q4_K_M-think-enforce | reject_placeholders | clarify | 0 | 0 |
| Q4_K_M-think-enforce | reject_placeholders | multi_turn | 0 | 0 |
| Q4_K_M-think-enforce | reject_placeholders | all | 0 | 0 |
| Q4_K_M-think-enforce | provenance | single | 0 | 0 |
| Q4_K_M-think-enforce | provenance | parallel | 0 | 0 |
| Q4_K_M-think-enforce | provenance | abstain | 0 | 0 |
| Q4_K_M-think-enforce | provenance | clarify | 0 | 0 |
| Q4_K_M-think-enforce | provenance | multi_turn | 0 | 0 |
| Q4_K_M-think-enforce | provenance | all | 0 | 0 |
| Q4_K_M-nothink-prov | one_call | single | 0 | 0 |
| Q4_K_M-nothink-prov | one_call | parallel | 12 | 10 |
| Q4_K_M-nothink-prov | one_call | abstain | 0 | 0 |
| Q4_K_M-nothink-prov | one_call | clarify | 2 | 0 |
| Q4_K_M-nothink-prov | one_call | multi_turn | 0 | 0 |
| Q4_K_M-nothink-prov | one_call | all | 14 | 10 |
| Q4_K_M-nothink-prov | reject_placeholders | single | 0 | 0 |
| Q4_K_M-nothink-prov | reject_placeholders | parallel | 0 | 0 |
| Q4_K_M-nothink-prov | reject_placeholders | abstain | 0 | 0 |
| Q4_K_M-nothink-prov | reject_placeholders | clarify | 0 | 0 |
| Q4_K_M-nothink-prov | reject_placeholders | multi_turn | 0 | 0 |
| Q4_K_M-nothink-prov | reject_placeholders | all | 0 | 0 |
| Q4_K_M-nothink-prov | provenance | single | 61 | 53 |
| Q4_K_M-nothink-prov | provenance | parallel | 24 | 22 |
| Q4_K_M-nothink-prov | provenance | abstain | 0 | 0 |
| Q4_K_M-nothink-prov | provenance | clarify | 13 | 0 |
| Q4_K_M-nothink-prov | provenance | multi_turn | 381 | 250 |
| Q4_K_M-nothink-prov | provenance | all | 479 | 325 |

Q3_K_M-nothink-enforce does not change single scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q3_K_M-nothink-enforce does not change single scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q3_K_M-nothink-enforce does not change parallel scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q3_K_M-nothink-enforce lowers parallel scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-nothink-enforce does not change single scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-nothink-enforce does not change single scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-nothink-enforce does not change parallel scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-nothink-enforce lowers parallel scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q5_K_M-nothink-enforce does not change single scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q5_K_M-nothink-enforce does not change single scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q5_K_M-nothink-enforce does not change parallel scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q5_K_M-nothink-enforce lowers parallel scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q6_K-nothink-enforce does not change single scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q6_K-nothink-enforce does not change single scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q6_K-nothink-enforce does not change parallel scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q6_K-nothink-enforce lowers parallel scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q8_0-nothink-enforce does not change single scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q8_0-nothink-enforce does not change single scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q8_0-nothink-enforce does not change parallel scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q8_0-nothink-enforce lowers parallel scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-think-enforce does not change single scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-think-enforce does not change single scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-think-enforce does not change parallel scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-think-enforce lowers parallel scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-nothink-prov does not change single scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-nothink-prov lowers single scores (executed); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-nothink-prov does not change parallel scores (strict); enforced run vs v3; differences include the harness limits listed separately. Q4_K_M-nothink-prov lowers parallel scores (executed); enforced run vs v3; differences include the harness limits listed separately. Adding provenance does not change single scores (strict) for Q4_K_M-nothink-prov; paired p=1.0 (not significant). Adding provenance lowers single scores (executed) for Q4_K_M-nothink-prov; paired p=7.450580596923828e-09. Adding provenance does not change parallel scores (strict) for Q4_K_M-nothink-prov; paired p=1.0 (not significant). Adding provenance does not change parallel scores (executed) for Q4_K_M-nothink-prov; paired p=1.0 (not significant).

Remaining harness differences (enforced run vs v3): dependent tasks have an eight-reply cap instead of expected-call count + 2; rejected calls receive error results and do not consume fixture ordinals. Single/parallel tasks stop after the first reply in both runners; accepted dependent calls use ordinal fixture dispatch in both. Executed success is an additional trajectory metric, independent of transport failure or completion. Enforced turn counts include failed completion requests; v3 counts returned responses.
