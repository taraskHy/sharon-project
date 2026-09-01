# Asymmetric risk — STRICT view (2026-09-02 02:26:16)

Policy `asymmetric_grading_risk_v1` `11e65e79e0f3…`; reference `ce78aed11563…`; 46 seen cases; descriptive development numbers, NOT independent validation.

| arm | total risk | mean | norm. | inv->val | pv->val | inv->pv | over-loss | under-loss | exact | macro-F1 | AUTO% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_8b_one_pass | 43 | 0.9348 | 0.0779 | 0 | 4 | 4 | 32 | 11 | 31/46 | 0.5063 | 95.7% |
| arm_a_q8_0 | 40 | 0.8696 | 0.0725 | 0 | 3 | 3 | 24 | 16 | 28/46 | 0.5069 | 87.0% |
| arm_b_two_pass | 43 | 0.9348 | 0.0779 | 0 | 4 | 4 | 32 | 11 | 31/46 | 0.5063 | 78.3% |

Constant baselines (same 46-case denominator):

| policy | total risk | mean | inv->val | pv->val | over-loss | under-loss | exact |
|---|---|---|---|---|---|---|---|
| always_invalid | 97 | 2.1087 | 0 | 0 | 0 | 97 | 5/46 |
| always_partially_valid | 43 | 0.9348 | 0 | 0 | 15 | 28 | 13/46 |
| always_valid | 125 | 2.7174 | 5 | 13 | 125 | 0 | 28/46 |

**Best constant policy: `always_partially_valid` at total risk 43.**
- baseline_8b_one_pass: 43 — does NOT beat it (margin 0).
- arm_a_q8_0: 40 — beats it (margin 3).
- arm_b_two_pass: 43 — does NOT beat it (margin 0).

No arm produced invalid->valid (the catastrophic cell) on seen data; the weighted comparison is therefore driven by the smaller overgrade cells and the undergrade profile. The invalid class has only 5 seen cases — low statistical power, seen-data only.
