# Local improvement arms — report (2026-09-02 01:34:54)

All-seen numbers are DESCRIPTIVE (seen development data), not independent validation.

| arm | exact | macro-F1 | bal.acc | over | under | AUTO% | gates passed |
|---|---|---|---|---|---|---|---|
| baseline_8b_one_pass | 31/46 = 67.4% | 0.5063 | 0.5062 | 8 | 7 | 95.7% | 1/7 |
| arm_a_q8_0 | 28/46 = 60.9% | 0.5069 | 0.5253 | 6 | 12 | 87.0% | 1/7 |
| arm_b_two_pass | 31/46 = 67.4% | 0.5063 | 0.5062 | 8 | 7 | 78.3% | 0/7 |

Gate columns: exact_agreement_pct >= 85 | macro_f1 >= 0.80 | balanced_accuracy >= 0.80 | harmful_undergrades <= 1/46 | harmful_overgrades <= 3/46 | evidence_or_schema_failure <= 2pct | prospective_AUTO_coverage >= 85pct

| arm | exact_agreement_pct >= 85 | macro_f1 >= 0.80 | balanced_accuracy >= 0.80 | harmful_undergrades <= 1/46 | harmful_overgrades <= 3/46 | evidence_or_schema_failure <= 2pct | prospective_AUTO_coverage >= 85pct |
|---|---|---|---|---|---|---|---|
| baseline_8b_one_pass | fail | fail | fail | fail | fail | fail | PASS |
| arm_a_q8_0 | fail | fail | fail | fail | fail | fail | PASS |
| arm_b_two_pass | fail | fail | fail | fail | fail | fail | fail |
