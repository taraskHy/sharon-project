# Asymmetric risk — SENSITIVITY grid (2026-09-02 02:26:16)

72 matrices (deterministic full grid; fixed invalid->partially_valid = 3). Winner frequency (ties count for both):

- always_partially_valid: wins/ties best in 16/72
- arm_a_q8_0: wins/ties best in 32/72
- arm_b_two_pass: wins/ties best in 36/72
- baseline_8b_one_pass: wins/ties best in 36/72

A CONSTANT policy is (co-)winner in **16/72** matrices — always_partially_valid, in matrices with adjacent_undergrade in [1] and partially_valid->valid in [5, 7] (full list in JSON).

arm_a beats baseline in 36/72 matrices; the flip is driven entirely by the adjacent-undergrade cost (arm A converts overgrades into adjacent undergrades).

Key pairwise stability (stable = same direction on ALL matrices):

- UNSTABLE baseline_8b_one_pass vs arm_a_q8_0: {'baseline_8b_one_pass_better': 24, 'arm_a_q8_0_better': 36, 'ties': 12, 'stable': False}
- UNSTABLE baseline_8b_one_pass vs always_partially_valid: {'baseline_8b_one_pass_better': 52, 'always_partially_valid_better': 16, 'ties': 4, 'stable': False}
- UNSTABLE arm_a_q8_0 vs arm_b_two_pass: {'arm_a_q8_0_better': 36, 'arm_b_two_pass_better': 24, 'ties': 12, 'stable': False}
- UNSTABLE arm_a_q8_0 vs always_partially_valid: {'arm_a_q8_0_better': 56, 'always_partially_valid_better': 16, 'ties': 0, 'stable': False}
- UNSTABLE arm_b_two_pass vs always_partially_valid: {'arm_b_two_pass_better': 52, 'always_partially_valid_better': 16, 'ties': 4, 'stable': False}
- UNSTABLE always_invalid vs always_valid: {'always_invalid_better': 49, 'always_valid_better': 22, 'ties': 1, 'stable': False}

Stable pairs (9): baseline_8b_one_pass vs arm_b_two_pass; baseline_8b_one_pass vs always_invalid; baseline_8b_one_pass vs always_valid; arm_a_q8_0 vs always_invalid; arm_a_q8_0 vs always_valid; arm_b_two_pass vs always_invalid; arm_b_two_pass vs always_valid; always_invalid vs always_partially_valid; always_partially_valid vs always_valid

Conclusion: the model-vs-model ranking (baseline vs q8_0) is NOT stable under plausible cost perturbations, and no arm robustly separates from `always_partially_valid`. Raw error counts (frozen v1 matrix) are reported alongside so weighted totals never hide count differences.
