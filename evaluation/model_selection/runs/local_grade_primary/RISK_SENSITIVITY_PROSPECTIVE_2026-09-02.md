# Prospective-policy sensitivity (2026-09-02 10:38:40)

72 matrices (same grid as the frozen semantic-layer sensitivity; invalid->partially_valid fixed at 3).

- best deployable policy by TOTAL AUTO risk: {'prospective_valid_only_v1': 72}
- best deployable policy by MEAN AUTO risk: {'prospective_valid_only_v1': 72}
- constant baseline (semantic layer) wins in 16/72 matrices
- deployable risk ordering: **ROBUST** — prospective_valid_only_v1 minimizes both total and mean AUTO weighted risk on every matrix — but ONLY because it refuses every partially_valid verdict; the choice between the deployable policies is a coverage-vs-risk tradeoff, not a dominance result
- model-arm ranking: **FRAGILE** — verified: the baseline-vs-q8_0 ordering flips on the adjacent-undergrade cost (expectation confirmed, not assumed)

Raw severe-event counts are carried per matrix in the JSON, independent of the weights.
