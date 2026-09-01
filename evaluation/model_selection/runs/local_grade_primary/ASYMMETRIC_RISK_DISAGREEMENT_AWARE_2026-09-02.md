# Asymmetric risk — DISAGREEMENT-AWARE view (2026-09-02 02:26:16)

`disagreement_aware_weighted_risk_v1`; policy `11e65e79e0f3…`; reference `ce78aed11563…`.

Included clean: **31** (21 agreed @ weight 1.0 + 10 adjacent @ weight 0.5). Excluded: 6 active evidence/source issues, 7 wide disagreements (raw wide = 8; 1 wide case is also issue-flagged and is counted in the issue bucket). Owner-repaired block: 2 (separate, never consensus).

| arm | clean weighted loss | per weight unit | per included case | strict-vs-aware flips |
|---|---|---|---|---|
| baseline_8b_one_pass | 20.0 (of weight 26.0) | 0.7692 | 0.6452 | 1 |
| arm_a_q8_0 | 18.0 (of weight 26.0) | 0.6923 | 0.5806 | 2 |
| arm_b_two_pass | 20.0 (of weight 26.0) | 0.7692 | 0.6452 | 1 |

Strict-vs-aware flips are cases scored as errors against the adjudicated verdict that match one of the two reviewer-supported verdicts (or vice versa) — boundary calls, not clear mistakes.

Wide-disagreement block (production recommendation: REVIEW):

- e002_q1_r6: reviewers ['valid', 'invalid'], adjudicated partially_valid, baseline model valid
- e002_q1_r7: reviewers ['valid', 'invalid'], adjudicated partially_valid, baseline model invalid
- e002_q2_r2: reviewers ['valid', 'invalid'], adjudicated valid, baseline model valid
- e003_q1_r6: reviewers ['valid', 'invalid'], adjudicated valid, baseline model partially_valid
- e003_q2_r8: reviewers ['valid', 'invalid'], adjudicated valid, baseline model valid
- e004_q1_r1: reviewers ['valid', 'invalid'], adjudicated valid, baseline model invalid
- e004_q1_r6: reviewers ['valid', 'invalid'], adjudicated partially_valid, baseline model invalid

Evidence/source-issue block (excluded until resolved):

- e002_q1_r5: flags ['rubric_official_solution'], adjudicated partially_valid, baseline model valid
- e002_q2_r4: flags ['transcription_evidence'], adjudicated partially_valid, baseline model invalid
- e002_q2_r7: flags ['transcription_evidence'], adjudicated valid, baseline model valid
- e004_q1_r5: flags ['rubric_official_solution'], adjudicated partially_valid, baseline model partially_valid
- e004_q2_r3: flags ['transcription_evidence'], adjudicated valid, baseline model valid
- e007_q1_r1: flags ['rubric_official_solution'], adjudicated partially_valid, baseline model partially_valid

Owner-repaired block (separate reference source):

- e004_q2_r6: owner reference valid, baseline model valid (strict loss 0)
- e004_q2_r8: owner reference valid, baseline model valid (strict loss 0)
