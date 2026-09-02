# Asymmetric-risk independent reproduction (2026-09-02 02:46:12)

**Verdict: REPRODUCED** — 58 checks, 0 failed. Recomputed from raw artifacts with self-contained logic (no import of scripts/asymmetric_risk.py metric code).

| check | reported | recomputed | pass |
|---|---|---|---|
| reference self-hash verifies | "required" | "ce78aed115633883" | PASS |
| 1. case count | 46 | 46 | PASS |
| 7a. no duplicate ids | "required" | 46 | PASS |
| 2. class distribution | {"invalid": 5, "partially_valid": 13, "valid": 28} | {"invalid": 5, "partially_valid": 13, "valid": 28} | PASS |
| 3. source distribution | {"two_reviewer_consensus": 22, "adjudicated_human_reference" | {"adjudicated_human_reference": 22, "two_reviewer_consensus" | PASS |
| 4. corrected r6/r8 active (reference pointer) | ["corrected_rerun_2026-09-02", "corrected_rerun_2026-09-02"] | ["corrected_rerun_2026-09-02", "corrected_rerun_2026-09-02"] | PASS |
| 5. stale r6/r8 rows preserved in run dirs but not used | "required" | ["e004_q2_r6", "e004_q2_r8"] | PASS |
| 6/7. all arms cover exactly the 46 reference ids | "required" | {"baseline": 46, "arm_a": 46, "arm_b": 46} | PASS |
| 15. no HELD_OUT writer id anywhere | "required" | ["e002", "e003", "e004", "e007"] | PASS |
| policy self-hash verifies + expected prefix | "required" | "11e65e79e0f36cf6" | PASS |
| policy matrix values | {"invalid": {"invalid": 0, "partially_valid": 3, "valid": 12 | {"invalid": {"invalid": 0, "partially_valid": 3, "valid": 12 | PASS |
| strict artifact self-hash verifies | "required" | "34770cc7a772b283" | PASS |
| 8. confusion cells baseline_8b_one_pass | {"invalid->partially_valid": 4, "invalid->valid": 0, "partia | {"invalid->partially_valid": 4, "invalid->valid": 0, "partia | PASS |
| 9. strict total loss baseline_8b_one_pass | 43 | 43 | PASS |
| 9b. strict total loss baseline_8b_one_pass (mission expectation) | 43 | 43 | PASS |
| exact agreement baseline_8b_one_pass | 31 | 31 | PASS |
| 8. confusion cells arm_a_q8_0 | {"invalid->partially_valid": 3, "invalid->valid": 0, "partia | {"invalid->partially_valid": 3, "invalid->valid": 0, "partia | PASS |
| 9. strict total loss arm_a_q8_0 | 40 | 40 | PASS |
| 9b. strict total loss arm_a_q8_0 (mission expectation) | 40 | 40 | PASS |
| exact agreement arm_a_q8_0 | 28 | 28 | PASS |
| 8. confusion cells arm_b_two_pass | {"invalid->partially_valid": 4, "invalid->valid": 0, "partia | {"invalid->partially_valid": 4, "invalid->valid": 0, "partia | PASS |
| 9. strict total loss arm_b_two_pass | 43 | 43 | PASS |
| 9b. strict total loss arm_b_two_pass (mission expectation) | 43 | 43 | PASS |
| exact agreement arm_b_two_pass | 31 | 31 | PASS |
| baseline mean weighted loss | 0.9348 | 0.9348 | PASS |
| baseline invalid->valid / partial->valid | [0, 4] | [0, 4] | PASS |
| baseline overgrade/undergrade cost | [32, 11] | [32, 11] | PASS |
| 11. constant always_invalid | 97 | 97 | PASS |
| 11b. constant always_invalid (mission expectation) | 97 | 97 | PASS |
| 11. constant always_partially_valid | 43 | 43 | PASS |
| 11b. constant always_partially_valid (mission expectation) | 43 | 43 | PASS |
| 11. constant always_valid | 125 | 125 | PASS |
| 11b. constant always_valid (mission expectation) | 125 | 125 | PASS |
| disagreement artifact self-hash verifies | "required" | "0a9ef6a7080ae9be" | PASS |
| 10. disagreement-aware clean loss baseline_8b_one_pass | 20.0 | 20.0 | PASS |
| 10. disagreement-aware clean loss arm_a_q8_0 | 18.0 | 18.0 | PASS |
| 10. disagreement-aware clean loss arm_b_two_pass | 20.0 | 20.0 | PASS |
| 10b. disagreement-aware included cases | 31 | 31 | PASS |
| sensitivity artifact self-hash verifies | "required" | "3a0d8ac0243354ab" | PASS |
| 12. sensitivity: all 72 matrices' totals reproduce | "required" | 72 | PASS |
| 12b. sensitivity winner frequency | {"always_partially_valid": 16, "arm_a_q8_0": 32, "arm_b_two_ | {"always_partially_valid": 16, "arm_a_q8_0": 32, "arm_b_two_ | PASS |
| 12c. sensitivity grid complete (4x3x3x2) | "required" | 72 | PASS |
| replay artifact self-hash verifies | "required" | "db7aa08f48da9464" | PASS |
| 13. replay AUTO_ALL (artifact) | {"auto": 44, "risk": 42, "cov": 95.7} | {"auto": 44, "risk": 42, "cov": 95.7} | PASS |
| 13b. replay AUTO_ALL (mission expectation) | {"cov": 95.7, "risk": 42, "pv_val": 4, "under": 6} | {"cov": 95.7, "risk": 42, "pv_val": 4, "under": 6} | PASS |
| 13. replay AUTO_VALID_ONLY (artifact) | {"auto": 25, "risk": 15, "cov": 54.3} | {"auto": 25, "risk": 15, "cov": 54.3} | PASS |
| 13b. replay AUTO_VALID_ONLY (mission expectation) | {"cov": 54.3, "risk": 15, "pv_val": 3, "under": 0, "review": | {"cov": 54.3, "risk": 15, "pv_val": 3, "under": 0, "review": | PASS |
| 13. replay AUTO_VALID_AND_PARTIAL (artifact) | {"auto": 35, "risk": 29, "cov": 76.1} | {"auto": 35, "risk": 29, "cov": 76.1} | PASS |
| 13b. replay AUTO_VALID_AND_PARTIAL (mission expectation) | {"cov": 76.1, "risk": 29, "pv_val": 3, "under": 2, "review": | {"cov": 76.1, "risk": 29, "pv_val": 3, "under": 2, "review": | PASS |
| 13. replay HUMAN_DISPUTE_AWARE_B (artifact) | {"auto": 22, "risk": 10, "cov": 47.8} | {"auto": 22, "risk": 10, "cov": 47.8} | PASS |
| 13b. replay HUMAN_DISPUTE_AWARE_B (mission expectation) | {"cov": 47.8, "risk": 10} | {"cov": 47.8, "risk": 10} | PASS |
| 13. replay HUMAN_DISPUTE_AWARE_C (artifact) | {"auto": 31, "risk": 23, "cov": 67.4} | {"auto": 31, "risk": 23, "cov": 67.4} | PASS |
| 13b. replay HUMAN_DISPUTE_AWARE_C (mission expectation) | {"cov": 67.4, "risk": 23} | {"cov": 67.4, "risk": 23} | PASS |
| 14. gate HARD_FALSE_FULL | "PASS" | "PASS" | PASS |
| 14. gate WEIGHTED_RISK | "FAIL" | "FAIL" | PASS |
| 14. gate GROUNDING | "FAIL" | "FAIL" | PASS |
| 14b. gates passed count | "5/9" | "5/9" | PASS |
| cross-check exact agreement vs improvement report | {"baseline": 31, "arm_a": 28, "arm_b": 31} | {"baseline": 31, "arm_a": 28, "arm_b": 31} | PASS |
