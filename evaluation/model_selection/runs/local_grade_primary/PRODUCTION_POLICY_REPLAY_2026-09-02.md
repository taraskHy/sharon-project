# Production policy replay (2026-09-02 02:26:16)

explanation-case automation ONLY: this 46-case reference contains no OCR-quality or deterministic-MC population, so NO full-exam automation claim is made.

## baseline_8b_one_pass

| policy | AUTO | cov% | REVIEW | AUTO prec% | AUTO risk | mean | false-full | pv->val | inv->pv | auto under | review/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUTO_ALL | 44 | 95.7 | 2 | 68.2 | 42 | 0.9545 | 0 | 4 | 4 | 6 | 4.3 |
| AUTO_VALID_ONLY | 25 | 54.3 | 21 | 88.0 | 15 | 0.6 | 0 | 3 | 0 | 0 | 45.7 |
| AUTO_VALID_AND_PARTIAL | 35 | 76.1 | 11 | 74.3 | 29 | 0.8286 | 0 | 3 | 4 | 2 | 23.9 |
| HUMAN_DISPUTE_AWARE_B | 22 | 47.8 | 24 | 90.9 | 10 | 0.4545 | 0 | 2 | 0 | 0 | 52.2 |
| HUMAN_DISPUTE_AWARE_C | 31 | 67.4 | 15 | 77.4 | 23 | 0.7419 | 0 | 2 | 4 | 1 | 32.6 |

## arm_a_q8_0

| policy | AUTO | cov% | REVIEW | AUTO prec% | AUTO risk | mean | false-full | pv->val | inv->pv | auto under | review/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUTO_ALL | 40 | 87.0 | 6 | 65.0 | 34 | 0.85 | 0 | 3 | 3 | 8 | 13.0 |
| AUTO_VALID_ONLY | 20 | 43.5 | 26 | 90.0 | 10 | 0.5 | 0 | 2 | 0 | 0 | 56.5 |
| AUTO_VALID_AND_PARTIAL | 34 | 73.9 | 12 | 67.6 | 25 | 0.7353 | 0 | 2 | 3 | 6 | 26.1 |
| HUMAN_DISPUTE_AWARE_B | 19 | 41.3 | 27 | 89.5 | 10 | 0.5263 | 0 | 2 | 0 | 0 | 58.7 |
| HUMAN_DISPUTE_AWARE_C | 30 | 65.2 | 16 | 70.0 | 23 | 0.7667 | 0 | 2 | 3 | 4 | 34.8 |

## arm_b_two_pass

| policy | AUTO | cov% | REVIEW | AUTO prec% | AUTO risk | mean | false-full | pv->val | inv->pv | auto under | review/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUTO_ALL | 36 | 78.3 | 10 | 66.7 | 32 | 0.8889 | 0 | 2 | 4 | 6 | 21.7 |
| AUTO_VALID_ONLY | 18 | 39.1 | 28 | 94.4 | 5 | 0.2778 | 0 | 1 | 0 | 0 | 60.9 |
| AUTO_VALID_AND_PARTIAL | 28 | 60.9 | 18 | 75.0 | 19 | 0.6786 | 0 | 1 | 4 | 2 | 39.1 |
| HUMAN_DISPUTE_AWARE_B | 17 | 37.0 | 29 | 94.1 | 5 | 0.2941 | 0 | 1 | 0 | 0 | 63.0 |
| HUMAN_DISPUTE_AWARE_C | 26 | 56.5 | 20 | 76.9 | 18 | 0.6923 | 0 | 1 | 4 | 1 | 43.5 |

Appeal-aware view (baseline, HUMAN_DISPUTE_AWARE_C): 1 automatic undergrade(s) (upper-bound appeal candidates), verdict-step deficit 1; 6 automatic overgrades, step excess 6. Counts only — no fabricated appeal probabilities.

Raw semantic verdicts, structural evidence fields, risk-policy version and review reasons are preserved per case in the JSON.
