# Prospective policy replay — SHADOW (2026-09-02 03:02:14)

Deployable policies use ONLY decision-time-observable inputs (typed, fail-closed). 46 seen explanation cases; baseline Q4 8B outputs; no inference.

## PROSPECTIVE_DEPLOYABLE (genuinely deployable)

| policy | AUTO | cov% | REVIEW | prec% | AUTO risk | mean | iv->v | pv->v | iv->pv | under | step+ | step- | rev/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prospective_valid_only_v1 | 27 | 58.7 | 19 | 85.2 | 20 | 0.7407 | 0 | 4 | 0 | 0 | 4 | 0 | 41.3 |
| prospective_noninvalid_v1 | 39 | 84.8 | 7 | 74.4 | 34 | 0.8718 | 0 | 4 | 4 | 2 | 8 | 2 | 15.2 |
| prospective_auto_all_structurally_valid_v1 | 44 | 95.7 | 2 | 68.2 | 42 | 0.9545 | 0 | 4 | 4 | 6 | 8 | 8 | 4.3 |

## ORACLE-ASSISTED RETROSPECTIVE UPPER BOUND — NOT DEPLOYABLE

These rankings use post-review human-disagreement data that does not exist before review on a new case. They bound what a future prospective dispute-predictor could add; they are NOT candidate production policies.

| policy | AUTO | cov% | REVIEW | prec% | AUTO risk | pv->v | under |
|---|---|---|---|---|---|---|---|
| retrospective_human_dispute_aware_b_v1 | 22 | 47.8 | 24 | 90.9 | 10 | 2 | 0 |
| retrospective_human_dispute_aware_c_v1 | 31 | 67.4 | 15 | 77.4 | 23 | 2 | 1 |

## Rare-event uncertainty (exact Clopper-Pearson)

invalid->valid automatic full credit: observed 0/5 on seen data, one-sided 95% upper bound **45.1%** — zero observed events over five invalid cases CANNOT demonstrate safety; the data only excludes rates above ~45%.

Minimum independent invalid examples (zero events observed) for a one-sided 95% upper bound below:

| bound | min invalid examples |
|---|---|
| 10% | 29 |
| 5% | 59 |
| 2% | 149 |
| 1% | 299 |

Formula: smallest n with (1-bound)^n <= 0.05. The bound choice is the owner's; none is selected here.

Shadow events: 138 rows in `SHADOW_REPLAY_2026-09-02.jsonl` (decision inputs and offline evaluation fields strictly separated).
