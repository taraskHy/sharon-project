# CALIBRATION quality — output-contract experiment (2026-08-28)

> **Ground-truth provenance.** actual instructor-assigned grades (final_labels.json, ground_truth_source=original_instructor_grade) + actual selection correctness + frozen production scoring policy. A/B/C/D human-audit decisions are diagnostic flags only and NEVER targets; no previous model output (Qwen/Gemini/Sonnet/Luna), model vote, or subjective ranking defines a target; no target reaches a model request

Experiment `e86845b9c85f…`; strict rule: strict model-quality denominators use calibration_strict_case_ids only; excluded cases still run for diagnostic completeness. Targets re-verified against final_labels.json at build time. CALIBRATION is ONE writer (e004); nothing here is a production-readiness claim.

## A. LOCAL EXPLANATION-GRADER QUALITY (strict 11 cases: 7 valid, 4 partially_valid)

| metric | `qwen3-vl:8b-instruct` | `qwen3-vl:30b-a3b-instruct` |
|---|---|---|
| verdict accuracy | 63.64% | 27.27% |
| balanced accuracy | 0.6071 | 0.3214 |
| macro-F1 (supported) | 0.7179 | 0.325 |
| valid P / R / F1 | 0.8333 / 0.7143 / 0.7692 | 1.0 / 0.1429 / 0.25 |
| partially_valid P / R / F1 | 1.0 / 0.5 / 0.6667 | 0.3333 / 0.5 / 0.4 |
| harmful upgrades / downgrades | 1 / 3 | 0 / 8 |
| uncertainty rate | 0.0% | 0.0% |
| AUTO / REVIEW | 100.0% / 0.0% | 100.0% / 0.0% |
| evidence-failure rate | 0.0% | 0.0% |
| evidence-engagement rate | 100.0% | 100.0% |
| schema-failure rate | 0.0% | 0.0% |

`qwen3-vl:8b-instruct` confusion (instructor-derived truth rows x model columns):

| truth \ model | invalid | partially_valid | valid |
|---|---|---|---|
| invalid | 0 | 0 | 0 |
| partially_valid | 1 | 2 | 1 |
| valid | 2 | 0 | 5 |

`qwen3-vl:30b-a3b-instruct` confusion (instructor-derived truth rows x model columns):

| truth \ model | invalid | partially_valid | valid |
|---|---|---|---|
| invalid | 0 | 0 | 0 |
| partially_valid | 2 | 2 | 0 |
| valid | 2 | 4 | 1 |

## B. END-TO-END TEST-GRADE AGREEMENT (strict 11; selection correct implied by credit)

| metric | `qwen3-vl:8b-instruct` | `qwen3-vl:30b-a3b-instruct` |
|---|---|---|
| exact final-score match | 7/11 (63.64%) | 3/11 (27.27%) |
| MAE | 1.0909 | 1.8182 |
| harmful overgrades / undergrades | 1 / 3 | 0 / 8 |

population contains no actual-0 case: the two zero-score CALIBRATION cases have unresolved selection correctness and are outside the frozen derivable subset.

## Trivial baselines (strict 11)

| baseline | accuracy | balanced acc | macro-F1 |
|---|---|---|---|
| always-valid | 63.64% | 0.5 | 0.3889 |
| always-partially_valid | 36.36% | 0.5 | 0.2667 |
| majority class | 63.64% | - | - |

invalid-class performance = NOT MEASURED (no authoritative invalid example exists).

## Per-case (strict + diagnostic)

`qwen3-vl:8b-instruct`:

| case | actual | derived verdict | model verdict | predicted score | decision | strict | latency |
|---|---|---|---|---|---|---|---|
| e004_q1_r1 | 4 | valid | invalid | 0 | AUTO | ERROR | 11.578s |
| e004_q1_r2 | 4 | valid | valid | 4 | AUTO | ok | 1.907s |
| e004_q1_r3 | 2 | partially_valid | partially_valid | 2 | AUTO | ok | 1.766s |
| e004_q1_r4 | 4 | valid | valid | 4 | AUTO | ok | 1.968s |
| e004_q1_r5 | 2 | partially_valid | partially_valid | 2 | AUTO | ok | 2.25s |
| e004_q1_r6 | 2 | partially_valid | invalid | 0 | AUTO | ERROR | 2.203s |
| e004_q1_r8 | 2 | partially_valid | valid | 4 | AUTO | ERROR | 2.016s |
| e004_q2_r1 | 4 | valid | valid | 4 | AUTO | ok | 2.64s |
| e004_q2_r2 | 4 | valid | valid | 4 | AUTO | ok | 2.219s |
| e004_q2_r3 | 4 | valid | valid | 4 | AUTO | ok | 1.875s |
| e004_q2_r6 | 4 | valid | invalid | 0 | AUTO | ERROR | 3.125s |
| e004_q2_r8 | 2 | partially_valid | invalid | 0 | AUTO | diagnostic (audit C) | 2.047s |

`qwen3-vl:30b-a3b-instruct`:

| case | actual | derived verdict | model verdict | predicted score | decision | strict | latency |
|---|---|---|---|---|---|---|---|
| e004_q1_r1 | 4 | valid | invalid | 0 | AUTO | ERROR | 21.375s |
| e004_q1_r2 | 4 | valid | partially_valid | 2 | AUTO | ERROR | 3.109s |
| e004_q1_r3 | 2 | partially_valid | invalid | 0 | AUTO | ERROR | 2.688s |
| e004_q1_r4 | 4 | valid | valid | 4 | AUTO | ok | 2.563s |
| e004_q1_r5 | 2 | partially_valid | invalid | 0 | AUTO | ERROR | 2.828s |
| e004_q1_r6 | 2 | partially_valid | partially_valid | 2 | AUTO | ok | 2.688s |
| e004_q1_r8 | 2 | partially_valid | partially_valid | 2 | AUTO | ok | 2.593s |
| e004_q2_r1 | 4 | valid | partially_valid | 2 | AUTO | ERROR | 3.157s |
| e004_q2_r2 | 4 | valid | partially_valid | 2 | AUTO | ERROR | 2.391s |
| e004_q2_r3 | 4 | valid | partially_valid | 2 | AUTO | ERROR | 3.094s |
| e004_q2_r6 | 4 | valid | invalid | 0 | AUTO | ERROR | 3.421s |
| e004_q2_r8 | 2 | partially_valid | invalid | 0 | AUTO | diagnostic (audit C) | 2.625s |

## Comparison

- both_correct: e004_q1_r4
- only_qwen3-vl:8b-instruct_correct: e004_q1_r2, e004_q1_r3, e004_q1_r5, e004_q2_r1, e004_q2_r2, e004_q2_r3
- only_qwen3-vl:30b-a3b-instruct_correct: e004_q1_r6, e004_q1_r8
- both_wrong: e004_q1_r1, e004_q2_r6
- shared_harmful_downgrades: e004_q1_r1, e004_q2_r6

## Classification (no production winner is selected here)

- `qwen3-vl:8b-instruct`: **MAYBE.** The output contract is fully solved (evidence engagement 100%, evidence failures 0%, schema failures 0%, AUTO 100%, ~2.1s median). Quality: matches the always-valid baseline on raw accuracy (63.64%) and clearly beats both trivial baselines on balanced accuracy (0.607 vs 0.5) and macro-F1 (0.718 vs 0.389) — real class discrimination — but 3 harmful undergrades in 11 strict cases now AUTO-finalize because validation passes. SAFETY TRADE-OFF to decide: the contract converted 'review noise' into confidently grounded wrong AUTOs; a deterministic policy knob (e.g. AUTO only for `valid`, route partial/invalid verdicts to REVIEW) would cap undergrade harm at zero at ~36% review cost on this population. Not proposed as a change tonight.
- `qwen3-vl:30b-a3b-instruct`: **DROP for GRADE_PRIMARY.** 27.27% accuracy, 0 upgrades / 8 harmful downgrades of 11 (the same harsh-downgrader pathology that disqualified qwen3.8:27b), below every trivial baseline on every metric, plus partial CPU offload (37%/63%), 47s cold load and 21s worst-case latency. Structural contract compliance was perfect — the failure is judgement, not format.

_Layer A and Layer B are never combined. invalid-class performance = NOT MEASURED. CALIBRATION is one writer; strict denominator excludes the audit-C evidence-issue case (target preserved)._
