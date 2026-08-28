# FullDev zero-inference audit — qwen3-vl:8b-instruct (2026-08-28)

Owner-directed audit of the FullDev run `dev__dev_verdict__all__qwen3-vl-8b-instruct__433146e4b1`, from persisted artifacts only: **0 local inference calls, 0 cloud calls, 0 HELD_OUT calls, $0 spend.** Ground truth = actual instructor grades; A/B/C/D audit decisions are flags only; nothing was relabelled. Span facts were computed with the production evidence matcher; categories are recorded audit judgement.

## 1. The 25 evidence failures

**Root cause is structural and uniform: the model returned `rubric_items: []` in all 26 outputs.** Every citation went into the free-text `evidence` field (25/25 with an 'R1:'-style prefix), which the validator checks only for length — grounding is verified exclusively on `rubric_items[].student_evidence`, so every credit-awarding output fired the fail-closed `ungrounded_credit` rule. In **25/25** failures a verbatim transcription span (>= 12 chars, production normalization) sits INSIDE the mis-placed field — the model found the right words and put them in the wrong place.

| category | cases |
|---|---|
| PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | 19 |
| RUBRIC_STRUCTURE_MISSING | 5 |
| WRONG_SOURCE_SPAN | 1 |
| EMPTY_REQUIRED_EVIDENCE | 0 (the one empty-evidence output scored 0, which demands no grounding) |
| NORMALIZATION_MISMATCH | 0 — **no validator loosening is justified** |
| OTHER | 0 |

Secondary: 20/25 also exceeded the 200-char evidence length limit (prose essays, not spans). Per-case rows (raw evidence field, frozen transcription, exact validator reasons, verdict correctness, span availability) are in `FULLDEV_AUDIT_2026-08-28.json`.

| case | category | verdict correct? | verbatim span in field (chars) |
|---|---|---|---|
| e002_q1_r2 | RUBRIC_STRUCTURE_MISSING | yes | 29 |
| e002_q1_r3 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 49 |
| e002_q1_r4 | RUBRIC_STRUCTURE_MISSING | yes | 29 |
| e002_q1_r5 | RUBRIC_STRUCTURE_MISSING | no | 28 |
| e002_q1_r6 | RUBRIC_STRUCTURE_MISSING | no | 28 |
| e002_q1_r7 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | no | 42 |
| e002_q1_r8 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 50 |
| e002_q2_r1 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 52 |
| e002_q2_r2 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 36 |
| e002_q2_r3 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 47 |
| e002_q2_r5 | WRONG_SOURCE_SPAN | no | 44 |
| e002_q2_r6 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | no | 45 |
| e002_q2_r7 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 49 |
| e002_q2_r8 | RUBRIC_STRUCTURE_MISSING | yes | 46 |
| e003_q1_r1 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 62 |
| e003_q1_r4 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 73 |
| e003_q1_r5 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 52 |
| e003_q1_r6 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 53 |
| e003_q1_r7 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | no | 36 |
| e003_q1_r8 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 35 |
| e003_q2_r1 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 61 |
| e003_q2_r2 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 69 |
| e003_q2_r7 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 64 |
| e003_q2_r8 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | yes | 69 |
| e007_q1_r1 | PARAPHRASED_INSTEAD_OF_EXACT_QUOTE | no | 14 |

## 2. The 8 verdict errors

| case | direction | truth -> model | classification |
|---|---|---|---|
| e002_q1_r5 | upgrade | partially_valid -> valid | model_too_generous |
| e002_q1_r6 | upgrade | partially_valid -> valid | model_too_generous |
| e002_q1_r7 | downgrade | valid -> partially_valid | rubric_instructor_practice_mismatch |
| e002_q2_r4 | downgrade | valid -> invalid | transcription_evidence_issue |
| e002_q2_r5 | downgrade | valid -> partially_valid | model_too_strict |
| e002_q2_r6 | downgrade | valid -> partially_valid | model_too_strict |
| e003_q1_r7 | downgrade | valid -> partially_valid | rubric_instructor_practice_mismatch |
| e007_q1_r1 | downgrade | valid -> partially_valid | rubric_instructor_practice_mismatch |

Tally: {'model_too_generous': 2, 'rubric_instructor_practice_mismatch': 3, 'transcription_evidence_issue': 1, 'model_too_strict': 2}. Full rationales in the JSON. Nothing was relabelled; the two rubric-vs-practice cases on q1 op 7 flag the pack's official-solution text (histogram-flavored on the frequency-decomposition question) for owner review — a flag, not a change.

**System finding:** the run's ONLY AUTO decision (e002_q2_r4) is a harmful downgrade — a 0-score output demands no evidence grounding, so invalid-side judgements auto-finalize while credit-awarding judgements almost all went to REVIEW. The asymmetry means the current gate reviews the model where it is right and trusts it where it is wrong.

## 3. Trivial always-valid baseline vs the 8B

| metric | always-valid baseline | qwen3-vl:8b-instruct |
|---|---|---|
| verdict accuracy (26) | **84.62%** (22/26) | 69.23% (18/26) |
| balanced accuracy | 0.5 | **0.6137** |
| macro-F1 (supported classes) | 0.4583 | **0.5818** |
| harmful upgrades / downgrades | 4 / 0 | 2 / 6 |
| end-to-end exact (32) | **87.5%** | 75.0% |
| end-to-end MAE | **0.25** | 0.5625 |

The 8B is WORSE than the trivial always-valid baseline on raw verdict accuracy (69.23% vs 84.62%), on end-to-end exact match (75.0% vs 87.5%), and on MAE (0.5625 vs 0.25). It beats the baseline only on balanced accuracy (0.6137 vs 0.5) and macro-F1 (0.5818 vs 0.4583), by recovering 2 of 4 partially_valid cases — at the cost of 6 harmful downgrades the baseline cannot make. 69.2% must not be read as strong: exceeding 50% is meaningless here, and the class-blind baseline beats it on every unbalanced metric.

_Sections 4-7 of the owner's directive (generic fix proposal, evaluation discipline, third candidate) are reported in the session summary; this artifact records the case-level audit._
