# Two-layer grading report — qwen3-vl:8b-instruct (CALIBRATION, subset calibration_verdict_v4)

> **Ground-truth provenance.** All benchmark targets in this report originate from the ACTUAL instructor-assigned grades on the original graded tests (final_labels.json, ground_truth_source=original_instructor_grade), inverted through the frozen production scoring policy. The owner's blind A/B/C/D audit decisions, model-majority votes and previous cloud-model predictions are NOT used as expected labels anywhere in this report; audit decisions appear only as diagnostic flags. Every target was re-derived from the instructor score at report time and matched the frozen label.

Run `calibration__calibration_verdict_v4__all__qwen3-vl-8b-instruct__0a865643c3` | prompt `grade-v4-charitable-local` | backend `ollama` @ `http://localhost:11434/v1` | adapter `grade-bench-v3` | commit `ac3227a44c`

## A. LOCAL GRADER QUALITY — explanation verdict vs instructor-derived verdict

Population: **12 derivable cases** (valid 7, partially_valid 5; invalid 0 — NOT MEASURED). Scored: **12**. The 0 wrong-selection cases are excluded by construction (no explanation ground truth).

| metric | value |
| --- | --- |
| verdict exact | 58.33% (7/12) |
| balanced accuracy | 0.5572 |
| macro-F1 | 0.6703 |
| harmful verdict upgrades | 1 |
| harmful verdict downgrades | 4 |

Confusion (instructor-derived truth rows x model columns):

| truth \ model | invalid | partially_valid | valid |
| --- | --- | --- | --- |
| invalid | 0 | 0 | 0 |
| partially_valid | 2 | 2 | 1 |
| valid | 2 | 0 | 5 |

## B. END-TO-END TEST-GRADE AGREEMENT — predicted final score vs actual instructor score

Population: **14 cases** (the whole CALIBRATION split). Predicted final score = model explanation verdict + actual selection correctness + frozen production scoring policy.

**Full system** (model-scored cases + the deterministic selection-gate zeros):

| metric | value |
| --- | --- |
| cases scored | 12 |
| exact final-score match | 58.33% (7/12) |
| mean absolute score error | 1.1667 |
| harmful overgrades (predicted > actual) | 1 |
| harmful undergrades (predicted < actual) | 4 |

**Model-scored subpopulation** (selection correct; the score depends on the model):

| metric | value |
| --- | --- |
| cases scored | 12 |
| exact final-score match | 58.33% (7/12) |
| mean absolute score error | 1.1667 |
| harmful overgrades (predicted > actual) | 1 |
| harmful undergrades (predicted < actual) | 4 |

Confusion by actual instructor score (rows) vs predicted final score (columns):

| actual \ predicted | 0 | 2 | 4 |
| --- | --- | --- | --- |
| 2 | 2 | 2 | 1 |
| 4 | 2 | 0 | 5 |

**Wrong-selection sub-report** (0 cases: ). wrong selection -> deterministic zero -> a local grading call should normally be avoided for these; a final-score match here proves nothing about explanation judgement, so these cases are reported separately and never enter Layer A.

Excluded (selection correctness unresolved): e004_q2_r4, e004_q2_r5.

## Audit decisions (diagnostic flags only)

audit decisions are diagnostic flags only (rubric-practice mismatch / evidence-transcription concern / ambiguity); they never replace, modify or determine an expected label, and no case in this report was excluded or relabelled because of one.

| case | decision | flag | meaning |
| --- | --- | --- | --- |
| e004_q1_r1 | B | rubric_practice_mismatch | instructor practice is more lenient than the literal encoded rubric |
| e004_q1_r3 | B | rubric_practice_mismatch | instructor practice is more lenient than the literal encoded rubric |
| e004_q1_r5 | A | none (consistent) | derived verdict is consistent with the rubric and instructor practice |
| e004_q1_r6 | A | none (consistent) | derived verdict is consistent with the rubric and instructor practice |
| e004_q2_r6 | A | none (consistent) | derived verdict is consistent with the rubric and instructor practice |
| e004_q2_r8 | C | evidence_transcription_concern | transcription/evidence/rubric artifact is incomplete or incorrect |

_Layer A and Layer B answer different questions and are never combined._
