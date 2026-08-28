# Two-layer grading report — qwen3-vl:8b-instruct (DEV)

> **Ground-truth provenance.** All benchmark targets in this report originate from the ACTUAL instructor-assigned grades on the original graded tests (final_labels.json, ground_truth_source=original_instructor_grade), inverted through the frozen production scoring policy. The owner's blind A/B/C/D audit decisions, model-majority votes and previous cloud-model predictions are NOT used as expected labels anywhere in this report; audit decisions appear only as diagnostic flags. Every target was re-derived from the instructor score at report time and matched the frozen label.

Run `dev__all__qwen3-vl-8b-instruct__72e19378d1` | prompt `grade-v4-charitable-local` | backend `ollama` @ `http://localhost:11434/v1` | adapter `grade-bench-v3` | commit `77a4c2b0aa`

## A. LOCAL GRADER QUALITY — explanation verdict vs instructor-derived verdict

Population: **26 derivable cases** (valid 22, partially_valid 4; invalid 0 — NOT MEASURED). Scored: **26**. The 6 wrong-selection cases are excluded by construction (no explanation ground truth).

| metric | value |
| --- | --- |
| verdict exact | 65.38% (17/26) |
| balanced accuracy | 0.5909 |
| macro-F1 | 0.5846 |
| harmful verdict upgrades | 2 |
| harmful verdict downgrades | 7 |

Confusion (instructor-derived truth rows x model columns):

| truth \ model | invalid | partially_valid | valid |
| --- | --- | --- | --- |
| invalid | 0 | 0 | 0 |
| partially_valid | 0 | 2 | 2 |
| valid | 3 | 4 | 15 |

## B. END-TO-END TEST-GRADE AGREEMENT — predicted final score vs actual instructor score

Population: **32 cases** (the whole DEV split). Predicted final score = model explanation verdict + actual selection correctness + frozen production scoring policy.

**Full system** (model-scored cases + the deterministic selection-gate zeros):

| metric | value |
| --- | --- |
| cases scored | 32 |
| exact final-score match | 71.88% (23/32) |
| mean absolute score error | 0.75 |
| harmful overgrades (predicted > actual) | 2 |
| harmful undergrades (predicted < actual) | 7 |

**Model-scored subpopulation** (selection correct; the score depends on the model):

| metric | value |
| --- | --- |
| cases scored | 26 |
| exact final-score match | 65.38% (17/26) |
| mean absolute score error | 0.9231 |
| harmful overgrades (predicted > actual) | 2 |
| harmful undergrades (predicted < actual) | 7 |

Confusion by actual instructor score (rows) vs predicted final score (columns):

| actual \ predicted | 0 | 2 | 4 |
| --- | --- | --- | --- |
| 0 | 6 | 0 | 0 |
| 2 | 0 | 2 | 2 |
| 4 | 3 | 4 | 15 |

**Wrong-selection sub-report** (6 cases: e002_q1_r1, e003_q1_r3, e003_q2_r3, e003_q2_r4, e003_q2_r5, e003_q2_r6). wrong selection -> deterministic zero -> a local grading call should normally be avoided for these; a final-score match here proves nothing about explanation judgement, so these cases are reported separately and never enter Layer A.

## Audit decisions (diagnostic flags only)

audit decisions are diagnostic flags only (rubric-practice mismatch / evidence-transcription concern / ambiguity); they never replace, modify or determine an expected label, and no case in this report was excluded or relabelled because of one.

No audited case is in this population (6 decision(s) exist on other splits).

_Layer A and Layer B answer different questions and are never combined._
