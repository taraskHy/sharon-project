# Two-layer grading report — qwen3-vl:8b-instruct (DEV, subset smoke)

> **Ground-truth provenance.** All benchmark targets in this report originate from the ACTUAL instructor-assigned grades on the original graded tests (final_labels.json, ground_truth_source=original_instructor_grade), inverted through the frozen production scoring policy. The owner's blind A/B/C/D audit decisions, model-majority votes and previous cloud-model predictions are NOT used as expected labels anywhere in this report; audit decisions appear only as diagnostic flags. Every target was re-derived from the instructor score at report time and matched the frozen label.

Run `dev__smoke__all__qwen3-vl-8b-instruct__2c27b7282b` | prompt `grade-v4-charitable` | backend `ollama` @ `http://localhost:11434/v1` | adapter `grade-bench-v2` | commit `9ac02197ff`

## A. LOCAL GRADER QUALITY — explanation verdict vs instructor-derived verdict

Population: **26 derivable cases** (valid 22, partially_valid 4; invalid 0 — NOT MEASURED). Scored: **2**; not in this run 24. The 6 wrong-selection cases are excluded by construction (no explanation ground truth).

| metric | value |
| --- | --- |
| verdict exact | 50.0% (1/2) |
| balanced accuracy | 0.5 |
| macro-F1 | 0.3333 |
| harmful verdict upgrades | 0 |
| harmful verdict downgrades | 1 |

Confusion (instructor-derived truth rows x model columns):

| truth \ model | invalid | partially_valid | valid |
| --- | --- | --- | --- |
| invalid | 0 | 0 | 0 |
| partially_valid | 0 | 1 | 0 |
| valid | 0 | 1 | 0 |

## B. END-TO-END TEST-GRADE AGREEMENT — predicted final score vs actual instructor score

Population: **32 cases** (the whole DEV split). Predicted final score = model explanation verdict + actual selection correctness + frozen production scoring policy.

**Full system** (model-scored cases + the deterministic selection-gate zeros):

| metric | value |
| --- | --- |
| cases scored | 8 |
| exact final-score match | 87.5% (7/8) |
| mean absolute score error | 0.25 |
| harmful overgrades (predicted > actual) | 0 |
| harmful undergrades (predicted < actual) | 1 |

**Model-scored subpopulation** (selection correct; the score depends on the model):

| metric | value |
| --- | --- |
| cases scored | 2 |
| exact final-score match | 50.0% (1/2) |
| mean absolute score error | 1.0 |
| harmful overgrades (predicted > actual) | 0 |
| harmful undergrades (predicted < actual) | 1 |

Confusion by actual instructor score (rows) vs predicted final score (columns):

| actual \ predicted | 0 | 2 | no_automated_score |
| --- | --- | --- | --- |
| 0 | 6 | 0 | 0 |
| 2 | 0 | 1 | 3 |
| 4 | 0 | 1 | 21 |

**Wrong-selection sub-report** (6 cases: e002_q1_r1, e003_q1_r3, e003_q2_r3, e003_q2_r4, e003_q2_r5, e003_q2_r6). wrong selection -> deterministic zero -> a local grading call should normally be avoided for these; a final-score match here proves nothing about explanation judgement, so these cases are reported separately and never enter Layer A.

Not executed in this run (24 cases outside the run's subset); they count in the population above but have no prediction here.

## Audit decisions (diagnostic flags only)

audit decisions are diagnostic flags only (rubric-practice mismatch / evidence-transcription concern / ambiguity); they never replace, modify or determine an expected label, and no case in this report was excluded or relabelled because of one.

No audited case is in this population (6 decision(s) exist on other splits).

_Layer A and Layer B answer different questions and are never combined._
