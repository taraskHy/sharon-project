# GRADE_PRIMARY: what the benchmark should actually measure

Status: **proposal — requires an owner decision before the dataset changes.**
Written 2026-08-24 after the first live smoke run. No dataset has been
modified. HELD_OUT has not been touched.

Everything below is derived from source and pinned by
`tests/test_grading_semantics_derivation.py`, not inferred from Erik's scores.

---

## 1. The scoring pipeline, exactly

All 67 GRADE_PRIMARY cases are `matching_with_explanation`, with
`explanation_required = True`, `explanation_weight = 0.0`, `max_score = 4.0`.

```
  MC selection            deterministic (CV / mc_resolve) — never the grading model
        |                 policies.decide_before_ocr gates whether we grade at all
        v
  explanation OCR         task ocr_primary (lazy, per item)
        |
        v
  grade_primary model --> GradeResult.score            [a bounded proposal]
        |
        v
  _verdict_from_score(score, max_score)                reliability.py:145
        ratio = score / max_score
        ratio >= 0.999 -> "valid"
        ratio <= 0.001 -> "invalid"
        else           -> "partially_valid"
        |
        v
  _verdict_factor(verdict, config)                     grade.py:307
        valid -> 1.0 | partially_valid -> 0.5 | everything else -> 0.0
        |
        v
  _grade_sub_item, branch `explanation_required and w == 0`   grade.py:471
        final = max_points * factor   if selection_correct
              = 0.0                   otherwise
        points_explanation = 0.0      (always, in this branch)
```

`_verdict_from_score` says it in its own docstring:

> *"Map the grader's proposal onto the existing explanation verdict, which the
> deterministic scorer then turns into points. **The model never supplies the
> number itself.**"*

`gradingpack.py:224` emits the rule string `explanation weight 0` **only when
`explanation_required` is True** — so its presence in all 67 packs is proof
that this gating branch is the one that applies.

## 2. Derivation of 0 / 2 / 4

`partial_explanation_factor = 0.5` (`config.py:27`), `max_points = 4.0`:

| final | exact condition |
|---|---|
| **4** | `selection_correct` **and** verdict `valid` |
| **2** | `selection_correct` **and** verdict `partially_valid` |
| **0** | `not selection_correct` (any verdict) **or** verdict ∈ {`invalid`, `missing`, `illegible`, `None`} |

The reachable set is exactly `{0, 2, 4}` — identical to the observed instructor
label set. That agreement is the strongest available evidence the derivation is
right.

### The three frozen smoke cases

| case | instructor | implied selection | implied verdict | unique? |
|---|---|---|---|---|
| `e007_q1_r1` | 4 | correct | `valid` | **yes** |
| `e002_q1_r8` | 2 | correct | `partially_valid` | **yes** |
| `e003_q2_r6` | 0 | unknown | unknown | **no** — 6 states produce 0 |

What both models actually returned was `score = 0.0` on every parsed case.
Forward through the pipeline: `_verdict_from_score(0.0, 4.0) = "invalid"` →
factor `0.0` → final `0.0` regardless of selection. The observed result is what
the corrected semantics *predict* for an input with the selection withheld.

## 3. What the model is responsible for

**The explanation verdict, and nothing else.**

The selection is resolved deterministically upstream and is not the grading
model's job. The final number is arithmetic. The only judgement in the pipeline
that requires a language model is: *given the rubric, the official solution and
the student's transcribed explanation, is this explanation `valid`,
`partially_valid`, or `invalid`?* (with `missing` / `illegible` reserved for
OCR-side states the grader never has to produce).

Answering question 6 directly — `GradeResult.score` is **(d) something else**:
a bounded proposal whose only consumer quantises it to three levels. It is
documented as (a) a final sub-item score by `GRADE_SYSTEM` and by
`grade_prompt`'s `Score range: 0..{max}` line, and it is consumed as (b/c) an
explanation ratio. That mismatch is itself a defect: the prompt asks for a
quantity the pipeline does not want.

## 4. Is `selected = None` valid?

| target | verdict |
|---|---|
| final sub-item score | **invalid** — the target is unreachable. All points ride on the selection; withholding it caps every honest grader at 0. |
| explanation verdict | **valid, and correct** — the verdict is a property of the explanation. Showing the selection would invite the grader to reward the choice instead of judging the reasoning, and would leak the answer whenever `version` is present. |

So `selected = None` was never the bug. Pairing it with a **final-score target**
was. The 2026-08-23 ruling ("selected=None is frozen dataset policy") is
correct about the input; it is the *label* that does not belong.

## 5. Recommended benchmark target

Replace the final-score comparison with the model's real responsibility.

**Target:** the explanation verdict, three classes: `valid` /
`partially_valid` / `invalid`.

**Schema:** keep `GradeResult` for the transport (it already carries
`rubric_items`, `uncertain`, `evidence`, all of which we want to score), but
score the *derived verdict*, not the raw number:

```python
predicted_verdict = _verdict_from_score(g.score, pack.max_score)
```

This needs **no dataset rebuild and no prompt change** — it is a scoring-side
change in the benchmark adapter. It also keeps the benchmark measuring the
exact function production calls, rather than a parallel definition.

**Metrics:**

| metric | why |
|---|---|
| 3-class accuracy on the verdict | the actual decision |
| macro-F1 | 2s are rare (4/32 DEV); accuracy alone hides them |
| confusion matrix | `valid`↔`partially_valid` and `→invalid` are different failures |
| **cost-weighted error**: implied final-score delta | a `valid`→`invalid` slip costs a student 4 points; `valid`→`partially_valid` costs 2 |
| AUTO/REVIEW rate at fixed safety | `uncertain` + `validate_grade` behaviour |
| evidence-grounding failures | already computed; unaffected by any of this |

Report accuracy **only over cases whose verdict is uniquely determined** (see
below), and report the excluded count explicitly — never silently.

## 6. Is Erik's ground truth sufficient?

Partly, and the boundary is mathematically exact.

Inverting `final = max * factor(verdict) if selection_correct else 0`:

- `score == 4` ⟹ `selection_correct = True`, verdict `valid` — **unique**
- `score == 2` ⟹ `selection_correct = True`, verdict `partially_valid` — **unique**
- `score == 0` ⟹ **six** distinct `(selection_correct, verdict)` states — **not unique**

A 0 tells us only "no credit". It cannot distinguish *wrong selection, good
explanation* from *correct selection, worthless explanation* — and those demand
opposite things of the grader. **Deriving a verdict label from a 0 would be
inventing ground truth.**

Derivable coverage (HELD_OUT deliberately not inspected):

| split | cases | verdict-derivable (`2` or `4`) | ambiguous (`0`) |
|---|---|---|---|
| DEV | 32 | **26 (81.2%)** | 6 |
| CALIBRATION | 14 | **12 (85.7%)** | 2 |

That is enough to run a real benchmark today, on labels we already own, with
zero new human work.

Caveat to state in any report: the derivable subset is **not** a random sample
— it contains no zero-credit cases at all. It measures "can the grader tell a
fully valid explanation from a partially valid one", not "can it withhold
credit". The `invalid` class is exactly the one it cannot yet measure.

## 7. What additional human ground truth would be needed

To close the `invalid` class — and only for that — the 6 DEV + 2 CALIBRATION
zero-score cases need **one** human field each:

- `selection_correct: bool` — was the student's chosen letter right?

That single bit is sufficient, and it is cheap: it is a lookup against the
answer key plus the student's mark, not a re-grade.

- if `selection_correct = False` → the case is **unusable** as an explanation
  label (the explanation was never the reason for the 0) and should be excluded
  from verdict accuracy;
- if `selection_correct = True` → the verdict is one of
  `invalid` / `missing` / `illegible`, and a human must pick which. For a
  grading benchmark `missing` and `illegible` are OCR-side states, so in
  practice this collapses to: confirm `invalid`.

Estimated human effort: 8 cases × (one key lookup + one glance at the crop).
No re-grading, no rubric work.

**Not recommended:** synthesising verdict labels for 0-score cases from any
model's output, or assuming `selection_correct = True` because it is the
common case. Both would fabricate the only class the benchmark currently
cannot see.

## 8. Decision required

1. Adopt the explanation-verdict target? (scoring-side change; no rebuild)
2. Report on the 26 DEV derivable cases now, or wait for the 8-case
   `selection_correct` pass to include the `invalid` class?
3. Fix the prompt/schema mismatch — `GRADE_SYSTEM` and `grade_prompt` ask for a
   final score the pipeline never uses. Changing that changes
   `prompt_version` and production grading semantics, so it is a separate,
   explicitly-versioned decision.

Nothing in this document has been applied to the dataset.

## 9. Resolution and standing owner directive (2026-08-28)

Decisions 1–2 were taken: the verdict target was adopted as a label-side
dataset revision (manifest revisions of 2026-08-24/25), the 8-case
`selection_correct` audit ran (`scripts/selection_audit_ui.py`), and all six
DEV zero-score cases turned out to have a **wrong selection** — so they are
EXCLUDED from verdict ground truth (not relabelled `invalid`) and the
`invalid` class remains unmeasured in every split.

The owner's standing directive for every grading evaluation:

- **The actual instructor-assigned grade from the original graded test is
  the only authoritative ground truth** (`final_labels.json`,
  `ground_truth_source=original_instructor_grade`). Explanation-verdict
  targets exist only where they are mathematically identifiable from
  (instructor score, selection correctness, frozen production policy) —
  DEV: 26 cases (22 valid, 4 partially_valid, 0 invalid).
- **Blind A/B/C/D audit decisions, model-majority votes and previous
  cloud-model predictions are never expected labels.** Audit decisions are
  reported as flags only (rubric-practice mismatch / evidence-transcription
  concern / ambiguity) and never silently alter an instructor grade.
- Results are reported in **two separated layers**
  (`autograder/benchmark/gradereport.py`, `bench grade-report`):
  **A** — model explanation verdict vs instructor-derived verdict (the 26);
  **B** — system predicted final score (model verdict + actual selection
  correctness + frozen production policy) vs the actual instructor score
  over the whole split, with exact match, absolute error, harmful
  overgrades/undergrades and a confusion by actual score 0/2/4. The six
  wrong-selection DEV cases appear only in Layer B's separate policy
  sub-report (wrong selection -> deterministic zero -> a local grading call
  is normally unnecessary) and never count toward explanation-model
  accuracy. The report re-derives every target from the instructor score at
  report time and refuses to run over labels that disagree.
