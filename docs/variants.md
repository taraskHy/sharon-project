# Exam variants and the cover-page flower marker

This exam family is printed in **three variants (A1/A2/A3)**. The variant of
a given student's booklet is identified by the **flower symbol printed on
the cover page** (bottom third, next to "בהצלחה!"). The variant decides
which answer-key column applies AND how the printed sub-item numbering maps
onto the key's canonical numbering, because **variants shuffle question
order and option order**.

## Hard policy

- The variant is decided from the cover marker plus the authoritative
  mapping file only. It is **never** inferred from the student's answers,
  the instructor's grade, or whichever key column scores highest — the
  detection call receives the cover image and marker descriptions only
  (`autograder/variant.py`, enforced by tests in `tests/test_variant.py`).
- A missing/cropped/illegible/ambiguous marker → the decision is
  **uncertain**, a review item is raised, and a provisional variant is
  chosen deterministically (first in the mapping — documented arbitrary
  choice, not score-based).
- Instructor ink never enters detection: batch runs mask red ink before any
  model call, and the detection prompt explicitly excludes handwritten ink.
- The marker config and any pinned `--version` are part of the exam
  fingerprint: artefacts graded under one variant interpretation are never
  resumed/reused under another.

## The authoritative mapping (this exam family)

Stored in [sample_data/Exam_solution.variants.json](../sample_data/Exam_solution.variants.json)
(auto-discovered as `<key>.variants.json`; override with `--variant-map`).

| Cover flower | Variant | Anchoring evidence (printed material only) |
|---|---|---|
| Four-petal clover/butterfly | **A1** | The instructor-verified representative exam carries it; its printed question order matches the key's canonical order exactly (e.g. #18 Canny, #19 Harris, #20 Rectification = key items 18/19/20); the answer key's own cover carries the same flower. |
| Five-petal star | **A2** | Prints the Hough-triangles question at **#20** with the "one side **parallel to the x-axis**" wording that the key's page-11 note attributes to *version 2* (answer: minimal dimension 3). |
| Many-petal daisy | **A3** | Prints the Hough-triangles question at **#18** with the "any orientation" wording (versions 1 and 3 per the key's note) and does **not** match the key's canonical order, so it is not A1. |

Key legend anchoring A-numbers: key page 3 — "הצבעים הם R,B,G לפי A1, A2, A3
בהתאם" (the key's answer colours R/B/G correspond to variants A1/A2/A3).

Derived 2026-07-13 by manual inspection of the printed forms and the key
(exams `test/002` (daisy), `test/003` (star), `sample_data/student_exam.pdf`
(clover)). No student answers were used.

## Question alignment (order shuffling)

Observed: the Hough-triangles question is printed at #16 (A1), #20 (A2),
#18 (A3); option orders differ too (e.g. the Laplacian-collapse options are
permuted between the key and variant A2's print).

Consequently, scoring cannot pair "printed row N" with "key item N" outside
A1. The pipeline derives a **per-variant alignment** (one model call per
variant, matching printed question CONTENT to key prompts — no answers
sent), validates it deterministically (bijection over the key's sub-items),
caches it persistently (`align_*.json` in the key cache, keyed by key
fingerprint + variant + prompt/schema hashes), and then:

1. extraction runs in the variant's **printed numbering** (what the student
   saw and filled into the answer sheet), and
2. results are remapped to key ids before reconciliation and scoring, with
   the printed number kept in each sub-item's provenance
   (`source_region: "... (printed #20)"`).

If the alignment cannot be validated, the pipeline uses identity numbering
and **flags every affected sub-item for human review** — misalignment is
never silent.

## Pipeline placement

```
key (cached) → variant detection (cover marker)  ── uncertain → review
            → survey (low-res) → sheet close-read (full-res)
            → alignment (cached per variant)
            → extraction (printed numbering → remap to key ids)
            → judging + scoring under the detected variant's key column
```

Recorded in every result (`result.variant_detection`): marker seen, matched
catalogue entry, selected variant, confidence, page + region, obstruction
notes, mapping source + config fingerprint, and how alignment was obtained
(cache/derived/identity-fallback). Batch outputs carry `detected_variant`,
`variant_uncertain` and `key_source` per exam.

## Legacy exams without a marker config

When no `<key>.variants.json` exists, the old answer-agreement detection
(with its uncertainty margin and review flag) still applies — that is the
only option for forms with no documented visual marker. For THIS exam
family the marker config exists and answer-agreement detection never runs.
