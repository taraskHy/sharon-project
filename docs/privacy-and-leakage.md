# Privacy and label-leakage prevention

## Threat model

Two distinct leakage channels exist when developing a grader on
already-graded exams:

1. **Filename labels** — `02_78.pdf` encodes the instructor's final grade.
   If the model ever sees the filename (in a prompt, a caption, a temp-file
   path, a log excerpt), it can "predict" the grade by copying it.
2. **Instructor annotations in the scans** — red-ink per-question scores,
   ticks/crosses, deductions, and final grades (e.g. "28/32") are visible on
   the pages themselves. A model that reads them is not grading; it is
   transcribing the answer.

## Filename-label prevention (implemented + tested)

- Exams get anonymized IDs (`exam-002`) at discovery; all downstream
  directories, result files, and the `exam_file` field use the anonymized ID.
- The model input contains page images only. `tests/test_offline_pipeline.py::
  test_grade_and_filename_never_reach_model_input` runs the full pipeline with
  a recording backend and asserts that no source filename or path token
  appears in any content block.
- Expected grades live in the manifests and are compared with predictions
  only after grading completes.
- No temp files are created from exam content; pages are rendered in memory.

## Instructor-annotation masking (implemented; effectiveness must be audited)

`autograder/masking.py` removes red-hued ink from rendered page images before
they are sent to any backend (`eval-batch` masks by default; `--no-mask`
disables for audits only):

- pixel-level: only red-dominant pixels are whitened, so student handwriting,
  print, and answer marks in other colours are preserved;
- every masked region and the per-page red-pixel fraction are recorded in
  `masking.json` next to the exam's results — masking is auditable;
- the original PDF is never modified.

**Explicit non-assumptions and limits:**

- *Not all red ink is instructor ink.* Pages with an unusually high red
  fraction are flagged with a warning instead of being trusted silently — a
  student writing in red would be caught by this flag and must be reviewed.
- *Not all instructor ink is red.* Pencil ticks or blue corrections are not
  removed by masking; the survey pass's ink-separation instructions (and its
  `grader_annotations_description`) are the second line of defence, and the
  extraction prompt forbids treating grader marks as student answers.
- Masking is colour heuristics, not understanding — it cannot remove a grade
  written in blue.

## Leakage audit (implemented; must be run on a live backend)

`autograder audit-leakage` sends grade-bearing pages (first 2 + last 3) to
the configured backend twice — unmasked and masked — with a probe prompt that
asks only for readable instructor grades/scores, and compares the probe's
guess against the manifest label:

```
autograder audit-leakage --backend openai --base-url ... --model ... --limit 5
```

Verdict logic:

- probe reads the correct grade from **unmasked** pages → masking is
  **required**; unmasked exams must not be used as ordinary grading input;
- probe still reads the grade from **masked** pages → red-only masking is
  **insufficient**; those scans must not be used as grading input at all
  (exit code 1).

The audit has **not yet been executed** (no inference backend was available
on the development machine when this was written) — run it before trusting
masked scans. Record results in `eval_out/leakage_audit.json` and update
PROJECT_STATUS.md.

## Privacy considerations for hosted APIs

Student exam scans are personal data. Before sending scans to ANY hosted API:

- verify the provider's data-retention policy and whether inputs are used for
  training (free tiers often reserve exactly that right);
- verify your institution's rules on processing student data with external
  services — anonymized IDs do not anonymize handwriting or content;
- prefer the self-hosted deployment (university server) for real student
  data; treat hosted free tiers as suitable for development on consenting /
  synthetic data only, unless the university's data-protection officer
  approves otherwise.

The provider-independent backend means this is a policy decision, not an
architectural one: the same pipeline runs against a local server with no data
leaving the machine.
