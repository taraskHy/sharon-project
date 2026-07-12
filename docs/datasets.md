# Datasets

## Corpus layout

| Location | Contents | Allowed uses |
|---|---|---|
| `sample_data/` | one representative exam + its answer key (13-page Hebrew image-processing exam, version A1, with instructor annotations) | prompt development, debugging, benchmarks |
| `test/` | 41 previously graded exams, filenames `<index>_<grade>.pdf` (e.g. `02_78.pdf` = exam 02, final grade 78) | training, development, validation, model selection, prompt development, calibration, fine-tuning, debugging |
| *(not yet present)* | 48 additional graded exams | **HELD-OUT FINAL TEST SET — completely unseen until the system is frozen** (see docs/evaluation.md) |

The original files are never renamed, modified, or written to; every pipeline
step reads bytes and works on rendered copies.

## Manifests

`autograder make-manifests` discovers `test/`, validates filenames (malformed
names and duplicate indices are reported to `datasets/discovery_issues.json`),
assigns anonymized IDs, splits deterministically, and writes:

- `datasets/train_manifest.json` — 25 exams (seed 42, 60 %)
- `datasets/validation_manifest.json` — 16 exams (40 %)
- `datasets/final_test_manifest.json` — placeholder with **no entries and no
  labels**; populated only after configuration freeze

Each entry records: anonymized ID, original relative path, expected final
grade, split, whether detailed per-question labels exist (currently none do),
instructor-annotation status, masking status, and data-quality warnings
(e.g. `exam-014` is a 7-page partial scan of the 13-page form).

Split properties:

- fixed seed (42), shuffle over sorted anonymized IDs → fully reproducible;
- one entry per exam index — duplicate indices are rejected at discovery, so
  no exam (or copy of it) can appear in both splits;
- the split is **not** to be regenerated with different seeds to improve
  results. If it must ever change (e.g. new data), record why in this file.

## Anonymization and label discipline

The filename grade is a **label**. Rules enforced by code and tests:

1. Exams are addressed by anonymized IDs (`exam-002`) everywhere downstream
   of discovery; output directories and result files use only the anonymized
   ID.
2. The model receives page images only — never filenames, paths, or metadata.
   A regression test drives the full pipeline with a recording backend and
   asserts no source filename/path token appears in any model input.
3. The expected grade is read from the manifest and compared with the
   prediction only **after** grading completes (`autograder/evalcli.py`).
4. Red-ink instructor annotations (including written grades) are masked from
   page images before inference by default (see docs/privacy-and-leakage.md).

## Label limitations

The filename grade is an exam-level label only. It supports total-score
calibration, end-to-end model comparison, and detecting systematic over/under
grading. It does **not** verify answer extraction, transcription quality,
explanation judging, per-question credit, or that instructor annotations were
ignored — a correct total can result from cancelling mistakes.

Stronger labels (per-question scores, final answers, explanation
transcriptions) exist in the scans themselves as red-ink annotations. The
intended path to derive them (documented, not yet executed): run extraction
on **unmasked** pages with a dedicated "read the instructor's marks" prompt,
store the result as *label* files (never as model input for grading), and
hand-verify a sample before use. Until that is done and verified,
`has_detailed_labels` stays `false` in the manifests.
