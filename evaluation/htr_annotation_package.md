# HTR pilot — annotation package report

Date: 2026-07-14. Built by `scripts/htr_pilot_build.py`; annotated with
`scripts/htr_annotation_app.py`; validated by
`scripts/htr_annotation_validate.py`. Status: **built, validated
(RESULT: PASS), smoke-tested — ready for owner annotation. 441 line
samples across 16 writers; no authoritative labels exist yet.**

## Split (writer-separated, deterministic)

One exam = one writer; assignment is by exam number, fixed in code
(`htr_pilot_build.SPLITS`) and covered by tests:

| split | writers (anonymized ids) | exams |
|---|---|---|
| train | e003–e012 | 10 |
| val | e013–e015 | 3 |
| internal_test | e016–e018 | 3 |

- **Exam 002 is excluded from every split** — it is the hidden
  transcription-benchmark writer (16 owner-verified cells); keeping it out
  of the pilot entirely means the benchmark stays a clean, unseen probe.
- The representative exam is excluded (different sheet layout; already
  used for grading audits). Exams e019–e042 are reserved for scale-up.
- The 48 held-out exams are not present in the repo and are never
  referenced; the validator enforces that no writer outside e003–e018
  appears anywhere in the package.
- No writer crosses splits (validated + tested).

## Assets (per explanation cell)

- original crop `images/<writer>/q{q}_r{r}_cell_orig.jpg` (red-masked scan);
- cleaned student-ink-only crop `…_cell_clean.png` (registered synthetic
  blank-template subtraction + blue-ink isolation — the pipeline validated
  in `evaluation/student_ink_isolation_experiment.md`);
- segmented line crops `…_l{i}.png` (annotation unit; from the ink mask);
- per-sheet contact sheets `contact/<writer>_q{1,2}.jpg` (title strip for
  sheet-identity QA + original|cleaned thumbnails per row);
- metadata `splits/{train,val,internal_test}.json` (anonymized writer ids,
  relative paths only — no scan filenames, no grades).

Sheet pages were located per exam by ECC page-search and labelled Q1/Q2 by
matched correlation of the printed title digit (the two forms are
otherwise identical; booklet page order varies per scan — e014's sheets
sit on pages 5–6). Cell geometry comes from the 10 printed table rules
detected on each exam's synthesized template, with a registration-mapped
fallback to exam-002's reference geometry when direct detection is not
exact; sheets using the fallback are listed in `summary.json`.

## Counts

441 line samples over 256 cells (16 writers × 16 explanation cells; every
sheet built, zero failures; no cell was flagged expected-blank):

| split | writers | cells | line samples |
|---|---|---|---|
| train | 10 | 160 | 264 |
| val | 3 | 48 | 83 |
| internal_test | 3 | 48 | 94 |
| **total** | **16** | **256** | **441** |

Per writer (cells are 16 everywhere): e003 22 · e004 27 · e005 39 ·
e006 21 · e007 28 · e008 20 · e009 31 · e010 25 · e011 24 · e012 27 |
e013 23 · e014 34 · e015 26 | e016 23 · e017 32 · e018 39 lines.

Five Q1 sheets (e003, e008, e012, e014, e018) used the registration
fallback geometry (direct rule detection on their synthesized templates
found no rules — donor-alignment blur); their contact sheets and full-res
cleaned cells were visually QA'd: rows correctly cut, handwriting intact,
printed structure removed. e014's answer sheets sit on booklet pages 5–6
(digit-verified), not 11–12.

## Annotation storage & isolation

`annotations/train/`, `annotations/val/`, `annotations/internal_test/` —
one JSON per sample, atomic writes, split directories fully separate.
Training code must read only `splits/train.json` +
`annotations/train/`; the val/test label directories are never needed at
training time. `evaluation/htr_pilot_sources.json` (writer → scan PDF,
grade-bearing filenames) lives OUTSIDE the package and is for recropping
maintenance only.

## Validation & tests

- `scripts/htr_annotation_validate.py` — duplicate ids, missing images,
  split leakage, exam-002/held-out references, grade-pattern leakage,
  verified-but-flagged records, empty `ok` transcriptions, encoding
  damage, NFC, bidi controls. **RESULT: PASS** on the built package
  (441 samples, 0 annotations).
- Test suite: **136/136 passing** (`tests/test_htr_annotation.py` covers
  the split constants, record rules, resume logic, and that the validator
  catches each seeded violation class; plus the pre-existing 122).
- App smoke test (`scripts/htr_annotation_smoke.py`, AppTest against a
  temp copy with dummy Hebrew text only): **PASS** — renders with resolving
  image paths, Save-and-next writes atomically and advances, a fresh
  session resumes at the first undecided sample, the unreadable button
  writes the exact token, navigation autosaves drafts, and the real
  package's annotation directories stay untouched. A headless
  `streamlit run` was also launched for real: health endpoint OK, page
  served, then stopped.

## Owner workload estimate

441 line samples; typical line is 4–9 handwritten words. At 20–40 s per
line (read against the original, type, one button press) plus flags:

- train (264 lines): ≈ **1.5–3 h**
- val (83 lines): ≈ 0.5–1 h
- internal_test (94 lines): ≈ 0.5–1 h
- **total ≈ 2.5–5 h**, comfortably splittable across sittings — resume is
  automatic, and every save is already on disk.

Suggested order: train first (the fine-tune needs it), then val, then
internal_test. Writers with heavy overflow (e008) will produce most of
the Bad-segmentation/Needs-recrop flags — flag rather than struggle.

## Launch

```
.venv\Scripts\python.exe -m streamlit run scripts/htr_annotation_app.py
```

Annotation rules, storage layout and the recovery/resume procedure are in
`evaluation/htr_pilot/README.md` (also shown in the app sidebar).

## Recrop queue

Empty at build time: all 32 sheets were generated (0 failures) and none
of the 256 cells is known-bad a priori. The queue is populated by the
owner during annotation via **Needs recrop** / **Bad segmentation**;
`evaluation/htr_pilot_sources.json` (outside the package) maps writers
back to scan PDFs and sheet pages for regeneration. Expected sources of
flags: right-margin overflow text cut by the column crop (notably writer
e008) and multi-line cells whose lines merged into one band.
