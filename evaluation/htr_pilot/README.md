# HTR pilot annotation package

Everything needed to hand-transcribe the pilot line crops, fully offline.
Built by `scripts/htr_pilot_build.py`; do not edit generated files by hand.

## Launch

```
.venv\Scripts\python.exe -m streamlit run scripts/htr_annotation_app.py
```

A browser tab opens at http://localhost:8501. Pick the split in the
sidebar (start with **train**). The app always resumes at the first
undecided sample — closing the tab or the terminal loses nothing that was
saved (every button press writes the record to disk immediately).

## What you annotate

One **line crop** (cleaned, student-ink-only) at a time. The cleaned cell
and the ORIGINAL cell are shown underneath — always cross-check against
the original: the cleaning removes table lines and red ink but must never
invent strokes.

## Rules

1. Copy EXACTLY what is written, character for character, right to left.
   No spelling fixes, no abbreviation expansion, no completing words.
2. A single unreadable word → insert `[לא קריא]` in its place (button
   available). The rest of the line is still transcribed.
3. Whole line unreadable → **Whole line unreadable** button.
4. No student writing in the crop → **Blank** button ("expected BLANK" in
   the header is a hint from ink statistics, still confirm visually).
5. Crop shows parts of two text lines, or cuts a line in half → **Bad
   segmentation** (do not transcribe fragments).
6. The cell box itself is wrong (wrong row, borders inside, half a cell)
   → **Needs recrop**.
7. Crossed-out but readable text: transcribe it and note "crossed out" in
   Notes. Overwritten text: transcribe the final (top) version.
8. Latin words and math (e.g. `Echo`, `2x`, `DC`) are copied as written,
   embedded in the Hebrew text.
9. Not sure right now → **Skip for now**; it stays in the undecided queue.

## Storage layout (autosave/recovery)

- `splits/{train,val,internal_test}.json` — sample metadata, one file per
  split. Training code must load ONLY its own split file + its own
  annotations directory.
- `annotations/<split>/<sample_id>.json` — one record per sample, written
  atomically on every save. Deleting a file returns that sample to the
  undecided queue; nothing else references it.
- `images/<writer>/…` — line crops (`*_lN.png`), cleaned cells
  (`*_cell_clean.png`), originals (`*_cell_orig.jpg`).
- `contact/<writer>_q{1,2}.jpg` — per-sheet contact sheets (title strip +
  original|cleaned thumbnails per row) for quick QA of geometry and
  sheet identity.
- `summary.json` — build stats; `evaluation/htr_pilot_sources.json`
  (OUTSIDE this package) maps writers to scan PDFs and must never be read
  by training code (scan filenames carry grades).

## Recovery / resume

Relaunch the same command; the app reopens at the first sample that is
missing or still `draft`/`skipped`. To revisit anything, use Previous/Next
or re-annotate — the newest save wins. If a record was saved wrongly, just
navigate back to it and save again.

## Validation

```
.venv\Scripts\python.exe scripts/htr_annotation_validate.py
```

Run after any annotation session. It checks duplicate ids, missing
images, split leakage, exam-002/held-out references, grade-pattern
leakage, verified-but-flagged records, empty `ok` transcriptions and
encoding damage. Must print `RESULT: PASS` before annotations are used.
