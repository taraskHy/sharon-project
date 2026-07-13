# Student-ink isolation experiment (template subtraction + blue-ink separation)

Date: 2026-07-13 (owner-directed follow-up to the closed 8-iteration
transcription campaign). Status: **complete — see Phase 4 decision.**

## Hypothesis

The recognizers fail partly because explanation crops still contain printed
text, table borders, grid lines and other document structure. A registered
blank-template subtraction followed by student-blue-ink isolation may
produce a substantially cleaner handwriting image and improve local HTR
accuracy.

Constraints honoured: local models only; the 48 held-out exams untouched
(everything below uses `test/` = the 41 sanctioned dev exams and exam 002
itself); grading policy unchanged; no training; the 16 human-verified
exam-002 cells used ONLY as hidden post-inference ground truth via
`scripts/hebrew_bench_eval.py`; preprocessing thresholds chosen from image
statistics and visual mask inspection, never from CER.

## Phase 1 — pipeline (`scripts/student_ink_isolation.py`)

### Template

No blank answer-sheet exists in the repo: the born-digital
`sample_data/Exam_solution.pdf` contains only the question booklet
(questions + solutions, 12 pages) — the answer-table pages students write
on exist only inside the scanned booklets. The blank template is therefore
**synthesized**: per-pixel median of 16 other dev exams' scans of the same
printed sheet, each red-masked and ECC-registered onto exam 002's own page
frame. Printed structure is common to all exams and survives the median;
per-student handwriting occupies different pixels per exam and drops out
(residual ghosting measures ≥ ~215 grey — far above every print threshold
used). The template is in the scan's own colour space (same paper tone,
print darkness and scanner blur), which subtracts more cleanly than a
born-digital render would.

### Registration

- Donor booklets vary in page count and sheet position, so each donor is
  searched: every page scored by quarter-scale ECC correlation against the
  reference page, winner refined at half scale (affine ECC, phase-corr
  init, 200 iters). Gate: cc >= 0.55, first 16 passing donors kept
  (numeric order, deterministic).
- Results: 16/16 donors kept per sheet. Q1 sheet cc 0.59–0.85; Q2 sheet
  cc 0.59–0.86. Non-standard sheet locations found automatically (e.g.
  donor 013's Q1 sheet on page 12, donor 014's on page 5). Two donors
  contributed their Q1 page to the Q2 template (the two printed forms are
  near-identical apart from the title digit); with 14/16 correct pages the
  median is unaffected — verified visually on the template PNGs.
- Full log incl. warp matrices:
  `evaluation/student_ink_isolation_artifacts/registration_report.json`.

### Masks (all thresholds frozen from statistics BEFORE any recognition)

| constant | value | justification (measured on e002 p11/p12 at 2200 px width) |
|---|---|---|
| T_BLUE (blue dominance b−max(r,g)) | 25 | paper pixels P99 = 12, P99.9 = 38 (JPEG fringing near print); blue-ink cores far above; 25 sits on the stable 20–30 plateau |
| T_dark | per-page Otsu | 157/163 (scan), 153/150 (template); clamped [100,200], percentile fallback |
| red rule | r−max(g,b) > 50 & r > 70 | identical to production `autograder.masking`; dilated 2 px for pen halo |
| template print dilation | 2 px | absorbs residual registration error (≤ ~1–2 px after page-level affine) |
| T_TEXTURE on 9 px box-blurred template | 200 | dithered header fill: local mean P50 = 188; page paper local-mean P1 = 207 — separates print texture from paper; catches dither the per-pixel test misses |
| despeckle | components < 8 px | stroke width at this zoom is 3–5 px; scanner dust 1–4 px |
| final mask dilation | 1 px | keeps anti-aliased stroke edges; only ORIGINAL scan pixels are copied |

Final masks:

- `E1 blue-only` = despeckle(blue & ~red) — colour separation alone.
- `E2 template-subtracted` = despeckle((blue | (dark & ~template-print)) & ~red)
  — blue strokes always kept (even where they cross print); non-blue dark
  pixels kept only where the registered template is clean. Nothing is
  reconstructed, sharpened or redrawn; E images copy original scan pixels
  onto white.

One iteration was needed after visual inspection (before any recognition):
the table header's dithered grey fill medians into a texture too light for
the per-pixel dark test, leaving a speckle band in `e002_q2_r1` (the only
crop that includes header fill). The T_TEXTURE term above fixed it;
`q2_r1`'s mask dropped 35.4k → 23.3k px while all other cells changed by
< 100 px.

### Line segmentation (F)

Row-profile bands computed on the E2 ink mask with the constants of
`scripts/segment_lines.py` (profile > 0.01 of width, merge gap 3.5 %, min
height 8 %, pad 2 %). 21 line crops over the 16 cells.

### Artifacts

`evaluation/student_ink_isolation_artifacts/`:
`templates/` (2 synthesized blank pages), `cells/<cell>/{A_original,
B_template, C_absdiff, D_mask, E1_blueonly, E2_templatesub}.png`,
`lines/<cell>/lineN.png`, `contact/<cell>.png` (A–F contact sheet per
cell), `manifests/manifest_{blueonly,templatesub,lines}.json`,
`registration_report.json`.

## Phase 2 — visual integrity check (done BEFORE recognition)

Contact sheets: `evaluation/student_ink_isolation_artifacts/contact/`.
Assessed per cell on E2 (the template-subtracted image):

| cell | handwriting | printed structure | red | segmentation (F) |
|---|---|---|---|---|
| q1_r1 | complete (incl. heavy cross-out) | removed | n/a | merged: cross-out bridges 2 lines → 1 band |
| q1_r2 | complete | removed | n/a | correct (1 line; bleed fragments dropped) |
| q1_r3 | complete | removed | n/a | correct (2 lines) |
| q1_r4 | complete | removed | n/a | merged: 2 staggered lines → 1 band |
| q1_r5 | complete | removed | removed (margin note edge) | correct (1 line) |
| q1_r6 | complete | removed | n/a | split-extra: neighbour-row bleed became a 2nd thin band |
| q1_r7 | complete | removed | removed (margin curl) | correct (2 lines) |
| q1_r8 | complete (incl. struck-through word) | removed | removed (pink remnant) | merged: 3 lines → 1 band |
| q2_r1 | complete | removed except 2–3 tiny corner dot clusters | n/a | split-extra: bleed band |
| q2_r2 | complete | removed | n/a | merged: 2 → 1 |
| q2_r3 | complete | removed | n/a | split-extra: bleed band |
| q2_r4 | complete (incl. overwritten patch) | removed | n/a | merged: 3 → 1 |
| q2_r5 | complete | removed | n/a | merged: 2 → 1 |
| q2_r6 | complete (incl. Latin "Echo") | removed | n/a | merged: 3 → 1 |
| q2_r7 | complete | removed | n/a | merged: 2 → 1 (bleed included) |
| q2_r8 | complete (incl. 2 crossed-out words, "x", "2x") | removed | n/a | merged: 2 → 1 |

- **E2: 16/16 handwriting preserved completely; printed structure removed
  completely in 15/16 (q2_r1 keeps 3 tiny corner clusters); red remnants
  removed 3/3.** No real text line was split mid-line; merges keep all ink
  inside one band; 3 cells produce an extra junk band from neighbour-row
  bleed (student ink of the adjacent row, present in the originals too).
- **E1 (blue-only): handwriting only PARTLY preserved in all 16 cells** —
  dark stroke cores with low blue dominance are visibly eroded/broken.
  Template subtraction is what recovers them (this is the E1 vs E2
  ablation, by design).

Gate passed → recognition allowed.

## Phase 3 — controlled recognition ablation

Fixed recognizer settings across all arms: `qwen3-vl:8b-instruct` (Q4),
`strict_fidelity` prompt, temperature 0, max_tokens 400, JSON-schema
output, 3 runs — identical to the campaign's it2/it4 except that no
runtime preprocessing is applied (the input IMAGE is the ablation
variable). Dedicated HTR: `sivan22/hdd-words-ocr` (it6 pipeline,
safetensors, greedy, 2 runs) — chosen over surya because surya's failure
is script-level (reads Hebrew cursive as English words), which no input
cleaning can address, while hdd-words is Hebrew-handwriting-trained.
`isol6` feeds it the pre-segmented E2 lines (its word segmentation kept).

Campaign reference rows (same 11 strict / 5 hard cells): it2 strict+none
CER .842, it4 strict+contrast CER .786 (best), usable 0 %.

Configs (outputs under `evaluation/hebrew_bench/outputs/<config_id>/`,
saved before ground truth was read; ledger
`evaluation/student_ink_isolation_results.csv`):

| config | input | model | runs |
|---|---|---|---|
| isol0_orig_e002 | A original crops (16 e002 cells) | qwen3-vl:8b-instruct | 3 |
| isol1_blueonly | E1 blue-ink-only | qwen3-vl:8b-instruct | 3 |
| isol2_tsub | E2 template-subtracted | qwen3-vl:8b-instruct | 3 |
| isol3_tsub_lines | F segmented lines of E2, joined | qwen3-vl:8b-instruct | 3 |
| isol4_hdd_blueonly | E1 | hdd-words-ocr | 2 |
| isol5_hdd_tsub | E2 | hdd-words-ocr | 2 |
| isol6_hdd_tsub_lines | F lines of E2 | hdd-words-ocr | 2 |

### Results (strict cells; hidden GT read post-inference by the evaluator)

| config | input | CER | WER | omission | halluc | usable | stability | honest hard flags | wall s |
|---|---|---|---|---|---|---|---|---|---|
| isol0_orig_e002 | original | .866 | 1.094 | .318 | .134 | **0** | .070 | 0/15 | 199 |
| isol1_blueonly | E1 blue-only | .870 | 1.267 | .129 | .245 | **0** | .126 | 0/15 | 147 |
| isol2_tsub | E2 template-sub | **.790** | 1.063 | **.051** | **.062** | **0** | .134 | 0/15 | 140 |
| isol3_tsub_lines | F lines of E2 | .828 | 1.094 | .126 | .119 | **0** | .073 | **3/15** | 166 |
| isol4_hdd_blueonly | E1 | .978 | 1.424 | .235 | .356 | **0** | .000 | 0/10 | 140 |
| isol5_hdd_tsub | E2 | .937 | 1.259 | .341 | .282 | **0** | .000 | 0/10 | 119 |
| isol6_hdd_tsub_lines | F | .936 | 1.259 | .329 | .279 | **0** | .000 | 0/10 | 111 |

Per-cell CER under isol2 (run 1): best .66 (q1_r4), worst 1.05 (q1_r1,
the crossed-out cell) — a flat failure plateau; no cell approaches the
usable threshold (.25). isol0 (.866) reproduces the campaign's it2 (.842)
within run-to-run server drift (pairwise stability ~.07), confirming a
comparable baseline on the same day/server.

Notable secondary effects (real, but not success by the gate):

- At an equal-or-better CER, the clean E2 input **changes the error
  profile dramatically**: omission .318 → .051 (6×) and hallucination
  .134 → .062 (2×) vs the same-day original baseline. The VLM attends to
  all of the ink and stops inventing content — it still decodes the
  cursive incorrectly.
- isol2 reaches the campaign-best CER (.786, it4-with-contrast) **without
  any contrast preprocessing** — cleaning reproduces the known ceiling by
  another path; it does not move the ceiling. (Stacking contrast on top of
  E2 was deliberately not tried post-hoc — that would be tuning
  preprocessing against CER.)
- The lines arm produced the **first honest [unreadable] flags of the
  entire campaign** (3 of 15 hard-cell evaluations; every prior config was
  85/85 confabulation).
- hdd-words improves slightly on clean input (.963 campaign → .937) yet
  remains catastrophic — the dedicated HTR cannot read this hand
  regardless of input cleanliness.

## Phase 4 — decision

**REJECT PREPROCESSING HYPOTHESIS.** The handwriting-only images are
visually faithful (Phase 2: 16/16 complete stroke preservation, printed
structure and red ink removed), yet recognition stays at the existing
plateau: CER .790 vs the campaign's .786, usable rate 0 %, best single
cell .66.

Failure attribution, per the required taxonomy:

- **Registration: not the failure.** 16/16 donors registered per sheet
  (cc .59–.86); print alignment verified by overlay; residuals absorbed by
  the 2 px template dilation.
- **Ink separation: not the failure.** Phase-2 inspection found no lost
  strokes in E2 (cross-outs, overwrites, Latin words and math notation all
  preserved; print and red removed).
- **Line segmentation: not the failure.** No mid-line splits; merges keep
  all ink in one band; the lines arm scores within noise of the full-cell
  arm (.828 vs .790), so segmentation quality is not what separates
  success from failure.
- **Recognition despite clean input: CONFIRMED.** With document structure
  gone and only student ink on white, the strongest local recognizer still
  misreads the cursive at CER ≈ .79 on every cell, and the dedicated
  Hebrew HTR stays ≈ .94. The bottleneck is the recognizer's inability to
  read this handwriting style, not scene clutter.

This does NOT show OCR of these exams is impossible — it shows input
cleaning cannot bridge the gap, which reinforces the campaign's structural
diagnosis (no available model knows this script style; HebHTR's
self-reported 4.76 % CER on comparable exams proves learnability). The
writer-separated kraken/PyLaia fine-tune pilot remains the recommended
next step; the E2 cell images and F line crops produced here are directly
reusable as its cleaned inputs.

Owner review requested before any further step.
