# Overnight assisted-annotation campaign — 2026-07-17

Bounded autonomous campaign (start 05:55, hard stop 10:55, actual end
see final section). Goal: determine whether the current owner-verified
train lines already carry writer-generalizable signal, and whether local
AI candidates can speed future annotation WITHOUT corrupting ground
truth. All inference local (Ollama qwen3-vl:8b-instruct + in-repo CRNN);
no hosted image APIs; no grading-policy change; Stage B/C untouched;
val/internal_test/held-out exams untouched.

Dataset state at campaign start: 111 train annotations on disk
(86 `ok` owner-verified — 14 more than the 72 in the campaign brief,
annotated by the owner just before start — 23 bad_segmentation,
2 needs_recrop), writers e003 16 / e004 23 / e005 30 / e006 16 /
e007 1 ok-lines.

## Phase 0 — backup

`evaluation/annotation_backups/20260717_055747/` : all 111 annotation
records + `SHA256SUMS.txt` (manifest itself hashes to `b62a237c…0861`).
Verified restorable, and live files verified byte-identical to the
backup at campaign end (`scripts/annotation_backup_verify.py` →
IDENTICAL). No annotation file was created, modified, or deleted by the
campaign.

## Phase 1 — retrospective assisted-annotation benchmark: **REJECT prefill**

Protocol pre-registered in `evaluation/htr_candidates/PROTOCOL.md`
BEFORE generation. Bench = 66 owner-verified train lines excluding the
20 overfit-test ids (44 clean GT + 22 with a partial [לא קריא] span).
Raw outputs saved before GT was opened; per-candidate metadata (model
digest, prompt id, image SHA256, latency, confidence) retained in
`evaluation/htr_candidates/outputs/`. Per-line ledger:
`evaluation/annotation_candidate_results.csv`.

| candidate (66 lines) | mean CER | median | exact | no-edit+minor | major (CER>.4) | omit | insert | halluc lines | wall |
|---|---|---|---|---|---|---|---|---|---|
| A `qwen_line` (strict, line crop) | .843 | .830 | 0 | **0** | **100 %** | .318 | .108 | .318 | 266 s |
| B `qwen_line_cell` (+cell context) | .865 | .863 | 0 | **0** | **100 %** | .430 | .119 | .227 | 323 s |
| C `crnn_overfit` (20-line checkpoint) | .833 | .829 | 0 | **0** | **100 %** | .445 | .060 | .091 | ~20 s CPU |

- Best single candidate line anywhere: CER 0.635. Not one of 198
  candidate evaluations reached even the `moderate` class boundary
  (CER ≤ 0.40).
- Cell context (B) does not help; it slightly increases omissions.
- A↔B agreement does NOT predict correctness: lines with agreement
  ≥ 0.9 average CER 0.94 (worse than the 0.84 overall) — consistent
  with the oracle-ensemble finding that agreement carries no
  correctness signal on this handwriting.
- Contamination tripwire: 0 overfit ids in the bench; CRNN caveat —
  4 of its 20 training lines share writer e004 with 19 bench lines,
  and it still scored CER .84 on that writer (no writer-level leak
  advantage observed).
- 5 VLM calls (of 132) died in the known degenerate-repetition loop and
  count as failed candidates.

**Pre-registered gate** (coverage ≥ 20 %, no-edit+minor ≥ 60 %, major
≤ 10 % in a model-visible subset): no agreement or confidence subset
comes close (best subset: helpful 0 %, major 100 %). **Decision:
REJECT_PREFILL — candidates are usually harmful; showing them would
risk anchoring errors into ground truth while saving zero typing.**

## Phase 2 — annotation-app assistance: NOT SHIPPED (per gate)

Because Phase 1 rejected prefill, no candidate display, no Copy A/B
buttons, and no candidate generation for untouched samples were added,
and the 40-sample candidate queue was NOT generated. The only Phase-2
item shipped is the ground-truth protection that is justified
regardless: **owner-verified records are now overwrite-locked in the
app** until a per-sample "Unlock this verified record" checkbox is
deliberately ticked (`locked_against_overwrite` in
`scripts/htr_annotation_lib.py`, enforced inside `commit()` and via
disabled buttons). Autosave/resume semantics unchanged (AppTest smoke
scenario 6 covers the lock; scenarios 1–5, 7 unchanged → PASS).

## Phase 3 — annotation priority queues

`evaluation/annotation_priority_queue.csv` — all 153 untouched train
samples, fully deterministic (fixed weights, sample_id tie-break),
built ONLY from model-visible signals: line-crop image statistics
(contrast, blur, ink fraction, edge-touch, text-band count,
strike-through run), split geometry, and the CRNN decode's confidence
and script mix (decode text used as a signal only — never as a label).
`candidate_agreement` is intentionally empty: prefill was rejected, so
no VLM candidates exist for untouched samples.

- `easy_rank` 1–153: easiest first (22 near-blank one-click confirms
  surface at the top).
- `info_rank`: active-learning value (low CRNN confidence, long,
  mixed-script, but well-segmented).
- `recrop_rank`: 2 samples flagged for geometry repair before typing.

## Phase 4 — writer-generalization diagnostic

Protocol pre-registered in `evaluation/htr_gen_diag/PROTOCOL.md`; 3
writer-grouped folds (hold out e005 / e004 / e003), fixed overfit-test
architecture and settings, EMPTY val list (selection on train loss
only), per-fold symbol table = predeclared base + fold-train chars,
aug ×3, uniform epoch/wall-clock caps. Results:
`evaluation/writer_generalization_diagnostic.md` (this file is written
by `scripts/writer_gen_diagnostic.py report` after the folds finish).

| fold (held-out) | train lines | epochs (35-min cap) | train CER | held-out CER | usable | omit | insert |
|---|---|---|---|---|---|---|---|
| e005 | 44 | 73 | .060 | **.747** | 0 | .376 | .000 |
| e004 | 46 | 72 | .025 | **.735** | 0 | .058 | .149 |
| e003 | 48 | 65 | .011 | **.797** | 0 | .393 | .051 |

**Interpretation (per the pre-registered guide): WEAK cross-writer
signal — real, but memorization-dominated.** Training writers are
memorized (CER .01–.06) while held-out writers sit at .73–.80 with 0 %
usable lines and no exact matches: the model has NOT learned to read
new writers. Yet the signal is not zero: two of three folds already
beat the best VLM ever measured on this handwriting (CER .786) from
only ~45 training lines, all folds beat the 20-line overfit checkpoint
on the same task (.833), and predictions are Hebrew-letter strings of
roughly correct length rather than garbage — transfer improves as
training lines/writers grow. Confidence correlates only marginally
with CER (e004/e005 higher-confidence buckets ~.04 lower CER; e003
flat) — not yet an abstention signal. Well short of the pilot's
CONTINUE gate (val CER ≤ .60), which argues for more writers/lines
before the official pilot, not for config tuning.

## Phase 5 — flagged-sample QA

`evaluation/flagged_sample_review/` — contact sheets grouped by
category (`bad_segmentation.jpg` 23 lines, `needs_recrop.jpg` 2;
skipped and blank/unreadable groups are empty), each panel showing the
ORIGINAL cell, the cleaned cell with CURRENT line bands (red) vs a
deterministic re-segmentation PROPOSAL (green; halved merge gap, halved
profile threshold, 0.6× min height), and the flagged line crop.
`proposals.json` records proposed band geometry for all 23 unique
flagged cells (23/23 proposals differ from current bands). **No
annotation status was changed**; applying any proposal requires an
owner-approved rebuild.

## Phase 6 — validation

- New safety tests: `tests/test_annotation_campaign.py` (13 tests):
  candidates can never be verified / generator cannot write annotations
  / verified-lock semantics / app commit guard / prompts contain only
  fixed text (no Hebrew, no GT) / bench selection stores ids only /
  candidate metadata + image hashes / fold writer-separation + empty
  val / diagnostic eligibility excludes flagged+span lines /
  contamination exclusion / internal_test decode refusal / backup
  restorability.
- Full suite: **154/154 passed**. Annotation validator: **RESULT: PASS**
  (441 samples / 111 annotations).
- App smoke (AppTest, temp package only): PASS incl. new lock scenario.
- Backup invariant at end of campaign: IDENTICAL (111/111 files).

## Final verdict

1. **Does local AI candidate prefill save owner effort without
   increasing label errors?** **No.** Every local candidate system
   (Qwen strict line, Qwen line+cell, CRNN) produced major-error output
   (CER > 0.4) on 100 % of the 66 benchmark lines; the single best
   candidate anywhere was CER 0.635. Reading and fixing such text is
   slower than typing, and displaying it creates anchoring risk with
   zero benefit. Prefill is rejected by the pre-registered gate.
2. **What fraction of samples can safely receive a candidate
   suggestion?** **0 %.** No model-visible subset (A↔B agreement at any
   threshold 0.7–0.99, CRNN confidence at any threshold) contains even
   one helpful candidate; high A↔B agreement actually predicts WORSE
   candidates (CER 0.94 at agreement ≥ 0.9).
3. **Does the current verified-line set show cross-writer learning
   signal?** **Weak but real.** Writer-grouped folds reach held-out CER
   .735–.797 (usable 0 %) vs train CER .01–.06: heavy writer
   memorization, no usable reading of unseen writers — but 2/3 folds
   beat the best VLM (.786) from ~45 lines, and transfer improved over
   the 20-line checkpoint (.833 → ~.76 mean). Signal grows with data;
   it is nowhere near the pilot's CONTINUE bar (.60) yet.
4. **Annotate more train lines, begin validation annotation, or collect
   more writers first?** **Annotate more TRAIN lines, prioritising the
   six writers that currently have zero or almost-zero verified lines
   (e007–e012)** — writer diversity, not line count per writer, is the
   binding constraint the diagnostic exposes. Use the easy-first queue;
   the 22 near-blank confirms take minutes. Begin val (e013–e015)
   annotation after train writers are covered, to unlock the official
   pilot. Collecting new scans beyond the 41 dev exams is NOT yet
   needed.
5. **Single highest-value next action?** Work through
   `evaluation/annotation_priority_queue.csv` in easy_rank order for
   writers e007–e012 (~60–80 lines, roughly 2–3 h at the observed
   pace), then annotate val and run the pre-registered pilot
   (`evaluation/htr_pilot_gates.md`) — its 6-trial budget is still
   fully intact, and this diagnostic consumed none of it.

Safety compliance: no annotation was created/modified/deleted by AI
(backup byte-identical at end); nothing was auto-verified; ground truth
was hidden during all inference; no hosted image API was touched; the
48 held-out exams, internal_test, val, and exam 002 were never read;
grading policy and Stage B/C untouched; splits and manifests preserved.
