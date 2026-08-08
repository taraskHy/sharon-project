# Unlimited-OCR frozen 5-item smoke — precommitted rule application

Config: `unlimited_ocr_gundam_eager` (baidu/Unlimited-OCR @
`07dea832e22aefee32ad281d4b80551282e1c168`, BF16, eager attention,
documented gundam preset base_size=1024 / image_size=640 / crop_mode=True,
prompt `<image>document parsing.`, no_repeat_ngram_size=35 +
ngram_window=128, temperature 0, offline local snapshot). Raw predictions
were persisted for all 5 items BEFORE any reference access
(`outputs/unlimited_ocr_gundam_eager/{run1,raw_predictions}/`).

## Measured (evaluation/unlimited_ocr/smoke5_eval.json)

| Metric | Value |
|---|---|
| n | 5/5 persisted, 0 runtime errors |
| Empty outputs | **4/5** (model labels the whole crop `<\|det\|>image [...]<\|/det\|>` and emits no text) |
| Mean CER | **0.9543** |
| Median CER | 1.0 (min 0.7714, max 1.0) |
| Usable CER<=0.25 | 0/5 |
| Usable CER<=0.50 | 0/5 |
| Mean WER | 1.0; omission 0.886 |
| Latency | mean 2.3 s/item; peak VRAM ~8.9 GiB total |

The single non-empty output (hl_e006_q1_r3__l1, CER 0.771) is
plausible-looking Hebrew with niqqud added — the plausible-but-wrong
failure class, not a usable transcription. QUALITATIVE LABEL, explicitly
a judgment: the model behaves as a printed-document parser; extreme
strip-shaped handwriting crops are classified as picture regions and
skipped.

## Integration diligence (no references used)

- Instruction-style prompts (incl. the repo's own commented `Free OCR.`,
  `Extract the text in the image.`) return EMPTY on this revision;
  only the README task prompt `document parsing.` produces output
  (evaluation/unlimited_ocr/diag_prompts.json).
- The ONLY other documented preset (base: image_size=1024,
  crop_mode=False) was probed on the same 5 items
  (`unlimited_ocr_diag_base`): 2 empty, 2 hallucinated LaTeX repetition
  loops, 1 identical Hebrew. Post-hoc CER mean 3.35 (insertion-driven,
  diag_base5_eval.json) — strictly worse. Both documented invocation
  modes were tried faithfully; the official interface cannot make this
  model transcribe these crops without prompt/model tuning, which is
  out of bounds.

## Precommitted classification: **U-A — NONVIABLE**

Rule triggers (any one suffices; three hold):
1. mean CER 0.9543 > 0.75 ✓
2. 4 of 5 empty/catastrophic >= 3 ✓
3. repeated structural failure of literal transcription (image-region
   refusal in gundam; LaTeX confabulation in base) that the official
   interface cannot fix without tuning ✓

**Action: Unlimited-OCR inference STOPPED. The remaining 15 frozen-gate
items were NOT run.** No Phase-10 expansion, no Phase-11
grading-decision-preservation run for this arm.

## Same-sample comparisons (identical items only)

| vs | paired n | Unlimited-OCR mean CER | Other mean CER | W/T/L |
|---|---|---|---|---|
| local Qwen (qwen8b_strict_contrast) | 5 | 0.954 | 0.801 | 0/0/5 |
| ML Kit (mlkit_ink_rtl_a1, top-1) | 5 | 0.954 | 0.459 | 0/0/5 |
| Gemini 3 Flash | 1 | 1.0 | 0.344 | 0/0/1 |

A 5-item smoke cannot prove or disprove general quality; it is
sufficient only for the precommitted viability gate above. Reference-side
comparisons use the same normalize/CER definitions as every other arm.
