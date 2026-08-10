# qwen_rag_ocr_v1 — real-course-material evaluation, decision record

Date: 2026-08-10. Course corpus: the owner's real 118-page image-processing
lecture notes (course id `CV`, 430 chunks, bge-m3, index config_hash
d563bfc2e895d71b; the ingestion content-screen flag on the file was a
verified false positive — math subscripts extract as "<n> <letter>" lines).
Arm config: frozen `outputs/qwen_rag_ocr_v1/config.json` — source
qwen8b_strict_contrast, top-k 4, repair qwen3-vl:8b-instruct temp 0,
fail-closed question context (none supplied). References joined strictly
after all repair records were persisted.

## OCR (paired, identical items)

| Metric | raw Qwen | Qwen + RAG repair |
|---|---|---|
| N | 101 | 101 |
| mean CER | 0.9690 | 0.9677 |
| median CER | 0.8043 | 0.8061 |
| usable CER<=0.50 | 0 | 0 |
| usable CER<=0.25 | 0 | 0 |

- 31/101 texts changed at all; among them 7 improved / 7 worsened /
  17 CER-neutral (best single item −0.351, worst +0.157).
- 5 repair-call failures degraded safely to raw_text (by construction).

## Grading-decision preservation (identical 12-cell subset, same fixed judge)

- decision preservation UNCHANGED: 0.0833 (both arms).
- safe abstention 0.6667 → 0.5833: **one formerly safe abstention became a
  silent wrong verdict** (partially_valid judged invalid instead of
  unintelligible). Rows: `evaluation/m2_grading_results.csv`.

This is NOT actual grading accuracy — both sides are the same fixed judge;
the reference-side decision is not ground truth.

## Decision

**`qwen_rag_ocr_v1` provides no useful improvement and must not be used
for grading.** The repair behaved conservatively (fidelity-first held; no
confabulation cascade), but it was not harmless: it introduced one
grading-safety regression on the 12-cell subset. With raw OCR at ~0.97
mean CER the input carries too little signal for terminology-level repair
to act on — the bottleneck is recognition quality, not vocabulary. This
independently re-confirms the settled Gemini + ML Kit direction.

Open (not started here): the same frozen repair arm applied to a stronger
raw-OCR source (e.g. the frozen Gemini transcriptions) is the natural
follow-up question.
