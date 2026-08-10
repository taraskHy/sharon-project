# gemini_rag_ocr_v1 — frozen Gemini raws × frozen RAG repair, decision record

Date: 2026-08-10/11. Identical frozen repair configuration as
qwen_rag_ocr_v1 (same course index `CV` 430 chunks / bge-m3 / top-k 4 /
repair model / prompt — prompt_sha256 matches; fail-closed question
context), applied to the committed immutable `gemini3_flash`
transcriptions via `--config-id`. **No new Gemini API calls.** References
joined strictly after all 32 repair records were persisted.

## OCR (paired, identical 32 items)

| Metric | raw Gemini | Gemini + RAG |
|---|---|---|
| N | 32 | 32 |
| mean CER | 0.4124 | **0.3546** |
| median CER | 0.4432 | **0.3370** |
| usable CER<=0.25 | 9/32 | **14/32** |
| usable CER<=0.50 | 20/32 | 21/32 |

- 16/32 texts changed: 11 improved / 4 worsened / 1 CER-neutral
  (mean delta on changed items −0.116; best −0.275, worst +0.118).
- 3 repair failures degraded safely to raw_text.
- 0 semantic-risk edits; 1 needs_review (uncertainty region).

## Grading-decision preservation (identical 12-cell subset, same fixed judge)

Every grading verdict remained **byte-identical** to the raw-Gemini
baseline (decision match 0.4167, safe rate 0.5, same verdict per cell):
**zero new silent upgrades and zero new silent downgrades** on the judged
subset. This is decision preservation under the same fixed judge — NOT
actual grading accuracy.

## Decision

**RAG materially improved transcription on this 32-item sample** — a 14%
relative mean-CER reduction and +5 items into the usable<=0.25 band —
confirming that course-aware repair helps when the input OCR carries real
signal (contrast: zero effect on the ~0.97-CER local-Qwen input,
`evaluation/qwen_rag_ocr_v1_decision.md`). **Grading safety is not yet
established**: only 12 cells were judged, and a 32-item / few-writer
sample bounds all conclusions. Before any grading use, the repairs'
image fidelity must be audited (are gains visually supported recovery or
course-context normalization?) — that audit is the next gate.
