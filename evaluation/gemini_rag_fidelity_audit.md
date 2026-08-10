# Gemini-RAG fidelity audit — REJECT semantic RAG OCR repair

Date: 2026-08-11. Image-first audit of all 16 items whose text
`qwen_rag_ocr_v1`-style frozen repair changed on the frozen Gemini raws
(`gemini_rag_ocr_v1`). Protocol: per-item stroke-level adjudication of the
handwriting crop against raw vs repaired text, classifications frozen
BEFORE any reference access; references joined post-hoc only. Per-item
evidence: `hebrew_bench_v2/outputs/gemini_rag_ocr_v1/fidelity_audit.json`.

## Findings

- **The apparent Gemini+RAG CER gains were dominated by deterministic
  protocol-wrapper cleanup**, not handwriting recovery. On the 16 changed
  items: raw Gemini mean CER **0.494**; wrapper-strip-ONLY **0.353**;
  full RAG **0.378** — a trivial deterministic strip of the
  `{"transcription": ...}` envelope outperforms the entire RAG repair,
  and RAG's word edits are net harmful beyond it.
- Classifications (16 changed items): **1** IMAGE_SUPPORTED_FIX,
  **0** PLAUSIBLE_BUT_AMBIGUOUS, **5** UNSUPPORTED_NORMALIZATION,
  **4** SEMANTIC_CORRECTION_RISK, **6** OTHER (5 pure protocol-wrapper
  strips, 1 line-wrap whitespace join).
- **All four CER-worsened items lacked image support** — chunk-driven
  normalization or outright fabrication (e.g. a visible capital E
  rewritten to the course luminance channel Y; a fabricated
  "ל-1/2N … נראית כהה יותר" completion sourced near-verbatim from a
  retrieved chunk while the page reads "הוזז ימינה באותו מרחק שהיה
  במקור"; a high→low frequency flip against clearly written הגבוה"ם).
- **RAG edit logs are not reliable enough to serve as an audit trail**:
  12/16 items carried undocumented rewrites, several edit entries were
  no-ops, and one documented edit was never applied (phantom edit).
- **Semantic-risk self-labels failed**: `semantic_change_risk` was false
  on every one of the 4 items the audit classified as
  SEMANTIC_CORRECTION_RISK.
- Where the image-first classification conflicted with a RAG edit, the
  post-hoc reference agreed with the strokes, independently corroborating
  the audit.

## Decision

**Do NOT use course-RAG to rewrite OCR text for grading.** It manufactures
exactly the failure a grading pipeline must never produce — student
mistakes silently rewritten toward course-correct answers — while its
gains reduce to a deterministic protocol cleanup that needs no model.
Follow-up (separate arm): `gemini_protocol_clean_v1`, a pure deterministic
envelope parser, to establish Gemini's true OCR baseline.
