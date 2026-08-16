# qwen38_rag_ocr_v1 — frozen course-RAG repair over raw Qwen3.8 OCR, decision record

Date: 2026-08-16. Frozen `m2_rag_ocr.py` (course CV, existing 430-chunk
index, bge-m3, top-k 4, frozen repair model/prompt — same prompt_sha256 as
qwen_rag_ocr_v1 / gemini_rag_ocr_v1, fail-closed question context) run
ONLY over the persisted `qwen38_27b_q4km` transcriptions (no image
inference, no reference/key/rubric/grader input). 102 eligible handwritten
items, 102 records persisted; references joined only afterwards.

## Paired OCR (identical 102 items, canonical CER)

| Metric | raw Qwen3.8 | Qwen3.8 + RAG |
|---|---|---|
| mean CER | 2.3909 | 2.3951 |
| median CER | 0.782 | 0.784 |
| usable <=0.25 | 0 | 0 |
| usable <=0.50 | 1 | 1 |

12/102 texts changed: **2 improved / 8 worsened / 2 CER-neutral**. 30
repair-call failures (repair model's own JSON broke on long/looping raws)
degraded safely to raw. Self-labels: semantic_change_risk 1, needs_review 2.

## Image-first fidelity audit (all 12 changed items; blind protocol,
references joined post-hoc; `outputs/qwen38_rag_ocr_v1/fidelity_audit.json`)

**0 IMAGE_SUPPORTED_FIX / 0 PLAUSIBLE_BUT_AMBIGUOUS /
9 UNSUPPORTED_NORMALIZATION / 1 SEMANTIC_CORRECTION_RISK / 2 OTHER**
(punctuation/whitespace only). 7/12 carried undocumented rewrites.

Dominant pattern — worse than the Gemini case: the raw Qwen3.8 text on
these items is frequently a wholesale hallucination of the strokes (legal
boilerplate, jazz genres, load-testing jargon), and the repair does not
detect that; it re-hallucinates the sentence into course-plausible
content pulled from retrieved chunks (`f`, `prior`, `הרציף`,
`לטרנספורם`, `x = x_w`, `רעש גאוסיאני`). It also destroyed the one
recoverable correct course token it saw (`carry` -> `קיימת`, where the
strokes clearly read `Canny`). The repair model's semantic-risk label
flagged 1 of the 12 changed items and missed the audited
SEMANTIC_CORRECTION_RISK case.

## Grading-decision preservation (identical 12 cells, fixed judge)

raw Qwen3.8: match 0.1667 / safe 0.5833; Qwen3.8+RAG: match 0.1667 /
safe **0.5000** — one safe abstention became a silent wrong verdict.
Same regression signature as qwen_rag_ocr_v1.

## Answer to the main question

**No.** Stronger-than-8B Qwen3.8 OCR still does not carry enough genuine
signal on Hebrew handwriting for course-RAG to help without rewriting
what the student wrote: zero image-supported fixes, net CER harm, and a
grading-safety regression. Combined with the Gemini-RAG fidelity audit,
this closes the question across three raw-OCR quality levels: course-RAG
text repair is REJECTED for grading regardless of the OCR source.
