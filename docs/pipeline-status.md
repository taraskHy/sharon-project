# Pipeline status — component readiness (2026-09-02)

The current end-to-end shape, with every component marked. **SHADOW is not
production activation**, and nothing here is activated.

```
scanned exam
    ↓
deterministic variant / MC logic                       READY
    ↓
OpenRouter OCR ONLY (ocr_primary)                      EXPERIMENTAL
  · two-layer cloud boundary, deny-by-default          READY
  · Gemini→Sonnet hard-failure fallback                EXPERIMENTAL
    ↓
immutable transcription (content-hashed artifacts)     READY
    ↓
local semantic grader (no cloud path)                  READY (infrastructure)
  · semantic quality                                   EXPERIMENTAL
    ↓
deterministic risk engine (risk-engine-v1)             SHADOW
    ├── candidate AUTO                                 BLOCKED (not activated)
    └── REVIEW
          ↓
    human adjudication / appeal                        READY
          ↓
    deterministic score                                READY
          ↓
    immutable audit trail                              READY
```

## Component table

| Component | Status | Evidence |
|---|---|---|
| deterministic MC path | **READY** | no LLM in the loop; covered by tests |
| variant detection | **READY** | deterministic; frozen 16-case dataset |
| cloud boundary | **READY** | two-layer authorization; `--research` no longer bypasses content safety; 34 boundary tests; all 64 paired payloads verified on the wire |
| OCR transcription | **EXPERIMENTAL** | best deployable configuration reaches 29/32 coverage at failure-aware CER 0.3560 on handwriting |
| OCR fallback policy | **EXPERIMENTAL** | `gemini_then_sonnet_hard_failure_fallback_v1`, reference-blind by construction, 46 adversarial stress tests; **not wired into production** |
| immutable transcription | **READY** | content-hashed, append-only run artifacts |
| local grading infrastructure | **READY** | runs offline; grade-validation-v2; no cloud path exists |
| semantic grading quality | **EXPERIMENTAL** | no arm beats always-partial on the frozen 46-case set |
| evidence grounding | **EXPERIMENTAL** | structural checks exist; never validated against live OCR output |
| risk engine | **SHADOW** | risk-engine-v1, matrix hash-pinned, decisions logged, **never acted on** |
| candidate AUTO | **BLOCKED** | 0/5 false-full observations bound the rate only below ~45% |
| human review UI | **READY** | shared blind review site used for the 92/92 SEEN-46 campaign |
| deterministic score + audit trail | **READY** | append-only, hash-chained |
| HELD_OUT | **BLOCKED** | untouched by design; final-eval path only |

## The two release blockers

Both are the same underlying fact, and both concern OCR rather than grading:

1. **A transcription can be confidently wrong in a way that changes meaning** —
   a changed digit, a dropped negation, a moved operator. Offline we detect
   these by comparing against an audited reference. **In production no
   reference exists**, so nothing detects them.
2. **Fabrication** (fluent text unrelated to the crop) has no automatic
   detector at all; it has only ever been caught by human adjudication.

Until one of those has a production detector — or a second independent read
that can disagree — no OCR output can feed an unreviewed grade.

## What the OCR evidence currently says

Measured on 32 frozen seen-DEV handwritten crops (16 line, 16 cell):

| Route | Coverage | Failure-aware CER | Failure-aware critical errors /32 |
|---|---|---|---|
| Gemini only | 14/32 | 0.6130 | 20 |
| Sonnet only | 27/32 | 0.5544 | 14 |
| Gemini → Sonnet fallback | **29/32** | **0.3560** | **9** |

The fallback is the best deployable configuration measured and is still not
good enough to ship unreviewed: 18 of 32 crops are touched by a human under it
(3 unresolved + 15 fallback rows flagged for review).

See `evaluation/model_selection/runs/ocr_primary/OCR_SEEN32_PAIRED_RESULT_2026-09-02.md`
for the full result and `OCR_SHIPMENT_READINESS_2026-09-02.json` for the
failure-mode inventory and checklist.
