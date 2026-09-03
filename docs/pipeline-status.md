# Pipeline status — component readiness (2026-09-03)

The current end-to-end shape, with every component marked. **SHADOW is not
production activation**, and nothing here is activated.

```
scanned exam
    ↓
deterministic variant / MC logic                       READY
    ↓
OpenRouter OCR ONLY (ocr_primary)                      EXPERIMENTAL — NO WINNER
  · two-layer cloud boundary, deny-by-default          READY
  · Gemini→Sonnet hard-failure fallback                REJECTED (withdrawn)
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
| OCR transcription | **EXPERIMENTAL — no winner** | every candidate measured has been dropped or is control-only; see the decision registry |
| OCR fallback policy | **REJECTED** | `gemini_then_sonnet_hard_failure_fallback_v1` withdrawn as strictly dominated; **never wired into production** |
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

**There is no current OCR winner.** `roles.ocr_primary` remains `UNSELECTED`.
Every candidate measured so far has been dropped or retained only as a control:

| Configuration | Status | Why |
|---|---|---|
| `openai/gpt-5.6-luna-pro` | **DROP** | 4/5 handwritten crops refused, 1 fabrication, failure-aware handwriting CER 0.9487 |
| `anthropic/claude-sonnet-5` | **HISTORICAL CONTROL ONLY** | 27/32 coverage but successful-output CER 0.4718 with 9/27 critical errors — not suitable for automatic OCR |
| `google/gemini-3.7-flash` + `m2-strict-v1` | **DROP as primary route** | 14/32 usable, 10 lost to provider content filtering |
| `google/gemini-3.7-flash` + `ocr-neutral-v2` | **DROP as primary route** | filtering got *worse* (10 → 14); pre-registered drop rule fired at 16/32 hard failures |
| `gemini_then_sonnet_hard_failure_fallback_v1` | **REJECTED** | strictly dominated; withdrawn |

These are recorded machine-readably in
`evaluation/model_selection/policies/ocr_decision_registry.json`, which refuses
to run a dropped arm without an explicit override naming a new experiment, and
refuses to report a dropped arm as a winner.

### Two lessons that generalise beyond OCR

**Good conditional CER is not enough when coverage is poor.** Gemini reads this
handwriting better than anything else measured anywhere in the project —
successful-output CER 0.1155, four times better than Sonnet and five times
better than the best local system — and it is still unusable on its own,
because it only produced usable output on 14 of 32 crops. A metric conditioned
on success silently excludes every case that never produced one. Always report
the failure-aware number and its denominator alongside it.

**Fluent fallback text can be more dangerous than an explicit failure.** A
content filter is loud and machine-detectable, so the crop routes to a human. A
fluent wrong transcription is silent and reaches the grader. The Gemini→Sonnet
composite raised coverage to 29/32 and was rejected precisely because it bought
that coverage by converting detectable failures into undetectable errors.

### Next step

`OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1` — a frozen, **not yet executed** 8-case
screen over three routes: Gemini pinned to `google-ai-studio`, Gemini pinned to
`google-vertex` (the catalog serves this slug from two distinct providers, both
prior arms ran automatic routing across a mix, and no content-filtered row in
any run records which provider produced it), and
`qwen/qwen3-vl-235b-a22b-instruct` pinned to `alibaba` as a genuinely different
family. Local OCR is not the fallback: 14 local configurations across
Qwen3-VL 8B/30B, a dedicated Hebrew HTR model and surya document OCR all
returned mean CER ≥ 0.94 on this corpus.

See `OCR_EXPERIMENT_LINEAGE_2026-09-03.json`,
`OCR_PROVIDER_ROUTE_FORENSICS_2026-09-03.json`,
`OCR_CANDIDATE_DISCOVERY_2026-09-03.json` and
`OCR_SHIPMENT_READINESS_2026-09-02.json` (failure-mode inventory).
