# Canonical OCR baselines — frozen 2026-08-11

Future OCR experiments MUST compare against these corrected baselines on
identical item subsets. Protocol-wrapper artifacts (truncated
`{"transcription": ...}` envelopes leaking into scored text) are
corrected everywhere via the deterministic parser of commit `8cd2748`
(`m2_bench_run.parse_declared_envelope`); historical raw outputs are
untouched and every cleanup op is annotated per record. Historical
wrapped numbers are SUPERSEDED and must not be cited as baselines.

## Canonical arms (text source for any downstream comparison)

| Provider | canonical arm | notes |
|---|---|---|
| Gemini 3 Flash | `gemini_protocol_clean_v1` | 13/32 records corrected |
| ML Kit Digital Ink | `mlkit_ink_rtl_a1` (top-1) | no protocol artifacts existed |
| local Qwen 8B | `qwen_protocol_clean_v1` | 2/101 records corrected |

## Frozen gate-20 (handwriting)

| Arm | mean CER | median | usable<=0.25 | usable<=0.50 |
|---|---|---|---|---|
| **Gemini (protocol-clean)** | **0.287** | **0.226** | 11/20 | 15/20 |
| ML Kit | 0.664 | 0.665 | 1/20 | 4/20 |

Gemini vs ML Kit: 19/0/1. (Historical wrapped Gemini 0.368/0.339 —
superseded.)

## Gemini eligible-32 (handwriting)

mean CER 0.3293 / median 0.2787; usable<=0.25 13/32, <=0.50 23/32;
WER mean 0.4885 / median 0.4000. Fixed-judge decision preservation
(12-cell subset): match 0.4167, safe 0.5.

## Local Qwen (qwen8b_strict_contrast, corrected)

Full-set (128 scored): mean CER 0.8169 -> **0.8128** after cleaning the
2 affected records (median 0.7812 unchanged; usable rates unchanged:
17/128 <=0.25, 20/128 <=0.50 — driven by printed categories).
Handwritten remains unusable: line mean CER 0.9042, cell 0.7971, usable
0. Fixed-judge decision preservation (12 cells): match 0.0833, safe
0.6667 — byte-identical to the pre-cleaning baseline. **Provider verdict
unchanged**: local Qwen handwriting stays parked; printed OCR remains
potentially useful.

## Standing decisions

- RAG text repair is REJECTED for grading
  (`gemini_rag_fidelity_audit.md`): deterministic protocol cleanup
  strictly dominates it on the evaluated overlap with zero fidelity risk.
- Comparisons are valid only on identical item subsets, with references
  joined strictly post-inference.
- Decision-preservation metrics are same-fixed-judge measurements, NOT
  actual grading accuracy; actual grading accuracy remains unestablished
  without independent human per-item grades.
