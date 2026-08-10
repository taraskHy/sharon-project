# gemini_protocol_clean_v1 — corrected Gemini OCR baseline, decision record

Date: 2026-08-11.

## Root cause

The benchmark runner's prompt declares a `{"transcription": ...}` JSON
envelope; responses TRUNCATED at max output tokens never close the
envelope, the `\{.*\}` parse fails, and the parser fallback scored the
WHOLE wrapper as the transcription. This was an adapter parser gap (live
in both the Gemini and OpenAI-compat fallbacks), affecting 13/34
committed gemini3_flash records and 2/129 qwen8b_strict_contrast records;
ML Kit records carry no such artifacts. **Historical raw outputs remain
untouched** — every frozen record's `raw` is preserved byte-for-byte.
The deterministic parser implementation + regression tests are commit
`8cd2748`; this arm re-extracts from `raw` with per-record operation
annotations and fails closed when no declared envelope parses.

## Gemini, identical 32 eligible items (raw vs protocol-clean)

| Metric | raw | protocol-clean |
|---|---|---|
| mean / median CER | 0.4124 / 0.4432 | **0.3293 / 0.2787** |
| mean / median WER | 0.5223 / 0.4356 | **0.4885 / 0.4000** |
| usable CER<=0.25 | 9/32 | **13/32** |
| usable CER<=0.50 | 20/32 | **23/32** |

13 outputs changed: **13 improved / 0 worsened / 0 neutral** (a
deterministic cleanup must be non-harmful, and is). Grading decisions on
the identical 12-cell fixed-judge subset: **unchanged** (match 0.4167,
safe 0.5, same verdict per cell).

## Corrected frozen gate-20

| Arm | mean CER | median CER | usable<=0.25 | usable<=0.50 |
|---|---|---|---|---|
| **protocol-clean Gemini (canonical)** | **0.287** | **0.226** | **11/20** | **15/20** |
| historical wrapped Gemini (superseded) | 0.368 | 0.339 | 8/20 | 13/20 |
| ML Kit (unchanged) | 0.664 | 0.665 | 1/20 | 4/20 |

Protocol-clean Gemini vs ML Kit: **19 wins / 0 ties / 1 loss**.

## Decision

The corrected protocol-clean numbers are the canonical Gemini baseline
going forward; the historical wrapped numbers understated Gemini and are
superseded. **RAG text repair remains rejected**
(`evaluation/gemini_rag_fidelity_audit.md`): deterministic protocol
cleanup strictly dominates it on the evaluated overlap (0.3293 vs 0.3546
mean CER on the same 32 items) with zero fidelity risk.
