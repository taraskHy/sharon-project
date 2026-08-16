# qwen38_27b_q4km — canonical raw benchmark, decision record

Date: 2026-08-16. Official `qwen3.8:27b-q4_K_M` (digest 25b843619e94,
27.3B, Q4_K_M, Ollama 0.32.13) through the CANONICAL runner
(scripts/m2_bench_run.py, backend ollama_native = Ollama /api/chat so that
`think:false` is honored — the OpenAI-compat endpoint ignores it), frozen
per arm_freeze.md: contrast/1100 preproc, canonical strict-fidelity
prompt/schema/parser, temperature 0, 500-token cap, repeat_penalty 1.0,
num_ctx 8192 (effective, per ollama ps), 17 GB in memory at 27%/73%
CPU/GPU. 129/129 records; smoke gate 5/5; ~13 s/item; 5 records hit the
token cap (repetition loops), 0 errors, no reruns; references joined only
by the canonical evaluator afterwards.

## Canonical evaluator (m2_bench_eval), by category

| category | n | mean CER | usable | notes |
|---|---|---|---|---|
| handwritten_line | 64* | 1.400 | 0.0 | 22/22 hard items hallucinated |
| handwritten_cell | 11* | 0.881 | 0.0 | 5/5 hard hallucinated |
| printed_rtl | 8 | 0.070 | 0.875 | strong |
| mixed_he_en | 7 | 0.113 | 1.0 | strong |
| formula_printed | 7 | 0.424 | 0.43 | mixed |
| option_row_association | 5 | — | pair acc 0.79 | |

*evaluator subset with reference text.

Handwritten (all 102 with references, canonical CER): mean 2.391 / median
0.782 / usable 0 @<=0.25, 1 @<=0.50 — the mean is inflated by 8 items with
CER>2 driven by repetition loops (worst 114.7 on a 9-char cell) and
insertions; median 0.78 is the honest central tendency.

## Same-item comparison vs canonical baselines

| pairing | n | Qwen3.8 mean/median | baseline mean/median | Qwen3.8 W/T/L |
|---|---|---|---|---|
| vs protocol-clean Gemini | 32 | 1.009 / 0.769 (usable 0/1) | 0.329 / 0.279 (13/23) | 2/0/30 |
| vs ML Kit | 20 | — | 0.664 / 0.665 | 3/0/17 |
| frozen gate-20 (3-way) | 20 | 1.173 / 0.775 (0/0) | Gemini 0.287/0.226 (11/15); ML Kit 0.664/0.665 (1/4) | — |

Grading-decision preservation (identical 12 cells, fixed judge): match
0.1667, safe 0.5833 (vs local Qwen 8B 0.0833/0.6667; Gemini 0.4167/0.5).

## Decision

Qwen3.8-27B (Q4_K_M, direct mode) is a strong PRINTED-text reader
(printed_rtl CER 0.07, mixed 0.11) but its Hebrew HANDWRITING transcription
is not competitive: it loses 30/32 to protocol-clean Gemini and 17/20 to
ML Kit on identical items, with 0/102 usable handwritten items at CER<=0.25
and a repetition-loop failure mode at repeat_penalty 1.0. Provider ranking
for handwriting is unchanged: Gemini >> ML Kit > local Qwen (8B or 27B).
Note the honest caveat: hardware forced Q4 + partial CPU offload; this is
the model as it can actually run on this PC, not an upper bound.
