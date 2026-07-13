# Project status — 2026-07-13 (strong-PC validation session)

This file reports **what was actually executed and verified**, on what
hardware, and what remains open. Nothing here is extrapolated from
benchmarks we did not run. The previous status (2026-07-12, weak CPU-only
PC) is superseded; its numbers must not be used.

## Environment used for validation

| Item | Value |
|---|---|
| Machine | Windows 11, AMD Ryzen 5 5600G, 64 GB RAM, **NVIDIA RTX 2000 Ada 15.4 GB**, driver 595.97 |
| Server | Ollama 0.31.2, `OLLAMA_CONTEXT_LENGTH` 8192 (probes) / 32768 (pipeline; see context table in docs/deployment.md) |
| Model | `qwen3-vl:8b-instruct` (ID 0533d74300e4, Qwen3-VL-8B **Q4_K_M**, Apache-2.0) — **never** the bare `qwen3-vl:8b` tag (thinking variant; reasoning consumes the output budget; `think:false` ineffective on 0.31.2) |
| GPU residency | 100 % GPU at both 8K (~6.5 GiB) and 32K (~10 GiB; 14.7/15.4 GiB total — the card's ceiling); **zero CPU offload**; decode 19–33 tok/s |
| App | this repository, `--backend openai`, json_schema structured mode, temperature 0 |

## Test suite

**116/116 offline tests pass** (no network, no keys): scoring policy,
backend transport/malformed/truncation, dataset split determinism, masking,
metrics, full pipeline + eval-batch on mocks, filename/grade leakage,
resume fingerprints (incl. crash-resume mid-extraction), answer-sheet
authority matrix (11 scenarios), close-read merge + swapped-tables
regression, chunked extraction + collapse tripwire, key cache (one parse
per batch; invalidation by key/rubric/model/render/prompt/schema;
corruption rejection), deterministic key repair (text-layer letter groups,
operator overrides, flattening detector), variant detection (each flower →
its variant; aliases; unclear → review; no score-maximisation; fingerprints
separate variants), question alignment (printed↔key numbering round-trip).

## What was executed against the real model and verified

| Check | Result |
|---|---|
| Probes A/B/C (single page, ≤1200 out tokens, temp 0, 8K ctx) | **PASS 4.6 s / 11.1 s / 4.8 s** (weak PC: 300 s / fail / fail). Root causes of the old "vision truncation" found live: (a) thinking-variant tag; (b) grammar-pressure repetition loop on an unbounded verdict field — both fixed structurally (instruct tag; schema discipline) |
| Answer-key parse | 6 model attempts logged (docs/validation, ledger): quality varies between runs at temp 0 — flattened version columns twice, missing columns twice, one client timeout. Resolution: **deterministic text-layer repair** decodes the per-version letter groups and overrides model columns on every load; colour-only values (this key's Q3 MC answers) come from the version-controlled operator override file or stay review-flagged. Final key: versions A1/A2/A3, Q1+Q2 columns text-layer-verified, 2.8 and 3.16 operator-verified (instructor-tick evidence), Q3 remainder flagged unverified pending instructor input |
| Persistent key cache | Live hit confirmed (~30 s vs ~12 min parse); repair re-applies on every load; defective parses never cached |
| Variant detection (cover flower) | Live on the representative exam: `variant_symbol_a1` (four-petal clover), bottom third, confident → A1 per the owner-confirmed authoritative mapping; recorded in the result with page/region/mapping-source |
| Per-variant question alignment | A1 derived live: exact identity (correct — the key's canonical order IS A1's print order). A2/A3 print orders differ (verified manually: Hough question at #16/#20/#18 in A1/A2/A3) |
| Answer-sheet detection | Pages 11–13 located structurally (headings/tables/position; no hardcoded count) in every live run |
| Two-resolution inference | Survey at 640 px over 13 pages + close-read of 3 sheet pages at 1400 px + extraction of only the relevant sheet pages at 1000 px — whole-exam cost ≈ 10 model calls, ~10–13 min end-to-end on this GPU with the cached key |
| Chunked extraction | The earlier live 20-row template collapse (all rows "B", identical rationale) is gone; per-row reads with mark observations; row-attribution drift between runs remains on messy handwriting (see limitations) |

## Representative exam (per-item audit: evaluation/representative_exam_audit.md)

Chain: flower→A1 ✓, key cached+repaired ✓, sheets 11–13 ✓, alignment
identity ✓, X-convention meaning ✓ (transcription garbage), instructor ink
excluded from grades ✓ (score fractions deterministically filtered from
conventions). **Open defect:** the student's answer-table swap (crossed-out
title digits + faint note) was missed by the close-read at 1000 px and at
1400 px — Q1/Q2 grade against each other's key columns. Mitigations shipped:
topic-anchored close-read (question topics vs handwritten explanation
topics) and a deterministic **swap tripwire** that review-flags strongly
crossed key agreement (never regrades). Per-item sheet-reading accuracy
measured manually: 8/8 exact on the neat sheet, 5/8 on the messy one;
explanation transcriptions largely skipped — Hebrew handwriting remains the
#1 model limitation.

## What requires human review (by design, observed live)

Uncertain variant markers; key values not deterministically verified (Q3
columns pending the operator override); suspected table swaps; uniform-
answer patterns; ambiguous/conflicting marks; explanation-dependent
reversals. Review flags fire liberally until the instructor fills the Q3
override — review-rate metrics reflect that honestly.

## Honest limitations (real-model-tested)

1. **Hebrew handwriting**: letter misreads on messy sheets (3/8 on the hard
   page), near-total failure to transcribe long handwritten justifications,
   row-attribution drift between identical runs (GPU nondeterminism).
   These bound end-to-end accuracy regardless of pipeline correctness.
2. **Fine-print corrections** (crossed-out title digits): below this
   8B model's reliable perception even at 1400 px; covered by the
   deterministic tripwire → human review, not silent misgrading.
3. **Colour-only key encodings** are not deterministically decodable from
   the text layer; they require the operator override once per exam form.
4. Key-parse output quality varies between runs at temperature 0; the
   validation+repair layer rejects/repairs rather than trusts.

## Hebrew transcription campaign (2026-07-13, closed)

Bounded 8-iteration campaign on a 16-cell owner-verified benchmark
(one writer): best local result qwen3-vl:8b + strict prompt + contrast
= CER 0.786, usable 0% - an order of magnitude from the acceptance
gate; all candidates confabulate rather than flag unreadable text.
Verdict STOP; structural diagnosis = insufficient training data
(HebHTR's 4.76% CER on this exact domain proves learnability).
Details: evaluation/hebrew_transcription_loop.md,
evaluation/local_hebrew_htr_benchmark.md. Next session starts with
the oracle-ensemble analysis (evaluation/NEXT_SESSION_HANDOFF.md).

## Batch evaluation (Stage A executed; full details: evaluation/)

Stage A (first 5 validation exams, masked, anonymized, sequential):
**5/5 processed, 0 failures, variant detection 5/5 correct and confident
across all three flowers, mean 906 s/exam, GPU 98.1 % avg utilization,
no CPU offload.** Accuracy: MAE 39.6 with uniformly NEGATIVE errors —
under-scoring driven by two 8B perception limits measured per-item against
owner-supplied ground truth (evaluation/exam003_audit.md): skipped Hebrew
explanation transcriptions (rubric gate zeroes correct selections; now
review-flagged per item) and chance-level dense bubble-grid reading (now
tripwire-flagged, incl. cyclic patterns). The A2/A3 question-order
misalignment found in Stage A is FIXED via operator-verified mappings.
Leakage audit: 0/10 probes could extract instructor grades (masked or
unmasked). Stage B/C intentionally NOT launched per the owner's audit gate
— re-measuring known model limits adds no decision value; next lever is
row-band crops / a dedicated transcription pass / the 32B bake-off.
Splits: train 25 / validation 16, seed 42 (datasets/ manifests;
evaluation/run_manifest.json lists the exact ids).
