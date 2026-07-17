# Retrospective assisted-annotation benchmark — pre-registered protocol

Written 2026-07-17 06:10, BEFORE any candidate generation or evaluation.
Part of the bounded overnight campaign (start 05:55, hard stop 10:55).

## Question

Can a locally generated candidate transcription (displayed as UNVERIFIED
text in the annotation app) save the owner typing effort without
increasing label errors?

## Benchmark set

All train-split lines whose annotation status is `ok` (owner-verified)
and whose sample_id is NOT among the 20 overfit-test ids
(`evaluation/htr_overfit_test/selected_ids.json`): **66 lines**
(e004×19, e005×30, e006×16, e007×1; 22 of them contain a partial
[לא קריא] span and are reported separately). Selection is recorded in
`bench_ids.json` (ids only — no transcription text) before generation.

## Candidate systems (all local, GPU, no hosted APIs)

- **A `qwen_line`** — qwen3-vl:8b-instruct (Q4, Ollama 0.32.0,
  OLLAMA_CONTEXT_LENGTH=16384), the campaign's `strict_fidelity` prompt,
  temperature 0, JSON-schema output, LINE crop only.
- **B `qwen_line_cell`** — same model/prompt plus the cleaned full-cell
  image as context; instruction states only the line crop is to be
  transcribed.
- **C `crnn_overfit`** — the overfit-test CRNN checkpoint
  (`evaluation/htr_overfit_test/ws/model/crnn_best.pt`, trained ONLY on
  the 20 excluded ids). All 66 bench lines are sample-level
  uncontaminated. Caveat reported: 4 of its 20 training lines share
  writer e004 with 19 bench lines (writer-level overlap, flagged in the
  report). The eval script refuses any decode row whose id is in the
  overfit-20 (contaminated) as a tripwire.

## Ground-truth hygiene

- Generation reads ONLY `splits/train.json` (paths/geometry) +
  `bench_ids.json` + image files. It never opens
  `annotations/`; prompts contain no transcription, no answer key, no
  rubric, no course vocabulary.
- Raw responses are saved to disk per sample BEFORE
  `annotation_candidates_eval.py` (the only GT reader) runs.
- Metadata retained per candidate: model + Ollama digest, prompt id,
  image SHA256(s), latency, confidence (CRNN only; VLM has none).

## Metrics (campaign definitions, `scripts/hebrew_bench_eval.py`)

Normalized CER (char lev / len(ref)); WER, omission rate (word
deletions / ref words), hallucinated-word rate (insertions / hyp
words); exact-match (normalized). Edit-burden classes per line:
- `no_edit`   CER = 0 (normalized)
- `minor`     0 < CER ≤ 0.15
- `moderate`  0.15 < CER ≤ 0.40
- `major`     CER > 0.40  (candidate presumed harmful: reading + fixing
  is slower than typing from scratch, and anchoring risk is real)

A↔B agreement = 1 − lev(norm A, norm B)/max(len); analysed as a
correctness predictor (bucket ≥0.9 / 0.7–0.9 / <0.7).

## Decision gate (fixed now, before results)

Let a *conservative subset* be any subset selected ONLY by model-visible
signals (A↔B agreement threshold and/or CRNN confidence threshold —
never by GT). Candidate display is justified iff there exists a
conservative subset with:

- coverage ≥ 20 % of bench lines, AND
- (no_edit + minor) rate ≥ 60 % within the subset, AND
- major rate ≤ 10 % within the subset.

Otherwise AI prefill is REJECTED and Phase 2 ships no candidate
generation for new samples. In all cases candidates are stored apart
from `transcription`, are labelled unverified, and bulk accept does not
exist.
