# Claude Vision assisted-annotation experiment

Status: **BLOCKED — no Anthropic API credential on this machine.**
No exam image has been sent anywhere; zero API calls have been made.
Everything up to the first API call is built, tested, and
pre-registered so the benchmark is a two-command run once a credential
exists.

## Privacy pre-flight (confirmed)

This experiment, when run, sends **anonymized handwriting crops to
Anthropic**: cleaned student-ink-only line crops, plus (pass B) the
same student's cleaned explanation-cell image. Payloads contain base64
image bytes and fixed instruction text only — no filenames, sample
ids, grades, student identifiers, instructor scores or ink
(`cell_orig` is never accessed — enforced by test), answer keys,
rubrics, question text, hidden ground truth, or neighboring students'
answers. Full statement: `evaluation/claude_candidates/PROTOCOL.md`.

## Pre-registered design (frozen before any inference)

- 30 owner-verified clean lines, 10 each from writers e004/e005/e006
  (deterministic; excludes the 20 overfit ids; ids-only record:
  `evaluation/claude_candidates/claude_bench_ids.json`).
- Two passes on `claude-opus-4-8` (most capable Opus tier,
  high-resolution vision): A = line crop; B = line crop + cleaned cell
  context. Owner's transcription-only instruction verbatim; adaptive
  thinking; no sampling parameters. Raw responses + model id + prompt
  version + image SHA256s + timestamps + latency + token usage are
  saved per call BEFORE the evaluator (the only GT reader) may run —
  it refuses to score a partial run (tested).
- Owner's acceptance gate, frozen in PROTOCOL.md and pinned by
  `tests/test_claude_candidates.py::test_gate_constants_match_owner_protocol`:
  exact-match ≥ 40 %, median CER ≤ 0.10, major hallucination (≥ 2
  inserted words) on ≤ 5 % of lines, and an A↔B-agreement subset
  (≥ 5 lines) with CER ≤ 0.10. ACCEPT iff at least one pass meets all
  four. On REJECT: no candidates for untouched lines, no app changes
  beyond recording the negative result.

## Safety guarantees already in force (9 tests, suite 163/163)

- `build_request()` accepts only image bytes — there is no code path
  by which a transcription or id can enter a payload.
- Claude outputs live outside the pilot package with `verified: false`;
  the runner cannot write annotation records.
- Verified labels remain overwrite-locked in the app (per-sample
  unlock checkbox, 2026-07-17 campaign).
- The annotation app and lib contain no candidate UI and will fail the
  test suite if one appears without an ACCEPT verdict — candidate
  generation is disabled by default, and a failed benchmark leaves the
  manual workflow byte-identical.

## Context: prior evidence

The same 2026-07-17 overnight campaign benchmarked local candidates
(qwen3-vl:8b strict prompts, overfit-CRNN) on 66 verified lines: 0/198
candidates below CER 0.4 → prefill REJECTED. Claude Opus is a far
stronger vision model, hence this separate owner-authorized test; the
bar (median CER ≤ 0.10) is however ~8× stricter than anything any
model has achieved on this handwriting to date (best local: 0.635 on
a single best line).

## To resume (owner action, one of):

1. `ant auth login` (preferred — no static key), or
2. set `ANTHROPIC_API_KEY` for the session.

Then:

```
.venv\Scripts\python.exe scripts/claude_candidates_run.py generate --config claude_line
.venv\Scripts\python.exe scripts/claude_candidates_run.py generate --config claude_line_cell
.venv\Scripts\python.exe scripts/claude_candidates_eval.py
```

(~60 calls, est. a few dollars on Opus 4.8; resume-safe — already-saved
samples are skipped.) The evaluator prints ACCEPT/REJECT against the
frozen gate and writes `evaluation/claude_annotation_candidate_results.csv`
+ `evaluation/claude_candidates/eval_summary.json`; this report is then
updated with the exact metrics.

## Results

*(pending — no inference has run)*
