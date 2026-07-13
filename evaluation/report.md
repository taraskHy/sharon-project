# Validation evaluation — Stage A report (2026-07-13)

Machine: RTX 2000 Ada 15.4 GB; model `qwen3-vl:8b-instruct` (Q4_K_M),
json_schema, temperature 0, 32K server context (required once per variant
for alignment derivation; 16K is the recommended steady-state default —
see performance.md). Masking ON, anonymized ids, expected grades loaded
only post-prediction. Split: the committed deterministic manifests
(validation = exams 002, 003, 004, 007, 008, 009, 010, 016, 017, 018, 019,
029, 035, 040, 041, 042; Stage A = the first five).

## Stage A results (5 validation exams, sequential)

| Exam | Variant (detected) | Predicted | Expected | Error | Review items | Runtime |
|---|---|---|---|---|---|---|
| exam-002 | A3 ✓ confident | 30 | 76 | −46 | 36 | 729 s |
| exam-003 | A2 ✓ confident | 14 | 70 | −56 | 19 | 957 s |
| exam-004 | A1 ✓ confident | 22 | 58 | −36 | 36 | 1010 s |
| exam-007 | A1 ✓ confident | 22 | 48 | −26 | 19 | 1027 s |
| exam-008 | A2 ✓ confident | 18 | 52 | −34 | 19 | 807 s |

Aggregates: processed 5/5, failures 0, MAE 39.6, within ±5 0 %, review
rate 100 %, mean runtime 906 s/exam. GPU: 98.1 % average utilization while
active, peak 14.8 GiB VRAM, no CPU offload (2,330 telemetry samples).

**Technical chain: stable.** Zero crashes, variant detection 5/5 correct
and confident across all three flowers, masking + anonymization active,
resume fingerprints correct, key served from the persistent cache
(parse paid once), per-exam isolation intact.

**Accuracy: systematically low, with known causes** (per-item evidence in
exam003_audit.md and representative_exam_audit.md):

1. **Explanation gating** — the model rarely transcribes handwritten Hebrew
   justifications; the rubric's explanation-required gate then zeroes
   CORRECT selections (measured: 13 correct selections worth ≥52 points
   zeroed on exam-003 alone). Every such item is now review-flagged.
2. **Bubble-grid reading** — on dense Q3 answer tables the 8B reads at
   chance level and emits degenerate patterns (a period-4 "A,B,C,D"
   staircase, now tripwire-flagged).
3. Alignment (A2/A3 question-order shuffling) was a third systematic cause
   during Stage A and is **fixed**: operator-verified mappings shipped with
   the key, applied and recorded ("operator-override"); model-derived
   alignments are never silently trusted anymore.

All errors are negative (under-scoring): the system loses points it cannot
perceive; it does not invent credit. This is the designed failure
direction.

## Stage B / Stage C decision

Per the owner's gate (A2 mapping manually verified ✓; per-item rerun audit
must pass ✗ — extraction quality on Q3-type grids and explanation
transcription are below usable), **the batch was not expanded past Stage A
plus the exam-003 rerun**. Expanding to 10 or 41 exams would re-measure the
same two model-perception limits at ~15 min/exam without adding decision
value. See exam003_audit.md for the prioritized fixes (row-band crops,
dedicated transcription pass, 32B bake-off, Q3 override completion).

## Evidence tiers

- **Unit-tested:** 122 offline tests (authority matrix, variant/alignment
  machinery incl. the committed real mapping, key cache/repair, tripwires,
  resume, leakage prevention, metrics).
- **Real-model-tested:** everything in this report; probe suite; the
  representative exam (8 live end-to-end runs across the session).
- **Manually verified:** flower↔variant mapping; A2/A3 printed↔canonical
  question mappings; exam-003 student answers (owner); representative-exam
  sheet contents incl. instructor marks.
- **Assumptions:** same-variant booklets are identical prints; Q1/Q2 order
  is canonical in every variant (content-verified on A2/A3 samples).
- **Untested:** hosted-API path; vLLM/32B path; the 48-exam held-out set
  (untouched by design).
