# Asymmetric-risk evaluation — source verification (Phase 0, 2026-09-02)

Written BEFORE any asymmetric metric was implemented. Zero-inference task:
no local model call, no cloud call, no OCR, no RAG, no HELD_OUT access.

## Repository

- branch `initial-prototype`, HEAD `cfe7a02a0145fd46fd6396cf8af0628d38bbd84b`
  (matches the expected pushed HEAD), working tree clean at task start.

## Final human reference — `FINAL_HUMAN_REFERENCE_2026-09-02.json`

- self-hash RECOMPUTED and MATCHED:
  `ce78aed115633883dd94bd14035ddc3e3f38d02c2484bbd0a7f49c5dbc384306`
  (sha256 over the sort_keys JSON payload without the `reference_sha256`
  field — the same convention `final_reference_freeze.py` wrote it with).
- exactly 46 unique current cases; writers e002 (16), e003 (15), e004 (14),
  e007 (1); no HELD_OUT writer id appears anywhere in the population.
- class distribution: valid 28, partially_valid 13, invalid 5 — as expected.
- reference sources kept distinct: 22 `two_reviewer_consensus`,
  22 `adjudicated_human_reference`, 2 `owner_adjudicated_after_source_repair`.
- every case preserves: both independent blind reviews (verdict, confidence,
  issue flag, note, revision), stale historical reviews where they exist,
  the adjudication record, the ORIGINAL instructor grade
  (`ground_truth_source=original_instructor_grade`), and the baseline model
  output pointer. Nothing in this task modifies any of them.
- source repair e004_q2_r6 <-> e004_q2_r8: both rows carry
  `reference_source=owner_adjudicated_after_source_repair`, 0 fresh reviews +
  2 stale historical reviews each (the stale reviews graded transposed source
  text and are never usable), and their baseline output pointer is
  `corrected_rerun_2026-09-02` — the corrected outputs, not the stale ones.

## Reviewer-pair structure (input to the disagreement-aware view)

Over the 44 non-repaired cases (2 fresh reviews each):

- 22 agreed pairs — exactly the 22 `two_reviewer_consensus` cases;
- 14 adjacent disagreements ({invalid, partially_valid} or
  {partially_valid, valid}) — all `adjudicated_human_reference`;
- 8 wide disagreements ({invalid, valid}) — all `adjudicated_human_reference`.

Issue flags on fresh reviews: 79 `none`, 3 `transcription_evidence`,
3 `rubric_official_solution`, 3 `genuinely_ambiguous` (9 flagged cases; one —
e004_q2_r3 — is both wide-disagreement and transcription-flagged).

## Model-output populations (all three arms cover the identical 46 ids)

| arm | source files | sha256 (first 16) | coverage |
|---|---|---|---|
| baseline Q4 8B one-pass | `grade_primary/dev__all__qwen3-vl-8b-instruct__72e19378d1/scored.jsonl.json` | `e6db6ab3a8fdb135` | 32 DEV |
| | `grade_primary/calibration__all__qwen3-vl-8b-instruct__e2a3cfc925/scored.jsonl.json` | `2df474847aa2f1f6` | 14 CALIBRATION (r6/r8 rows excluded as stale) |
| | `CORRECTED_RERUN_2026-09-02.jsonl` | `feb6d7febb3a3457` | 2 corrected (r6/r8) |
| arm A q8_0 | `ARM_A_Q8_2026-09-02.jsonl` | `60a871a37d11a352` | 46 |
| arm B two-pass | `ARM_B_VERIFY_2026-09-02.jsonl` | `4bab0dee665cba3b` | 46 |

- id-set equality reference == baseline == arm A == arm B: VERIFIED (46).
- stale outputs: the 14 rows registered in `STALE_MODEL_OUTPUTS_2026-09-01.json`
  (r6/r8 across 7 historical runs) are excluded by construction; the corrected
  r6/r8 rerun rows are the active baseline outputs. Arm A and arm B were run
  after the repair, against the corrected sources.
- arm B changed NO verdict vs pass-1 (0 `verifier_upgrade` firings); its
  confusion matrix is identical to baseline, only AUTO/REVIEW routing differs.
- no other arm covers the current 46-case reference with non-stale outputs:
  the 27B and 30B candidates were DROPPED by owner decision (2026-08-28), their
  runs and the cloud grade-v3/v4 runs cover CALIBRATION subsets only and
  contain stale r6/r8 rows. They are therefore NOT comparable and are omitted
  (denominators would differ).

## Baseline cross-check (against committed artifacts, recomputed from rows)

exact 31/46 = 67.4%; macro-F1 0.5063; balanced accuracy 0.5062; overgrades 8
(4 invalid->partially_valid, 4 partially_valid->valid, 0 invalid->valid);
undergrades 7 (3 partially_valid->invalid, 2 valid->partially_valid,
2 valid->invalid); schema failures 0; evidence failures 2 (e002_q1_r7,
e002_q2_r7 — both already routed REVIEW by the validator); AUTO 44/46.
Matches `BASELINE_CLASS_METRICS_2026-09-02.json`,
`LOCAL_IMPROVEMENT_REPORT_2026-09-02.json` and the error audit exactly.

## HELD_OUT

Untouched. No HELD_OUT id, row, file or content was loaded, listed or read.
The reference is SEEN development data only (DEV + CALIBRATION).
