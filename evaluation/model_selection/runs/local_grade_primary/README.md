# local_grade_primary runs — the PRODUCTION grader selection

Result root for the LOCAL grade_primary benchmark. Experiment records:

- **SEEN-46 diagnostic campaign** (2026-08-28):
  `experiments/LOCAL_GRADE_PRIMARY_SEEN_46_CAMPAIGN_2026-08-28.json` — the
  whole DEV (32) + CALIBRATION (14) splits, one candidate
  (qwen3-vl:8b-instruct, grade-v4-charitable-local), model outputs FROZEN
  (`SEEN46_MODEL_RUN_2026-08-28.*`, gate PASS, 46/46) and compared against a
  blind independent human consensus via the `review46_app` website
  (2 reviews/case + adjudication; instructor grade = reference source, not
  infallible truth; every source preserved separately). Zero-leakage proof:
  `SEEN46_LEAKAGE_VERIFICATION_2026-08-28.json`. HELD_OUT sealed.

- **ACTIVE** (output-contract phase, 2026-08-28):
  `evaluation/model_selection/experiments/LOCAL_GRADE_CONTRACT_FREEZE_2026-08-28.json`
  — prompt `grade-v4-charitable-local` (v4 semantics verbatim + mechanical
  output contract), `grade-bench-v3` / `grade-validation-v2` (symmetric
  zero-side grounding: an ungrounded invalid verdict on non-empty text
  routes to REVIEW). Structural DEV smoke (2 cases, NO quality claim) →
  CALIBRATION quality population (12 cases; strict metrics exclude the
  audit-C case `e004_q2_r8`, whose instructor-derived target is preserved).
  Candidates: `qwen3-vl:8b-instruct` (development candidate),
  `qwen3-vl:30b-a3b-instruct`. `qwen3.8:27b-q4_K_M` was DROPPED by owner
  decision 2026-08-28 and may not run again.
- **COMPLETED** (FullDev phase, immutable history):
  `evaluation/model_selection/experiments/LOCAL_GRADE_PRIMARY_FREEZE_2026-08-27.json`
  — results in `FULLDEV_2026-08-28.md`, audit in
  `FULLDEV_AUDIT_2026-08-28.{json,md}`, counterfactual structural replay
  (NOT actual performance) in `REPLAY_STRUCTURAL_2026-08-28.{json,md}`.
Runs are produced on the strong PC by `scripts/run_local_grade_primary.ps1`
(preflight-gated, `-Execute` required, local backend only, cloud grading
cost $0) through the standard bench runner, so every run directory carries:

- `run.json` — run id, candidate/backend/base_url, config hash, git commit,
  dataset + prompt + schema hashes, case selection sha, dry/live history;
- `outputs.jsonl` — per case: raw structured model output, usage, latency,
  cache_hit, fingerprint, attempt (append-only; resume skips done cases —
  an earlier run is never overwritten);
- `scored.jsonl.json` / `metrics.json` — parsed GradeResult, normalized
  verdict vs expected verdict (production conversion), confusion matrix,
  macro-F1, balanced accuracy, per-class recalls, harmful up/downgrades,
  uncertainty, schema failures, latency stats;
- `usage.json` — token/cost accounting (cloud grading cost is $0 here).

`machine_profile_<timestamp>.json` files record the executing machine
(RAM/VRAM/GPU/CPU, hashed hostname, git commit) next to the runs.

Policy: never commit model weights or Ollama caches; result artifacts are
committed after review per repository policy.

## Ground truth (owner directive 2026-08-28)

The authoritative evaluation target is the **actual instructor-assigned
grade from the original graded test** (`final_labels.json`,
`ground_truth_source=original_instructor_grade`). The owner's blind A/B/C/D
audit decisions, model-majority votes and previous cloud-model (Gemini /
Sonnet) predictions are **diagnostic metadata only** — they may flag a
rubric-practice mismatch, an evidence/transcription concern or an ambiguity,
but they never replace, modify or determine an expected label. (The frozen
CALIBRATION strict-metrics policy for the C-decided `e004_q2_r8` lives in
`scripts/calibration_audit_recompute.py` and is a separate artifact; the
two-layer reports below exclude or relabel nothing on audit grounds.)

Every executed run gets a **two-layer report**
(`bench grade-report --run-dir <dir>` -> `two_layer_report.{json,md}`),
which re-derives every target from the instructor score at report time and
refuses on any disagreement:

- **A. LOCAL GRADER QUALITY** — model canonical explanation verdict vs the
  instructor-derived verdict, only over the mathematically derivable cases
  (DEV: 26 = 22 valid + 4 partially_valid; `invalid` has no support and is
  NOT MEASURED).
- **B. END-TO-END TEST-GRADE AGREEMENT** — system predicted final score
  (model verdict + actual selection correctness + frozen production policy)
  vs the actual instructor score, over the whole split (DEV: 32). The six
  audited wrong-selection DEV cases score a deterministic 0 through the
  production selection gate and appear ONLY here (never in Layer A): their
  zero was decided by the selection, so a final-score match on them proves
  nothing about explanation judgement.

The two layers are never combined.
