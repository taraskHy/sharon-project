# local_grade_primary runs — the PRODUCTION grader selection

Result root for the LOCAL grade_primary benchmark (frozen experiment:
`evaluation/model_selection/experiments/LOCAL_GRADE_PRIMARY_FREEZE_2026-08-27.json`).
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
committed after review per repository policy. The case decided **C** in the
human audit (`e004_q2_r8`, evidence problem) still runs but is excluded from
strict-accuracy denominators until repaired.
