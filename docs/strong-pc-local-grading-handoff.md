# Strong-PC handoff — the LOCAL grade_primary experiment

*Written 2026-08-27 on the laptop. Everything needed to run the local-grader
selection on the strong PC is committed on `initial-prototype`; this page is
the runbook. The exact pushed HEAD is recorded in the freeze record's
`git_commit` field (see below) — `git log -5 --oneline` after pulling shows
the finalize commits.*

## The architecture (settled 2026-08)

**OpenRouter = OCR transcription only. Grading, RAG, scoring = local.**
Enforced in code by `autograder/cloudboundary.py` at the gateway choke point
(task allowlist `ocr_primary`/`ocr_verify` + registered-OCR-prompt check +
grading-content payload tripwire, classified by effective backend+URL).
models.toml cannot override it. A dead/malformed local grader parks items as
`REVIEW / LOCAL_GRADER_UNAVAILABLE`; there is **no cloud grading fallback**.
Cloud-grader results (grade-v3/v4, Sonnet/Gemini, DEV+CALIBRATION) are
**research baselines** behind `bench ... --research` — they do NOT select
the production grader. Full picture: docs/architecture.md §"The production
rule".

## What the laptop session completed

- the six-case CALIBRATION human audit, decided **blind** and committed:
  `e004_q1_r1=B · e004_q1_r3=B · e004_q2_r6=A · e004_q2_r8=C ·
  e004_q1_r5=A · e004_q1_r6=A`
  (artifact: `evaluation/model_selection/runs/grade_primary/CALIBRATION_AUDIT_2026-08-26.json`);
- the cloud boundary + research mode + independent ocr_verify contract +
  local-grading route, with the full test suite green;
- the frozen local experiment + this runbook.

## The frozen experiment

`evaluation/model_selection/experiments/LOCAL_GRADE_PRIMARY_FREEZE_2026-08-27.json`
binds (verify with `python scripts/local_grade_freeze.py --verify`):

| what | value |
|---|---|
| prompt | `grade-v4-charitable` (sha256 recorded; **do not create a v5 from CALIBRATION**) |
| target | canonical explanation verdict via the production conversion |
| DEV population | 26 derivable cases (valid 22 / partially_valid 4 / invalid 0), frozen ids + order |
| CALIBRATION population | 12 derivable cases (valid 7 / partially_valid 5 / invalid 0) |
| smoke | 2 pre-registered DEV cases |
| dataset | FROZEN `grade_primary` (inputs/labels/final-labels/manifest sha256 recorded) |
| HELD_OUT | e005/e006 — never read during selection; one final `bench final-eval` later |
| limitation | **invalid-class performance = NOT MEASURED** |
| evidence review | `e004_q2_r8` was decided **C** in the audit: excluded from strict-accuracy denominators until its transcription/evidence is repaired (it still runs; raw outputs persist) |

Candidates: `evaluation/model_selection/candidates.toml
[roles.grade_primary_local]` — `qwen3-vl:8b-instruct` (laptop+strong PC) and
`qwen3.8:27b-q4_K_M` (strong PC only). **No winner is selected**; installed
models are discovered at runtime, never assumed and never downloaded by any
script here.

## Prerequisites on the strong PC

1. `git pull --ff-only origin initial-prototype`;
2. the project venv (`.venv`) with the repo installed (as before);
3. Ollama running locally with the candidate model(s) pulled — pulling is a
   deliberate operator action (`ollama pull qwen3-vl:8b-instruct`), no script
   does it;
4. no OpenRouter key needed for any of this (grading is local; OCR is not
   part of this experiment).

## Commands (in order)

```powershell
# 1. PREFLIGHT — zero inference: freeze + boundary + installed models
.\scripts\run_local_grade_primary.ps1

# 2. SMOKE — 2 frozen DEV cases, one candidate (plan first, then -Execute)
.\scripts\run_local_grade_primary.ps1 -Smoke -Candidate qwen3-vl:8b-instruct
.\scripts\run_local_grade_primary.ps1 -Smoke -Candidate qwen3-vl:8b-instruct -Execute

# 3. FULL DEV — 26 frozen cases; requires the same candidate's failure-free smoke
.\scripts\run_local_grade_primary.ps1 -FullDev -Candidate qwen3-vl:8b-instruct -Execute
```

Repeat 2–3 per candidate. CALIBRATION finalists afterwards go through the
bench CLI directly (same guarantees, same runs root):

```powershell
.\.venv\Scripts\python.exe -m autograder bench run --role grade_primary --split calibration --subset calibration_verdict_v4 --candidate <finalist> --backend ollama --base-url http://localhost:11434/v1 --runs-root evaluation\model_selection\runs\local_grade_primary --i-understand-this-spends-money
```

Without `-Execute` every mode prints the plan and exits with **zero
inference**. Preflight failure (exit 2 = freeze mismatch, 3 = boundary
problem) refuses execution.

## Results

Everything lands under `evaluation/model_selection/runs/local_grade_primary/`
(see its README for the per-run schema: raw outputs, parsed GradeResult,
normalized-vs-expected verdicts, confusion matrix, macro-F1, balanced
accuracy, recalls, harmful up/downgrades, schema failures, latency, usage,
machine profile). Runs are append-only: a rerun resumes or gets a new
config-hash run id — earlier results are never overwritten. Inspect with:

```powershell
.\.venv\Scripts\python.exe -m autograder bench report --role grade_primary
```

(or read `metrics.json` per run directly). Commit reviewed result artifacts
per repository policy; never commit model weights or Ollama caches.

## Guarantees / what NOT to run

- **No cloud grading is possible from these commands** — production-mode
  gateway + role-independent `--research` gate + `is_remote_route` URL check;
  cloud grading cost is $0 by construction.
- Do **not** pass `--research` here; it exists only to reproduce the
  historical cloud baselines.
- Do **not** run `bench final-eval` / touch HELD_OUT until one candidate is
  frozen in models.toml and the owner explicitly decides the final run.
- Do **not** re-tune `grade-v4-charitable` on CALIBRATION cases.
- RAG stays `RAG_DISABLED` for selection (a local-grader RAG A/B is a later,
  separate experiment).

## After the runs

1. Read the per-candidate reports; pick the winner on DEV, confirm on
   CALIBRATION finalists.
2. Write the winner into `models.toml [models.grade_primary]` (backend
   ollama, local base_url) — replacing `UNSELECTED`.
3. Only then consider the single HELD_OUT confirmation run.
4. `e004_q2_r8`: repair its transcription/evidence (labeling app on the
   strong PC) before treating strict CALIBRATION metrics as final.

## Troubleshooting / rollback

- Freeze mismatch (exit 2): someone changed a frozen artifact — `git status`
  + `python scripts/local_grade_freeze.py --verify` names the field; restore
  the artifact rather than regenerating the freeze, unless the change was a
  deliberate, reviewed re-freeze.
- Preflight cannot see Ollama: start it (`ollama serve`) — preflight only
  lists metadata and never loads a model.
- A crashed run: rerun the same command — the runner resumes, skipping
  completed cases (failures are re-attempted only with `--retry-failed`).
- Nothing in this experiment writes to the live labeling DB or to any
  benchmark label file.
