# Exam Autograder

Automatic grading of **scanned student exams** — Hebrew/RTL forms with
handwritten answers, matching questions with justifications, multiple-choice
answer tables, messy corrections, and instructor annotations mixed into the
scan — against an official answer key and rubric, using **open-weight
vision-language models** through a provider-independent inference layer.

The finished system requires **no Anthropic key, no Claude subscription, and
no paid proprietary API**: it runs against any OpenAI-compatible server
(local Ollama, university vLLM/TGI server, llama.cpp, or a free hosted API
serving an open model). An optional Anthropic backend exists for development
comparison only.

## Settled architecture (2026-08)

Measured on the frozen Hebrew benchmark (`evaluation/hebrew_bench_v2`,
129 items, owner-verified + born-digital references):

| Task | Component | Status |
|---|---|---|
| Multiple-choice answer grids | **Deterministic CV** (`autograder/tablecrop.py`) + local model for variant symbol only | **Production.** Live: 13/13 variants, 120/120 auto-decided rows correct, 0 silent errors |
| Printed Hebrew / mixed He-En | Local `qwen3-vl:8b-instruct` (Ollama) | **Production-adequate** (CER 0.066 printed / 0.148 mixed) |
| Handwritten Hebrew explanations | **Gemini 3 Flash** (optional `GEMINI_API_KEY`) — strongest measured (CER 0.315 vs 0.53-0.68 alternatives); ML Kit Digital Ink + line-split router as the local/offline research arm | **Research-stage — NOT wired into the grading pipeline.** All handwriting reads require human review; see `evaluation/m2_grading/` for the decision-preservation evidence |
| Handwriting local fallback | ML Kit strike-aware router (`scripts/m2_linesplit_v3.py`, `android/mlkit-ink-runner`) | Experimental; needs an Android runtime; ~0.53 median CER |

The core product a client runs today is the **multiple-choice autograder +
web UI**, fully local and offline. Handwritten-explanation grading remains
review-gated by design.

## Run the local web application

```powershell
.\.venv\Scripts\python.exe -m autograder ui
```

Opens the lecturer-facing interface at http://localhost:8501: upload an
answer key and student exams (single files, many files, or a ZIP), pick an
exam template (multiple-choice-only / with-explanations / mixed), start
grading, watch live progress, pause/stop/resume safely, and download
per-exam JSON/Markdown plus combined CSV/JSON/ZIP reports. Grading runs in
a detached process, so closing the browser or the app never loses work —
see [docs/ui.md](docs/ui.md).

## Quick start (fresh machine)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m pytest                 # offline tests, no network/model needed

# with a local open model (see docs/deployment.md):
ollama pull qwen3-vl:8b-instruct    # NOT bare qwen3-vl:8b — that tag is the thinking
                                    # variant and burns max_tokens on reasoning
autograder doctor --backend openai --base-url http://localhost:11434/v1 --model qwen3-vl:8b-instruct
autograder grade  --backend openai --base-url http://localhost:11434/v1 --model qwen3-vl:8b-instruct `
    --exam sample_data/student_exam.pdf --key sample_data/Exam_solution.pdf --out out
```

## Commands

| Command | Purpose |
|---|---|
| `autograder doctor` | backend reachability + model availability check |
| `autograder parse-key` | answer key PDF → structured `answer_key.json` |
| `autograder grade` | grade one exam end to end (JSON + Markdown report) |
| `autograder make-manifests` | discover graded exams in `test/`, build the deterministic train/validation split |
| `autograder eval-batch` | grade a whole split, compare totals to instructor grades, write combined reports |
| `autograder audit-leakage` | probe whether the model can read instructor grades from (un)masked scans |

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | pipeline stages, backend layer, grading-policy decisions |
| [docs/variants.md](docs/variants.md) | cover-page flower → exam variant (A1/A2/A3), question-order alignment, mapping evidence |
| [docs/reliability-layer.md](docs/reliability-layer.md) | grading modes (legacy/reliability/shadow), evidence-grounded grading, batch anomalies, grade invariants, review reason codes/priority/grouping, decision traces, package preflight, cost estimates, privacy, integration status |
| [docs/model-comparison.md](docs/model-comparison.md) | open-model research, licenses, hardware, hosted-API survey, final choice |
| [docs/deployment.md](docs/deployment.md) | local / university-server / hosted deployment, hardware requirements |
| [docs/datasets.md](docs/datasets.md) | dataset layout, manifests, split, label discipline |
| [docs/privacy-and-leakage.md](docs/privacy-and-leakage.md) | filename-label & instructor-annotation leakage prevention, masking, audit |
| [docs/training.md](docs/training.md) | training/calibration strategy comparison |
| [docs/evaluation.md](docs/evaluation.md) | benchmarks, metrics, held-out final-test procedure and freeze |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | what has actually been executed, verified, and what remains blocked |
| [evaluation/](evaluation/report.md) | live validation results: Stage A batch, per-item audits vs ground truth, error taxonomy, leakage audit, performance |

## Output contract

Per sub-item: identifier, type, interpreted final answer, transcription of
the explanation, selection correctness, explanation evaluation, points per
component, total and maximum, a concise reason, and an uncertainty flag.
Exam-level: totals, per-question breakdown, unanswered list,
ambiguous-for-review list, version-detection record, the log of how
conflicting/corrected markings were interpreted, and the exact
backend/model/configuration used.

Grading policy highlights: ambiguous ≠ incorrect ≠ unanswered; student
marking-convention notes govern interpretation; instructor red-ink
annotations are masked before inference and excluded by prompt; explanations
gate credit where the rubric says so; answers are never invented — unclear
items are flagged for human review.

Answer-source authority: dedicated answer sheets (detected structurally,
never by a hardcoded page count) are the only normal source of final
answers; student ink on question pages is scratch work and can stand in
only as review-flagged secondary evidence when the sheet is missing, blank,
damaged, or unreadable. Exam variants are detected from the cover-page
flower marker BEFORE grading (never from answers or score maximisation) and
each variant grades against its own key column with per-variant question
alignment — see [docs/variants.md](docs/variants.md).

The expensive answer-key parse is cached persistently (key document +
rubric + model + prompts + render settings fingerprint) — a batch parses
each unique key once; `--no-key-cache` disables it.
