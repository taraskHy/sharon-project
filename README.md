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

## Quick start

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m pytest                 # 64 offline tests, no network/model needed

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
| [docs/model-comparison.md](docs/model-comparison.md) | open-model research, licenses, hardware, hosted-API survey, final choice |
| [docs/deployment.md](docs/deployment.md) | local / university-server / hosted deployment, hardware requirements |
| [docs/datasets.md](docs/datasets.md) | dataset layout, manifests, split, label discipline |
| [docs/privacy-and-leakage.md](docs/privacy-and-leakage.md) | filename-label & instructor-annotation leakage prevention, masking, audit |
| [docs/training.md](docs/training.md) | training/calibration strategy comparison |
| [docs/evaluation.md](docs/evaluation.md) | benchmarks, metrics, held-out final-test procedure and freeze |
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | what has actually been executed, verified, and what remains blocked |

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
