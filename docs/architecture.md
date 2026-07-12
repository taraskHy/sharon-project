# Architecture

## Pipeline

The grading pipeline separates *reading* from *grading* so ambiguity is
preserved instead of silently resolved:

```
answer key PDF ──► 1. KEY PARSING     structured key: questions, sub-items,
   (+ rubric)                         per-version answers, points, caps,
                                      reference reasoning, rubric rules
student scan  ──► 0. MASKING          red-ink instructor annotations removed
                                      from rendered pages (auditable regions)
              ──► 2. SURVEY           whole-document pass: page inventory,
                                      student marking-convention notes,
                                      instructor-ink identification,
                                      authoritative answer locations
              ──► 3. EXTRACTION       per question, blind to correct answers:
                                      marks observed, final answer under the
                                      conventions, verbatim transcription,
                                      answered / unanswered / ambiguous
              ──► 4. JUDGING          model compares each written explanation
                                      against the key's reference reasoning
              ──► 5. SCORING          plain Python: normalisation, version
                                      detection, points, caps, review flags
```

Stages 1–4 are model calls; stage 5 is deterministic and unit-tested offline.
Every stage's output is written to the run directory as JSON, and `--resume`
reuses a stage only when a fingerprint over *all* inputs (exam bytes, key
bytes, rubric, backend, model, generation parameters, render size) matches.

## Provider-independent inference layer

The pipeline depends only on `autograder.backends.VisionBackend`:

```
parse(system=..., content_blocks=[text|image...], output_model=PydanticModel) -> instance
health_check() -> HealthReport
describe()     -> exact backend/model/config identity (recorded in results
                  and in the resume fingerprint; never contains secrets)
```

| Backend (`--backend`) | Serves | Notes |
|---|---|---|
| `openai` | **Ollama, vLLM, HF TGI, llama.cpp server, LM Studio, OpenRouter, Groq, Mistral La Plateforme, …** | one implementation for every OpenAI-compatible chat-completions server; images as base64 data URLs |
| `mock` | offline tests, plumbing checks | fixture-driven; records every request for leakage tests |
| `anthropic` | development comparison only | optional extra (`pip install -e .[anthropic]`); the finished system does not require it |

Structured output: the backend requests constrained decoding where the server
supports it (`--structured-mode json_schema`, the default; `json_object` and
`prompt` for weaker servers), **always** validates the reply against the
Pydantic schema locally, retries malformed output a bounded number of times
with the validation error fed back, and then fails loudly. Truncation
(`finish_reason=length`) and refusals are hard errors. There is **no silent
fallback to a different model** — the exact backend and model that produced a
result are recorded in `result.json` under `backend_info`.

Switching deployment target is a configuration change only:

```
# local Ollama            --backend openai --base-url http://localhost:11434/v1 --model qwen3-vl:8b
# university vLLM server  --backend openai --base-url http://gpu-server:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct
# hosted open-model API   --backend openai --base-url https://openrouter.ai/api/v1 --model <open model> --api-key-env OPENROUTER_API_KEY
# offline tests           --backend mock --model tests/fixtures
```

or a TOML file (`--config grader.toml`, CLI flags win):

```toml
[backend]
backend = "openai"
model = "qwen3-vl:8b"
base_url = "http://localhost:11434/v1"
structured_mode = "json_schema"
timeout_s = 600.0
transport_retries = 2
validation_retries = 2

[grading]
max_image_edge = 1600
max_tokens = 8000
```

## Modules

| Module | Responsibility |
|---|---|
| `autograder/backends/` | provider-independent inference (this page, above) |
| `autograder/ingest.py` | PDF/images → rendered page images (+ text layer when present) |
| `autograder/masking.py` | red-ink annotation masking with auditable region reports |
| `autograder/key_parser.py` | answer key + rubric → structured `AnswerKey` |
| `autograder/survey.py` | whole-document survey pass |
| `autograder/extract.py` | per-question extraction + reconciliation (duplicates, id drift) |
| `autograder/grade.py` | version detection, explanation judging, deterministic scoring |
| `autograder/report.py` | Markdown report rendering |
| `autograder/dataset.py` | dataset discovery, anonymization, deterministic split, manifests |
| `autograder/metrics.py` | total-score evaluation metrics |
| `autograder/evalcli.py` | `make-manifests`, `eval-batch`, `audit-leakage` commands |
| `autograder/cli.py` | `doctor`, `parse-key`, `grade`, config resolution, fingerprints |

## Key grading-policy decisions

- **Ambiguous ≠ incorrect ≠ unanswered** — ambiguous items earn 0 points
  *pending human review* and are listed separately from unanswered ones.
- **Document-level conventions rule** — student notes like "X marks are
  final" are captured once in the survey and applied everywhere; every
  application is logged in `mark_interpretations`.
- **Explanations gate credit** where the rubric says so; an explanation that
  correctly justifies a *different* option than the one selected is flagged
  for a human, not silently decided.
- **Multi-version keys** — per-version answers; the version is detected from
  answer agreement and the decision (with margins) is recorded; thin margins
  flag the exam for review.
- **Key defects are reviewable** — a sub-item with no accepted answers for
  the graded version yields a review flag, not a wrong answer.
