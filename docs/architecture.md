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
              ──► 1b. VARIANT         cover-page marker (flower) → exam
                                      variant A1/A2/A3 per the authoritative
                                      mapping (<key>.variants.json); never
                                      inferred from answers or scores;
                                      unclear marker → human review
                                      (docs/variants.md)
              ──► 2. SURVEY           whole-document pass at LOW resolution
                                      (survey_image_edge, default 640px):
                                      page classification (question pages /
                                      dedicated answer sheets / mixed /
                                      instructor-only) + functional regions
                                      (answer_table, explanation_area,
                                      scratch_work, instructor_grading,
                                      convention_note), answer-sheet policy,
                                      student convention notes, instructor-ink
                                      identification, version hints
              ──► 3. EXTRACTION       per question, blind to correct answers,
                                      at FULL resolution but only on the
                                      authoritative answer-sheet pages (plus
                                      convention notes; question pages only
                                      when no dedicated sheet exists): marks
                                      observed, final answer under the
                                      conventions + its origin (answer sheet
                                      vs question page), verbatim
                                      transcription, answered / unanswered /
                                      ambiguous, answer-sheet condition
              ──► 2b. ALIGNMENT       per-variant map of PRINTED sub-item
                                      numbering onto the key's canonical
                                      numbering (variants shuffle question/
                                      option order); one model call per
                                      variant, validated deterministically,
                                      cached persistently; invalid → identity
                                      + every sub-item review-flagged
              ──► 3b. AUTHORITY       plain Python (autograder/authority.py):
                                      question-page scratch never silently
                                      overrides a usable answer sheet — such
                                      answers are demoted to ambiguous and
                                      routed to review; when the sheet is
                                      missing/blank/damaged/ambiguous,
                                      question-page evidence may stand only
                                      as flagged secondary evidence
              ──► 4. JUDGING          model compares each written explanation
                                      against the key's reference reasoning
              ──► 5. SCORING          plain Python: normalisation, version
                                      detection, points, caps, review flags
```

Stages 1–4 are model calls; stages 3b and 5 are deterministic and
unit-tested offline. Every stage's output is written to the run directory as
JSON, and `--resume` reuses a stage only when a fingerprint over *all*
inputs (exam bytes, key bytes, rubric, backend, model, generation
parameters, render sizes) matches.

### Where final answers are read from

Most pages of a scanned exam are the question booklet; students write
circles, notes, calculations and tentative answers on them. Those markings
are scratch work, not final answers. Exams in this corpus provide a small
number of dedicated answer-sheet pages (typically near the end) carrying the
matching answers, short explanations, the multiple-choice answer table,
convention notes and numbering corrections — the survey DETECTS them from
headings, instructions, table layouts, repeated question identifiers and
position (never a hardcoded count or location), and records them in
`answer_sheet_policy`. Extraction then reads final answers from those pages
only, and reports per-sub-item provenance (`answer_origin`) plus the sheet's
condition per question (`answer_sheet_status`). Exams whose printed
instructions say booklet markings are not checked are honoured strictly;
exams with no dedicated sheet at all use the question pages' response areas
as the normal authoritative source. The deterministic authority pass (3b)
enforces all of this structurally, so a prompt-ignoring model cannot smuggle
scratch work into grades: sheet usable → question-page answers are demoted
to ambiguous + human review; sheet broken → they stand only as
low-confidence secondary evidence flagged for review.

This split is also the inference-cost optimisation: one cheap low-resolution
pass over everything to LOCATE, then full-resolution vision only on the few
pages that carry gradeable content. Question pages are consulted (in full
resolution) only for questions the survey could not place or exams without
answer sheets — not resent for every question.

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
