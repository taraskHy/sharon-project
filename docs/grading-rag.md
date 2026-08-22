# Grading-side course RAG — architecture and boundaries

Local course retrieval feeding the **grader** with supplemental context.
Structurally separate from the frozen experimental OCR-repair arm
(`scripts/m2_rag_ocr.py`): grading RAG never touches student text, and the
OCR arm never touches grading packs (pinned by
`tests/test_gradingpack.py::test_no_rag_text_reaches_ocr_repair_path`).

## The core safety rule

The student's OCR transcription is immutable evidence. RAG influences the
grader's *understanding of correctness*; it never changes the student
transcription, the selected MC option, the exam images, OCR status, or the
student evidence spans a grader cites. Course text can never satisfy
student-evidence validation (`evidence.py` verifies spans against the frozen
transcription only; a span that exists only in a course chunk is fabricated).

## Retrieval query

Question-level and identity-free — built from the pack's canonical question
text + rubric + official solution, capped at 1,500 chars
(`gradingpack.rag_query`). Never student OCR text (a bad reading would
retrieve the wrong material and bias the grade), never names/ids/paths.
Retrieval runs against the persistent local course index (bge-m3 embeddings
via local Ollama; `courses.retrieve`). No cloud call, no cloud embedding, no
index rebuild per question or student.

## Preparation vs. use (important cost distinction)

- **PREPARATION** happens once per exam package at pack build: the top-k
  chunks are retrieved locally and cached on the pack (`rag_prepared`).
  Free — no provider tokens, no repeated per-student searches.
- **USE** is when chunks enter a grading request (`rag_evidence` →
  `to_grader_context`). Only use costs OpenRouter input tokens, and only the
  policy decides when: `RAG_ALWAYS` at build; the lazy policies activate the
  prepared cache (`activate_rag`) only when the policy fires.

## Policies (default: RAG_DISABLED)

| Policy | Primary grader | Escalation | Retrieval |
|---|---|---|---|
| `RAG_DISABLED` | no context | no context | none at all |
| `RAG_ALWAYS` | cached context | inherits | once, at build |
| `RAG_ON_UNCERTAIN` | no context; ONE retry with context after an unclean primary | context if reached | once, at build (prepared) |
| `RAG_ON_ESCALATION` | no context | context | once, at build (prepared) |

Default stays `RAG_DISABLED` until the frozen A/B measures a benefit. When
an optional-RAG policy finds no course/retriever/index, grading continues
WITHOUT context and records `rag_available=false` — never a REVIEW by itself.
An `OCR_UNRESOLVED` reading never reaches grading RAG: OCR trouble routes to
OCR resolution/REVIEW, not to a grader with course vocabulary.

## Budgets

`top_k=2`, `max_rag_chars=1200` (`gradingpack.DEFAULT_*`), deterministic
truncation with provenance preserved (chunk id, source, page, score, excerpt
hash). Packs record chars + estimated tokens; ledger rows carry
`rag_chars`/`rag_chunks`/`rag_policy`/`pack_hash` per call; run accounting
separates RAG overhead and what the MC early exit skipped.

## Pack lifecycle (persistent, per-package)

`run_grade_pipeline` (non-legacy) loads packs from `PackStore`
(`--packs-root`, default `<out>/../packs` — shared by all students of a job)
keyed by `source_fingerprint(key bytes, course index hash, policies, top_k,
char budget, pack schema version, retrieval version, RAG policy)`. Any input
change rebuilds; otherwise every student reuses the identical persisted pack.

## Lazy explanation OCR (reliability mode)

Extraction defers gradeable explanations (`legibility="deferred"`,
structural); the reliability route transcribes per item through gateway task
`ocr_primary` only after MC extraction → MC resolution → grading-policy gate
prove the transcription is needed. `wrong_choice_zero`/`choice_only` early
exits therefore make ZERO model calls (pinned by raising-mock tests in
`tests/test_lazy_ocr.py`). Shadow mode stays eager (the authoritative legacy
judge needs transcriptions); legacy mode is unchanged.

## Cloud/gateway boundary

Every OpenRouter-capable production route passes privacy → cache → budget →
provider → ledger inside `ModelGateway.call`. Classification uses the
EFFECTIVE provider (`usage.effective_provider`): `backend="openai"` +
`base_url=openrouter.ai` is OpenRouter for budget/ledger/metadata. The
legacy direct-backend path (local extraction/survey/judging) **refuses**
OpenRouter-effective configurations outright
(`cli.guard_direct_cloud_backend`); it remains available for local/mock/
anthropic-dev backends as the validated compatibility path. Remaining known
non-gateway cloud surfaces: `--backend anthropic` (documented dev-comparison
only) and the frozen `scripts/m2_*` bench harnesses (curated data, not the
student pipeline).

## Canonical data locations (2026-08-22)

- **Course store:** `<repo>/courses` (anchored to the repository by
  `courses.courses_root()`; `GRADER_COURSES_DIR` overrides). The CV index
  (`courses/CV`: 1 source, 430 chunks, bge-m3, built 2026-08-10) was migrated
  by COPY from the old PyCharm working copy on 2026-08-22 — no rebuild, no
  embedding call; `courses.index_status("CV")` reports `indexed=True,
  stale=False`. The active working copy no longer depends on the old copy.
- **Jobs:** `<repo>/jobs` (anchored; `GRADER_JOBS_DIR` overrides).
- **Exam packages:** `grader.toml [ui] package_dirs` (local machine config,
  gitignored; `["packages", "sample_data", "prob_data"]` here) or
  `GRADER_PACKAGE_DIRS`; the prob package (`prob_data/sol.answer_key.template.json`)
  is discovered through that line.
- **Model config:** `models.toml` (local, gitignored, all cloud roles
  UNSELECTED); campaign state `evaluation/model_selection/state/`.
