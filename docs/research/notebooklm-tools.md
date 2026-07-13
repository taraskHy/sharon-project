# Evaluation: open-notebook & notebooklm-py (2026-07-13)

Requested evaluation of two NotebookLM-style repositories for possible use by
this project. Context: the custom grading pipeline is not up for replacement;
the question is whether either repo helps with (a) the grading pipeline,
(b) resumable rate-limited batch orchestration (~100 exams per occasional
batch, free hosted APIs acceptable), or (c) something adjacent.

**Bottom line: adopt neither.** Both are document-RAG/chat products in the
NotebookLM mold — neither is a batch-processing framework, and neither
addresses the pipeline's hard problem (vision-model reading of Hebrew
handwritten scans), which the pipeline already solves more directly.

## 1. lfnovo/open-notebook — https://github.com/lfnovo/open-notebook

Self-hosted, open-source NotebookLM clone: web app for organizing research
sources, chatting with citations, notes, "transformations", podcast
generation. Aimed at privacy-conscious researchers doing RAG over their own
documents with their own models.

| Attribute | Finding |
|---|---|
| License | MIT |
| Maturity / activity | 35.5k stars; created Oct 2024; very active (v1.12.0 released 2026-07-12) |
| Providers | 18+ via `esperanto`: OpenAI, Anthropic, Groq, **Ollama**, Mistral, OpenRouter, LM Studio, any OpenAI-compatible endpoint — same provider space as the autograder |
| Vision input | **None.** Design issue [#331](https://github.com/lfnovo/open-notebook/issues/331): "Open Notebook can't yet *understand* images — neither images embedded in documents (PDF figures, scanned/watermarked pages) nor pure image sources." Images ignored at ingestion; vision PR #791 unmerged, "needs-design" |
| OCR | Text-layer extraction only (`content-core`). OCR toggle is an open request ([#1104](https://github.com/lfnovo/open-notebook/issues/1104)); image OCR earlier closed "not planned" (#881). Planned path is Docling — would not handle Hebrew handwriting anyway |
| Batch / rate limits / resume | Internal SurrealDB-backed job queue (`surreal-commands`) welded to the app; nothing reusable as external batch infra |
| Hebrew/RTL | Zero evidence; no Hebrew UI locale |
| Deployment | Heavy: Docker Compose, SurrealDB + FastAPI + Next.js; no pip-library mode |

**Verdict: (d) not useful for the core problem; weakly (c).** It literally
cannot see the exam scans (image-only PDFs ingest as empty). Plausible only
as a self-hosted Q&A UI over *typed* Hebrew course materials, reusing the
same Ollama/Groq credentials — a convenience, not infrastructure.

## 2. teng-lin/notebooklm-py — https://github.com/teng-lin/notebooklm-py

Unofficial Python API/CLI/MCP server for **Google's hosted NotebookLM**,
built on reverse-engineered, undocumented `batchexecute` endpoints. Auth via
Playwright browser login / cookie import / master token.

| Attribute | Finding |
|---|---|
| License | MIT |
| Maturity / activity | 17.6k stars; created Jan 2026; very active (v0.7.3 stable 2026-06-30; v0.8.0a3 pre-release 2026-07-06); unusually rigorous engineering (ADRs, VCR cassettes, CI) |
| Providers | **Google/Gemini only** — no Ollama, no OpenAI-compatible endpoints, no model choice; requires a Google account |
| Vision input | Indirectly yes: NotebookLM does server-side OCR/vision on uploads including image-only scans; Hebrew among ~80 supported languages. Quality on messy Hebrew handwriting unverified and uncontrollable |
| Batch / rate limits / resume | Client retries (3 attempts); hard server-side quotas — free tier **50 chat queries/day**, 50 sources/notebook, 10 studio generations/day ([quota-limits.md](https://github.com/teng-lin/notebooklm-py/blob/main/docs/quota-limits.md)); no resumable job queue. A 100-exam batch exceeds free-tier quota on day one |
| Deployment | Light: pip/uv install (+ optional Playwright ~170 MB) |
| Risk | README's own words: undocumented Google APIs "can change without notice"; "best for prototypes, research, and personal projects". Student-exam PII through an unofficial Google API is disqualifying for institutional grading |

**Verdict: (d) not useful for the core problem; narrowly (c).** No model
pinning, no prompt/structured-output control, quotas below one batch, PII
governance failure. Defensible only as (1) scripted Q&A over course handouts
or (2) a one-off *anonymized* benchmark of Gemini-grade OCR on the specific
handwriting as a reference point for the autograder's own accuracy.

## 3. Side-by-side

| Criterion | open-notebook | notebooklm-py |
|---|---|---|
| What it is | Self-hosted NotebookLM clone (web app) | Unofficial client for Google's hosted NotebookLM |
| Ollama / OpenAI-compatible | Yes (all major providers) | No — Google only |
| Local / private | Yes | No (Google cloud) |
| Reads image-only scanned PDFs | No (images ignored) | Yes, via server-side OCR (uncontrolled) |
| Hebrew handwriting viability | None | Unverified, "varies" |
| Batch / retry / resume | Internal queue only, not reusable | 3 retries; hard quotas; no resume |
| Deployment | Heavy (Docker, SurrealDB, web stack) | Light (pip) |
| Useful for grading (a) / batch infra (b) | No / No | No / No |

## 4. Consequence for the project

The stated need — resumable, rate-limit-tolerant batching of ~100 exams
against free hosted APIs — is already substantially covered **inside this
repository**: `eval-batch` continues after per-exam failures and resumes via
fingerprint-guarded stage reuse (`--resume`), and the OpenAI-compatible
backend retries HTTP 429/5xx with exponential backoff honouring
`Retry-After` (`autograder/backends/openai_compat.py`). If free-tier daily
caps (e.g. Groq) ever bite mid-batch, the missing piece is not a framework —
it is at most a `--retry-failed` convenience loop around the existing
failed-exam list, re-invoking `eval-batch` until the failed list is empty.
