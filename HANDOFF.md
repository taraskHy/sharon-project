# HANDOFF — continue on the stronger PC

Written 2026-07-12 at the end of the development session on the **weaker PC**
(Windows 11, Intel Core Ultra 7 265U, 63 GB RAM, **no discrete GPU** — CPU-only
inference). Local-model experiments on that machine are **stopped** per the
project owner's instruction; do not run further inference there.

> ⚠️ **Performance caveat:** every runtime number in this document and in
> PROJECT_STATUS.md was measured on the weaker, CPU-only PC. **Do not use
> those timings to estimate performance on the stronger machine.** Re-measure
> everything there.

---

## 1. Original project objective

Automatic grading of scanned Hebrew university exams (multipage image-only
PDFs; RTL printed text + handwritten Hebrew + English terms + math; matching
questions with written justifications; multiple-choice with a separate answer
table) against an official answer key and rubric. Core difficulties: deciding
which visible marking is the student's *final* answer (circles, X marks,
filled bubbles, cross-outs, overwrites, document-level convention notes such
as "answers marked with X are final"), ignoring red instructor annotations,
judging short explanations semantically, and never silently inventing an
answer — ambiguous ≠ incorrect ≠ unanswered, with human-review flags.

A later requirement update added: the finished system must need **no
Anthropic key, no Claude subscription, and no paid proprietary API** — only
open-source / openly licensed models, with a provider-independent inference
layer (free hosted open-model API / local Ollama / university vLLM server /
mock for tests), dataset management with strict label-leakage prevention, a
held-out-test discipline, benchmarking, and honest validation reporting.
The full requirement text is in the conversation that produced this repo;
its operative parts are restated across `docs/`.

## 2. What is implemented (all committed on `initial-prototype`)

- **Full grading pipeline** (`autograder/`): answer-key parsing (multi-version
  keys, colour-coded versions, accepted alternatives, caps) → whole-document
  survey (marking conventions, instructor-ink description, authoritative
  answer locations) → per-question extraction (blind to correct answers;
  answered/unanswered/ambiguous; verbatim transcription; duplicate/id-drift
  reconciliation) → explanation judging → deterministic scoring (Hebrew
  letter normalisation, automatic version detection with margins, explanation
  gating, partial credit, caps, review flags) → JSON result + Markdown report.
- **Provider-independent backends** (`autograder/backends/`): `openai`
  (any OpenAI-compatible server — Ollama/vLLM/TGI/llama.cpp/LM Studio/
  OpenRouter/Groq/Mistral), `mock` (fixtures; records all inputs), optional
  `anthropic` (dev comparison only; optional extra, lazily imported — a test
  proves the app never imports it). Structured output: `json_schema` /
  `json_object` / `prompt` modes + local Pydantic validation + bounded repair
  retries; truncation and refusals are hard errors; no silent model fallback;
  exact backend/model/config recorded in every result (`backend_info`).
- **Config**: CLI flags or TOML (`--config`, see `grader.example.toml`);
  timeouts, retries, temperature, max tokens, image resolution, structured
  mode, concurrency; API keys only via env var names.
- **Dataset tooling** (`autograder/dataset.py`, `datasets/`): filename parsing
  (`<index>_<grade>.<ext>`), malformed/duplicate reporting, anonymized IDs
  (`exam-002` — never contain the grade), deterministic split (seed 42,
  25 train / 16 validation), version-controlled manifests, held-out
  `final_test_manifest.json` placeholder with the freeze rules.
- **Leakage prevention** (`autograder/masking.py`, `evalcli.py`): red-ink
  masking with auditable per-page regions (originals never modified),
  `audit-leakage` command (masked-vs-unmasked grade-reading probe —
  implemented, **not yet run**), tests proving no filename/grade token
  reaches model input.
- **Batch evaluation** (`eval-batch`): manifest-driven, anonymizes, masks by
  default, continues after per-exam failures, fingerprint-guarded resume,
  per-exam results + combined JSON/CSV + `summary.md` + failed list + review
  list; metrics: exact/±2/±5/±10, MAE, median AE, RMSE, mean signed error,
  max abs error, review rate, runtime.
- **Safety of resume**: `--resume` re-uses stage outputs only when a SHA-256
  fingerprint over exam bytes + key bytes + rubric + backend + model +
  generation config + render size matches (tests cover exam/key/model/config
  changes).
- **Docs**: `docs/architecture.md`, `model-comparison.md` (+ raw research in
  `docs/research/`), `deployment.md`, `datasets.md`, `privacy-and-leakage.md`,
  `training.md`, `evaluation.md`, `PROJECT_STATUS.md`, this file.

## 3. Current architecture (one paragraph)

`cli.py` resolves backend + grading config (CLI/TOML), builds a
`VisionBackend`, and runs `run_grade_pipeline`: pages are rendered by
`ingest.py` (PyMuPDF, no temp files), optionally masked by `masking.py`,
then the four model passes produce Pydantic-validated JSON at each stage
(persisted per stage for audit/resume), and `grade.py` computes the scores
deterministically. `evalcli.py` wraps this per manifest entry for batches.
The model never receives filenames, paths, or grades — images and
key-derived text only.

## 4. Files added/changed in this session

Everything on branch `initial-prototype` (already pushed). Top level:
`autograder/` (package, incl. `backends/`), `tests/` (6 files, 64 tests),
`docs/` (7 docs + `research/` ×7 + `validation/` ×3 probe logs),
`datasets/` (4 manifests/issue files), `scripts/smoke_probe.py`,
`grader.example.toml`, `pyproject.toml`, `README.md`, `PROJECT_STATUS.md`,
`.gitignore`, `HANDOFF.md`. Data directories: `sample_data/` (representative
exam + key), `test/` (41 graded exams — **committed; contains real student
scans; keep the GitHub repo private**).

## 5. Models researched (details + URLs: docs/model-comparison.md)

Serious candidates, all license-checked live on 2026-07-12: **Qwen3-VL
8B/32B/30B-A3B** (Apache-2.0; only family with published Hebrew-OCR
evidence), **Gemma 4** (Apache-2.0 since 2026-03), **Qwen3.6-27B**
(Apache-2.0, multimodal), **Gemma 3 27B** (Gemma ToU — automated-decision
gray zone), **Mistral Small 3.2-24B / Ministral 3 14B** (Apache-2.0),
**InternVL3.5-8B/38B**, **MiniCPM-V 4.5**. Excluded: Pixtral Large
(research license), Mistral OCR (API-only), all Llama vision (language
grounds), OCR+LLM route (no open Hebrew-handwriting OCR exists; OCR can't do
marks/ink/conventions). Free hosted APIs with acceptable data policies:
**Groq** (`qwen/qwen3.6-27b`, no-training/no-retention), **Cerebras**
(`gemma-4-31b`, preview). OpenRouter free routes / NVIDIA NIM / Google
unpaid tier train on or human-review inputs — do not send student scans.
**Adopted decision** (docs/model-comparison.md §FINAL DECISION): single-VLM
architecture; Qwen3-VL-32B on university vLLM; Qwen3-VL-8B for local dev;
bake off Gemma 4 and Qwen3.6-27B before freezing.

## 6. Exact model downloaded on the weaker PC

- Ollama 0.31.2 (installed via `winget install Ollama.Ollama`, official
  package). **Server processes were stopped at session end**, but Ollama
  remains installed and may auto-start its tray app on login (uninstall via
  Windows Apps, or disable autostart in Task Manager → Startup).
- Model: `qwen3-vl:8b` — Qwen3-VL-8B Q4_K_M GGUF, 6.1 GB download,
  Ollama model ID `901cae732162`, Apache-2.0. ~10 GB RAM resident with 32K
  context, 100 % CPU on that machine.

## 7. Probe configurations and results (weaker PC — do NOT extrapolate)

Script: `scripts/smoke_probe.py`; raw logs: `docs/validation/`.

| Run | Config | Result |
|---|---|---|
| doctor | `--backend openai --base-url http://localhost:11434/v1 --model qwen3-vl:8b` | OK (server reachable, model listed) |
| Probe A (text-only judging of a real Hebrew student explanation) — 1st run | default ctx **4096**, json_schema, temp 0, max_tokens 2000 | model call OK in 207.7 s; script crashed **printing** Hebrew to a cp1252 console (fixed: UTF-8 reconfigure) |
| Probe A — 2nd run | `OLLAMA_CONTEXT_LENGTH=32768`, json_schema, max_tokens 2000 | **PASS in 300.3 s** — verdict `valid`, correct fluent Hebrew reasoning (recognised "בהירות" ≙ DC component). Full output in `docs/validation/smoke-2026-07-12-probes-ABC.txt` |
| Probe B (vision: printed-Hebrew MC page 6, 1200 px) | 32K ctx, json_schema, max_tokens 2000 | **FAIL — truncation** after 1132.5 s. Server log: image prefill 3 batches ≈ 2.2 min (1364 vision + 252 text tokens); generation then consumed the whole budget without closing the JSON |
| Probe C (vision: bubble-sheet page 13 with X-convention note) | 32K ctx, json_schema, max_tokens 2000 | **FAIL — truncation** after 910.0 s |
| Probe C retry | max_tokens 4000, `extra_generation {"think": false}` (possibly ignored by Ollama's OpenAI endpoint) | **FAIL — truncation** after 1798.3 s |
| Probe C diagnostic (prompt mode, 800 px, no grammar) | `--structured-mode prompt --max-image-edge 800 --max-tokens 800/1600 --no-think` | **NEVER COMPLETED** — first attempt's background task was killed; second attempt stopped by owner's halt order. Root cause of truncation (thinking-token consumption vs constrained-decoding repetition loop) is **undetermined** |

Known pitfall confirmed live: Ollama's runtime context defaulted to **4096**
(`ollama ps`) despite the model's 262K max — must set
`OLLAMA_CONTEXT_LENGTH` (or a Modelfile `num_ctx`) or multipage calls
silently truncate.

## 8. Known failures / open problems

1. **Vision structured output truncates on local Ollama + qwen3-vl:8b**
   (three reproductions above). Candidate causes: (a) the Ollama tag runs
   with thinking enabled (`ollama show` lists a `thinking` capability) and
   reasoning tokens consume `max_tokens` before/outside the constrained
   JSON; (b) documented grammar-pressure repetition loops
   (docs/research/structured-inference.md). Next diagnostics (on the strong
   PC): run the prompt-mode probe to see raw output; try Ollama native API
   `think:false`; try `--structured-mode json_object`; or skip Ollama and
   use vLLM, where constrained decoding (xgrammar) is the recommended path.
2. **CPU runtime** on the weak PC: ~5 min/text call, ~2 min image prefill —
   hours per exam. Weak-PC numbers are smoke-level only.
3. **Leakage audit not yet executed** (needs a working vision backend).
4. **Hebrew handwriting quality unmeasured** — no published numbers for any
   open model; project's #1 risk; measure on the representative exam +
   validation split.
5. Ollama's OpenAI endpoint ignores unknown fields (e.g. `think`) silently —
   verify effective options server-side (`ollama ps`, server logs).

## 9. Tests currently passing

`pytest` → **64/64**, fully offline (network access is blocked inside key
tests). Coverage: deterministic scoring policy (incl. the sample exam's
swapped-tables / X-convention / version-detection behaviours), backend
transport errors, malformed-output repair + hard failure, truncation
errors, health checks, no-Anthropic-import proof, dataset parsing/split
determinism/manifests, masking (red removed, blue/black preserved, regions
recorded, originals untouched), metrics math, full pipeline + eval-batch
end-to-end on a mock backend, filename/grade leakage prevention, resume
fingerprint invalidation across exam/key/model/config changes, batch
continuation after per-exam failure.

## 10. Remaining tasks (priority order)

1. Strong PC: serve a model and re-run `scripts/smoke_probe.py`; resolve the
   vision structured-output truncation (see §8.1).
2. Grade the representative exam end-to-end; compare with ground truth
   (version A1; swapped answer tables; X-marks-final; instructor Q1=24/32,
   Q2=28/32 — details in docs/evaluation.md and the memory notes).
3. Run `audit-leakage` (masked vs unmasked) and record the verdict.
4. `eval-batch --split train` for calibration, then `--split validation`
   for reportable metrics; inspect `extraction.json` transcriptions manually
   (Hebrew handwriting quality).
5. Bake off Qwen3-VL-32B vs Gemma 4 vs Qwen3.6-27B (same commands, different
   `--model`); decide, then consider QLoRA only if extraction dominates the
   error budget (docs/training.md).
6. Freeze configuration per docs/evaluation.md, then (and only then) obtain
   and evaluate the 48 held-out exams once.

## 11. Exact commands for the stronger PC

```powershell
# setup
git clone https://github.com/taraskHy/sharon-project.git && cd sharon-project
git checkout initial-prototype
python -m venv .venv && .\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m pytest                      # expect 64 passed

# option 1 — GPU + vLLM (recommended; Linux/WSL)
pip install vllm
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --limit-mm-per-prompt image=16
python -m autograder doctor --backend openai --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct

# option 2 — Ollama (Windows-friendly)
winget install Ollama.Ollama
setx OLLAMA_CONTEXT_LENGTH 32768      # then restart the Ollama service/app
ollama pull qwen3-vl:8b               # or qwen3-vl:32b on a 24GB+ GPU
python -m autograder doctor --backend openai --base-url http://localhost:11434/v1 --model qwen3-vl:8b

# validation sequence (replace BASE/MODEL to match the option chosen)
python scripts/smoke_probe.py --base-url BASE --model MODEL
python -m autograder grade --backend openai --base-url BASE --model MODEL `
    --exam sample_data/student_exam.pdf --key sample_data/Exam_solution.pdf --out out
python -m autograder audit-leakage --backend openai --base-url BASE --model MODEL --limit 5
python -m autograder eval-batch --split validation --backend openai --base-url BASE --model MODEL `
    --key sample_data/Exam_solution.pdf --out eval_out
```

## 12. Uncommitted changes / background processes

- **Working tree:** clean after the final commit accompanying this file
  (`git status` to confirm). Untracked-and-ignored only: `.venv/`, `out/`,
  `.idea/`, `.pytest_cache/`.
- **Background processes on the weak PC:** all inference stopped; every
  `ollama`/`ollama app` process was killed at session end. Ollama itself and
  the 6.1 GB `qwen3-vl:8b` blob remain on disk (`%USERPROFILE%\.ollama`) and
  the tray app may auto-start on next login — harmless, but uninstall/disable
  if unwanted.
- **Remote:** `origin` = https://github.com/taraskHy/sharon-project.git,
  branch `initial-prototype`. Note that `test/` (real student scans) is in
  the repository — keep it private.
