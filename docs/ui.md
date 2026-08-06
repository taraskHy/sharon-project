# Web interface guide

Start:

```powershell
.\.venv\Scripts\python.exe -m autograder ui
```

(equivalent: `autograder ui` when the package is installed; add `--port N`
to change the port). The app opens at http://localhost:8501.

Prerequisite: an inference backend. For local grading run Ollama with the
project model (see docs/deployment.md):

```powershell
$env:OLLAMA_CONTEXT_LENGTH="16384"; ollama serve   # once, in its own terminal
ollama pull qwen3-vl:8b-instruct                   # once
```

## Workflow

1. **Sidebar — model backend.** Defaults come from `grader.toml` when
   present. "Check backend" verifies the server is reachable and the model
   is available before any grading starts.
2. **New batch tab.**
   - *Configured exam package* lists every key that ships with a
     `.template.json` sidecar (e.g. the probability exam in `prob_data/`,
     the image-processing exam in `sample_data/`), showing its grading mode,
     answer-sheet rule, and — when variants exist — the authoritative
     marker→variant mapping table.
   - *Upload key & configuration* accepts a new answer key (PDF or parsed
     JSON), an optional rubric, an exam mode (`multiple_choice` /
     `with_explanation` / `mixed` with per-question modes), the answer-sheet
     rule (structural detection vs fixed pages), and an optional variant
     mapping JSON.
   - Upload student exams as individual PDFs/images, several at once, or ZIP
     archives (malformed ZIPs and duplicate/unsupported entries are reported;
     nested paths are flattened; nothing is ever written outside the job).
   - **Create job** copies everything into `jobs/<job-id>/`, anonymizes the
     exam files (`exam-001.pdf`, …) so no original filename or private path
     ever reaches the model, and reports intake issues.
3. **Jobs & results tab.**
   - Pick a job; the status panel auto-refreshes: discovered/pending/
     processing/completed/failed/rate-limited counts, human-review count,
     the exam and pipeline stage currently processing, elapsed time, the
     backend+model in use, and whether the answer-key cache was reused.
   - **Start / Resume** launches the runner as a *detached* process:
     closing the browser or the whole app does not interrupt grading, and
     completed work is never lost. **Pause**/**Stop** finish safely: the
     current exam's subprocess is terminated, its finished pipeline stages
     remain on disk (fingerprint-guarded), and the exam returns to pending.
     Resuming re-runs only pending/failed exams, reusing finished stages.
   - Per-exam details: anonymized id, original filename (UI only), detected
     variant with the marker evidence, answer-sheet pages used, extracted
     answers with per-item confidence, per-question points, explanations
     (when the template grades them), ambiguous items, human-review reasons,
     processing errors, and the raw processing log.
   - Downloads: per-exam `result.json` / `report.md`; combined CSV, combined
     JSON, batch Markdown report, and a ZIP of all reports.

## Job directories

Everything lives under `jobs/<job-id>/` (git-ignored): `job.json` (config),
`state.json` (live status), `uploads/` (anonymized inputs + operator-only
name map), `exams/<anon>/` (per-exam pipeline artefacts incl. `grade.log`),
combined reports at the top level. A job can equally be driven headless:

```powershell
.\.venv\Scripts\python.exe -m autograder run-job --job-dir jobs\<job-id>
```

## Privacy

Model-visible inputs are the anonymized page images and the answer-key
structure only. Original filenames (which may encode grades), private paths,
`grades.csv`-style label files, and instructor-grade information never enter
any model request — enforced by tests (`tests/test_mission_ui_modes.py`).
