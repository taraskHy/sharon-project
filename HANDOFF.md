# HANDOFF — current state (2026-07-13, strong-PC validation session)

The 2026-07-12 handoff (weak CPU-only PC) is superseded. This session ran
on the validation machine: Windows 11, Ryzen 5 5600G, 64 GB RAM, **RTX 2000
Ada 15.4 GB**, Ollama 0.31.2, model `qwen3-vl:8b-instruct` (Q4_K_M,
Apache-2.0). Everything below is committed on `initial-prototype`.

## Read these first

- [PROJECT_STATUS.md](PROJECT_STATUS.md) — what is verified, on what
  hardware, honest limitations.
- [docs/variants.md](docs/variants.md) — cover-flower → variant system,
  question-order alignment, authoritative mapping + evidence.
- [docs/validation/smoke-2026-07-13-strongpc-diagnosis.md](docs/validation/smoke-2026-07-13-strongpc-diagnosis.md)
  — probe results, truncation root causes, key-parse attempt ledger.
- [evaluation/representative_exam_audit.md](evaluation/representative_exam_audit.md)
  — per-item audit of the representative exam with instructor-reference truth.
- [docs/deployment.md](docs/deployment.md) — context-length table for the
  15.4 GB card (16K batch default; 32K only for the cached one-time calls).

## How to run on this machine

```powershell
# server (detached); 32K only needed to (re)seed key/alignment caches
$env:OLLAMA_CONTEXT_LENGTH="16384"; & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve

.\.venv\Scripts\python.exe -m pytest                 # offline suite
.\.venv\Scripts\python.exe -m autograder grade `
    --backend openai --base-url http://localhost:11434/v1 --model qwen3-vl:8b-instruct `
    --exam sample_data/student_exam.pdf --key sample_data/Exam_solution.pdf `
    --out out --max-image-edge 1000 --survey-image-edge 640 --timeout 1800

.\.venv\Scripts\python.exe -m autograder eval-batch --split validation --limit 5 `
    --backend openai --base-url http://localhost:11434/v1 --model qwen3-vl:8b-instruct `
    --key sample_data/Exam_solution.pdf --out eval_out `
    --max-image-edge 1000 --survey-image-edge 640 --timeout 1800 --resume
```

Key facts: the answer-key parse is cached persistently
(`%LOCALAPPDATA%\autograder\key_cache`; `GRADER_KEY_CACHE` overrides); the
parsed key's version columns are deterministically verified/repaired from
the PDF text layer on every load; colour-only values come from
`sample_data/Exam_solution.versions-override.json` (Q3 items 1–15/17–20
still need instructor entries — until then they are review-flagged on every
exam). Variant detection reads the cover flower only; per-variant question
alignment is cached next to the key cache.

> **Fresh-session entry point: [evaluation/NEXT_SESSION_HANDOFF.md](evaluation/NEXT_SESSION_HANDOFF.md)**
> (state, metrics, retained outputs, and the mandated first task).

## Known open items

1. **Q3 override entries** — instructor should fill the remaining Q3
   per-version letters (template + evidence format in the override file).
2. **Answer-table swap perception** — crossed-out title digits are below
   the 8B model's reliable perception (missed at 1000 px and 1400 px on the
   representative exam). The deterministic swap tripwire review-flags the
   crossed-agreement signature instead; a stronger model (Qwen3-VL-32B on
   the university vLLM server) should re-test close-read recall.
3. **Hebrew handwriting** — letter misreads on messy sheets and skipped
   explanation transcriptions bound accuracy; row-attribution drift between
   identical runs (GPU nondeterminism). Candidate next steps: row-band
   crops for tables, higher extraction resolution for sheets only, the
   32B model bake-off (docs/model-comparison.md).
4. **Batch metrics** — staged validation batch results land in
   `evaluation/`; see PROJECT_STATUS for the current stage.
5. The 48-exam held-out set remains untouched (frozen procedure in
   docs/evaluation.md).
