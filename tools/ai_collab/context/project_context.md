# Reviewer context — sharon-project exam autograder (v1, 2026-08)

Stable architecture facts for the independent code reviewer. Keep this file
compact and versioned; task-specific detail belongs in each run's `task.md`.

## What the product is

Automatic grading of scanned student exams (Hebrew/RTL, handwritten marks)
against an official answer key, using open-weight vision-language models via
a provider-independent inference layer. The finished system must run with no
Anthropic key and no paid proprietary API (any OpenAI-compatible server; an
optional Anthropic backend exists for development comparison only).

## Non-negotiable design rules

- **Deterministic-first.** CV table cropping, policy gates, authority rules,
  scoring and validation are plain Python; model calls are the exception.
  Multiple-choice early exits (`choice_only`, `wrong_choice_zero`) skip
  explanation OCR and grading entirely.
- **The student's OCR transcription is immutable evidence.** Grading-side
  course RAG informs the grader's understanding of correctness only; it must
  never alter transcriptions, selected MC options, images, OCR status, or
  student evidence spans. RAG is never OCR repair; the frozen OCR-repair
  experiment arm is separate and must stay separate.
- **ModelGateway is the only cloud boundary.** `autograder/gateway.py` routes
  task -> backend/model from `models.toml` (env-expanded, no model ids in
  code) and wraps every call with request cache, usage ledger, budget
  (soft warn / hard pause, never silent downgrade) and privacy rules
  (crops only, identity-free prompts, no filenames/labels to providers).
  New cloud calls that bypass the gateway are architecture violations.
- **Grading modes:** legacy (validated pipeline, hooks off), reliability
  (gateway runtime on), shadow (reliability runs read-only next to legacy;
  shadow output is non-authoritative). Hooks default OFF; absent
  `models.toml`, the validated pipeline must be unchanged.
- **Minimal human review is the product goal; review is never silently
  dropped.** Ambiguity is preserved and routed to REVIEW, not resolved by
  guessing. Handwritten-explanation grading stays review-gated.
- **Minimal OpenRouter usage.** Caches, packs, early exits, tiny structured
  outputs, crops-only images; identical requests must never be paid twice.
- **Frozen experiment artifacts** (`evaluation/*` benchmarks, manifests,
  ground truth) must not be rewritten by feature work.
- **Graphify is navigation only.** The static code graph aids exploration;
  it misses DI/runtime wiring; source and tests are authoritative.

## Review priorities for this repo

1. Correctness against the task; no silent behavior change to the validated
   legacy pipeline (default-off discipline for new hooks/flags).
2. Privacy/leakage: nothing student-identifying to providers or logs.
3. Provider boundaries: no hardcoded model ids/keys; env-config only;
   gateway not bypassed.
4. Resource behavior: bounded calls/tokens; caches used; no per-student
   rebuild of shared artifacts.
5. Offline test adequacy: the default pytest suite runs with no network and
   no models; live paths need explicit opt-in markers/scripts.
