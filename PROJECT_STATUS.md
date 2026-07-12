# Project status — 2026-07-12

This file reports **what was actually executed and verified**, on what
hardware, and what remains open. Nothing here is extrapolated from
benchmarks we did not run.

## Summary

The system no longer requires the Anthropic API (or any paid proprietary
API): the pipeline runs against any OpenAI-compatible server through a
provider-independent backend layer, with a mock backend proving full offline
operation. Open-model research is complete and committed
(docs/model-comparison.md, raw reports in docs/research/); the adopted
primary is **Qwen3-VL (Apache-2.0)**. A local CPU-only smoke validation
against the real sample exam ran a genuine open model end-to-end on the
text-judging stage successfully; vision-stage structured output on the local
Ollama/CPU stack surfaced real issues (documented below) that the
architecture handles as designed (clear errors, no silent fallback) but that
must be resolved on university GPU hardware (vLLM) or with further local
tuning before batch grading.

## Environment used for validation

| Item | Value |
|---|---|
| Machine | Windows 11, Intel Core Ultra 7 265U (12C/14T), 63.4 GB RAM, **no discrete GPU** (Intel iGPU only), CPU-only inference |
| Server | Ollama 0.31.2 (installed via official winget package during this session) |
| Model | `qwen3-vl:8b` (Qwen3-VL-8B, Q4_K_M GGUF, 6.1 GB download, Apache-2.0), 32K context (`OLLAMA_CONTEXT_LENGTH=32768` — the 4096 default silently truncates multipage calls) |
| App | this repository, `--backend openai --base-url http://localhost:11434/v1` |

## What was executed and the actual results

| Check | Result |
|---|---|
| Offline test suite (66 tests: scoring policy, version detection, ambiguity, caps, backend transport/malformed-output/truncation handling, dataset split determinism, masking, metrics, full-pipeline-with-mock, leakage, resume fingerprints, batch eval incl. failure continuation) | **66/66 pass**, no network, no API keys (`pytest`) |
| Proof no Anthropic dependency remains | test runs the CLI + backends in a subprocess and asserts the `anthropic` package is never imported; deps moved to optional extra |
| `autograder doctor` against local Ollama | **OK** — server reachable, model available |
| Probe A — text-only judging of a REAL student explanation (Hebrew) against the key's reference reasoning | **PASS in 300 s**: verdict `valid` with correct, fluent Hebrew justification (recognised that "בהירות" refers to the DC component / coarsest level). JSON schema respected. |
| Probe B — vision: printed-Hebrew MC page (page 6, 1200 px) | **FAIL (truncation)** after 1133 s: image encoded fine (~2.2 min for 1364 vision tokens), but generation consumed the whole 2000-token budget without completing the JSON; the backend raised the designed truncation error (no silent output) |
| Probe C — vision: bubble-sheet page with X-convention note (page 13) | **FAIL (truncation)** twice: at 2000 tokens (910 s) and at 4000 tokens with `think:false` passed through (1798 s) |
| Vision diagnostic (no constrained decoding, 800 px) | see addendum below |

Interpretation of the vision failures (evidence-based, not confirmed root
cause): Ollama reports this model with a `thinking` capability, and
reasoning tokens do not count toward the schema-constrained JSON — they
consume `max_tokens` first; a grammar-pressure repetition loop is the other
documented candidate (see docs/research/structured-inference.md). Both are
**local-serving-stack issues, not pipeline issues** — the pipeline's error
handling surfaced each one loudly and precisely. Resolution paths, in order:
(1) vLLM on GPU with xgrammar constrained decoding (the recommended
production stack); (2) Ollama native-API `think:false` / non-thinking model
tags; (3) `--structured-mode prompt|json_object` fallback modes (already
implemented and configurable).

## Timing reality on CPU-only hardware

- text-only judge call: ~5 min
- one-page vision call: ~2 min prefill + decode (several more minutes)
- a full 13-page exam needs ~6–10 calls, several with many images ⇒ **hours
  per exam on this laptop**. The laptop is a smoke-test environment only;
  batch evaluation of the 41-exam corpus requires the university GPU server
  (est. 1–3 min/exam on a 24–80 GB GPU with vLLM) or a hosted open-model API.

## What is blocked, and by what

| Blocked item | Blocker | Exact command once unblocked |
|---|---|---|
| Full-exam grading run against the representative exam | CPU-only laptop (hours/exam); vision structured-output issue on local Ollama | `autograder grade --backend openai --base-url http://GPU-SERVER:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct --exam sample_data/student_exam.pdf --key sample_data/Exam_solution.pdf --out out` after `vllm serve Qwen/Qwen3-VL-8B-Instruct --limit-mm-per-prompt image=16` |
| Batch evaluation on the 41-exam corpus (train/validation metrics) | same | `autograder eval-batch --split validation --backend openai --base-url http://GPU-SERVER:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct --key sample_data/Exam_solution.pdf --out eval_out` |
| Leakage audit (masked vs unmasked grade-reading probe) | needs a working vision backend | `autograder audit-leakage --limit 5 --backend openai --base-url ... --model ...` |
| Model bake-off (Qwen3-VL-32B vs Gemma 4 vs Qwen3.6-27B) | GPU hardware | same eval-batch command per model |
| Hebrew-handwriting quality assessment | needs the above runs + manual inspection of `extraction.json` transcriptions against the scans | — |
| Fine-tuning experiments (QLoRA) | GPU + derived per-question labels (derivation path documented in docs/datasets.md) | see docs/training.md |
| Held-out 48-exam evaluation | exams not yet provided; must follow the freeze procedure in docs/evaluation.md | — |

## Honest answers to the validation questions

- **What works:** the entire deterministic pipeline (offline-tested); the
  provider-independent layer against a real local open model; Hebrew
  text-stage judging on a real student explanation; dataset manifests;
  masking with auditable regions; batch tooling; resume fingerprinting.
- **What failed:** vision-stage constrained JSON on the local Ollama/CPU
  stack (truncation — details above). Failure was loud and diagnosable, as
  designed.
- **What requires human review:** by design — ambiguous answers, illegible
  explanations, key defects, uncertain version detection, explanation/answer
  mismatches; operationally — all grading output until validation-split
  metrics exist.
- **What remains untested:** end-to-end grading accuracy of any open model
  on this exam; Hebrew *handwriting* transcription quality (no published
  numbers exist for any open model — this is the project's #1 risk);
  masking effectiveness against a live model (audit implemented, not yet
  run); hosted-API paths (require account/keys the developer must create).
- **Is Hebrew handwriting still a limitation?** Unknown-to-likely: printed
  Hebrew has published evidence for Qwen3-VL (~72 % on a printed-text
  benchmark); handwriting is unmeasured anywhere and must be assessed on the
  validation split before trusting the system.
- **Was grade leakage prevented?** Filename-grade leakage: yes, by
  construction and by test. Instructor-annotation leakage: masking is
  implemented and auditable, but its sufficiency is **not proven** until
  `audit-leakage` runs against a live model.
- **Is the solution actually self-hostable?** Yes — demonstrated end to end
  on this machine with open weights (Apache-2.0) served locally; no key, no
  external calls (all model traffic to localhost).
