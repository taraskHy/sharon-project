# qwen38_27b_q4km — arm freeze (declared BEFORE any reference scoring)

Date: 2026-08-13. Purpose: run the official qwen3.8:27b-q4_K_M through the
CANONICAL benchmark runner unchanged; only the provider/model inference
layer differs from the frozen local-Qwen arm.

## Model (verified locally)
- tag `qwen3.8:27b-q4_K_M`, digest `25b843619e94` (== official registry)
- 27.3B params, Q4_K_M, arch qwen35, projector clip 460.73M, requires
  Ollama 0.32.12; installed Ollama 0.32.13
- capabilities: completion, vision, tools, thinking

## Frozen invocation (canonical scripts/m2_bench_run.py, backend qwen_local)
- --preproc contrast --max-edge 1100 (mirrors the frozen qwen8b arm)
- prompt/schema/parser: canonical strict-fidelity per-category PROMPTS,
  {"transcription"} json_schema, parse_declared_envelope — untouched
- temperature 0, max_tokens 500 (canonical ChatVLM defaults)
- --extra-body '{"think": false, "options": {"num_ctx": 8192,
  "repeat_penalty": 1.0}}'
  * think:false = non-thinking/direct mode, CONFIRMED at request level:
    sanity call returned a 22-token direct answer, no <think>/reasoning
    content in the raw body
  * repeat_penalty 1.0 explicitly frozen here, pre-scoring
  * num_ctx 8192 REQUESTED; EFFECTIVE context is 16384 because the
    server-level OLLAMA_CONTEXT_LENGTH=16384 takes precedence on model
    load in this Ollama version (verified: identical 16384 after a
    stop/reload with the 8192 request). Recorded honestly; not
    re-configured (server env is a shared production setting).
- no RAG, no course material, no reference/key/rubric/grader input
  (runner never reads references — canonical guarantee)

## Sanity call (canonical smoke item hl_e004_q1_r3__l1, reference NOT opened)
- image accepted; real Hebrew transcription returned via the frozen
  schema; latency 145.4 s; prompt 679 / completion 22 tokens
- ollama ps during inference: 18 GB in memory, 29% CPU / 71% GPU
  (partial CPU offload — the q4 weights exceed the 15.4 GB card),
  context 16384

## Item sets (canonical, not recreated)
- smoke-5: evaluation/unlimited_ocr/gate_items.json `smoke_5_frozen`
- full: hebrew_bench_v2/items.json (129 items)
- comparison baselines: gemini_protocol_clean_v1 (canonical Gemini),
  mlkit_ink_rtl_a1 (top-1), per-item persisted records via the canonical
  evaluator only

## Scoring discipline
ONLY scripts/m2_bench_eval.py produces canonical CER/WER; if it cannot
load, inference outputs are preserved and scoring stops.

## Amendment 2026-08-16 (before any reference scoring): transport = Ollama native /api/chat

The first launch attempt (OpenAI-compat backend qwen_local) produced, on the
first smoke item, status 200 / 500 completion tokens / EMPTY content: the
model spent its whole budget in `reasoning`. Reproduced deterministically:
Ollama's /v1/chat/completions IGNORES `"think": false` (with or without
response_format), while Ollama's native /api/chat honors it (same crop,
same frozen prompt+schema: direct JSON answer in 26 s / 40 tokens,
thinking_len 0). The earlier sanity call had merely fit a direct answer
inside the 500-token budget on an easier crop.

Fix (transport only): backend `ollama_native` = OllamaNativeVLM — same
frozen prompt, {"transcription"} schema (as /api/chat `format`),
temperature 0, num_predict 500 (== max_tokens 500), think:false,
options {num_ctx 8192, repeat_penalty 1.0}, same result schema + parser +
resume. Nothing in the frozen experimental configuration changed. The one
invalid empty record produced under the ignored-think path was deleted
before relaunch; no reference was opened at any point.
