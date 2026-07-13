# Strong-PC session 2026-07-13 — root cause of the vision truncation failures

Machine: Windows 11, AMD Ryzen 5 5600G, 64 GB RAM, **NVIDIA RTX 2000 Ada
(15.4 GB VRAM)**, driver 595.97. Ollama 0.31.2 (winget), server started with
`OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_KEEP_ALIVE=30m`.

## Environment verification

- `pytest`: **64/64 pass** (fresh Python 3.12.10 venv, `pip install -e .[dev]`).
- `ollama pull qwen3-vl:8b` → ID `901cae732162`, arch qwen3vl, 8.8B params,
  **Q4_K_M**, Apache-2.0. `ollama show` lists capabilities:
  completion, vision, tools, **thinking**; sampling defaults temp=1,
  top_p=0.95, top_k=20 (Qwen3 *thinking-variant* presets).
- `ollama ps` after warm-up: `qwen3-vl:8b … 6.5 GB 100% GPU 8192` —
  **fully GPU-resident, zero CPU offload**, model+KV ≈ 7.5 GiB VRAM
  (nvidia-smi 9604 MiB total incl. ~2 GiB desktop baseline).
- Decode speed on this GPU: 50 tokens in 1.52 s ≈ **33 tok/s**; prompt eval
  11 tokens in 0.26 s. Trivial call round-trip ≈ 2 s (vs ~300 s weak PC).

## Root cause: the default Ollama tag is the THINKING variant

The weak-PC failures (probes B/C truncating at any max_tokens; handoff §8.1
"root cause undetermined") are now fully explained by two live experiments:

1. **Native API** `/api/generate`, `think:false`, `num_predict=50`, prompt
   "Say OK.": `response=""`, `thinking="<think>\nOkay, the user just said…"`,
   `done_reason="length"`, eval_count=50. The model emits reasoning tokens
   regardless of `think:false`; they consume the entire budget.
2. **OpenAI endpoint** `/v1/chat/completions`, `think:false`,
   `response_format=json_schema`, `max_tokens=200`, temp 0: `content=""`,
   the message carries a `reasoning` field with 200 tokens of meta-analysis,
   `finish_reason="length"` in 8.9 s.

Conclusions:

- `qwen3-vl:8b` (the bare tag pulled on both machines) behaves as the
  **Thinking** variant. The adopted model per docs/model-comparison.md is
  **Qwen3-VL-8B-Instruct** — a different fine-tune, tag
  `qwen3-vl:8b-instruct`.
- Ollama 0.31.2 accepts `think:false` on both APIs for this tag but it is
  **ineffective** — reasoning is emitted anyway, outside the
  schema-constrained content, eating `max_tokens` first. Constrained
  decoding applies only to `content`, so the JSON never starts.
- The weak-PC "truncation" was therefore **not** CPU slowness and **not** a
  grammar-pressure repetition loop; it was thinking-token consumption
  (handoff §8.1 candidate (a) — confirmed; candidate (b) — ruled out for
  this failure).
- The pipeline's designed behaviour (hard truncation error, no silent
  output) reported exactly the right symptom on both machines.

**Action:** switch all local runs to `qwen3-vl:8b-instruct`; keep
`think:false` in the payload as belt-and-braces. Probes A/B/C re-run against
the instruct tag follow below.

## Second root cause (probe C only): repetition loop under constrained decoding

With the instruct tag, probes A and B passed but C still truncated at
max_tokens=1200 — reproducibly (40.6 s / 40.9 s, full budget generated).
Capturing the raw content (scripts/diag_probe_c.py, byte-identical request
without the truncation guard) showed the mechanism: the page contains a row
where one option is BOTH circled and X-marked, and the probe schema forced a
free-text `question_1_final_answer` string. At temperature 0 under grammar
constraint the model cycled the same contradictory deliberation sentence
("So B is reported… but it is invalidated… However…") verbatim ~15 times
until the budget died. This is the grammar-pressure repetition loop
documented in docs/research/structured-inference.md — triggered by a
SEMANTIC conflict with no schema escape hatch, not by grammar syntax.

**Fix (schema discipline, matches model-comparison.md §4.5):** free-form
observation fields first; every verdict field bounded (enum) with explicit
escape values (`ambiguous` / `unanswered`); "do not enumerate every row"
instruction. After changing `TableReadProbe` accordingly, probe C passes in
4.8 s choosing `"B"`. The production pipeline schemas already carry
`status="ambiguous"` escape values throughout; the probe now follows the
same discipline. Both weak-PC candidate causes were therefore REAL, on
different calls: (a) thinking-tag budget consumption (probes B/C on the old
tag), (b) constrained-decoding repetition loop (probe C's original schema).

## Probe results — constrained config (this machine, GPU)

Config: `qwen3-vl:8b-instruct`, 8192 server context, json_schema mode,
max_tokens 1200, temperature 0, single page per request, images at 1000 px,
no thinking (capability absent on the instruct tag).

| Probe | Result | Weak PC (thinking tag, CPU) |
|---|---|---|
| A — text-only Hebrew judging | **PASS 4.6 s**; verdict `partially_valid`, fluent coherent Hebrew reasoning (wanted the explicit term "DC component") | PASS 300.3 s |
| B — printed Hebrew MC page 6 + marks | **PASS 11.1 s**; valid JSON; marks located on options A–D; transcription READABLE but in reversed (visual) character order — a known VLM RTL failure mode; `marked_option_of_first_question` null | FAIL (truncation) 1132.5 s |
| C — bubble table p.13 + X-convention note | **PASS 4.8 s** (after schema fix); marks description correct (circles + cross-outs + convention); note transcription garbage ("X 75 \|W'0") but note MEANING inferred correctly ("only X counts"); row-1 verdict "B" | FAIL (truncation) 910–1798 s |

GPU telemetry over the probe window (421 samples @ 2 s, `nvidia-smi`):
peak utilization **100 %**, mean ~45 % when active, peak VRAM **10 030 MiB**
/ 15 356 (≈7.5 GiB model+KV above the ~2 GiB desktop baseline), peak power
69.5 W. `ollama ps` throughout: **100 % GPU, zero CPU offload**, context
8192.

Reading quality summary for the grading risk register:
- Hebrew PRINTED text: readable but char-order-reversed transcriptions at
  1000 px (extraction prompts must not rely on transcription order;
  scoring never does).
- Hebrew HANDWRITING: still the #1 risk — the convention note transcribed
  as garbage at 1000 px, though its meaning was correctly inferred from
  visual context.
- X/circle/correction detection: marks and their kinds detected on both
  probe pages; per-row attribution not yet validated against ground truth
  (that is the full-pipeline test's job).
- Instructor-annotation separation: not exercised by these probes (masking
  audit + survey ink separation cover it; see the pipeline run).

## Answer-key parse attempts — ledger (model: qwen3-vl:8b-instruct, GPU)

The key parse (~19.6 K prompt tokens, 7–9 K output tokens) proved the least
reliable model stage: output quality varies BETWEEN runs even at
temperature 0. Six attempts were spent before switching to a deterministic
path; per the owner's directive, no further model retries are made for this
structural parsing task.

| # | When (session) | Config | Wall time | Outcome |
|---|---|---|---|---|
| 1 | grade run 3 | timeout 900 s, no validation | ~10–11 min | "Success", but version columns FLATTENED (every version = A1's letter) — root cause of the margin-0 version detection in that run |
| 2 | grade run 4 | timeout 900 s | 900 s → client timeout | Server still decoding at ~19 tok/s (>7 K tokens, likely a repair round); no silent retry (by design) |
| 3 | parse-key #1 | timeout 1800 s, validation not yet implemented | ~13 min | Parsed but `versions=['default']` — defective; cache entry deleted |
| 4+5 | parse-key #2 | + version validation, versions hint | ~25 min (2 attempts) | Both REJECTED: Q1 sub-items had no per-version answers; candidates preserved as `answer_key.rejected-*.json` |
| 6 | parse-key #3 | + worked decode example, flattening detector | ~12 min | **Accepted**: versions A1/A2/A3, Q1+Q2 columns exactly match the text-layer letter groups; Q3 (colour-only) columns version-uniform → handled below |

**Resolution (deterministic, no more model retries):**
`autograder/keyrepair.py` decodes the per-version letter groups (e.g.
"F/F/G") straight from the key's born-digital PDF **text layer** in legend
order, overriding model columns on disagreement, on every load (including
cache hits). Colour-only values with no text-layer group (this key's Q3
multiple-choice answers) are taken from the one-time operator override file
(`Exam_solution.versions-override.json`; item 3.16's A2=B entered with
evidence from the key's own note + the printed A2 form) or flagged
`versions_unverified` → every affected exam sub-item is routed to human
review until the instructor fills the override. The validated, repaired key
is what the persistent cache stores; nothing exam-specific is hardcoded in
code.

## Context-length note for pipeline runs

The probes honour the 8K directive. The full pipeline's whole-document
calls exceed it: the answer-key parse (all key pages at full resolution +
text layer, one call) measured **19 601 prompt tokens** — over 16K too —
so pipeline runs on this machine use `OLLAMA_CONTEXT_LENGTH=32768`
(matching the weak-PC configuration; ~6.5 GiB model + KV still fits the
15.4 GiB card fully on GPU). Ollama 0.31.2 fails such overruns loudly
(`exceed_context_size_error`) rather than silently truncating — verified
live at 16K. Per-question extraction stays small by design: the survey
locates the dedicated answer sheets and extraction sends only those pages
at full resolution (docs/architecture.md).
