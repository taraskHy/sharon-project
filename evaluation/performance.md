# Performance evaluation — RTX 2000 Ada (15.4 GB), 2026-07-13

Model: `qwen3-vl:8b-instruct` = Qwen3-VL 8.8B **Q4_K_M** GGUF (Ollama ID
0533d74300e4, Apache-2.0), json_schema constrained decoding, temperature 0.
Host: Windows 11, Ryzen 5 5600G, 64 GB RAM, driver 595.97, Ollama 0.31.2.

## Context-length configurations (measured)

| Context | VRAM (model+KV) | Total VRAM observed | CPU offload | Verdict |
|---|---|---|---|---|
| 8192 | ~6.5 GiB | 10.0 GiB peak (incl. ~2 GiB desktop) | none — `ollama ps` 100 % GPU | fits every single-page call (probes, variant detection, judging) |
| 16384 | ~8 GiB | ~11–12 GiB | none | **recommended batch default**: covers every per-exam call under the two-resolution architecture with ~3 GiB margin; the 19.6 K-token key parse does NOT fit (fails loudly: `exceed_context_size_error`, measured) |
| 32768 | ~10 GiB | **14.6–14.7 GiB — the card's ceiling** | none | required ONLY for the two cached one-time calls: answer-key parse (~19.6 K prompt tokens) and per-variant alignment derivation (~12–18 K). Never run in parallel |

## Throughput and latencies (measured live)

| Metric | Value |
|---|---|
| Decode speed | 19–21 tok/s during 19.6 K-context key parse; ~33 tok/s at small context |
| Prompt eval | 11 text tokens ≈ 0.26 s; one 1000 px page ≈ 1 K vision tokens, seconds |
| Probe A (text-only Hebrew judging) | 4.6 s |
| Probe B (one printed page, 1000 px) | 11.1 s |
| Probe C (bubble table, 1000 px) | 4.8 s |
| Variant detection (cover page) | ~5–10 s |
| Survey (13 pages @ 640 px) | ~1.5–2.5 min |
| Sheet close-read (3 pages @ 1400 px) | ~1–2 min |
| Alignment derivation (once per variant, 32 K) | ~3–5 min, then cached (hit ≈ instant, verified live) |
| Answer-key parse (once per key/config, 32 K) | **~12 min** (7–9 K output tokens at ~20 tok/s + ~20 K-token prefill); persistent cache hit ≈ 30 s incl. deterministic repair (verified live) |
| Chunked extraction (Q1/Q2 1 call each, Q3 3 calls) | ~4–6 min total |
| Explanation judging (2 questions) | ~1–2 min |
| **Whole exam, cached key + cached alignment** | **~10–14 min** (measured on the representative exam, runs 6–7) |

GPU utilization during probe window: peak 100 %, ~45 % mean while active,
peak power 69.5 W (421 samples @ 2 s). Batch-window telemetry: appended
after the staged batch (gpu_batch.csv summary).

## Key-parse attempt ledger (why the cache + deterministic repair exist)

Six model attempts on one structural parse (details:
docs/validation/smoke-2026-07-13-strongpc-diagnosis.md): two flattened
version columns, two missing columns, one client timeout at 900 s, one
accepted. Per operational policy no further model retries are spent on this
task: the text-layer repair layer decodes/verifies the version columns
deterministically on every load, and defective parses are rejected and
never cached.

## Feasibility projection for ~100 exams (stated need)

With the key and all three alignments cached: ~10–14 min/exam sequential on
this card ⇒ **~17–24 h for 100 exams** — feasible unattended with `--resume`
(per-exam isolation, per-stage fingerprints, bounded retries), but not
fast. Practical accelerators, in order: university vLLM server with
Qwen3-VL-32B (minutes per exam, better accuracy, xgrammar constrained
decoding); free hosted open-model APIs for non-sensitive stages; reduced
judging scope. CPU offload never occurred in any measured configuration.

## Configuration guidance

Use 16 K as the batch default; run one 32 K pass only to seed the key and
alignment caches when a key/model/prompt changes. 8 K suffices for probes
and single-page diagnostics. Do not run parallel requests at 32 K (the card
sits at 14.7/15.4 GiB).
