<!-- Generated from live web research on 2026-07-12 (7 parallel research passes
     + synthesis). Sources and access dates inline. See docs/ for how this feeds
     deployment and training decisions. -->

# Open-Model Replacement for the Anthropic API in the Hebrew Exam Autograder

**Status:** Research complete (all sources verified live on **2026-07-12** unless noted; every load-bearing claim carries its URL; unverifiable claims are flagged). Decision adopted below; empirical bake-off on the graded-exam corpus still pending (see PROJECT_STATUS.md).

> **Live addendum (2026-07-13, RTX 2000 Ada 15.4 GB):** the decision below is
> confirmed workable on a 15 GB-class GPU — `qwen3-vl:8b-instruct` Q4_K_M runs
> 100 % GPU-resident (~19–33 tok/s; single-page vision calls 5–20 s). Traps
> measured live: (1) Ollama's bare `qwen3-vl:8b` tag is the THINKING variant —
> reasoning tokens consume the whole `max_tokens` budget outside constrained
> JSON, and `think:false` is ineffective on Ollama 0.31.2; always pull
> `qwen3-vl:8b-instruct`. (2) At temperature 0 under json_schema, open-ended
> verdict string fields can enter verbatim repetition loops on genuinely
> conflicting inputs — schemas must put observation fields first and bound
> every verdict with enums incl. explicit escape values
> (docs/validation/smoke-2026-07-13-strongpc-diagnosis.md). (3) Key-parse
> output quality varies BETWEEN runs even at temperature 0 — parsed keys are
> validated deterministically (required versions present) and re-parsed or
> rejected rather than trusted.

## FINAL DECISION (adopted 2026-07-12)

- **Architecture: one vision-language model** end-to-end (survey → extraction →
  judging), keeping the existing deterministic scoring layer. The OCR+LLM
  route is rejected — no open OCR component reads modern Hebrew cursive, and
  OCR cannot do mark detection, ink separation, or convention reasoning (§3).
  Specialized auxiliary detectors remain an option later if benchmarks show
  mark-reading dominates the error budget (docs/training.md).
- **Primary model family: Qwen3-VL (Apache-2.0)** — the only open family with
  published quantified Hebrew OCR evidence, official GGUF/Ollama/vLLM support:
  - university GPU server: **Qwen3-VL-32B-Instruct** on vLLM
    (`response_format json_schema`, xgrammar);
  - development laptop (CPU-only): **Qwen3-VL-8B-Instruct** Q4 via Ollama
    (installed and smoke-validated here — see PROJECT_STATUS.md; requires
    `OLLAMA_CONTEXT_LENGTH` ≥ 32768 to avoid silent truncation).
- **Bake-off fallbacks** before freezing: Gemma 4 (26B-A4B / 31B) and
  Qwen3.6-27B, both Apache-2.0 — run `eval-batch` head-to-head on the train
  split; decide on validation split.
- **Free hosted option** (development / no-hardware fallback): **Groq
  `qwen/qwen3.6-27b`** — GA, no-training/no-retention policy, no card
  required; Cerebras `gemma-4-31b` for free volume. Real student scans go
  only to providers with a no-training/no-retention policy, and preferably to
  the self-hosted deployment (docs/privacy-and-leakage.md).
- **Biggest open risk:** no published handwritten-Hebrew numbers exist for
  ANY open model. This must be closed empirically with the representative
  exam's ground truth and the validation split before the system is trusted.

**Task profile the model must handle:** multipage scanned Hebrew university exams — printed Hebrew (RTL) + handwritten Hebrew + mixed English technical terms + math notation + tables/diagrams; MCQ mark detection (circles, X marks, filled bubbles, cross-outs, overwrites); document-level handwritten marking conventions; blue student ink vs. red instructor ink separation; transcription of short handwritten Hebrew explanations; semantic judging against reference reasoning; reliable structured JSON output.

**Deployment targets:** (a) free hosted API for an open model; (b) self-hosted on a university Linux server (GPU possible); (c) local Windows laptop, CPU-only, Intel iGPU, 63 GB RAM (development).

---

## 0. Executive summary

- **A single VLM architecture is the correct route.** The OCR+LLM route is not viable: no open OCR engine can read modern Hebrew cursive handwriting, and OCR inherently cannot do mark detection, ink separation, or convention reasoning (§3).
- **Qwen3-VL is the only open family with published, quantified Hebrew OCR evidence** (~72% on the tech report's 39-language printed-text benchmark — above the authors' 70% "practical usability" bar, but bottom-five of the supported set). No open model anywhere has published **handwritten**-Hebrew numbers. That gap is the project's #1 risk and must be closed with an in-house bake-off on the ground-truth sample exams (human scores 24/32 and 28/32).
- **Recommended primary:** Qwen3-VL-32B-Instruct (Apache-2.0) on the university GPU server via vLLM; Qwen3-VL-8B-Instruct official GGUF Q4 on the laptop. **Fallbacks to bake off head-to-head:** Gemma 4 (Apache-2.0 since March 2026) and Qwen3.6-27B (Apache-2.0, natively multimodal) — both have broad free-hosted availability but unpublished Hebrew OCR.
- **Free hosted:** Groq (`qwen/qwen3.6-27b`, GA, no-training/no-retention policy) for development; Cerebras (`gemma-4-31b`, preview) for free volume; ModelScope for free Qwen3-VL access (real-name verification hurdle). **Do not send real student exams to any free tier that trains on or retains inputs** (OpenRouter's current free vision routes, NVIDIA NIM trial, Google unpaid tier, Mistral free tier unless opted out).
- **JSON reliability is an engine problem, not a model problem:** schema-constrained decoding in vLLM/llama.cpp/Ollama/LM Studio guarantees syntax on any of these models; design the schema so reasoning fields precede verdict fields (§4.5).

---

## 1. Serious candidates — comparison

Rows are ordered by recommendation priority. The table is split into two parts (1a: model/license/deployment; 1b: capabilities/risks) with identical row order.

### Table 1a — Model, license, deployment

| # | Model (exact version) | License: code / weights | University use OK? | Weights downloadable / offline | Engines | Quantized variants | Hardware (GPU/VRAM · CPU/RAM · disk) | Windows support |
|---|---|---|---|---|---|---|---|---|
| 1 | **Qwen3-VL-32B-Instruct** (rel. 2025-10-21) | Apache-2.0 / Apache-2.0 ([repo](https://github.com/QwenLM/Qwen3-VL), [HF card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)) | Yes — fully clean | Yes / yes | vLLM ≥0.11.0, SGLang, transformers ≥4.57, llama.cpp ([PR #16780](https://github.com/ggml-org/llama.cpp/pull/16780)), Ollama ≥0.12.7 (`qwen3-vl:32b`, 21 GB) | Official FP8; official GGUF (Thinking variant confirmed: [Qwen/Qwen3-VL-32B-Thinking-GGUF](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking-GGUF)); community AWQ/GPTQ | fp16 ~66 GB (1×A100-80G or 2×48G); Q4/AWQ fits 24–32 GB card; CPU Q4 ~21 GB RAM (slow) | Yes (Ollama/llama.cpp native builds) |
| 2 | **Qwen3-VL-8B-Instruct** (rel. 2025-10-15) | Apache-2.0 / Apache-2.0 ([HF card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)) | Yes | Yes / yes | Same as above; official GGUF repo with llama-server instructions ([Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)); Ollama `qwen3-vl:8b` (6.1 GB) | Official GGUF Q4_K_M 5.03 GB / Q8_0 8.71 GB / F16 16.4 GB + mmproj; FP8 | fp16 ~18–20 GB VRAM; Q4 runs comfortably on the 63 GB-RAM laptop | Yes |
| 3 | **Qwen3-VL-30B-A3B-Instruct** (MoE, 3B active; rel. 2025-10-04) | Apache-2.0 / Apache-2.0 | Yes | Yes / yes | vLLM, llama.cpp (MoE supported in PR #16780), Ollama `qwen3-vl:30b` (20 GB) | Official FP8; GGUF Q4 ~20 GB | fp16 ~65 GB total; Q4 fits 63 GB RAM, fast decode (3B active) but slow ViT image prefill on CPU | Yes |
| 4 | **Gemma 4** (E2B/E4B, 12B, 26B-A4B MoE, 31B dense; rel. 2026-04-02) | Apache-2.0 ([model card](https://ai.google.dev/gemma/docs/core/model_card_4), updated 2026-06-26; [Google OSS blog](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html)) | Yes — Apache-2.0 removes all Gemma-ToU concerns | Yes / yes | vLLM day-0 incl. Intel XPU ([vLLM blog](https://vllm.ai/blog/2026-04-02-gemma4)); llama.cpp vision GGUFs listed in [multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) | GGUFs available (official/community); QAT status of Gemma 4 not verified | 31B ~Q4 fits ~18–20 GB; 26B-A4B MoE (3.8B active) attractive for CPU; exact numbers not yet verified per size | Yes (llama.cpp/Ollama) |
| 5 | **Qwen3.6-27B** (rel. 2026-04-22) / Qwen3.5-27B (2026-02) | Apache-2.0 / Apache-2.0 ([Qwen3.6 repo](https://github.com/QwenLM/Qwen3.6), [HF](https://huggingface.co/Qwen/Qwen3.6-27B)) | Yes | Yes / yes | llama.cpp (text+vision), Ollama, vLLM, SGLang, MLX; unsloth GGUF; 212 community quants for Qwen3.5-27B | GGUF Q4 ~16–18 GB | fp16 ~55 GB VRAM; Q4 on 24 GB card or CPU with 63 GB RAM | Yes |
| 6 | **Gemma 3 27B-it** (rel. Mar 2025) | Gemma Terms of Use (updated 2026-04-01, [terms](https://ai.google.dev/gemma/terms)) / same | Mostly — but Prohibited Use Policy bans "automated decisions in domains that affect… individual rights or well-being"; grading w/ human review defensible, fully-automated grading is a gray zone ([policy](https://ai.google.dev/gemma/prohibited_use_policy)) | Yes / yes | Ollama (17 GB), llama.cpp (first-class vision), vLLM | Official QAT int4: 27B = 14.1 GB, 12B = 6.6 GB ([Google blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)) | 27B-int4 on one 24 GB GPU; 12B-QAT ~8 GB usable on laptop CPU | Yes |
| 7 | **Mistral Small 3.2-24B-Instruct-2506** | Apache-2.0 ([HF](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)) | Yes | Yes / yes (API endpoint retires 2026-07-31; weights stay) | vLLM ≥0.9.1 (+mistral_common), llama.cpp (3.1-twin proven in multimodal.md) | Unsloth GGUF Q4_K_M 14.3 GB ([HF](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF)) | bf16 ~55 GB; Q4 on laptop OK; FP8/AWQ ~24–30 GB GPU | Yes |
| 8 | **Ministral 3 14B Instruct 2512** (rel. 2025-12-02) | Apache-2.0 ([HF](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512), [announcement](https://mistral.ai/news/mistral-3/)) | Yes | Yes / yes | vLLM ≥0.12.0; Ollama ≥0.13.1 (`ministral-3:14b`, 9.1 GB, vision confirmed — [library](https://ollama.com/library/ministral-3)); llama.cpp support merged Dec 2025 | Unsloth GGUF Q4_K_M 8.24 GB ([HF](https://huggingface.co/unsloth/Ministral-3-14B-Instruct-2512-GGUF)); **mmproj presence in GGUF repo unverified — check before relying on llama.cpp vision** | FP8 fits 24 GB VRAM; Q4 easy on laptop | Yes |
| 9 | **InternVL3.5-8B / -38B** (rel. 2025-08-26) | MIT (code) / Apache-2.0 (weights) ([HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B), [repo](https://github.com/OpenGVLab/InternVL)) | Yes | Yes / yes | vLLM ([recipe](https://docs.vllm.ai/projects/recipes/en/latest/InternVL/InternVL3_5.html)), LMDeploy ≥0.9.1, SGLang; llama.cpp via **community** GGUF+mmproj only ([bartowski](https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-8B-GGUF)); no official Ollama entry | Community GGUF Q4_K_M 5.03 GB (8B); no official AWQ for 3.5 yet | 8B bf16 ~17 GB (one 24 GB card); 38B needs 2×A100 | Partial (no Ollama packaging; llama.cpp side-load) |
| 10 | **MiniCPM-V 4.5** (8B; rel. 2025-08-26) | Apache-2.0 / Apache-2.0 (HF metadata verified; **stale custom-license text remains in the [GGUF repo card](https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf) — flag to legal**) | Yes | Yes / yes | vLLM ≥0.10.2, SGLang, llama.cpp (official GGUF + mmproj), Ollama ≥0.30 (`minicpm-v4.5`) ([cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook)) | GGUF Q4_K_M 5.03 GB; official int4 (~9 GB GPU); AWQ/BNB | One 24 GB GPU bf16; Q4 fine on laptop | Yes |

### Table 1b — Capabilities and risks (same row order)

| # | Model | Multi-image / PDF pages | Printed Hebrew | Handwritten Hebrew | Layout understanding | Structured JSON | Fine-tuning | Known weaknesses / risks |
|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3-VL-32B | Yes, native multi-image; 256K ctx (→1M YaRN) | **Best evidence of any open model:** Hebrew is one of 32 supported OCR languages at **~72%** printed-text accuracy — above the 70% usability bar, bottom-five of the set (Tech report [arXiv:2511.21631](https://arxiv.org/abs/2511.21631), Fig. 2 p.17, verified from the PDF; corroborated: [Alibaba blog](https://www.alibabacloud.com/blog/qwen3-vl-sharper-vision-deeper-thought-broader-action_602584)) | **No published evidence.** Latin-script exam-grading eval exists ([arXiv:2606.11477](https://arxiv.org/pdf/2606.11477)) showing systematic handwriting failure modes — Hebrew untested | Strong: OCR/KIE + document-parsing cookbooks ([repo](https://github.com/QwenLM/Qwen3-VL)); outperforms Gemini-2.5-Flash/GPT-5-mini on most doc metrics per tech report | Model claims + engine-level guarantee (vLLM xgrammar / Ollama format / GBNF) | vLLM/LLaMA-Factory/ms-swift ecosystem | Repetition loop on long extractions observed in one head-to-head ([Analytics Vidhya, Nov 2025](https://www.analyticsvidhya.com/blog/2025/11/deepseek-ocr-vs-qwen-3-vl-vs-mistral-ocr/)) — mitigate with per-page prompting + repetition penalty; Hebrew is its weakest supported OCR language |
| 2 | Qwen3-VL-8B | Same | Same family evidence (per-size Hebrew number not published) | Same gap | Same cookbooks | Same | Unsloth/LLaMA-Factory | Smaller model → weaker semantic judging; treat as dev-loop model, validate before trusting grades |
| 3 | Qwen3-VL-30B-A3B | Same | Same family evidence | Same gap | Same | Same | MoE FT less mature | CPU image-prefill is the bottleneck (tens of seconds/page); MoE quantization quality less studied |
| 4 | Gemma 4 (31B/26B-A4B/12B) | Yes; variable aspect-ratio/resolution input (better for A4 scans than Gemma 3's fixed 896×896); 128K–256K ctx | "35+ languages out of the box, pretrained on 140+" — **Hebrew not individually verified** ([model card](https://ai.google.dev/gemma/docs/core/model_card_4)) | No evidence | Native function calling / structured tool use; doc-layout benchmarks not verified | Engine-level; AI-Studio hosted route advertises `response_format` | Unsloth guide exists ([docs](https://unsloth.ai/docs/models/gemma-4/train)) | Newest family — least field history; Hebrew OCR quality entirely unmeasured; per-size hardware numbers still to confirm |
| 5 | Qwen3.6-27B / Qwen3.5-27B | Yes; 262K ctx; text+image+video | Higher aggregate OCR scores than Qwen3-VL (OCRBench 93.1 / OmniDocBench 90.8 for Qwen3.5-397B) but **no per-language table → Hebrew unverified** ([Qwen3.5 blog](https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894), search-corroborated) | No evidence | Strong on OmniDocBench (family claim) | Engine-level; JSON mode GA on Groq | llama.cpp/unsloth ecosystem | Hebrew regression vs Qwen3-VL is possible and unmeasured — must bake off |
| 6 | Gemma 3 27B | Yes, interleaved multi-image, 128K ctx, pan-and-scan ([HF blog](https://huggingface.co/blog/gemma3)) | "140+ languages"; Hebrew-capable per community [Hebrew LLM leaderboard](https://huggingface.co/spaces/hebrew-llm-leaderboard/chat-leaderboard) (text tasks); Gemini-derived 262K tokenizer good for Hebrew | No benchmark; generic HTR reports say usable but degrades on irregular cursive ([guide](https://www.arsturn.com/blog/gemma-3-handwritten-text-recognition-guide)) | Good doc OCR (Roboflow eval [6/7 vision tasks](https://blog.roboflow.com/gemma-3/)) | Engine-level | Official QLoRA docs + Unsloth | Fixed 896×896 tiles lose small handwriting detail (pre-crop per question); 8K output cap; ToU automated-decisions clause; superseded by Gemma 4 |
| 7 | Mistral Small 3.2 | vLLM caps 10 images/prompt; 128K ctx | **Hebrew absent from the 25-language list** ([3.1 card](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)) | No evidence | **DocVQA 94.86, ChartQA 87.4** — best documented doc-VQA of the self-hostable set ([3.2 card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)) | Engine-level + Mistral API [custom structured outputs](https://docs.mistral.ai/capabilities/structured-output/custom_structured_output) | Standard | Hebrew undocumented; hosted endpoint retiring 2026-07-31 |
| 8 | Ministral 3 14B | Multi-image not specified; 256K ctx | "Dozens of languages" incl. Arabic — **Hebrew not listed** | No evidence | Not benchmarked publicly for doc layout | Engine-level + Mistral API structured outputs | Standard | Hebrew undocumented; llama.cpp mmproj unverified; smallest LM of the serious set |
| 9 | InternVL3.5 | Native multi-image API (`num_patches_list`) | **No Hebrew claim anywhere**; MTVQA covers 9 languages, none Hebrew ([MTVQA](https://github.com/bytedance/MTVQA)) | No evidence | Good OCRBench/DocVQA; OCRBench-v2 analysis: struggles with overlapping/rotated/low-frequency text — 46.7% vs 79.1% ([arXiv:2501.00321](https://arxiv.org/html/2501.00321v2)) — directly relevant to messy exam pages | Engine-level | Official docs + ms-swift; official 3.5 LoRA scripts still requested ([issue #1145](https://github.com/OpenGVLab/InternVL/issues/1145)) | Hebrew unknown; community-only GGUF; low-frequency-text weakness |
| 10 | MiniCPM-V 4.5 | Multi-image chat supported | ">30 languages" — enumerations are European+CJK, **Hebrew never listed**; anti-hallucination OCR training ([paper](https://arxiv.org/html/2509.18154v1)) | No evidence | OCRBench-leading class; **OmniDocBench SOTA for PDF parsing among general MLLMs**; 1.8 MP any-aspect input | Engine-level | LLaMA-Factory (official) + SWIFT | Hebrew unknown; historical repetition loops on dense docs (V-2.6 era); no LMDeploy; stale license text in GGUF repo |

**Backbone note:** InternVL3.5 and MiniCPM-V 4.5 both use Qwen3 LLM backbones, which cover Hebrew as a text language ([Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)) — so semantic judging of a Hebrew transcript is plausible for all candidates; the risk is concentrated in *visual* Hebrew recognition, RTL layout, and handwriting.

### Disqualified (with reasons)

| Model | Reason | Source |
|---|---|---|
| Llama 3.2 11B/90B Vision | **"For image+text applications, English is the only language supported"** (model-card verbatim); single image per prompt; vLLM dropped mllama after 0.9.1 | https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct ; https://github.com/vllm-project/vllm/issues/8826 |
| Llama 4 Scout/Maverick | Hebrew absent from the 12-language supported list; tested to only 5 images; Scout Q4 (67 GB) exceeds the laptop; EU multimodal license exclusion on the whole family | https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct ; https://ollama.com/library/llama4 |
| Qwen2.5-VL (all sizes) | Only 10 non-EN/ZH OCR languages, Hebrew not documented; 3B is research-only license; 72B non-OSI | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE |
| Pixtral Large 2411 | Mistral Research License (not clean for operational grading); retired from API 2026-02-27 | https://docs.mistral.ai/models/overview |
| Mistral OCR 4 | API-only, weights not open ($4/1k pages) — though notably it **explicitly lists Hebrew among 170 languages** and OCR 3 claimed 88.9% handwriting accuracy; a useful paid benchmark reference, not an open solution | https://mistral.ai/news/ocr-4/ ; https://mistral.ai/news/mistral-ocr-3/ |
| Qwen3-Omni-30B-A3B | Apache-2.0 but audio-focused, weak GGUF/Ollama ecosystem, no OCR-language table | https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct |

---

## 2. Free hosted APIs (deployment target a)

All checked **2026-07-12**. Workload sizing reference: ~50 exams × ~15 vision calls = ~750 calls, ~2.5–4.5M tokens.

| Provider | Model(s) | Quota / rate limits | Image support | Context | Retention / training policy | Payment method | Self-hosted equivalent | Sources |
|---|---|---|---|---|---|---|---|---|
| **Groq** ★ | `qwen/qwen3.6-27b` (**GA**); `llama-4-scout` (preview) | Free: 30 RPM / 1K RPD / **8K TPM / 200K TPD** (≈3–4 exams/day; full batch ~2–3 weeks free, or cheap Developer tier) | URL ≤20 MB or base64 ≤4 MB; ≤5 images/request; JSON mode + tools | 131K | **Cleanest:** no training on inputs/outputs, no default retention, ZDR toggle | None for free tier | Qwen3.6-27B (Apache-2.0, GGUF exists) | https://console.groq.com/docs/vision ; /docs/rate-limits ; /docs/your-data |
| **Cerebras** ★ | `gemma-4-31b` (**preview**, vision); `gpt-oss-120b` (prod, text) | 5 RPM / 30K input TPM / **1M tokens/day** (full batch ~3–5 days free) | base64 PNG/JPEG data-URI only (no URLs) | 65K | No retention of prompts/responses; no training on customer data | None | Gemma 4 31B (Apache-2.0) | https://inference-docs.cerebras.ai/models/overview ; https://support.cerebras.net/articles/1811589793-does-cerebras-retain-my-data |
| **OpenRouter** | 5 free VLMs: `google/gemma-4-31b-it:free`, `gemma-4-26b-a4b-it:free`, `nvidia/nemotron-nano-12b-v2-vl:free`, `nemotron-3-nano-omni-30b-a3b:free`, (+safety classifier) | 20 RPM; 50 req/day (<$10 lifetime credits) or **1,000 req/day after one-time $10 purchase** | Yes (varies by route); AI-Studio route has `response_format` | 128–262K | **Worst privacy of the viable set:** free routes either train+retain (OpenInference, NVIDIA) or pass through Google's unpaid tier (human review, "do not submit… personal information" — https://ai.google.dev/gemini-api/terms); live uptime 71–99% | None ($10 unlock optional) | Same Apache-2.0 weights | https://openrouter.ai/docs/api/reference/limits ; live `GET /api/v1/models` |
| **ModelScope API-Inference** | Qwen3-VL models incl. 235B-A22B — **the only free host of the recommended primary family** | 2,000 calls/day (500/model/day, resets 00:00 UTC+8) | Yes | Model-dependent | Alibaba account + **real-name verification** required; retention terms not independently verified — treat as dev-only | None | Qwen3-VL (identical weights) | https://modelscope.ai/docs/model-service/API-Inference/limits (search-corroborated — re-verify at signup) |
| **Alibaba Model Studio** | Qwen3-VL family, qwen-vl-ocr | New-user ~1M tokens/model, 90 days, then paid — trial, not a free tier | Yes | Large | Standard cloud terms | Account | Qwen3-VL | https://www.alibabacloud.com/help/en/model-studio/new-free-quota |
| **InternVL/InternLM official** | `internvl3.5-latest`, `internvl3.5-241b-a28b` | ~10 RPM free, keys 6 months (portal is JS-rendered — re-verify in browser) | Yes (OpenAI-compatible) | — | China-hosted, no DPA — **dev-only, no real exams** | None | InternVL3.5 (Apache-2.0) | https://internlm.intern-ai.org.cn/api/document ; https://internvl.readthedocs.io/en/latest/get_started/internvl_chat_api.html |
| **OpenBMB MiniCPM API** | `MiniCPM-V-4.6-Instruct` / `-Thinking` (1.3B only) | Publicly posted shared trial key — zero QoS/privacy guarantees | Yes | — | Unspecified — **dev-only** | None | MiniCPM-V 4.6 | https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/api.md |
| **Mistral La Plateforme (Experiment)** | Ministral 3 14B, Mistral Small 4, Large 3, Medium 3.5 (vision via generalists; OCR endpoint inclusion unconfirmed) | ~1 RPS / 500K TPM / ~1B tokens/month (third-party trackers; console shows exact) | Yes | 256K | **Free-subscription data trainable by default unless opted out** (Commercial ToS eff. 2026-05-28: https://legal.mistral.ai/terms/commercial-terms-of-service); opt-out toggle: https://help.mistral.ai/en/articles/455207 | Phone verification, no card | Ministral 3 / Small 4 / Large 3 (all Apache-2.0) | https://docs.mistral.ai/admin/user-management-finops/tier |
| **NVIDIA NIM trial** | Nemotron Nano 12B v2 VL, Nemotron OCR v2, Gemma-4-26B, Qwen3.5/3.6 | ~1,000 trial credits (≈1 credit/request), ~40 RPM | Yes | — | **ToS: trains on user content AND contractually bars personal data** — verbatim clauses 3.3 & 4.3 in https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf — anonymized samples only | None | NIM containers on university GPU | https://build.nvidia.com |
| **Google AI Studio** | `gemma-3-27b-it` free (Gemma 4 also served) | Undocumented; 429s reported at ~20K-token inputs | Yes | 128K | Unpaid tier: human review + product-improvement use; "do not submit… personal information" | None | Gemma 3/4 weights | https://ai.google.dev/gemini-api/terms ; https://discuss.ai.google.dev/t/gemma-3-27b-rate-limits/73700 |
| SambaNova | `gemma-4-31B-it` (preview) | **20 req/DAY** free → unusable (37+ days/batch) | Yes | 128K | — | None | Gemma 4 | https://docs.sambanova.ai/docs/en/models/rate-limits |
| HF Inference Providers | Qwen3.6-27B, gemma-4-31b-it et al. via partners | **$0.10/month free credits** — not a practical free tier | Yes | — | Pass-through to provider | Card for real use | Same weights | https://huggingface.co/docs/inference-providers/pricing |
| Together AI | — | **No free tier** (2024 Llama-Vision-Free promo over) | — | — | — | — | — | https://www.together.ai/pricing |

**Cross-cutting free-tier risks:** every provider checked changed limits or lineup within the last ~90 days; OpenRouter's own blog (2026-06-15) concedes free tiers can tighten or die "without warning" (https://openrouter.ai/blog/tutorials/free-llm-apis-compared/). **Architect against an OpenAI-compatible client with provider-agnostic config**, so Groq/Cerebras/ModelScope/self-hosted are a config change. Note the awkward reality: the two privacy-clean free hosts (Groq, Cerebras) serve **Qwen3.6-27B and Gemma 4** — not the primary-recommended Qwen3-VL — which strengthens the case for including both in the bake-off.

---

## 3. OCR-route assessment: NOT viable as primary architecture

Verdict from a dedicated component-by-component investigation (all checked 2026-07-12): **no open OCR component can transcribe modern Hebrew cursive exam handwriting, and OCR inherently cannot do the mark/ink/convention subtasks.**

| Component | License | Printed Hebrew | Handwritten Hebrew | Verdict |
|---|---|---|---|---|
| Tesseract 5.x | Apache-2.0 | ~92–96% on clean print (practitioner reports) | FAQ: handwriting "won't work very well" (https://tesseract-ocr.github.io/tessdoc/FAQ.html) | Optional auxiliary for printed question text only |
| Kraken / eScriptorium | Apache-2.0 / MIT | Via models | Only **medieval square-script** models (BiblIA, Sofer Mahir) — useless for modern cursive; training your own = multi-month data-labeling project (https://kraken.re/6.0.0/advanced/repo.html) | Only license-clean DIY fallback path; not an integration |
| Surya 2 (Datalab) | Code Apache-2.0; **weights OpenRAIL-M restricting >$5M-revenue orgs to "research purposes" + share-alike on outputs** (https://github.com/datalab-to/surya) | 90.9% Hebrew pass rate (internal 91-lang benchmark) — strongest open printed-Hebrew OCR | No Hebrew handwriting evidence | License-unclean for institutional grading; also: it is itself a VLM now |
| PaddleOCR / PaddleOCR-VL | Apache-2.0 | **No Hebrew, period** (109-language appendix excludes it — https://ar5iv.labs.arxiv.org/html/2510.14528) | — | Eliminated |
| EasyOCR / docTR | Apache-2.0 | No Hebrew models | — | Eliminated |
| HF TrOCR-Hebrew fine-tunes | Mostly unlicensed experiments | — | Author of the main effort states it failed for lack of data ("30K lines, while Microsoft used… over 600M" — https://huggingface.co/spaces/sivan22/TrOCR-handwritten-hebrew/discussions/1) | Dead end |

**Structural argument (independent of Hebrew):** OCR outputs text lines + boxes. Circles, X marks, filled bubbles, cross-outs, overwrites, blue-vs-red ink, and document-level convention inference are not text. HSV-threshold ink separation via OpenCV is possible but brittle (scanner color shift, overlapping strokes). Even the OCR field has moved to VLMs — Surya 2 is a VLM; the only active Hebrew-HTR work on HF consists of Qwen3-VL / GLM-4.6V fine-tunes (`kohelet-splendour/*`). Even commercial engines don't support handwritten Hebrew (Microsoft Document Intelligence: https://learn.microsoft.com/en-us/answers/questions/1639127/hebrew-handwritten-text-optimal-solution-for-recog).

**Retained role for OCR:** an optional Tesseract pass over printed question text as anchor/index input to the VLM prompt. Nothing more.

---

## 4. Serving stacks and structured output (verified 2026-07-12)

### 4.1 vLLM — university GPU server (recommended)
- Structured outputs via **xgrammar/guidance** backends; use `response_format: {"type":"json_schema",…}` or `structured_outputs` in `extra_body` — old `guided_json` fields **removed in v0.12.0** (https://docs.vllm.ai/en/latest/features/structured_outputs.html).
- Multi-image per request on the OpenAI endpoint (`image_url`, base64 or URL); cap with `--limit-mm-per-prompt.image N` sized to max exam page count (https://docs.vllm.ai/en/latest/features/multimodal_inputs.html).
- Constrained decoding operates on output logits — orthogonal to vision inputs; routinely used with Qwen-VL. Gotcha is schema-feature support, not vision: avoid `minItems`/`maxItems`/`pattern`/length constraints (historic xgrammar gaps, e.g. https://github.com/vllm-project/vllm/issues/16880, open 2026 issue #45592).
- Linux-only — server target, not the laptop.

### 4.2 llama.cpp (llama-server) — best scriptable Windows/CPU option
- Vision via mtmd: `-m model.gguf --mmproj mmproj.gguf` or `-hf <repo>`; supported VLMs include Qwen3-VL (dense+MoE), Gemma 3/4, InternVL 2.5/3, Pixtral, Mistral Small 3.1, MiniCPM-V (https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).
- JSON: GBNF grammars, `--json-schema`, OpenAI-compatible `response_format` — note nonstandard shape (`schema` key directly, not nested `json_schema.schema`) (https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
- Official Windows binaries for CPU, **Vulkan** (Intel iGPU path), SYCL, OpenVINO — release b9977, 2026-07-12 (https://github.com/ggml-org/llama.cpp/releases). Community reports of Vulkan buffer failures with some VLMs on iGPUs — validate, fall back to CPU build (63 GB RAM is ample).
- CPU expectations: Q4 7–8B decodes ~5–12 tok/s; the real cost is **image prefill** (tens of seconds to minutes per scanned page) — downscale scans (long side ~1000–1300 px) and/or pre-crop per question.

### 4.3 Ollama — convenient, with engineering traps
- `format` accepts a full JSON schema, explicitly works with vision models; temperature 0 recommended (https://ollama.com/blog/structured-outputs ; https://docs.ollama.com/capabilities/structured-outputs).
- **Critical trap:** the OpenAI-compatible endpoint cannot set context size; default ~4K and **overflow is silently truncated from the start — no error**. Multipage scans will blow past 4K and the model will "grade" with pages silently dropped. Bake `num_ctx` into a Modelfile or set `OLLAMA_CONTEXT_LENGTH` (https://docs.ollama.com/context-length).
- OpenAI endpoint: base64 images only; no `tool_choice`/`logprobs` (https://docs.ollama.com/openai).
- Intel-iGPU Vulkan support is soft; known Windows hybrid-graphics device-selection bug (https://github.com/ollama/ollama/issues/16667).

### 4.4 TGI and LM Studio
- **TGI: do not adopt** — maintenance mode since ~Dec 2025; HF recommends vLLM/SGLang/llama.cpp instead (https://github.com/huggingface/text-generation-inference).
- **LM Studio:** free for work use since 2025-07-08 (https://lmstudio.ai/blog/free-for-work); OpenAI-compatible server with `response_format.json_schema` (GGUF → llama.cpp grammar; MLX → Outlines) (https://lmstudio.ai/docs/developer/openai-compat/structured-output). Proprietary freeware app over open engines — fine for dev, arguably not for the production server. Reported more stable than Ollama on Intel/Vulkan Windows today.

### 4.5 JSON-reliability pattern (applies to every model above)
1. **Constrained decoding as transport guarantee** (vLLM json_schema / Ollama format / GBNF / LM Studio) — guarantees syntax, not semantics.
2. **Schema design: reasoning before verdicts** — put free-text `transcription`/`reasoning` fields *before* `answer`/`score` in property order; strict JSON constraints measurably tax reasoning ("Let Me Speak Freely?", EMNLP 2024: https://aclanthology.org/2024.emnlp-industry.91.pdf ; format-tax follow-ups https://arxiv.org/pdf/2604.03616, CRANE https://arxiv.org/html/2502.09061v3).
3. **Also state the schema in the prompt**; temperature ~0.
4. **Keep schemas simple** — types/enums/required only; avoid minItems/pattern/length (backend feature gaps).
5. **Validate-and-retry layer** — Pydantic validation against the exam's known question list, one re-ask on failure.
6. Watch for repetition/whitespace loops under grammar pressure → repetition/frequency penalty.
7. Prefer a **two-call pipeline** (transcribe/observe first, judge second) over one mega-call — mitigates both the format tax and the observed long-extraction repetition failure mode.

---

## 5. Recommendation

### 5.1 Architecture: single VLM (not OCR+LLM, not hybrid-OCR)
One VLM sees the page and does transcription, mark detection, ink reasoning, and judging — because (a) no open OCR handles handwritten Hebrew at all (§3), and (b) the grade-deciding subtasks (marks, ink, conventions) are vision-reasoning tasks outside OCR's scope. Pipeline shape: **per-page or per-question two-call flow** — call 1 extracts observations (transcription, detected marks, ink attribution) into a loose schema; call 2 judges against the reference solution into a strict schema. An optional Tesseract pass over printed question text may be added later as anchor text; it is not load-bearing.

### 5.2 Model choice
- **Primary: Qwen3-VL-32B-Instruct** (Apache-2.0). It is the only open model with quantified Hebrew OCR evidence (~72% printed, tech report Fig. 2 — https://arxiv.org/abs/2511.21631), near-frontier document benchmarks, official FP8/GGUF, and first-class vLLM + llama.cpp + Ollama support. Be clear-eyed: 72% printed-Hebrew is the *floor of acceptability*, Hebrew is its weakest supported language, and handwriting is unmeasured.
- **Dev model (laptop): Qwen3-VL-8B-Instruct** official GGUF Q4_K_M (5.03 GB + mmproj) via llama-server or Ollama; step up to Qwen3-VL-30B-A3B Q4 (20 GB, 3B active) when quality matters more than page-encode latency.
- **Fallbacks / mandatory bake-off entrants:** **Gemma 4 (31B or 26B-A4B)** — Apache-2.0, variable-resolution input (better for A4 scans), best free-hosted availability, Hebrew unverified; **Qwen3.6-27B** — Apache-2.0, newer architecture, GA on Groq, Hebrew unverified. If either beats Qwen3-VL-32B on the sample exams, switch — the serving stack is model-agnostic.
- Second-tier fallbacks if all above disappoint: MiniCPM-V 4.5 (best PDF-parsing benchmarks per size), Mistral Small 3.2 (best DocVQA number), InternVL3.5 — all Hebrew-unverified.
- **Escape hatch if open models fail on handwriting:** paid Mistral OCR 4 ($4/1k pages, Hebrew explicitly supported) as transcription layer + open LLM as judge — abandons the fully-open requirement, so only after the bake-off proves necessity.

### 5.3 Deployment recommendation
- **(b) University Linux GPU server (production):** vLLM ≥0.12 + Qwen3-VL-32B-Instruct — FP8 on a 48–80 GB GPU, or AWQ/Q4 on a 24–32 GB card; `response_format json_schema` (xgrammar), `--limit-mm-per-prompt.image` sized to page count. Self-hosting also sidesteps every free-tier privacy problem: **real student exams are personal data and should be graded on university hardware.**
- **(c) Windows laptop (dev):** llama-server (CPU or Vulkan build) or LM Studio with Qwen3-VL-8B Q4 GGUF; Ollama acceptable if `num_ctx` is baked into a Modelfile. Downscale/pre-crop scans; expect tens of seconds per page.
- **(a) Free hosted (dev + spot-grading of the anonymized sample only):** Groq `qwen/qwen3.6-27b` (clean data policy) as the default; Cerebras `gemma-4-31b` for free volume; ModelScope for free access to actual Qwen3-VL (dev-only pending retention verification). OpenRouter with the $10 unlock as a multiplexer for prompt development only. Build everything against an OpenAI-compatible client with a provider config switch.

### 5.4 What to validate experimentally (in priority order)
The repo already has ground truth: sample exam version A1, swapped-tables layout, X marking convention, human scores **24/32 and 28/32** (`sample_data/student_exam.pdf`, `sample_data/Exam_solution.pdf`). No public benchmark covers any of items 1–4 — this bake-off is the actual decision procedure.

1. **Handwritten-Hebrew transcription accuracy** (biggest risk, zero public evidence for any candidate) — char/word error rate on the sample's handwritten explanations, per model.
2. **MCQ mark detection**: circles, X marks, cross-outs, overwrites — per-question detection accuracy, including the document-level "X means selected" convention inference.
3. **Blue/red ink separation** — does the model correctly attribute student vs. instructor marks? Test against scanner color shift.
4. **End-to-end score agreement** — does each candidate reproduce 24/32 and 28/32? Report per-question deltas, not just totals.
5. **JSON schema compliance under constrained decoding** — failure/retry rate per engine (vLLM vs llama.cpp vs Ollama), with the reasoning-before-verdict schema.
6. **Resolution/cropping ablation** — full-page vs per-question crops; downscaling threshold at which handwriting degrades (critical for Gemma 3's 896×896 tiles and for CPU prefill cost).
7. **Repetition-loop incidence** on long extractions (observed for Qwen3-VL: https://www.analyticsvidhya.com/blog/2025/11/deepseek-ocr-vs-qwen-3-vl-vs-mistral-ocr/) and mitigation (per-page prompting, repetition penalty).
8. **Model bake-off matrix:** Qwen3-VL-8B (local) and Qwen3-VL-32B, Gemma 4 31B/26B-A4B, Qwen3.6-27B (hosted/GPU) on items 1–5.
9. If zero-shot falls short: **LoRA fine-tuning feasibility** on annotated exam crops (Unsloth/LLaMA-Factory/ms-swift paths exist for all primary candidates; HebHTR-style data: https://github.com/Lotemn102/HebHTR).

### 5.5 Honest uncertainty register
- **Hebrew handwriting: no model, open or otherwise, has a published number.** The single strongest datapoint anywhere is Mistral OCR 3's *self-reported* 88.9% generic handwriting accuracy — closed weights, and not Hebrew-specific. Everything hinges on the in-house bake-off.
- Qwen3-VL's ~72% Hebrew figure is **printed** text on the vendor's own benchmark; treat as an optimistic upper anchor for print, uninformative for cursive.
- Gemma 4 and Qwen3.5/3.6 have **no per-language OCR tables at all** — their higher aggregate OCR scores may or may not transfer to Hebrew.
- Ink-color separation and marking-convention inference have **zero published evidence for any model** — genuinely novel territory.
- Ministral 3's llama.cpp vision path (mmproj in the GGUF repo) is unverified; ModelScope free-tier retention terms and InternVL free-API limits were search-corroborated, not fully fetched — re-verify before depending on them.
- Free-tier quotas, lineups, and privacy flags churned materially in the 90 days before 2026-07-12 at every provider checked — recheck all Table 2 numbers before any batch run.
