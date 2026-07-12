# Open-Model Replacement for the Anthropic API — Gemma 3 vs. Llama Vision (verified live, 2026-07-12)

All claims below were verified against live sources on **2026-07-12** (access date applies to every URL). Verdict up front: **Gemma 3 27B is the only viable candidate of the requested set for Hebrew exam grading; every Llama vision model is disqualified on language grounds alone.** Also flagged: **Gemma 4 shipped under Apache 2.0** (see §3) and likely supersedes Gemma 3 for this project.

---

## 1. GOOGLE GEMMA 3 (4B / 12B / 27B multimodal)

### 1.1 Exact versions and sources
- Family: 270M, 1B, 4B, 12B, 27B; pre-trained (`-pt`) and instruction-tuned (`-it`); **only 4B/12B/27B are multimodal** (270M/1B text-only). Images normalized to 896×896, encoded to 256 tokens each (SigLIP encoder); 128K context (4B+), 8,192-token output cap; knowledge cutoff Aug 2024; released March 2025. Sources: official model card https://ai.google.dev/gemma/docs/core/model_card_3 ; HF https://huggingface.co/google/gemma-3-27b-it (requires transformers ≥4.50).
- Weights: Hugging Face (`google/gemma-3-{4b,12b,27b}-{pt,it}`), Kaggle, plus official **QAT** int4/Q4_0 GGUF variants.

### 1.2 License — Gemma Terms of Use (IN DETAIL)
- Terms: https://ai.google.dev/gemma/terms — **last updated April 1, 2026**. Grants use, reproduction, modification, distribution for any purpose subject to (a) the Gemma Prohibited Use Policy (incorporated by reference) and (b) applicable law. **Commercial and academic use are not restricted per se.** Google claims no rights in outputs: "Google claims no rights in Outputs you generate using Gemma." Distribution/derivatives must carry the use restrictions downstream, include a copy of the terms, mark modified files, and ship the notice "Gemma is provided under and subject to the Gemma Terms of Use…". Note: the terms page states they cover Gemma 1–3 and variants; **Gemma 4 is governed by separate licensing** (Apache 2.0 — §3).
- Prohibited Use Policy: https://ai.google.dev/gemma/prohibited_use_policy (page shows last updated Feb 21, 2024). Categories: rights violations; dangerous/illegal activity; unlicensed professional practice (legal/medical/accounting/financial); abuse/circumvention; harmful content; deception; sexually explicit content; and — the one relevant clause — **"Making automated decisions in domains that affect material or individual rights or well-being (e.g., finance, legal, employment, healthcare, housing, insurance, and social welfare)."**
- **Assessment for university grading:** education/grading is *not* in the enumerated high-risk list, and academic use is otherwise unrestricted, so grading with human instructor review (which this workflow has — red-ink instructor scores exist) is defensible; a **fully automated grade-assignment pipeline with no human in the loop** could arguably fall under "automated decisions affecting individual rights/well-being." Recommendation: keep an instructor-review step, or use Apache-2.0 Gemma 4 to remove the question entirely. (Not legal advice; the university counsel should sign off.)

### 1.3 Engines, quantized sizes, hardware
- **Ollama** (https://ollama.com/library/gemma3): 4B = 3.3 GB, 12B = 8.1 GB, 27B = 17 GB downloads; vision enabled for 4B/12B/27B; QAT variants for 1B/4B/12B/27B ("similar quality as BF16… 3x less memory").
- **QAT int4 VRAM** (official blog, Apr 18, 2025: https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/): 27B = 14.1 GB (vs 54 GB BF16), 12B = 6.6 GB, 4B = 2.6 GB. 27B-int4 runs on a single RTX 3090-class GPU. Formats: int4 + Q4_0 GGUF for Ollama/llama.cpp/LM Studio/MLX.
- **llama.cpp**: Gemma 3 4B/12B/27B are first-class supported vision models via libmtmd (`llama-server`, `--mmproj`): https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md.
- **vLLM**: Gemma 3 supported (and vLLM ships day-0 Gemma 4 support incl. Intel XPUs/AMD/TPU: https://vllm.ai/blog/2026-04-02-gemma4); supported-models list: https://docs.vllm.ai/en/latest/models/supported_models/. vLLM is **Linux-only** — fine for the university server, not the Windows laptop.
- **Windows / CPU-only laptop (63 GB RAM, Intel iGPU)**: Ollama has native Windows builds; llama.cpp builds on Windows with MSVC/clang and supports **Intel iGPU via SYCL** ("Data Center Max, Flex, Arc, Built-in GPU and iGPU") and **Vulkan** (https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md). Practical dev setup: Gemma 3 12B-QAT (~8 GB) at usable CPU speed; 27B-QAT (~17 GB) fits RAM easily but will be slow on CPU (~1–3 tok/s class) — fine for single-exam debugging, not batch grading.

### 1.4 Multi-image support (critical for multipage exams)
Confirmed: Gemma 3 supports **interleaved multiple images per prompt** (each image = 256 tokens after encoding; "pan & scan" adaptive cropping for non-square/high-res pages — important for A4 exam scans). Source: https://huggingface.co/blog/gemma3. With 128K context, a full multipage exam fits in one prompt (e.g., 10 pages ≈ 2,560 image tokens + text). This directly supports the document-level convention detection requirement (marking conventions on page 1 applied to later pages).

### 1.5 Multilingual / Hebrew evidence
- Official: "over 140 languages" for 4B+ models (model card, above). Hebrew is not individually enumerated by Google.
- Tokenizer: same 262K-entry SentencePiece tokenizer as Gemini 2.0, rebalanced for non-English (https://huggingface.co/blog/gemma3) — materially better Hebrew tokenization than Llama's tokenizer.
- Hebrew text ability is independently measurable: the community **Hebrew LLM chat leaderboard** (https://huggingface.co/spaces/hebrew-llm-leaderboard/chat-leaderboard) scores `gemma3-27b-it` across Hebrew summarization/translation/Winogrande/trivia/nikud tasks — Gemma 3 is a standard Hebrew-capable baseline there; the DictaLM 3.0 paper (https://arxiv.org/abs/2602.02104) confirms Hebrew-sovereign models are still built by adapting open bases, i.e., stock open models trail Hebrew-tuned ones.
- **Hebrew HANDWRITING OCR: no public benchmark exists for any of these models.** Generic evidence that Gemma 3 is strong at printed OCR and usable for handwriting: Roboflow eval (passed 6/7 vision tasks incl. document OCR: https://blog.roboflow.com/gemma-3/); HTR setup guides note it beats Tesseract-class pipelines on messy handwriting but struggles with irregular cursive/ink noise (https://www.arsturn.com/blog/gemma-3-handwritten-text-recognition-guide). **You must validate on your own ground-truth exams** (the repo already has the A1 sample with human scores 24/32 and 28/32 — run that as the acceptance test).
- Blue/red ink separation and MCQ mark semantics (circles, X per document convention, cross-outs, overwrites): zero published evidence for any open model — strictly an empirical question for your sample set; expect this to be the hardest capability gap vs. Claude.

### 1.6 Structured JSON reliability
Engine-enforced, model-independent — this de-risks JSON regardless of model choice:
- Ollama `format` = JSON schema, works with any model incl. vision, temperature 0 recommended (https://ollama.com/blog/structured-outputs, Dec 6, 2024).
- vLLM `structured_outputs` (xgrammar/guidance backends; `guided_json` deprecated in v0.12) via OpenAI-compatible API (https://docs.vllm.ai/en/latest/features/structured_outputs.html).
- llama.cpp: GBNF grammars / `json_schema` on llama-server.
Caveat: constrained decoding guarantees *syntactic* schema validity, not semantic correctness of grades.

### 1.7 Fine-tuning
- Official vision QLoRA fine-tuning docs exist (ai.google.dev Gemma docs), and **Unsloth FastVisionModel supports Gemma 3 vision QLoRA** (rank-32 LoRA on all linear layers; 4B trains on free-tier Colab GPUs): https://unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune. Realistic path: fine-tune 4B/12B on a few hundred annotated exam crops (checkbox states, ink colors) if zero-shot is insufficient.

### 1.8 Known weaknesses
Not a knowledge base; struggles with nuance/sarcasm (model card); handwriting HTR degrades on irregular cursive and noisy scans; fixed 896×896 tiles mean small handwriting on full-page scans loses detail (mitigate: pan & scan, or pre-crop per question — your existing pipeline's page-region cropping matters more than the model); 8K output cap (fine for per-question JSON); community-reported 429s even on paid tier for the hosted Gemma endpoint (below).

### 1.9 Free hosted API (deployment target a)
- **Google AI Studio / Gemini API** hosts `gemma-3-27b-it` **free of charge — Google staff: "Gemma is free of cost, even if you use paid tier"**, but limits are undocumented and 429 RESOURCE_EXHAUSTED at ~20K-token inputs is reported (https://discuss.ai.google.dev/t/gemma-3-27b-rate-limits/73700, resolution Apr 4, 2025; rate-limit page defers to the AI Studio dashboard: https://ai.google.dev/gemini-api/docs/rate-limits). Multi-image Hebrew exams at 20K+ tokens may hit this — test early.
- **OpenRouter** `google/gemma-3-27b-it:free` — active, 131K context, image input (https://openrouter.ai/google/gemma-3-27b-it:free); free-variant limits: **20 req/min; 50 req/day (1,000 req/day after a one-time $10 credit purchase)** (https://openrouter.ai/docs/api-reference/limits). 1,000/day covers a realistic grading batch.
- NVIDIA NIM also exposes hosted `gemma-3-27b-it` endpoints (https://build.nvidia.com/google/gemma-3-27b-it/modelcard).
- Privacy note: hosted free tiers generally reserve the right to use inputs; student exam scans are personal data — prefer self-hosting for production grading.

---

## 2. META LLAMA VISION MODELS

### 2.1 Llama 3.2 11B/90B Vision — **disqualified**
- Released Sept 25, 2024; cross-attention vision adapter on a frozen Llama 3.1 backbone. Model card: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct.
- **Fatal for this project — verbatim from the model card:** "For text only tasks, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported… **Note for image+text applications, English is the only language supported.**" Hebrew is unsupported even for text-only.
- **Single-image limit:** officially one image per prompt; Meta staff say multi-image "doesn't work well" (HF discussions https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/43 and …/Llama-3.2-11B-Vision/discussions/45; vLLM bug "focuses only on first image": https://github.com/vllm-project/vllm/issues/10983). Multipage exams would need page-by-page prompting with no cross-page convention context.
- License: Llama 3.2 Community License, Sept 25, 2024 (https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/LICENSE): free below **700M MAU**, "Built with Llama" attribution, model names must start with "Llama", AUP incorporated by reference. **EU restriction (in the official llama.com/HF gated license text, absent from the GitHub mirror):** "With respect to any multimodal models included in Llama 3.2, the rights granted under Section 1(a)… are not being granted to you if you are an individual domiciled in, or a company with a principal place of business in, the European Union" — end users of products incorporating them are exempt (analysis with verbatim quotes: https://www.zansara.dev/posts/2025-05-16-llama-eu-ban/). Irrelevant if the university is in Israel; fatal if any EU campus deploys the weights.
- Engines: **Ollama** `llama3.2-vision` 11B = 7.8 GB, 90B = 55 GB (https://ollama.com/library/llama3.2-vision — custom mllama support); **mainline llama.cpp does NOT support it** (not in https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md); **vLLM dropped it — versions after 0.9.1 cannot run encoder-decoder mllama** (https://github.com/vllm-project/vllm/issues/8826 and V1 blog https://vllm.ai/blog/2025-01-27-v1-alpha-release). Fine-tuning: Unsloth supports it (https://unsloth.ai/blog/llama3-2) — but you cannot fine-tune in Hebrew what was never trained for Hebrew vision.

### 2.2 Llama 4 Scout / Maverick — **disqualified for Hebrew; impractical locally**
- Released Apr 5, 2025. Scout: MoE, 17B active / 109B total, 16 experts, 10M context; Maverick: 17B active / ~400B total, 128 experts. Natively multimodal (early-fusion). Model card: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct.
- **Languages (verbatim list): "Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese" — Hebrew is NOT supported.**
- **Image limit:** "tested for image understanding up to 5 input images" — beyond that you own the risk (model card). Worse than Gemma for multipage exams.
- License: Llama 4 Community License, effective Apr 5, 2025 (https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama4/LICENSE): same 700M-MAU gate, "Built with Llama", AUP (https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama4/USE_POLICY.md — bans unlicensed professional practice and inferring sensitive personal data; nothing specific to education/grading). **EU clause (official license on llama.com, verbatim):** "With respect to any multimodal models included in Llama 4, the rights granted under Section 1(a)… are not being granted to you if you are an individual domiciled in, or a company with a principal place of business in, the European Union" — and since **all Llama 4 models are multimodal, the whole family is off-limits to EU-domiciled entities** (https://www.compliance-made-simple.ch/llama/, https://ioplus.nl/en/posts/european-union-excluded-from-llama-4-multimodal-models).
- Size/hardware: Ollama Scout = **67 GB**, Maverick = **245 GB** (https://ollama.com/library/llama4). Scout q4 does not fit the 63 GB laptop; Maverick needs multi-GPU H100-class. vLLM supports Llama 4 multimodal; llama.cpp supports **Scout vision** (multimodal.md above), Maverick impractical.
- Hosted: Groq serves Scout (`meta-llama/llama-4-scout-17b-16e-instruct`, 131K ctx, $0.11/$0.34 per 1M tokens — not free, Maverick not listed: https://console.groq.com/docs/models); OpenRouter has had `:free` Scout/Maverick variants under the same 50–1,000 req/day free limits.
- Reputation/weaknesses: LM Arena benchmark controversy — Meta submitted a non-public "Llama-4-Maverick-03-26-Experimental"; the released Maverick fell to ~32nd (https://www.techtimes.com/articles/309909/20250407/meta-faces-backlash-over-experimental-maverick-ai-version-used-benchmark-rankings-why.htm); no small dense variant; vision fine-tuning ecosystem immature relative to Gemma.

---

## 3. FLAG: GEMMA 4 (released by mid-2026) — likely the better answer
Verified on Google's official docs (accessed 2026-07-12): **Gemma 4** exists — sizes E2B/E4B (edge), 12B unified, 26B-A4B MoE (3.8B active), 31B dense; text+image with **variable aspect ratio/resolution** (better for A4 scans than Gemma 3's fixed 896×896), audio/video on smaller models, 128K–256K context, **native function calling / structured tool use**, and — decisive — the model card header says **license: Apache 2.0** (https://ai.google.dev/gemma/docs/core/model_card_4, updated 2026-06-26; overview: https://ai.google.dev/gemma/docs/core). Apache 2.0 eliminates every Gemma-ToU concern incl. the automated-decisions clause. Day-0 vLLM support incl. Intel XPU (https://vllm.ai/blog/2026-04-02-gemma4); llama.cpp already lists Gemma 4 vision GGUFs (multimodal.md above); Unsloth has a Gemma 4 fine-tuning guide (https://unsloth.ai/docs/models/gemma-4/train). "35+ languages out of the box, pretrained on 140+" — Hebrew still needs in-house validation. Strongly recommend evaluating Gemma 4 (12B/26B-A4B) head-to-head with Gemma 3 27B on your sample exam.

---

## 4. DECISION MATRIX vs. YOUR THREE DEPLOYMENT TARGETS

| Requirement | Gemma 3 27B/12B | Llama 3.2 11B/90B Vision | Llama 4 Scout/Maverick |
|---|---|---|---|
| Hebrew image+text | 140+ langs, Gemini tokenizer; unbenchmarked for handwriting — testable | **English-only for image+text — hard fail** | **Hebrew not supported — fail** |
| Multipage (multi-image) | Interleaved images, 128K ctx | 1 image/prompt | ≤5 images tested |
| (a) Free hosted API | AI Studio free + OpenRouter :free | rare | Groq paid; OpenRouter :free |
| (b) University Linux GPU | vLLM/Ollama; 27B-int4 on one 24 GB GPU | vLLM dropped it (>0.9.1) | vLLM OK but ≥H100-class |
| (c) Windows CPU laptop | Ollama/llama.cpp (SYCL for Intel iGPU); 12B-QAT ≈ 8 GB | Ollama only | Does not fit 63 GB |
| License for academic grading | OK w/ human-in-loop caveat | OK below 700M MAU; EU vision ban | Same + EU ban on whole family |
| Guaranteed JSON | Engine-level (Ollama/vLLM/llama.cpp) | Ollama only (of the three) | Ollama/vLLM/llama.cpp-Scout |
| Vision fine-tuning | Unsloth/official QLoRA docs | Unsloth (English-only vision) | Impractical size |

**Recommendation:** adopt **Gemma 3 27B-it (QAT int4)** now — vLLM or Ollama on the university GPU server, Gemma 3 12B-QAT via Ollama/llama.cpp on the Windows laptop, Google AI Studio or OpenRouter `:free` for the hosted tier — and immediately benchmark **Gemma 4 (Apache 2.0)** as the probable production choice. Drop both Llama families from consideration: Llama 3.2 Vision is English-only for images and single-image; Llama 4 omits Hebrew, caps at 5 images, and can't run on the laptop. Gate go/no-go on an in-house test against the existing ground-truth sample exam (A1 version, swapped tables, X-convention, human scores 24/32 & 28/32), since no public benchmark covers Hebrew handwriting, MCQ mark disambiguation, or blue/red ink separation for any open model.