# Open-Model Replacement for Anthropic API — InternVL vs MiniCPM-V (verified July 12, 2026)

All claims below were verified against live sources on 2026-07-12. URLs cited inline; anything I could not confirm from a primary source is explicitly flagged.

## Bottom line

- Both candidate families are now **license-clean for university use: Apache-2.0 weights** (this changed recently for MiniCPM — older repos still carry stale custom-license text, see Risks).
- **Neither InternVL nor MiniCPM-V has ANY documented Hebrew OCR evidence** — printed or handwritten. Their multilingual-OCR claims are real but Hebrew-free. The text backbones (Qwen3/Qwen3.5) do cover Hebrew, so semantic judging of transcribed Hebrew is plausible; visual recognition of Hebrew script — especially handwriting — is unproven and is the project's main technical risk.
- **Qwen3-VL (Apache-2.0) claims 32-language OCR and is the only open family with a reported Hebrew OCR claim** — I could not find Hebrew enumerated in the official list, so treat as unconfirmed, but it should be added as a third candidate and likely front-runner for the vision-OCR core.
- Recommended pair to prototype: **MiniCPM-V 4.5 (8B)** for CPU-laptop dev + easy Ollama/llama.cpp path, and **InternVL3.5-8B (or 38B on the university GPU server)** via vLLM/LMDeploy with structured-output decoding. Build a Hebrew gold-set eval (you already have human-scored sample exams) before committing.

---

## 1. InternVL family (OpenGVLab)

**Current versions (as of July 2026)**
- **InternVL3.5** — released 2025-08-26; sizes 1B, 2B, 4B, 8B, 14B, 30B-A3B, 38B, 241B-A28B (+ a GPT-OSS-20B-A4B variant). This is the current flagship *understanding* family. Repo: https://github.com/OpenGVLab/InternVL ; paper: https://arxiv.org/abs/2508.18265 ; blog: https://internvl.github.io/blog/2025-08-26-InternVL-3.5/
- **InternVL-U** — 2026-03-06, a 4B *unified understanding+generation* model (multi-image understanding added 2026-03-19). Not the right tool for grading; noted for completeness. https://github.com/OpenGVLab/InternVL-U
- InternVL3 (Apr 2025) and InternVL2.5 (Dec 2024) remain relevant because **llama.cpp officially supports them** (see engines).

**Licenses**
- Code: MIT (repo). Weights: **InternVL3.5 is Apache-2.0**. Model card quote: "This project is released under the apache-2.0 License. This project uses the pre-trained Qwen3 as a component, which is licensed under the apache-2.0 License." HF metadata tag `license:apache-2.0` confirmed via API. https://huggingface.co/OpenGVLab/InternVL3_5-8B and https://huggingface.co/api/models/OpenGVLab/InternVL3_5-8B (lastModified 2025-08-29). Fully academic/commercial-clean.
- (Older InternVL2.5 weights were MIT — also clean.)

**Architecture**: InternViT-300M vision encoder + Qwen3 LLM (8B = 0.3B + 8.2B), dynamic high-resolution tiling, native multi-image API (`num_patches_list`), thinking mode (requires transformers >= 4.52.1).

**Engines**
- **LMDeploy** (>= 0.9.1): InternVL3.5 supported on TurboMind, FP16/BF16 + KV int8/int4 (no W4A16/AWQ row for 3.5 yet). https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html
- **vLLM**: supported, "T + I+" (multi-image); dedicated recipe: https://docs.vllm.ai/projects/recipes/en/latest/InternVL/InternVL3_5.html and https://docs.vllm.ai/en/latest/models/supported_models.html
- **SGLang**: supported per model card.
- **llama.cpp**: official docs list **InternVL2.5 and InternVL3** with ggml-org pre-converted GGUFs (1B/2B/8B/14B incl. mmproj): https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md . For **InternVL3.5**, community GGUFs exist **with mmproj vision files** (bartowski: 20 quants, Q4_K_M 5.03GB, Q8_0 8.71GB, bf16 16.4GB, `mmproj-...-bf16/f16.gguf` present): https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-8B-GGUF — works because 3.5 keeps the InternViT+Qwen3 architecture, but it is community-converted, not ggml-org official; verify vision output quality locally.
- **Ollama**: no official InternVL entries in the Ollama library (checked search — none). Can side-load HF GGUFs.

**Hardware**: 8B bf16 ~17GB VRAM (fits one 24GB card); card guidance: models <= 30B fit a single A100, 38B needs 2x A100, 241B-A28B needs 8x A100. CPU: Q4_K_M 5GB GGUF is feasible on the 63GB-RAM laptop.

**OCR/document evidence**: OCRBench, ChartQA, DocVQA, MTVQA reported on the 3.5 card; multilingual demos in EN/ZH/JA/AR/IT. **MTVQA covers 9 languages (AR, DE, FR, IT, JA, KO, RU, TH, VI) — no Hebrew**: https://github.com/bytedance/MTVQA . No Hebrew claim anywhere in InternVL materials. OCRBench v2 analysis notes InternVL3 struggles with overlapping/rotated text and low-frequency text (46.7% on low-frequency vs 79.1% high-frequency): https://arxiv.org/html/2501.00321v2 — directly relevant to messy handwritten exam pages.

**Fine-tuning**: official finetune docs (full + LoRA; LoRA for 8B ~= 2x 32/40GB GPUs): https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html ; InternVL3.5 LoRA via **ms-swift** (explicitly lists InternVL3.5): https://github.com/modelscope/ms-swift ; also XTuner. Open issue asking for official 3.5 LoRA scripts: https://github.com/OpenGVLab/InternVL/issues/1145

**Free hosted API**: official InternVL/InternLM API at `https://chat.intern-ai.org.cn/api/v1` (OpenAI-compatible, image input supported), models include `internvl3.5-latest` and `internvl3.5-241b-a28b`; **free tier ~10 RPM, keys valid 6 months** per docs portal https://internlm.intern-ai.org.cn/api/document (portal is JS-rendered — WebFetch returned empty; rate-limit specifics came from search snippets of that page, re-verify in a browser) and the readthedocs pointer page ("Welcome to the free API"): https://internvl.readthedocs.io/en/latest/get_started/internvl_chat_api.html . OpenRouter also lists InternVL3 78B/14B/2B (paid, no free variant): https://openrouter.ai/opengvlab

## 2. MiniCPM-V family (OpenBMB)

**Current versions**
- **MiniCPM-V 4.5** — 2025-08-26, **8B** (SigLIP2-400M + Qwen3-8B). The quality flagship. https://huggingface.co/openbmb/MiniCPM-V-4_5 ; paper https://arxiv.org/abs/2509.18154
- **MiniCPM-V 4.6** — 2026-05-11, **1.3B** (SigLIP2-400M + Qwen3.5-0.8B), mixed 4x/16x token compression; + `-Thinking` variant. https://huggingface.co/openbmb/MiniCPM-V-4.6 (HF lastModified 2026-07-01, 975K downloads)
- MiniCPM-o 4.5 (2026-02-03, 9B omni/audio) — not needed here. Repo/news: https://github.com/OpenBMB/MiniCPM-V

**Licenses**
- **Weights: Apache-2.0** for both 4.5 and 4.6 — HF metadata `license:apache-2.0` confirmed via API for both: https://huggingface.co/api/models/openbmb/MiniCPM-V-4_5 , https://huggingface.co/api/models/openbmb/MiniCPM-V-4.6 . Card statement: "The MiniCPM-o/V model weights and code are open-sourced under the Apache-2.0 license."
- **Caveat**: the GGUF repo card (https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf) still contains stale text saying weights "must strictly follow MiniCPM Model License.md" (free academic, registration for commercial). The authoritative Apache-2.0 tag + main-card statement supersede it, but flag this to whoever signs off on licensing.

**Engines** (all verified): vLLM (>= 0.10.2 for 4.5), SGLang, **llama.cpp** (since b6282 for 4.5; official GGUFs *with* `mmproj-model-f16.gguf` for both 4.5 and 4.6 — confirmed in repo file list via HF API), **Ollama** (official library entries `minicpm-v4.5` and `minicpm-v4.6`, updated ~June 2026, Ollama >= 0.30; the old `minicpm-v` entry = 2.6 and is stale): https://ollama.com/search?q=minicpm , https://ollama.com/library/minicpm-v4.5 . **LMDeploy supports only up to MiniCPM-V 2.6 — no 4.5/4.6.**
- Deployment cookbook (vLLM/Ollama/llama.cpp recipes, Windows Ollama installer): https://github.com/OpenSQZ/MiniCPM-V-CookBook

**Quantized variants + sizes**
- 4.5 (8B): GGUF Q4_K_M **5.03GB**, Q8_0 8.71GB, F16 16.4GB (https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf); official int4 (~"9GB GPU" class), AWQ, BNB.
- 4.6 (1.3B): GGUF Q4_K_M **529MB**, Q8_0 812MB, F16 1.52GB + mmproj (https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf); AWQ/GPTQ/BNB variants exist.

**OCR/document evidence**: OCRBench leading ("surpasses GPT-4o-latest"), **OmniDocBench SOTA for PDF parsing among general MLLMs**, handles 1.8MP any-aspect-ratio input; "multilingual capabilities in more than 30 languages" — enumerated lists in prior versions are European+CJK; **Hebrew never listed**. Anti-hallucination OCR training (dynamic text-region corruption) and RLAIF-V trust tuning described in https://arxiv.org/html/2509.18154v1 . Multi-image chat supported (both 4.5 and 4.6).

**Fine-tuning**: LoRA via **LLaMA-Factory** (official support since 2025-08-26) and **SWIFT**; multi-image SFT supported; recipes in the CookBook.

**Free hosted API**: OpenBMB launched an official API 2026-05-17 at `https://api.modelbest.cn/v1` (OpenAI-compatible chat/completions, vision requests) with a **publicly posted free API key** for trial use; model IDs `MiniCPM-V-4.6-Instruct` / `MiniCPM-V-4.6-Thinking`: https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/api.md . Note it serves the 1.3B/thinking 4.6, not the 8B 4.5, and a shared public key gives zero privacy/QoS guarantees.

## 3. Fit to your three deployment targets

**(a) Free hosted API**: InternVL official free API (10 RPM, includes the 241B flagship — best free-tier quality) or OpenBMB's free 4.6 API. **Hard caveat: both are China-hosted free tiers with no data-processing agreement — do not send real student exams; use only for prompt development on the anonymized sample pair.** OpenRouter has InternVL3 cheap-but-paid; its current free vision models are other families (Gemma, Nemotron): https://openrouter.ai/collections/free-models

**(b) University Linux GPU server**: InternVL3.5-8B or -38B via vLLM or LMDeploy; MiniCPM-V 4.5 via vLLM >= 0.10.2. One 24GB GPU runs either 8B bf16; 48-80GB runs 38B or AWQ'd larger. Both fully offline-capable (weights freely downloadable, no gating).

**(c) Windows laptop, CPU-only, Intel iGPU, 63GB RAM**: MiniCPM-V 4.5 Q4_K_M (5GB) via Ollama-for-Windows or llama.cpp is the practical dev loop; MiniCPM-V 4.6 (0.5GB) for fast iteration. InternVL3.5-8B GGUF+mmproj also runnable (community conversion). Intel iGPU acceleration exists via llama.cpp SYCL backend (tested on 11th-gen+ iGPUs; needs oneAPI toolkit; vision/mtmd-on-SYCL not explicitly documented — treat as bonus, not plan-of-record): https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md , https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html

## 4. Structured JSON reliability

Do not rely on the model; **enforce schema at the decoder**: vLLM structured outputs (xgrammar/guidance via OpenAI `response_format` json_schema — https://docs.vllm.ai/en/latest/features/structured_outputs.html ; vision-model compatibility not explicitly documented, verify once); **Ollama `format` json-schema works with vision models** (documented image example): https://ollama.com/blog/structured-outputs ; llama.cpp GBNF/json_schema grammars. This makes JSON reliability engine-level and near-model-agnostic; note that grammar-constrained decoding can't fix wrong *content*, and constraining can slightly degrade reasoning — keep a "transcribe first, then judge" two-call pipeline.

## 5. Hebrew — the honest picture

- **Zero direct evidence for either candidate.** No Hebrew in MiniCPM's 30+-language claims, none in InternVL's materials, none in MTVQA/OCRBench-style benchmarks they report. No relevant GitHub issues found.
- **Backbone text side is fine**: Qwen3 (both models' LLM) trained on 119 languages **including Hebrew**: https://qwenlm.github.io/blog/qwen3/ — so Hebrew instruction-following + semantic judging of a given transcript should work; the risk is concentrated in *visual* Hebrew glyph recognition, RTL layout, and handwriting.
- **Hebrew handwriting is unsolved even commercially**: Microsoft Document Intelligence doesn't support handwritten Hebrew (https://learn.microsoft.com/en-us/answers/questions/1639127/hebrew-handwritten-text-optimal-solution-for-recog); a 2026 practitioner benchmark (MF-SR, https://mf-sr.com/en/blog/ocr-hebrew-2026-practitioner-guide.html — page 403'd on direct fetch; per search snippet: Tesseract 92-96% on clean *print*, handwriting unsupported by major engines).
- **Qwen3-VL** (Apache-2.0, 2B-235B, incl. 8B dense): "Supports 32 languages" OCR, up from 19 (https://github.com/QwenLM/Qwen3-VL , https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). Secondary sources state the 32 include **Hebrew**; the official card/repo do not enumerate the list — verify against the Qwen3-VL blog before relying on it. It has first-class vLLM + llama.cpp support. Strongly recommend adding it to the bake-off.
- **Mitigations**: (1) the MCQ-mark-detection subtasks (circles, X marks, ink-color separation, crossed-out answers) are vision tasks, not Hebrew OCR — likely transferable; (2) run your own gold-set eval with the human-scored sample exams (24/32, 28/32) before choosing; (3) LoRA on synthetic + real Hebrew handwriting (HebHTR-style data, https://github.com/Lotemn102/HebHTR) is a documented path on both families (ms-swift / LLaMA-Factory).

## 6. Known weaknesses summary

- InternVL3.5: overlapping/rotated/low-frequency text degradation (OCRBench v2); no official Ollama packaging; 3.5-specific AWQ absent from LMDeploy matrix; official LoRA scripts for 3.5 still community-requested.
- MiniCPM-V 4.5/4.6: no LMDeploy; 4.6 is small (1.3B) — expect weaker reasoning for semantic judging (use it for dev only); mixed EN/ZH response drift reported for the o-family; stale license text in GGUF repos; historical repetition loops on dense documents in the V-2.6 era (4.5's anti-hallucination OCR training targets this, but re-test on full exam scans).
- Both: no Hebrew-handwriting evidence; free hosted APIs unsuitable for real student data (privacy).

## Key sources (accessed 2026-07-12)
- https://github.com/OpenGVLab/InternVL | https://huggingface.co/OpenGVLab/InternVL3_5-8B | https://arxiv.org/abs/2508.18265 | https://docs.vllm.ai/projects/recipes/en/latest/InternVL/InternVL3_5.html | https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html | https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-8B-GGUF | https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html | https://internlm.intern-ai.org.cn/api/document | https://openrouter.ai/opengvlab
- https://github.com/OpenBMB/MiniCPM-V | https://huggingface.co/openbmb/MiniCPM-V-4_5 | https://huggingface.co/openbmb/MiniCPM-V-4.6 | https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf | https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf | https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/api.md | https://github.com/OpenSQZ/MiniCPM-V-CookBook | https://ollama.com/library/minicpm-v4.5 | https://arxiv.org/abs/2509.18154
- https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md | https://docs.vllm.ai/en/latest/features/structured_outputs.html | https://ollama.com/blog/structured-outputs | https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md
- https://github.com/bytedance/MTVQA | https://arxiv.org/html/2501.00321v2 | https://qwenlm.github.io/blog/qwen3/ | https://github.com/QwenLM/Qwen3-VL | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | https://learn.microsoft.com/en-us/answers/questions/1639127/hebrew-handwritten-text-optimal-solution-for-recog | https://github.com/Lotemn102/HebHTR | https://mf-sr.com/en/blog/ocr-hebrew-2026-practitioner-guide.html (403 on fetch; snippet-only)