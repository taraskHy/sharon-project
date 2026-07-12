# Qwen Vision-Language Family for Hebrew Exam Grading — Research Report
All sources fetched live on 2026-07-12. Claims marked [verified] were confirmed by directly fetching the cited page; [search-corroborated] means confirmed via search-result summaries of the cited page but not a full page fetch; [inference] is my analysis.

## 1. Family overview and what's current as of July 2026

The Qwen VL line has moved through three generations relevant to this project:

1. **Qwen2.5-VL** (Jan 28, 2025; 3B/7B/32B/72B) — older, mixed licensing, only 10 non-English/Chinese OCR languages (Hebrew NOT documented).
2. **Qwen3-VL** (Sep 23 – Oct 21, 2025; 2B/4B/8B/32B dense + 30B-A3B/235B-A22B MoE, each Instruct + Thinking) — all Apache-2.0, 32-language OCR **with Hebrew officially evaluated**, best-documented engine support (vLLM, llama.cpp, Ollama, official GGUFs). This is the sweet spot today.
3. **Qwen3.5** (Feb 16 – Mar 2, 2026; natively multimodal, early-fusion: 0.8B/2B/4B/9B/27B dense + 35B-A3B/122B-A10B/397B-A17B MoE) and **Qwen3.6** (Apr 16–22, 2026; 27B dense + 35B-A3B, both with vision) — all open weights Apache-2.0, higher OCR benchmark scores, but no published per-language OCR breakdown (Hebrew OCR quality unverified).

There is no "Qwen3-VL" branded release newer than Oct 2025; the successors are the natively-multimodal Qwen3.5/3.6 series. Qwen3-Omni (Sep 2025) adds audio, is Apache-2.0, but is not the best fit for document grading (see §8).

Timeline source [verified]: QwenLM/Qwen3-VL README news section — 235B-A22B 2025-09-23; 30B-A3B + FP8 2025-10-04; 4B/8B 2025-10-15; 2B/32B 2025-10-21; tech report 2025-11-27; also Qwen2.5-VL dates (series 2025-01-28, AWQ models 2025-02-20, 32B 2025-03-25). https://github.com/QwenLM/Qwen3-VL (raw README fetched). Qwen3.5/3.6 lineup + dates [verified]: https://github.com/QwenLM/Qwen3.6 (Qwen3.6-27B 2026-04-22; Qwen3.6-35B-A3B 2026-04-16; Qwen3.5 series 2026-02-24–03-02; "All our open-weight models are licensed under Apache 2.0").

## 2. Licenses (the load-bearing detail for university use)

| Model | Weight license | University use? | Source (fetched 2026-07-12) |
|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | **Qwen RESEARCH License** (2024-09-19): "FOR NON-COMMERCIAL PURPOSES ONLY", non-commercial = "research or evaluation purposes only" | Research/eval OK; production grading of real exams is a gray area — avoid | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE [verified] |
| Qwen2.5-VL-7B-Instruct | Apache-2.0 | Yes | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct [verified] |
| Qwen2.5-VL-32B-Instruct | Apache-2.0 | Yes | https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct [verified] |
| Qwen2.5-VL-72B-Instruct | **Qwen LICENSE Agreement** (2024-09-19): commercial use allowed, but >100M-MAU products need a separate license; attribution ("Built with Qwen") required | Yes — a university is nowhere near 100M MAU, but it is not OSI-approved | https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/blob/main/LICENSE [verified] |
| Qwen3-VL — ALL sizes (2B/4B/8B/32B/30B-A3B/235B-A22B, Instruct & Thinking) | Apache-2.0 (code repo also Apache-2.0) | Yes, fully clean | Repo: https://github.com/QwenLM/Qwen3-VL [verified]; model cards spot-checked: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct [verified], https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct [verified] |
| Qwen3.5 (all open weights incl. 27B) / Qwen3.6 (27B, 35B-A3B) | Apache-2.0 | Yes | https://huggingface.co/Qwen/Qwen3.5-27B [verified]; https://github.com/QwenLM/Qwen3.6 [verified]; https://huggingface.co/Qwen/Qwen3.6-27B [verified] |
| Qwen3-Omni-30B-A3B-Instruct | Apache-2.0 | Yes | https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct [verified] |

Warning: Wikipedia-derived summaries claim "all Qwen2.5-VL except 72B are Apache-2.0" — that is wrong; the 3B is research-only. Verified directly from the LICENSE file above.

All Apache-2.0 weights are downloadable from Hugging Face/ModelScope and fully offline-capable.

## 3. Hebrew evidence (the decisive finding)

- **Primary evidence**: The Qwen3-VL Technical Report (arXiv:2511.21631, 2025-11-27, https://arxiv.org/abs/2511.21631) evaluates multilingual OCR on a 39-language self-built test set; 32 languages exceed the 70% accuracy bar the authors call "practical for real-world usability". I downloaded the PDF and rendered Figure 2 (p. 17): **Hebrew is explicitly one of the 32 supported languages, at ~72% accuracy** — above the usability threshold but in the bottom five (cf. Arabic ~87%, Swedish ~98%). [verified — figure extracted from the PDF locally]. The report also states the expansion "from the 10 non-English/Chinese languages supported by Qwen2.5-VL to 39 languages in Qwen3-VL" — i.e., **Hebrew is effectively a Qwen3-VL-and-later capability; do not use Qwen2.5-VL for Hebrew**.
- Official blog corroboration: "OCR now supports 32 languages (up from 10)… recognition accuracy for rare characters, ancient scripts, and technical terms has also improved" — https://www.alibabacloud.com/blog/qwen3-vl-sharper-vision-deeper-thought-broader-action_602584 [verified]; same claim on model cards and https://ollama.com/library/qwen3-vl [verified].
- Alibaba's hosted qwen-vl-ocr documentation shows Hebrew text in its sample output ("בר מולד") alongside Arabic/Russian etc. — https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr [verified].
- **Handwritten Hebrew: no public evidence found.** Searches (2026-07-12) for Qwen + Hebrew handwriting benchmarks/community reports returned nothing. The tech-report figure is printed-text OCR. A June 2026 paper, "Towards Fully Automated Exam Grading: Fairness-Aware Recognition of Handwritten Answers with Foundation Models" (https://arxiv.org/pdf/2606.11477 [verified]), evaluates Qwen3VL among Gemini/GPT/Grok for handwritten exam grading — but Latin script only, with an error taxonomy showing systematic failure modes on handwriting. Conclusion [inference]: Hebrew handwriting, ink-color separation, and mark-convention reasoning must be validated on your own sample exams (you already have ground truth: 24/32 & 28/32 human scores); there is no published number to lean on.
- Qwen3.5/3.6 claim "201 languages and dialects" (text) and higher OCR scores (OCRBench 93.1, OmniDocBench 90.8 for Qwen3.5-397B) but publish no per-language OCR table — https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894 [search-corroborated].

## 4. Model-by-model details (Qwen3-VL — the primary candidates)

Common to all Qwen3-VL: native 256K context (expandable to 1M via YaRN), multi-image + video input, Instruct and Thinking variants, Apache-2.0, transformers ≥4.57, vLLM ≥0.11.0, SGLang, Docker image qwenllm/qwenvl [verified: repo + HF cards]. Cookbooks include OCR/key-information-extraction, document parsing, and long-document understanding — https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/ocr.ipynb [verified via repo README].

| Model (HF id) | Q4 GGUF size (Ollama tag) | fp16 VRAM (approx) | Notes |
|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | 1.9 GB (qwen3-vl:2b) | ~5 GB | Too weak for semantic judging [inference] |
| Qwen/Qwen3-VL-4B-Instruct | 3.3 GB (qwen3-vl:4b) | ~9 GB | FP8 variant official |
| Qwen/Qwen3-VL-8B-Instruct | 6.1 GB (qwen3-vl:8b); official GGUF Q4_K_M 5.03 GB, Q8_0 8.71 GB, F16 16.4 GB + mmproj | ~18–20 GB | Best laptop-CPU dev model; official GGUF repo with llama-server/llama-mtmd-cli instructions — https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF [verified] |
| Qwen/Qwen3-VL-30B-A3B-Instruct (MoE, 3B active) | 20 GB (qwen3-vl:30b) | ~65 GB total fp16 | Tech report: outperforms Gemini-2.5-Flash & GPT-5-mini on most metrics [verified, p.17]; MoE = fast CPU decode with 63 GB RAM [inference] |
| Qwen/Qwen3-VL-32B-Instruct | 21 GB (qwen3-vl:32b) | ~66 GB (fits A100-80G; Q4/AWQ fits 24–32 GB card) | 33B params; also outperforms Gemini-2.5-Flash/GPT-5-mini per tech report; official GGUF exists (Qwen/Qwen3-VL-32B-Thinking-GGUF) [verified] |
| Qwen/Qwen3-VL-235B-A22B-Instruct | 143 GB (qwen3-vl:235b) | multi-GPU only | Flagship; SOTA MMLongBench-Doc 57.0% [verified, tech report] |

Ollama sizes/context [verified]: https://ollama.com/library/qwen3-vl (requires Ollama ≥0.12.7; all sizes local, text+image, 256K).

## 5. Engines and Windows/CPU story

- **vLLM** ≥0.11.0: recommended server engine for the university GPU box [verified: Qwen3-VL README]. Structured outputs (JSON-schema-constrained decoding via xgrammar/guidance backends, `response_format`/`structured_outputs` in the OpenAI-compatible server) — https://docs.vllm.ai/en/latest/features/structured_outputs.html [verified]. Constrained decoding is logit-level, so it applies to VL models too [inference].
- **llama.cpp**: Qwen3-VL support (dense + MoE) merged 2025-10-30, PR #16780 (MRoPE-interleave, deepstack, separate mmproj vision-encoder GGUF) — https://github.com/ggml-org/llama.cpp/pull/16780 [verified]. Qwen2.5-VL also supported (ggml-org GGUFs; see docs/multimodal.md) [search-corroborated: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md]. Windows binaries standard; `llama-server` exposes an OpenAI-compatible API with JSON-schema/GBNF grammar constraints.
- **Ollama**: qwen3-vl all sizes, Windows-native, and `format` parameter accepts a JSON schema — structured outputs demonstrated with vision models — https://ollama.com/blog/structured-outputs [verified].
- **LM Studio / Jan**: listed as compatible on the HF model cards [verified].
- **LMDeploy**: NOT listed among supported engines for Qwen3-VL/3.5/3.6 in official docs [verified: absence in both repos' README fetches]. It supported Qwen2.5-VL, but don't plan on it for Qwen3-VL+.
- **Quantized variants**: official FP8 (Qwen3-VL, since 2025-10-04) and official GGUF repos; AWQ officially released for Qwen2.5-VL (2025-02-20), community AWQ/GPTQ for Qwen3-VL; 40–87 community quantizations linked per model card [verified].
- **CPU-only laptop (63 GB RAM, Intel iGPU)**: Qwen3-VL-8B Q4_K_M (5.03 GB + mmproj) runs comfortably; Qwen3-VL-30B-A3B Q4 (20 GB) fits RAM easily and decodes fast (3B active params), though the ViT image-encode step is compute-heavy on CPU — expect tens of seconds per scanned page; fine for development, not for batch grading [inference from verified sizes/architecture].
- Qwen3.5-27B / Qwen3.6-27B: llama.cpp ("text & vision"), Ollama, MLX, KTransformers, vLLM, SGLang; 212 community quants for Qwen3.5-27B; unsloth GGUF for Qwen3.6-27B [verified: HF card + Qwen3.6 repo].

## 6. Hosted API options (deployment target a)

- **Truly free, hosted, open-weight Qwen VL**: **ModelScope API-Inference** — 2,000 free API calls/day (500/model/day, resets 00:00 UTC+8), includes Qwen3-VL models (e.g., Qwen3-VL-235B-A22B-Instruct on modelscope.ai); requires Alibaba Cloud account binding + real-name verification — https://modelscope.ai/docs/model-service/API-Inference/limits [search-corroborated]. This is the best free-hosted fit, with the verification hurdle noted.
- **Alibaba Cloud Model Studio**: new-user free quota (~1M tokens per model, Singapore region), valid 90 days, real-time inference only, then paid; no perpetual free tier — https://www.alibabacloud.com/help/en/model-studio/new-free-quota [verified].
- **OpenRouter**: no Qwen-VL `:free` endpoints as of July 2026 (the 26 free models include vision-capable Gemma 4 and Nemotron entries, but Qwen free slots are text-only qwen3-coder / qwen3-next-80b) — https://costgoat.com/pricing/openrouter-free-models [verified]. Paid Qwen VL is cheap: qwen3-vl-8b-instruct $0.117/M in, $0.455/M out — https://openrouter.ai/qwen/qwen3-vl-8b-instruct [search-corroborated]; qwen3.6-35b-a3b $0.14/M in, $1.00/M out — https://openrouter.ai/qwen [verified]. OpenRouter free tier limits: ~20 req/min, 200 req/day.
- Together AI and others host Qwen3-VL-32B paid (https://www.together.ai/models/qwen3-vl-32b-instruct, [search-corroborated]).

## 7. Structured JSON reliability

- Model-level: Qwen2.5-VL card explicitly claims "stable JSON outputs for coordinates and attributes" and structured outputs for invoices/forms/tables [verified: 3B card]; Qwen3-VL ships OCR/KIE and document-parsing cookbooks [verified: repo].
- Engine-level (the real guarantee): vLLM xgrammar/guidance constrained decoding [verified], Ollama JSON-schema `format` [verified], llama.cpp GBNF/JSON-schema. With schema-constrained decoding, syntactically-valid JSON is guaranteed; semantic correctness still needs your validation layer [inference].

## 8. Qwen3-Omni

Qwen3-Omni-30B-A3B-Instruct: Apache-2.0, MoE Thinker–Talker, image+audio+video input, text+speech output, 119 text languages; vLLM via custom branch, transformers from source — https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct [verified]. Not recommended here: adds audio complexity, weaker ecosystem (no mainstream GGUF/Ollama path), no OCR-language table [inference].

## 9. Known weaknesses (verified reports + assessment)

- Hebrew is bottom-5 of Qwen3-VL's 32 supported OCR languages (~72% vs 87% Arabic / 98% Swedish) — printed text; handwriting unmeasured [verified: tech report Fig. 2].
- Repetition/latency failure mode: a Nov 2025 head-to-head (DeepSeek-OCR vs Qwen3-VL vs Mistral-OCR) found Qwen3-VL "delivered the best character-level OCR" and captured checkbox marks, but in one long-extraction test degenerated into "infinite dots" (repetition loop) — https://www.analyticsvidhya.com/blog/2025/11/deepseek-ocr-vs-qwen-3-vl-vs-mistral-ocr/ [verified]. Mitigate with repetition penalties, page-at-a-time prompting, and schema-constrained decoding [inference].
- GGUF conversion pitfalls existed for Qwen2.5-VL (e.g., '@'-spam after conversion, llama.cpp issue #15870) — prefer the official Qwen GGUFs for Qwen3-VL [verified issue exists; inference on mitigation].
- Version gates: transformers ≥4.57, vLLM ≥0.11.0, Ollama ≥0.12.7 [verified].
- Qwen2.5-VL: only 10 non-EN/ZH OCR languages (no documented Hebrew), 32K default context, 3B license research-only, 72B non-OSI license [verified].
- No public evidence on blue-vs-red ink separation or exam-mark conventions for any Qwen model — must be validated empirically on your sample PDFs [inference from absence].

## 10. Recommendation for the autograder (C:\Users\ethan\PycharmProjects\sharon-project)

1. **Primary candidate: Qwen3-VL-32B-Instruct** (Apache-2.0, documented Hebrew OCR, near-frontier document benchmarks, official GGUF/FP8, vLLM-servable on a single 48–80 GB GPU or Q4 on 24–32 GB). **Alternative with newer architecture: Qwen3.5-27B** (Apache-2.0, higher OCR benchmark scores, 262K context) — but its Hebrew OCR is unpublished, so benchmark both on the ground-truth exams in sample_data/.
2. **Laptop dev (CPU, 63 GB RAM)**: Qwen3-VL-8B-Instruct official GGUF Q4_K_M (5.03 GB) via Ollama ≥0.12.7 or llama-server; step up to Qwen3-VL-30B-A3B Q4 (20 GB, 3B active) when quality matters more than page-encode latency.
3. **Free hosted**: ModelScope API-Inference (2,000 calls/day) if the Alibaba real-name verification is acceptable; otherwise Model Studio's 90-day 1M-token trial for evaluation, then OpenRouter paid (~$0.12/M input for 8B) — grading a full course would cost cents [inference].
4. Enforce JSON with engine-level schema-constrained decoding (vLLM `structured_outputs` / Ollama `format`), not prompt-only.
5. Critical open risk: handwritten-Hebrew accuracy and ink-color separation are unproven anywhere in the literature — run the existing sample exams (known scores 24/32, 28/32) through Qwen3-VL-8B (local) and Qwen3-VL-32B/Qwen3.5-27B (hosted trial) before committing.

Artifacts saved during research: Qwen3-VL tech report PDF and rendered Figure 2 (Hebrew bar chart) at C:\Users\ethan\AppData\Local\Temp\claude\C--Users-ethan-PycharmProjects-sharon-project\8a1ef58b-0e5d-40ce-a09d-db1e0d86c119\scratchpad\qwen3vl_fig2_p17.png and fig2_left_zoom.png.