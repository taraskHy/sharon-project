# Free Hosted APIs for Open-Weight VLMs — Hebrew Exam Grading (verified live 2026-07-12)

All claims below were verified today (2026-07-12) via live fetches of official docs/APIs unless marked otherwise. Workload sizing used throughout: 50 exams x ~15 vision calls = **~750 vision calls**; at ~3–6K tokens per call (scanned A4 page as image + prompt + JSON output) that is **~2.5–4.5M tokens total**.

## Landscape headline (changed since 2025)

The dominant open VLMs on free tiers right now are **Gemma 4** (released 2026-04-02 under **Apache 2.0** — Google switched from the restrictive Gemma license; confirmed at https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ and https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html, checked 2026-07-12) and **Qwen3.6-27B** (released 2026-04-22, Apache 2.0, dense 27B, native text+image+video, 262K context, official GGUF w/ mmproj exists — https://huggingface.co/Qwen/Qwen3.6-27B). Both are license-clean for university use and both are self-hostable, including CPU-only via llama.cpp GGUF. Qwen2.5-VL-era :free endpoints are gone from OpenRouter.

---

## 1. OpenRouter `:free` variants

**Source (live API, 2026-07-12):** `GET https://openrouter.ai/api/v1/models` — **23 total `:free` models, 5 with image input:**

| Model | Ctx | Modalities |
|---|---|---|
| `google/gemma-4-31b-it:free` | 262K | text+image+video |
| `google/gemma-4-26b-a4b-it:free` | 262K | text+image+video |
| `nvidia/nemotron-nano-12b-v2-vl:free` | 128K | text+image+video |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 256K | text+audio+image+video |
| `nvidia/nemotron-3.5-content-safety:free` | 128K | text+image (safety classifier, not a grader) |

**Rate limits** (official: https://openrouter.ai/docs/api/reference/limits, 2026-07-12): free variants = **20 req/min**; **50 req/day** if you've purchased <$10 lifetime credits; **1,000 req/day** once you've purchased **$10+ (all-time, one-off)**. No card needed for the free tier itself; the $10 unlock requires a normal payment (card/crypto). 750 calls fits in one day only with the $10 unlock.

**Who actually serves the free endpoints & data policy** (live: `GET https://openrouter.ai/api/v1/models/google/gemma-4-31b-it:free/endpoints` + `GET https://openrouter.ai/api/frontend/all-providers`, 2026-07-12):
- `gemma-4-31b-it:free` → **OpenInference** (`training: true, retainsPrompts: true` — your exam images may be used for training) and **Google AI Studio** (`training: false, retainsPrompts: true, retentionDays: 55` per OpenRouter's table — BUT Google's own Gemini API unpaid-tier terms say: "Google uses the content you submit... to provide, improve, and develop Google products and services and machine learning technologies" and "human reviewers may read, annotate, and process your API input and output"; "Do not submit sensitive, confidential, or personal information to the Unpaid Services" — https://ai.google.dev/gemini-api/terms, 2026-07-12. Treat Google's terms as authoritative over OpenRouter's flag.)
- `nemotron-nano-12b-v2-vl:free` → **NVIDIA trial infra** (`training: true, retainsPrompts: true`), and its live uptime today was **71–77% (status −5, deranked)**. The OpenInference gemma endpoint showed **82.6% 1-day uptime**; the Google AI Studio route 99.9%.
- OpenRouter account settings let you block routing to training providers ("separate settings for paid and free models" — https://openrouter.ai/docs/features/privacy-and-logging), but for the free vision models that leaves essentially only the Google AI Studio route.
- Structured output note: the AI-Studio route advertises `response_format`; the OpenInference route does not (tools only, max_completion 8192).

**Reliability reputation:** OpenRouter's own blog (2026-06-15, https://openrouter.ai/blog/tutorials/free-llm-apis-compared/) is candid: "Free tiers can tighten rate limits without warning, increase latency during peak hours, or experience complete outages with no compensation."

**Self-host equivalents:** identical open weights — `google/gemma-4-31b-it` / `gemma-4-26b-a4b-it` (Apache 2.0) and NVIDIA Nemotron Nano 12B v2 VL (open weights on HF).

## 2. Groq

**Vision models today** (https://console.groq.com/docs/vision, 2026-07-12): **`qwen/qwen3.6-27b` (GA — production)** and `meta-llama/llama-4-scout-17b-16e-instruct` (still *preview*). Both: images via URL (≤20MB) or base64 (≤4MB), ≤5 images/request, tool use + JSON mode. Llama 4 Maverick is no longer in the model list.

**Free-tier limits** (https://console.groq.com/docs/rate-limits, 2026-07-12): qwen3.6-27b = **30 RPM / 1K RPD / 8K TPM / 200K TPD**; llama-4-scout = 30 RPM / 1K RPD / 30K TPM / 500K TPD. No credit card required for the free tier (secondary sources; console signup is free). Upgrade path: self-serve **Developer tier** (pay per token, much higher limits, Batch/Flex).

**Data policy — the cleanest of all** (https://console.groq.com/docs/your-data + DPA, 2026-07-12): Groq "is not permitted to use Inputs or Outputs for training"; **no retention by default** (up to 30-day logs only for abuse/error investigation); **Zero Data Retention toggle** available in console Data Controls. OpenRouter's provider table independently lists Groq as `training: false, retainsPrompts: false`.

**Feasibility math:** 8K TPM means ~1–2 page-image calls/minute; 200K TPD ≈ 40–60 page calls/day ≈ 3–4 exams/day → full 50-exam batch takes ~2–3 weeks on free, or minutes-to-hours on the (cheap) Developer tier.

**Self-host equivalent:** Qwen3.6-27B is Apache 2.0 with official weights + GGUF (runs on your 63GB-RAM laptop via llama.cpp; Q4 ≈ ~16–18GB). Llama 4 Scout is Llama Community License (not OSI-clean; note its EU multimodal restrictions if relevant).

## 3. Cerebras — the sleeper option

**Catalog** (https://inference-docs.cerebras.ai/models/overview + /models/gemma-4-31b, 2026-07-12): `gpt-oss-120b` (prod), **`gemma-4-31b` (preview) — accepts image input** ("base64 PNG or JPEG data URI only; external URLs not supported"), `zai-glm-4.7` (preview). Free tier for gemma-4-31b: **5 req/min, 30K input tokens/min, 1M tokens/day, 65K context**, no credit card.

**Data policy:** Cerebras states it does **not retain** prompt content/API requests/responses and does not train on customer data (https://support.cerebras.net/articles/1811589793-does-cerebras-retain-my-data + https://www.cerebras.ai/privacy-policy, 2026-07-12).

**Feasibility:** 1M tokens/day ≈ 150–250 page calls/day → **full batch in ~3–5 days, free, with a no-training/no-retention policy**. Risks: vision on Cerebras is brand new and *preview*; only one VLM; 8K-era context caps have been reported on free tiers; hardware vendor whose model list churns (their catalog had different entries as recently as May 2026).

## 4. NVIDIA NIM / build.nvidia.com

**Access:** free NVIDIA Developer account (phone verification, no card), **1,000 trial credits** (1 credit ≈ 1 request), up to ~5,000 via requests; rate limit ~**40 RPM** (200 RPM on application) — https://build.nvidia.com + developer forums, checked 2026-07-12. Catalog includes relevant open VLMs: **Nemotron Nano 12B v2 VL** (document intelligence/OCR focus), Gemma-4-26B, Qwen3.5/3.6 series, and **Nemotron OCR v2** (https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-nano-12b-v2-vl).

**Disqualifying for real exams — read the ToS** (extracted verbatim today from the official PDF, https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf):
- 3.3: NVIDIA collects "...(iv) User Content and Generated Content **to improve NVIDIA products and services, including AI models**."
- 4.3: "you will not upload any personal information relating to an identifiable individual... or any other information which may be subject to data privacy or data security laws... NVIDIA does not represent, and specifically disclaims, that NVIDIA servers are appropriate for processing of any data including personal data..."

So: fine for development against the anonymized sample exam; **contractually off-limits for real student exams**. ~1,000 total credits also ≈ only one 750-call batch. Self-host equivalent: the same NIM containers/weights run on the university GPU server (Nemotron models are open weights).

## 5. SambaNova

Free tier exists ("forever free" + $5 signup credits, no card) and its only vision model is **`gemma-4-31B-it` (preview, image+video input, 128K ctx)** — but free limits are **20 RPM / 20 requests per DAY / 200K TPD** (https://docs.sambanova.ai/docs/en/models/rate-limits + /cloud/docs/get-started/supported-models, 2026-07-12). 750 calls ÷ 20/day = **37+ days → unusable**. Developer (paid) tier: 60 RPM / 12K RPD. Their own community forum has a thread titled "Is free tier going away?" — treat as unstable.

## 6. Hugging Face Inference Providers

Official pricing doc (https://huggingface.co/docs/inference-providers/pricing, 2026-07-12): free users get **$0.10/month** in credits ("subject to change"), PRO ($9/mo) gets $2.00/month; pass-through provider pricing after that. 273 image-text-to-text models are servable (Qwen3.6-27B via Featherless, gemma-4-31B-it via Novita, etc. — https://huggingface.co/models?pipeline_tag=image-text-to-text&inference_provider=all). Verdict: excellent unified *paid* aggregator, but **$0.10/month is not a free tier in any practical sense** (~1M tokens of gemma-4 at best ≈ a fraction of one batch).

## 7. Together AI

Pricing page (https://www.together.ai/pricing, 2026-07-12) advertises **no free tier and no free models**; the 2024-era `Llama-Vision-Free` (11B) endpoint promotion is over, and third-party trackers report signup credits were retired (small $1–5 credits reported inconsistently). Not a candidate.

## 8. Google AI Studio (excluded per your rule, with one nuance)

Gemini models are proprietary → excluded. Nuance: AI Studio also serves **open Gemma 4** free, and that's exactly what OpenRouter's `gemma-4-31b-it:free` non-training route uses — but Google's unpaid-tier terms (human review, product improvement, "do not submit... personal information") make it inappropriate for real student exams either way.

---

## Honest assessment for ~750 vision calls

**Best overall free option: Groq (`qwen/qwen3.6-27b`, GA)** — only provider combining a production-status open VLM (Apache 2.0), a genuinely clean data policy (no training, no default retention, ZDR toggle — important because student exams are personal data), JSON mode, and no card. Constraint: 200K TPD means a full batch takes ~2–3 weeks free; realistically you use the free tier for development + spot-grading and pay a few dollars on Developer tier for batch runs.

**Best free *volume*: Cerebras (`gemma-4-31b` preview)** — 1M tokens/day + no-retention policy → full batch in ~3–5 days free. But vision there is preview-status, base64-only, 65K ctx, unproven reliability.

**OpenRouter** is the best *development multiplexer* (one key, 5 free VLMs, $10 unlocks 1,000 req/day = full batch in a day) but is the **worst privacy story**: today's free vision routes either train on your data (OpenInference, NVIDIA) or pass through Google's unpaid tier (human review). Live uptime today on free vision endpoints was 71–99%. Use it for prompt development on the sample exam, not for real student pages.

**NVIDIA NIM**: great model catalog for experiments (incl. Nemotron OCR v2, interesting for Hebrew handwriting), contractually barred from personal data and trains on inputs. **SambaNova** (20 req/day), **HF free credits** ($0.10/mo), **Together** (none): not viable.

**Risks common to all free tiers:** limits and lineups changed materially within the last 90 days at every provider checked (Groq dropped Maverick; Cerebras' catalog turned over; OpenRouter's free vision set is entirely different from 2025; SambaNova users openly ask if free is ending). OpenRouter's own June 2026 blog concedes free tiers can be tightened or killed "without warning." **Do not architect the grader against any single free endpoint.** The hedge is that all three serious candidates serve Apache-2.0 models (Gemma 4 31B/26B-A4B, Qwen3.6-27B) that you can self-host bit-for-bit: vLLM on the university GPU server, llama.cpp GGUF (Qwen3.6-27B and Gemma 4 both have official/community GGUF + vision projectors) on the 63GB Windows laptop for CPU-only dev. Build against an OpenAI-compatible client with a provider-agnostic config and structured-output validation + retry, and switching between Groq/Cerebras/OpenRouter/self-hosted is a config change.

**Fit-to-task caveat (unverified by any vendor doc):** none of these providers documents Hebrew handwriting accuracy. Gemma 4 claims 140+ languages; Qwen3.6 is strongly multilingual; both must be validated empirically against your sample exam ground truth (the A1 version, swapped tables, X-convention, 24/32 & 28/32 human scores) before trusting either.

## Key sources (all accessed 2026-07-12)
- https://openrouter.ai/docs/api/reference/limits — free-variant RPM/RPD, $10 threshold
- https://openrouter.ai/api/v1/models and .../models/{slug}/endpoints and /api/frontend/all-providers — live free-VLM list, per-provider training/retention flags, live uptime
- https://openrouter.ai/docs/features/privacy-and-logging — free/paid training opt-out settings
- https://openrouter.ai/blog/tutorials/free-llm-apis-compared/ (2026-06-15) — free-tier reliability caveat
- https://console.groq.com/docs/vision, /docs/rate-limits, /docs/your-data, /docs/legal/customer-data-processing-addendum — Groq models, limits, no-training/ZDR policy
- https://inference-docs.cerebras.ai/models/overview and /models/gemma-4-31b — Cerebras vision support + free limits
- https://support.cerebras.net/articles/1811589793-does-cerebras-retain-my-data, https://www.cerebras.ai/privacy-policy — Cerebras retention
- https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf — verbatim training + personal-data clauses
- https://docs.sambanova.ai/docs/en/models/rate-limits and /cloud/docs/get-started/supported-models — 20 RPD free tier, gemma-4 vision preview
- https://huggingface.co/docs/inference-providers/pricing — $0.10/mo free credits
- https://www.together.ai/pricing — no free tier
- https://ai.google.dev/gemini-api/terms — unpaid-tier data use / human review
- https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/, https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html — Gemma 4 Apache 2.0
- https://huggingface.co/Qwen/Qwen3.6-27B — Qwen3.6-27B Apache 2.0, multimodal, GGUF