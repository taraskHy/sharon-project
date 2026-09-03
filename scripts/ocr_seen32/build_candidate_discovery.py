import json, hashlib, pathlib, subprocess, time, re, statistics

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ts = time.strftime("%Y-%m-%d %H:%M:%S")
CAT = json.loads(pathlib.Path("evaluation/model_selection/runs/ocr_primary/OCR_OPENROUTER_CATALOG_SNAPSHOT_2026-09-03.json").read_text(encoding="utf-8"))
data = CAT["models"]
by = {m["id"]: m for m in data}


def seal(doc, field="content_sha256"):
    body = json.dumps({k: v for k, v in doc.items() if k != field},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    doc[field] = hashlib.sha256(body.encode()).hexdigest()
    return doc


def img(m): return "image" in (m.get("architecture") or {}).get("input_modalities", [])
def so(m): return "structured_outputs" in (m.get("supported_parameters") or [])
def priced(m):
    p = m.get("pricing") or {}
    try: return float(p.get("prompt", -1)) > 0 and float(p.get("completion", -1)) > 0
    except Exception: return False


n_img = sum(1 for m in data if img(m))
n_img_so = sum(1 for m in data if img(m) and so(m))
n_elig = sum(1 for m in data if img(m) and so(m) and priced(m))

# local prior art, recomputed
local = {}
for d in sorted(pathlib.Path("evaluation/hebrew_bench/outputs").iterdir()):
    f = d / "eval_detail.txt"
    if not f.exists(): continue
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cers = [float(x) for x in re.findall(r"cer=([0-9.]+)", f.read_text(encoding="utf-8", errors="replace"))]
    if cers:
        local[d.name] = {"model": cfg.get("model"), "preproc": cfg.get("preproc"), "n": len(cers),
                         "mean_cer": round(statistics.mean(cers), 3),
                         "median_cer": round(statistics.median(cers), 3),
                         "best_case_cer": round(min(cers), 3)}

doc = seal({
 "artifact": "ocr_candidate_discovery_and_shortlist",
 "created_at": ts, "git_commit": commit,
 "provider_inference_calls": 0,
 "method": {
   "endpoints_used": ["GET https://openrouter.ai/api/v1/models (public catalog)",
                      "GET https://openrouter.ai/api/v1/models/{slug}/endpoints",
                      "GET https://openrouter.ai/api/v1/providers"],
   "these_are_metadata_only": "no generation request, no crop transmitted, no token billed",
   "fetched_at_utc": CAT["fetched_at_utc"],
   "catalog_size": CAT["models_in_full_catalog"],
   "primary_sources": "OpenRouter catalog metadata and each model's own published description. "
                      "Benchmark blogs, marketing pages and leaderboards were NOT used."},

 "funnel": {"models_in_catalog": CAT["models_in_full_catalog"], "image_input_capable": n_img,
            "image + structured_outputs": n_img_so,
            "image + structured_outputs + priced": n_elig,
            "official_description_documents_ocr_or_document_parsing": 1},

 "eligibility_filter_applied": [
   "accepts image input", "can return sufficient exact text (max_completion_tokens >> the longest "
   "frozen reference)", "route can be priced BEFORE execution", "output budget supports the "
   "longest frozen reference (116 chars)", "gateway can enforce the OCR-only payload boundary",
   "no existing result has disqualified the identical model/route",
   "meaningfully distinct from the failed candidates", "freezable by exact slug + route config",
   "8-case cost is reasonable", "no automatic fallback can silently change model/provider"],

 "rejected_with_reason": [
   {"candidate": "openai/gpt-5.6-luna-pro", "reason": "already disqualified - registry DROP "
    "(4/5 handwritten refusals, 1 fabrication, failure-aware handwritten CER 0.9487)"},
   {"candidate": "anthropic/claude-sonnet-5", "reason": "already measured on the exact target "
    "population - HISTORICAL_CONTROL_ONLY (27/32 usable but successful CER 0.4718, 9/27 critical). "
    "Not shortlisted merely to fill the list."},
   {"candidate": "google/gemini-3.7-flash under AUTOMATIC routing", "reason": "already measured "
    "under two prompts and dropped both times. Only a PINNED route is a new configuration."},
   {"candidate": "all text-only models", "reason": "no image input - 169 of 425 catalog entries"},
   {"candidate": "models without structured_outputs", "reason": "the frozen BenchTranscription "
    "JSON-schema contract cannot be enforced, so a failure could not be distinguished from a "
    "formatting artefact (e.g. mistralai/mistral-small-3.1-24b-instruct, "
    "meta-llama/llama-4-scout on its Novita endpoint)"},
   {"candidate": "models with zero or absent pricing (incl. ':free' tiers)", "reason": "an unpriced "
    "route makes predicted_call_cost 0, so the pre-call budget gate could not refuse a call that "
    "crosses the ceiling. require_priced_candidate refuses these by design."},
   {"candidate": "meta-llama/llama-4-scout", "reason": "its three endpoints disagree on capability "
    "(Novita reports structured_outputs false) and span three unrelated providers, so the arm "
    "would not be reproducible without a pin, and pinning to the one Google-Vertex endpoint "
    "reintroduces the provider whose behaviour is under investigation"},
   {"candidate": "qwen/qwen3-vl-235b on DeepInfra / Venice / Parasail / Novita", "reason": "served "
    "at fp8/bf16 quantisation by four different providers; unpinned this is non-reproducible, and "
    "quantisation is a live quality risk for this task"},
   {"candidate": "a same-route prompt variation", "reason": "explicitly excluded - two prompt "
    "variants have already been measured on this route and prompt engineering there is exhausted"}],

 "shortlist": [
  {"rank": "primary", "candidate": "google/gemini-3.7-flash pinned to google-ai-studio",
   "why_genuinely_different": "a deterministic single-provider route, never previously measured; "
     "the two 32-crop arms both ran AUTOMATIC routing across a mix of two providers",
   "scores": {"ocr_suitability": "highest measured in this project (successful CER 0.1155)",
              "hebrew_script_plausibility": "demonstrated on this exact corpus",
              "provider_reliability": "unknown per-provider; that is the question",
              "refusal_risk": "HIGH - this is the failure mode under test",
              "route_determinism": "high once pinned (allow_fallbacks=false)",
              "structured_output_reliability": "good - 0 schema failures in 32",
              "cost": "$0.038 worst case for 8", "latency": "6.7-8.7 s/crop measured"},
   "evidence": "OCR_PROVIDER_ROUTE_FORENSICS_2026-09-03.json",
   "unknowns": ["whether the pin is honoured without silent fallback"]},

  {"rank": "secondary", "candidate": "google/gemini-3.7-flash pinned to google-vertex",
   "why_genuinely_different": "the paired half; without it a filter count cannot be attributed",
   "scores": {"ocr_suitability": "same model, same expected quality",
              "refusal_risk": "HIGH - under test",
              "route_determinism": "high once pinned", "cost": "$0.038 worst case for 8"},
   "unknowns": ["Vertex safety thresholds are not surfaced through OpenRouter"]},

  {"rank": "third", "candidate": "qwen/qwen3-vl-235b-a22b-instruct pinned to alibaba",
   "why_genuinely_different": "different family, vendor and serving provider; the ONLY catalog "
     "model whose official description documents document parsing and chart/table extraction; not "
     "moderated by OpenRouter",
   "scores": {"ocr_suitability": "documented for documents; UNMEASURED on Hebrew handwriting",
              "hebrew_script_plausibility": "WEAK PRIOR - see local_prior_art below",
              "provider_reliability": "unknown", "refusal_risk": "low (is_moderated false, no "
              "Google-style native safety layer expected, but unverified)",
              "route_determinism": "high once pinned to the single Alibaba endpoint",
              "structured_output_reliability": "declared; unverified here",
              "cost": "$0.020 worst case for 8", "latency": "unknown"},
   "unknowns": ["image tokenisation on 1287px crops", "Hebrew handwriting at this scale"]}],

 "candidate_count": 3,
 "no_accuracy_number_invented": "no expected CER is claimed for any untested candidate",

 "local_prior_art_that_constrains_this_choice": {
   "why_it_matters": "the project already measured local OCR on this corpus in 2026-07, including "
     "TWO DEDICATED OCR SYSTEMS and four preprocessing strategies. Every configuration failed "
     "catastrophically. This is the evidence that a cloud VLM is not merely a convenience, and it "
     "is also the strongest caution against the Qwen candidate.",
   "configurations": local,
   "best_local_result": "mean CER 0.944 (dedicated Hebrew word-level HTR); best single case 0.62",
   "cloud_comparison": {"gemini_successful_only_cer": 0.1155, "sonnet_successful_only_cer": 0.4718},
   "reading": "every local option - Qwen3-VL 8B (Q4_K_M and q8_0), Qwen3-VL 30B-a3b, a dedicated "
     "Hebrew handwriting HTR model, and surya document OCR - produced mean CER at or above 0.94 "
     "with fluent hallucinated Hebrew, across blue-channel isolation, text subtraction, line "
     "pre-segmentation and contrast preprocessing. The best local case is still 5x worse than "
     "Gemini's MEAN successful-only CER.",
   "consequence_for_the_qwen_candidate": "the same family at 8B and 30B-a3b failed on this exact "
     "handwriting. The 235B-a22b is a different capacity class and unquantised at Alibaba, which "
     "is why it remains worth one 8-case screen - but the prior is genuinely unfavourable and is "
     "recorded as the arm's main risk rather than omitted."},

 "if_no_candidate_survives": "see OCR_ALTERNATIVE_ARCHITECTURES_2026-09-03.json"
})
p = R / "OCR_CANDIDATE_DISCOVERY_2026-09-03.json"
p.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

# ------------------------------------------------------- architectures ------
arch = seal({
 "artifact": "ocr_alternative_architectures",
 "created_at": ts, "git_commit": commit, "provider_calls": 0,
 "status": "ANALYSIS ONLY - nothing implemented, nothing called, no architecture changed",
 "context": "A credible OpenRouter candidate does exist (the shortlist), so these are the "
            "fallbacks if the 8-case screen fails, not a present recommendation.",
 "options": [
  {"id": 1, "option": "direct OCR/document API instead of a general vision model "
                      "(e.g. Google Document AI, Azure Document Intelligence, AWS Textract)",
   "expected_benefit": "purpose-built handwriting OCR with per-word confidence scores, which is "
     "exactly the signal the current blocker needs: wrong digits inside a USABLE transcription are "
     "presently undetectable in production",
   "privacy": "student exam images leave the country/vendor boundary to a NEW third party; a fresh "
              "DPA question, not covered by the existing OpenRouter decision",
   "engineering_cost": "high - a new backend, new auth, new payload boundary, new ledger mapping",
   "recurring_cost": "per-page pricing, typically well above $0.0016/crop",
   "lock_in": "high (proprietary response schema)",
   "violates_openrouter_only_product_decision": True,
   "requires_owner_approval": True},

  {"id": 2, "option": "direct provider API (Google AI Studio / Vertex) instead of OpenRouter",
   "expected_benefit": "Vertex exposes CONFIGURABLE safety thresholds that OpenRouter does not "
     "surface. If the screen shows the filter is provider-specific, this is the mechanism that "
     "would let it be turned down for a legitimate educational corpus.",
   "privacy": "same vendor already receiving the crops; removes OpenRouter as an intermediary, "
              "which arguably IMPROVES the data path",
   "engineering_cost": "medium - one new backend behind the existing gateway seam",
   "recurring_cost": "similar or lower (no OpenRouter margin)",
   "lock_in": "medium",
   "violates_openrouter_only_product_decision": True,
   "requires_owner_approval": True,
   "note": "the 8-case screen is the cheap precondition: pin the two providers first, and only "
           "pursue this if the filter turns out to be provider-specific"},

  {"id": 3, "option": "deterministic image preprocessing plus OCR",
   "expected_benefit": "NONE DEMONSTRATED - already measured and refuted in this project",
   "evidence": "four preprocessing strategies (blue-channel isolation, printed-text subtraction, "
     "line pre-segmentation, contrast) across 7 configurations: mean CER 1.00-1.23. Preprocessing "
     "moved the number by ~0.2 CER at best and never approached usability.",
   "privacy": "none (local)", "engineering_cost": "already spent", "recurring_cost": "$0",
   "lock_in": "none", "violates_openrouter_only_product_decision": False,
   "requires_owner_approval": False,
   "verdict": "REFUTED BY MEASUREMENT - do not re-attempt without a new idea"},

  {"id": 4, "option": "local OCR/VLM for transcription",
   "expected_benefit": "NONE DEMONSTRATED - already measured and refuted",
   "evidence": "qwen3-vl 8B (Q4_K_M, q8_0), qwen3-vl 30B-a3b, sivan22/hdd-words-ocr (a dedicated "
     "Hebrew handwriting HTR), and surya-ocr: best mean CER 0.944, all with fluent hallucination "
     "or degenerate repetition. Cloud Gemini reads the same corpus at 0.1155 when it answers.",
   "privacy": "BEST - nothing leaves the machine", "engineering_cost": "already spent",
   "recurring_cost": "$0", "lock_in": "none",
   "violates_openrouter_only_product_decision": False, "requires_owner_approval": False,
   "verdict": "REFUTED BY MEASUREMENT at the model scales tested. A materially larger local VLM on "
              "the RTX 2000 Ada is the only untested variant and is bounded by VRAM."},

  {"id": 5, "option": "human OCR verification for only machine-detected failures",
   "expected_benefit": "HIGH AND ALREADY HALF-BUILT. This is the one option the measurements "
     "actively support: Gemini's failures are LOUD (content_filter, parse, truncation) and "
     "machine-detectable, so they can be routed to a human deterministically. It converts an "
     "accuracy problem into a bounded review-queue problem.",
   "cost_shape": "at 16/32 usable, roughly half of all crops enter review - expensive in human "
                 "time, and the review burden is the honest headline number",
   "privacy": "none beyond the current design", "engineering_cost": "medium - the review UI and "
              "the labeling app already exist", "recurring_cost": "human time",
   "lock_in": "none", "violates_openrouter_only_product_decision": False,
   "requires_owner_approval": True,
   "critical_caveat": "this does NOT address the standing blocker. A wrong digit inside a USABLE "
     "transcription is silent: it is not flagged, so it never reaches the review queue. Failure "
     "routing fixes coverage, not correctness."},

  {"id": 6, "option": "hybrid local/cloud OCR with explicit provenance",
   "expected_benefit": "low on current evidence - the local half has no measured accuracy to "
     "contribute, so a hybrid inherits the cloud arm's numbers plus complexity",
   "privacy": "partial improvement only if the local arm handles real cases, which it cannot yet",
   "engineering_cost": "high", "recurring_cost": "cloud cost unchanged", "lock_in": "low",
   "violates_openrouter_only_product_decision": False, "requires_owner_approval": True,
   "verdict": "premature - revisit only if a local model ever reaches usable CER"}],

 "require_owner_approval": [1, 2, 5, 6],
 "no_architecture_changed_in_this_task": True
})
p2 = R / "OCR_ALTERNATIVE_ARCHITECTURES_2026-09-03.json"
p2.write_text(json.dumps(arch, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("wrote", p); print("wrote", p2)
print("discovery sha", doc["content_sha256"][:16], "| arch sha", arch["content_sha256"][:16])
print("local configs summarized:", len(local))
