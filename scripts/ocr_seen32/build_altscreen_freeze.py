"""Freeze OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1. ZERO provider calls."""
import json, hashlib, pathlib, subprocess, time

from autograder.benchmark.roles import load_ocr_prompts

BASE = json.loads(pathlib.Path("evaluation/model_selection/runs/ocr_primary/"
                               "OCR_SMOKE_STAGE1_BASELINE_2026-09-02.json").read_text(encoding="utf-8"))
CAT = json.loads(pathlib.Path("evaluation/model_selection/runs/ocr_primary/OCR_OPENROUTER_CATALOG_SNAPSHOT_2026-09-03.json").read_text(encoding="utf-8"))
by_model = {m["id"]: m for m in CAT["models"]}
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

P = load_ocr_prompts("m2-strict-v1")
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
CATS = ["handwritten_line", "handwritten_cell", "formula_printed", "mixed_he_en",
        "option_row_association"]

GEM, QWEN = "google/gemini-3.7-flash", "qwen/qwen3-vl-235b-a22b-instruct"
GIN, GOUT = 0.75e-6, 3.75e-6
QIN, QOUT = 0.21e-6, 1.90e-6
GEM_IN, GEM_OUT = 1388, 298          # measured on THESE 8 crops (stage1b/1c ledger)
QWEN_IN_EST = 3000                   # conservative; qwen image tokenisation is unmeasured here
MAXTOK = 1000


def arm_cost(inp, outp, pin, pout, n=8):
    return {"worst_case_usd": round(n * (inp * pin + MAXTOK * pout), 6),
            "expected_all_succeed_usd": round(n * (inp * pin + outp * pout), 6)}


arms = [
 {"arm_id": "gemini_pinned_ai_studio",
  "model": GEM, "provider_pin": "google-ai-studio",
  "provider_routing": {"order": ["google-ai-studio"], "allow_fallbacks": False},
  "why_genuinely_different": "Not a new model but a DETERMINISTIC ROUTE. The live catalog lists 6 "
    "endpoints across 2 distinct serving providers for this slug; both 32-crop arms ran under "
    "AUTOMATIC routing and were served by a mix (the control arm recorded 3 rows from Google AI "
    "Studio, the neutral arm 0), and NO content-filtered row in ANY Gemini run records which "
    "provider produced it. Whether the filter is provider-specific is therefore UNMEASURED. This "
    "arm and its Vertex twin are the only way to find out, and they preserve the best conditional "
    "transcription quality measured anywhere in this project (0.1155).",
  "route": {"task": "ocr_primary", "backend": "openrouter", "model": GEM,
            "structured_mode": "json_schema", "max_tokens": MAXTOK, "temperature": 0.0,
            "reasoning": {"effort": "low"}, "prompt_version": "m2-strict-v1"},
  "reasoning_requirement": "MANDATORY (catalog: reasoning.mandatory=true, lowest supported effort "
                           "'low'); effort=none is rejected with HTTP 400",
  "estimated_cost": arm_cost(GEM_IN, GEM_OUT, GIN, GOUT),
  "main_risk": "the filter may be model-level rather than provider-level, in which case both "
               "Gemini arms fail together and the route hypothesis is closed - which is itself a "
               "decisive result",
  "unknowns": ["whether OpenRouter honours the pin without silent fallback (allow_fallbacks=false "
               "is set; the recorded provider field is checked per row after the run)"]},

 {"arm_id": "gemini_pinned_vertex",
  "model": GEM, "provider_pin": "google-vertex",
  "provider_routing": {"order": ["google-vertex"], "allow_fallbacks": False},
  "why_genuinely_different": "The paired half of the route question. Without both pins a filter "
    "count cannot be attributed to a provider, so this arm is what makes the other interpretable.",
  "route": {"task": "ocr_primary", "backend": "openrouter", "model": GEM,
            "structured_mode": "json_schema", "max_tokens": MAXTOK, "temperature": 0.0,
            "reasoning": {"effort": "low"}, "prompt_version": "m2-strict-v1"},
  "reasoning_requirement": "MANDATORY, effort 'low'",
  "estimated_cost": arm_cost(GEM_IN, GEM_OUT, GIN, GOUT),
  "main_risk": "same as the AI Studio arm",
  "unknowns": ["Vertex exposes configurable safety thresholds that OpenRouter does not surface; a "
               "difference between the two arms would be suggestive, not mechanistic"]},

 {"arm_id": "qwen3_vl_235b_pinned_alibaba",
  "model": QWEN, "provider_pin": "alibaba",
  "provider_routing": {"order": ["alibaba"], "allow_fallbacks": False},
  "why_genuinely_different": "A different model family, vendor and serving provider from every "
    "dropped arm. It is the ONLY model in the live catalog whose official description documents "
    "document parsing and chart/table extraction, and it is not moderated by OpenRouter. Pinned to "
    "Alibaba because the slug is served by 5 providers at differing quantisations (DeepInfra fp8, "
    "Venice fp8, Parasail fp8, Novita bf16), which would otherwise make the arm non-reproducible.",
  "route": {"task": "ocr_primary", "backend": "openrouter", "model": QWEN,
            "structured_mode": "json_schema", "max_tokens": MAXTOK, "temperature": 0.0,
            "reasoning": None, "prompt_version": "m2-strict-v1"},
  "reasoning_requirement": "NONE (catalog: no reasoning object; reasoning is not a supported "
                           "parameter). max_tokens 1000 is therefore all visible budget - roughly "
                           "4x the 248 tokens the longest frozen reference (116 chars) needs at "
                           "the worst observed 1.78 tok/char, so truncation is not expected.",
  "estimated_cost": arm_cost(QWEN_IN_EST, 512, QIN, QOUT),
  "main_risk": "RECORDED AGAINST IT, NOT HIDDEN: this project measured the SAME FAMILY on this "
    "corpus locally in 2026-07 - qwen3-vl 8b (Q4_K_M and q8_0) and 30b-a3b, across 14 "
    "configurations including blue-channel isolation, text subtraction, line pre-segmentation and "
    "contrast preprocessing - at mean CER 1.0-1.6 with fluent hallucinated Hebrew. The 235b-a22b "
    "is a different capacity class (22B active, unquantised at Alibaba) and that is the "
    "hypothesis; it is NOT evidence the family works on this handwriting.",
  "unknowns": ["qwen image tokenisation on 1287px-wide crops is unmeasured here, so the input-token "
               "estimate is a conservative guess and the predicted ceiling absorbs it",
               "Hebrew handwriting performance at this scale is untested"]},
]

screen_worst = round(sum(a["estimated_cost"]["worst_case_usd"] for a in arms), 6)
screen_expected = round(sum(a["estimated_cost"]["expected_all_succeed_usd"] for a in arms), 6)

doc = {
 "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1",
 "status": "FROZEN - NOT EXECUTED. Requires explicit owner authorization and a budget grant.",
 "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
 "git_commit": commit,
 "provider_calls_made_preparing_this": 0,
 "catalog_snapshot": {"source": "https://openrouter.ai/api/v1/models and /models/{slug}/endpoints "
                                "and /providers (PUBLIC metadata; no inference)",
                      "fetched_at_utc": CAT["fetched_at_utc"],
                      "models_in_catalog": CAT["models_in_full_catalog"]},

 "question": "Is Gemini's content-filter loss a property of the MODEL, of the SERVING PROVIDER "
             "route, or of this content generally - and does a genuinely different model family "
             "read this handwriting at all?",

 "population": {
   "source": "the exact frozen Stage-1 eight cases, unchanged",
   "smoke_selection_sha256": BASE["smoke_selection_sha256"],
   "manifest_hashes": BASE["manifest_hashes"],
   "n": 8, "handwritten": 5, "printed_or_text_layer": 3,
   "splits": ["DEV"], "CALIBRATION": 0, "HELD_OUT": 0,
   "ordered_case_ids": BASE["ordered_case_ids"],
   "cases": BASE["cases"],
   "case_order_sha256": hashlib.sha256("\n".join(BASE["ordered_case_ids"]).encode()).hexdigest(),
   "logical_order_diagnostic_view": "unchanged - OCR_AUDITED_LOGICAL_ORDER_2026-09-02.json"},

 "prompt": {
   "version": "m2-strict-v1",
   "why_this_and_not_a_new_one": "The screen changes MODEL and ROUTE. Introducing a new prompt "
     "would add a second variable and break comparability with Stage-1, Stage-1b, Stage-1c and the "
     "32-crop control, all of which ran m2-strict-v1. ocr-neutral-v2 is deliberately NOT carried "
     "forward: it raised filtering 10 -> 14 and its 'ignore any marks written in a different "
     "colour of ink' clause is LIVE on 19/32 crops of the seen population, where the ink is blue "
     "handwriting over printed black rules. m2-strict-v1's 'ignore any red instructor ink' clause "
     "is a proven no-op here (a deterministic pixel audit found 0/32 crops with red), so it adds "
     "no instruction the model can act on.",
   "colour_wording_inherited": False,
   "identical_semantic_contract_across_all_three_arms": True,
   "prompt_sha256_by_category": {c: sha(P[c]) for c in CATS},
   "contract": ["copy only visible text", "preserve spelling, grammar, incorrect terminology and "
                "mathematical mistakes", "preserve digits, signs, operators and negation",
                "[?] for one unreadable word, [unreadable] for all of it",
                "struck-through text is cancelled", "logical RTL order",
                "no invention, no completion, no paraphrase, no grading"],
   "case_specific_instructions": 0,
   "rubric_or_solution_or_grade_context": 0},

 "schema": {"name": BASE["schema_name"], "sha256": BASE["schema_sha256"],
            "shape": "{\"transcription\": \"<text>\"} - minimal; unreadable status carried by the "
                     "frozen [?] / [unreadable] markers; no explanation or reasoning prose requested"},

 "adapter_version": BASE["adapter_version"],
 "candidates": arms,
 "candidate_count": len(arms),

 "live_pricing_snapshot": {
   GEM: {"input_per_M": 0.75, "output_per_M": 3.75,
         "is_moderated": (by_model[GEM].get("top_provider") or {}).get("is_moderated"),
         "max_completion_tokens": (by_model[GEM].get("top_provider") or {}).get("max_completion_tokens"),
         "context_length": by_model[GEM].get("context_length")},
   QWEN: {"input_per_M": 0.21, "output_per_M": 1.90,
          "is_moderated": (by_model[QWEN].get("top_provider") or {}).get("is_moderated"),
          "max_completion_tokens": (by_model[QWEN].get("top_provider") or {}).get("max_completion_tokens"),
          "context_length": by_model[QWEN].get("context_length")}},

 "budget": {
   "screen": {"predicted_worst_case_usd": screen_worst,
              "expected_all_succeed_usd": screen_expected,
              "proposed_predicted_ceiling_usd": 0.12,
              "proposed_actual_ceiling_usd": 0.08,
              "ceiling_rule_applied": "the PREDICTED ceiling is set above the dry-run worst case "
                "(every response filling max_tokens) and the ACTUAL ceiling above the all-succeed "
                "projection - the lesson from OCR_PROMPT_V2, where an $0.08 actual ceiling would "
                "have aborted the arm at 26 of 32 crops precisely when the treatment worked."},
   "follow_up_32_case_per_surviving_candidate": {
     "gemini_worst_case_usd": round(32 * (GEM_IN * GIN + MAXTOK * GOUT), 6),
     "gemini_expected_usd": round(32 * (GEM_IN * GIN + GEM_OUT * GOUT), 6),
     "qwen_worst_case_usd": round(32 * (QWEN_IN_EST * QIN + MAXTOK * QOUT), 6),
     "qwen_expected_usd": round(32 * (QWEN_IN_EST * QIN + 512 * QOUT), 6),
     "proposed_predicted_ceiling_usd": 0.20,
     "proposed_actual_ceiling_usd": 0.14},
   "project_wide_unchanged": {"warn_usd": 8.0, "hard_usd": 10.0,
                              "cumulative_spend_to_date_usd": 0.703232},
   "historically_lower_actual_was_NOT_used_to_lower_the_predicted_ceiling": True},

 "advancement_and_drop_rules_stated_in_advance": {
   "scope": "This is an ENGINEERING SCREEN on 8 cases. Passing authorizes ONE thing: the 32-crop "
            "seen experiment. It is not production proof and n=8 cannot establish a failure rate.",
   "operational_coverage": {"usable_total": ">= 7/8", "usable_handwritten": ">= 4/5"},
   "reliability": {"repeated_provider_filter_pattern": "not permitted",
                   "hard_provider_failures": "<= 1",
                   "repeated_model_text_refusal": "not permitted",
                   "fabrication": "0"},
   "handwriting_quality": {
     "successful_only_handwritten_mean_cer": "<= 0.20",
     "threshold_provenance": "the same 0.20 bar the OCR_PROMPT_V2 pre-registration used as its "
       "quality veto, so the screens stay comparable. It is materially below Sonnet's measured "
       "successful-output CER of 0.4718 and above Gemini's 0.1155, i.e. it separates the two.",
     "total_line_loss": "0",
     "labelled": "engineering screen, NOT production proof"},
   "critical_errors": {"max_critical_among_usable_handwriting": 1,
                       "fabricated_mathematical_content": 0},
   "gates_are_frozen": "these thresholds are fixed before any output and must not be weakened "
                       "after seeing results",
   "passing_authorizes": "the 32-crop seen OCR experiment for that candidate only",
   "passing_does_not_establish": "production readiness, a route selection, or any claim about "
                                 "HELD_OUT"},

 "payload_and_boundary_verification": {
   "payloads_built_offline": 24, "provider_calls": 0,
   "image_blocks_per_payload": 1, "text_blocks_per_payload": 0,
   "reference_leakage": 0, "grading_vocabulary_hits": 0, "secret_patterns": 0,
   "all_five_category_prompts_registered_in_cloud_boundary": True,
   "research_authorization": "exact task + exact model; wildcards rejected by construction",
   "held_out_in_payloads": 0},

 "prohibitions": ["no HELD_OUT", "no CALIBRATION", "no grading on OCR output", "no RAG",
                  "no OCR verification", "no composite fallback", "no reference edits",
                  "no prompt tuning after observing results", "no candidate added after results"],

 "not_executed": "This freeze makes ZERO provider calls. The exact command for each arm is "
                 "recorded in prepared_commands and has deliberately NOT been run.",
 "prepared_commands": [
   {"arm": a["arm_id"],
    "command": ("python -m autograder.cli bench run --role ocr_primary --split dev "
                f"--candidate {a['model']} --subset smoke --prompt-version m2-strict-v1 "
                "--research --models-config models.toml "
                "--runs-root evaluation/model_selection/runs_altscreen "
                f"--provider '{json.dumps(a['provider_routing'], separators=(',', ':'))}' "
                "--warn-usd <ledger+0.08> --hard-usd <ledger+0.12> "
                "--i-understand-this-spends-money")}
   for a in arms],
}
body = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True, default=str)
doc["experiment_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p = pathlib.Path("evaluation/model_selection/experiments/OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1_2026-09-03.json")
p.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("wrote", p)
print("experiment_sha256", doc["experiment_sha256"])
print("screen predicted worst case $%.6f | expected all-succeed $%.6f" % (screen_worst, screen_expected))
for a in arms:
    print(f"  {a['arm_id']:32s} worst ${a['estimated_cost']['worst_case_usd']:.6f}  "
          f"expected ${a['estimated_cost']['expected_all_succeed_usd']:.6f}")
