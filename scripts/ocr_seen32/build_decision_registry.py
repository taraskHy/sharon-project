import json, hashlib, pathlib, subprocess, time

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
doc = {
 "artifact": "ocr_decision_registry",
 "version": 1,
 "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
 "git_commit": commit,
 "purpose": "Research guard. Records the OCR model/prompt/route configurations that have been "
            "measured and ruled out, so a dropped arm cannot be re-run or reported as a winner by "
            "accident. This is NOT production routing and does not select anything.",
 "route_identity_rule": "A configuration is (model, prompt_version, provider_pin). provider_pin "
                        "null means AUTOMATIC OpenRouter routing; '*' matches any routing. An "
                        "explicit single-provider pin with allow_fallbacks=false is a DIFFERENT "
                        "configuration and does not inherit a drop recorded against automatic "
                        "routing - see distinct_routes_explicitly_not_dropped, which lists the "
                        "pins that are open, so a pin can never quietly evade a drop either.",
 "current_winner": None,
 "current_winner_note": "No OCR route is selected. roles.ocr_primary remains UNSELECTED. Dropping "
                        "candidates does not select one.",
 "entries": [
  {"id": "LUNA_ALL_ROUTES",
   "status": "DROP",
   "match": {"model": "openai/gpt-5.6-luna-pro", "prompt_version": "*", "provider_pin": "*"},
   "reason": "Refused 4 of 5 handwritten crops with an unreadable marker against readable audited "
             "references, and fabricated the fifth as fluent Hebrew unrelated to the image.",
   "evidence": "OCR_SMOKE_STAGE1C_RESULT_2026-09-02.json - failure-aware handwritten CER 0.9487, "
               "6/8 classification criteria passed, classification DROP",
   "measured": {"handwritten_failure_aware_cer": 0.9487, "unreadable_on_readable": 4,
                "fabrications": 1, "handwritten_denominator": 5},
   "superseded_by": None},

  {"id": "SONNET_ALL_ROUTES",
   "status": "HISTORICAL_CONTROL_ONLY",
   "match": {"model": "anthropic/claude-sonnet-5", "prompt_version": "*", "provider_pin": "*"},
   "reason": "Higher output coverage than Gemini but its successful-output handwriting CER stays "
             "near 0.47 with frequent critical errors, and it was not a useful fallback for "
             "Gemini's hard failures. Retained as a reliability/comparison arm only; NOT suitable "
             "for automatic OCR.",
   "evidence": "OCR_SEEN32_PAIRED_RESULT_2026-09-02.json - 27/32 usable, successful CER 0.4718, "
               "failure-aware CER 0.5544, 9/27 critical errors, 4 model-text refusals",
   "measured": {"usable": "27/32", "successful_cer": 0.4718, "failure_aware_cer": 0.5544,
                "critical_errors": "9/27", "model_text_refusals": 4},
   "superseded_by": None},

  {"id": "GEMINI_M2_STRICT_V1_AUTOROUTE",
   "status": "DROP_AS_PRIMARY_ROUTE",
   "match": {"model": "google/gemini-3.7-flash", "prompt_version": "m2-strict-v1",
             "provider_pin": None},
   "reason": "Best conditional transcription quality measured anywhere in this project, but only "
             "14 of 32 handwritten crops produced usable output; 10 were lost to provider content "
             "filtering. Coverage, not capability, is the disqualifier.",
   "evidence": "OCR_SEEN32_PAIRED_RESULT_2026-09-02.json",
   "measured": {"usable": "14/32", "provider_content_filter": 10, "other_hard_failures": 8,
                "successful_cer": 0.1155, "failure_aware_cer": 0.6130, "critical_errors": "2/14"},
   "superseded_by": None},

  {"id": "GEMINI_OCR_NEUTRAL_V2_AUTOROUTE",
   "status": "DROP_AS_PRIMARY_ROUTE",
   "match": {"model": "google/gemini-3.7-flash", "prompt_version": "ocr-neutral-v2",
             "provider_pin": None},
   "reason": "The neutral-framing hypothesis was refuted in the wrong direction: provider content "
             "filtering ROSE from 10 to 14 of 32. The pre-registered drop rule (>= 10 hard "
             "failures of 32) fired at 16/32. The small coverage gain came from fewer parse and "
             "truncation failures, not fewer filter outcomes.",
   "evidence": "OCR_PROMPT_V2_PAIRED_RESULT_2026-09-03.json, drop rule pre-registered in "
               "OCR_PROMPT_V2_NEUTRAL_FRAMING_2026-09-02.json (05185839e204ca61...)",
   "measured": {"usable": "16/32", "provider_content_filter": 14, "other_hard_failures": 2,
                "successful_cer": 0.1608, "failure_aware_cer": 0.5804, "critical_errors": "2/16",
                "annotation_inclusion_errors": 0, "exact_mcnemar_p": 0.726562},
   "superseded_by": None},

  {"id": "GEMINI_THEN_SONNET_FALLBACK_V1",
   "status": "REJECTED",
   "match": {"model": "gemini_then_sonnet_hard_failure_fallback_v1", "prompt_version": "*",
             "provider_pin": "*"},
   "reason": "Strictly dominated. It did not reduce the human-review requirement, its output on "
             "triggered cases was identical to the Sonnet-only arm, it recovered about 20% of "
             "reference words with several zero-word recoveries, no triggered case met a 10% CER "
             "bar, it roughly doubled cost, and it converted explicit machine-detectable Gemini "
             "failures into fluent but potentially undetectable transcription errors.",
   "evidence": "OCR_SEEN32_PAIRED_RESULT_2026-09-02.json decision_revised.composite = "
               "'NOT USEFUL - strictly dominated by Gemini-only. Withdrawn as a recommendation.'",
   "measured": {"rescue_rate": 0.8333, "rescue_quality_succ_mean_cer": 0.4516,
                "met_10pct_cer_bar": 0},
   "do_not_revive_without": "genuinely new evidence, not a re-analysis of these runs",
   "superseded_by": None}
 ],

 "distinct_routes_explicitly_not_dropped": [
  {"model": "google/gemini-3.7-flash", "prompt_version": "m2-strict-v1",
   "provider_pin": "google-ai-studio",
   "why_distinct": "A single pinned serving provider with allow_fallbacks=false. The catalog lists "
                   "6 endpoints across 2 distinct providers for this slug; both 32-crop arms ran "
                   "under AUTOMATIC routing and were served by a mix, and no content-filtered row "
                   "in any arm records its provider. Whether the filter is provider-specific is "
                   "therefore UNMEASURED, not settled.",
   "status": "OPEN - to be measured by OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1"},
  {"model": "google/gemini-3.7-flash", "prompt_version": "m2-strict-v1",
   "provider_pin": "google-vertex",
   "why_distinct": "The paired half of the same route question; without both pins the filter "
                   "cannot be attributed to a provider.",
   "status": "OPEN - to be measured by OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1"},
  {"model": "qwen/qwen3-vl-235b-a22b-instruct", "prompt_version": "m2-strict-v1",
   "provider_pin": "alibaba",
   "why_distinct": "A different model family, vendor and serving provider entirely.",
   "status": "OPEN - to be measured by OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1"}
 ],

 "standing_conclusions": {
  "no_current_ocr_winner": True,
  "held_out_sealed": True,
  "production_grading_is_local_and_unchanged": True,
  "good_conditional_cer_is_not_enough_when_coverage_is_poor":
      "Gemini reads better than anything else measured (0.1155 successful-only CER) and is still "
      "unusable alone, because it only produced usable output on 14 of 32 crops. A metric "
      "conditioned on success hides the cases that never produced one.",
  "fluent_fallback_text_can_be_more_dangerous_than_an_explicit_failure":
      "A content_filter is loud and machine-detectable, so the crop routes to a human. A fluent "
      "wrong transcription is silent and reaches the grader. The composite fallback traded the "
      "first for the second and was rejected for it."
 },
 "prohibitions_reaffirmed": [
  "the Gemini->Sonnet composite fallback stays withdrawn",
  "AI prefill remains rejected by the gate",
  "instructor grade = reference truth; audits are flags-only"
 ]
}
body = json.dumps({k: v for k, v in doc.items() if k != "content_sha256"},
                  ensure_ascii=False, indent=1, sort_keys=True, default=str)
doc["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p = pathlib.Path("evaluation/model_selection/policies/ocr_decision_registry.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("wrote", p)
print("content_sha256", doc["content_sha256"][:16])
