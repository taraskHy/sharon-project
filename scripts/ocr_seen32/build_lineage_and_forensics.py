import json, hashlib, pathlib, subprocess, time, collections, statistics, re

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
E = pathlib.Path("evaluation/model_selection/experiments")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ts = time.strftime("%Y-%m-%d %H:%M:%S")
CAT = json.loads(pathlib.Path("catalog_raw.json").read_text(encoding="utf-8"))


def seal(doc, field="content_sha256"):
    body = json.dumps({k: v for k, v in doc.items() if k != field},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    doc[field] = hashlib.sha256(body.encode()).hexdigest()
    return doc


led = [json.loads(l) for l in pathlib.Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl")
       .read_text(encoding="utf-8").splitlines() if l.strip()]


def route_forensics(job_id):
    rows = [r for r in led if r.get("job_id") == job_id]
    ct = collections.Counter((r.get("provider"), r.get("finish_reason")) for r in rows)
    provs = collections.Counter(r.get("provider") for r in rows if r.get("provider"))
    filtered = [r for r in rows if r.get("finish_reason") == "content_filter"]
    return {
      "ledger_rows": len(rows),
      "provider_x_finish_reason": {f"{p}|{f}": n for (p, f), n in sorted(ct.items(), key=lambda kv: str(kv[0]))},
      "distinct_providers_observed_on_SUCCESSFUL_rows": dict(provs),
      "content_filtered_rows": len(filtered),
      "content_filtered_rows_with_a_provider_recorded": sum(1 for r in filtered if r.get("provider")),
      "provider_of_filtered_rows": "UNKNOWN",
      "request_ids_preserved": sum(1 for r in rows if r.get("request_id")),
      "http_status": {str(k): v for k, v in collections.Counter(r.get("http_status") for r in rows).items()},
      "routing_mode": "AUTOMATIC (no provider pin: extra_generation carried no provider object)",
    }


# ------------------------------------------------------------------ lineage --
lineage = seal({
 "artifact": "ocr_experiment_lineage",
 "created_at": ts, "git_commit": commit, "provider_calls": 0,
 "purpose": "One ordered record of every OCR experiment, what it measured, and what superseded it.",
 "stages": [
  {"stage": "Stage-1", "date": "2026-09-02",
   "experiment_sha256": json.loads((R / "OCR_SMOKE_STAGE1_BASELINE_2026-09-02.json").read_text(encoding="utf-8"))["campaign_sha256"],
   "models": ["openai/gpt-5.6-luna-pro", "anthropic/claude-sonnet-5", "google/gemini-3.7-flash"],
   "prompt": "m2-strict-v1", "case_set": "frozen smoke 8 (5 handwritten, 3 printed)",
   "route": "openrouter, automatic provider routing",
   "intended": 24, "usable": "24/24 schema-valid responses",
   "failure_taxonomy": "Luna 4 unreadable-on-readable + 1 fabrication; Gemini arm INVALID - the "
                       "route recorded reasoning effort=none, which this endpoint rejects with "
                       "HTTP 400 (8/8 pre-inference rejections, nothing billed)",
   "cost_usd": 0.0268, "outcome": "Luna DROP; Sonnet MAYBE; Gemini UNMEASURED",
   "superseded_by": "Stage-1b (re-ran Gemini with a legal reasoning effort)"},

  {"stage": "Stage-1b", "date": "2026-09-02",
   "experiment_sha256": "4de29894cc25b0cc53c5268d2a2cee385151ddb094862d5d95f6eb0c6e7a1626",
   "models": ["google/gemini-3.7-flash"], "prompt": "m2-strict-v1",
   "case_set": "the same frozen smoke 8", "route": "openrouter, automatic",
   "intended": 8, "usable": "5/8",
   "failure_taxonomy": "harness defect discovered by this arm: the runner sent max_tokens 600 "
                       "while the route recorded 1000, so the cap under test never reached the wire",
   "outcome": "INVALID as a measurement of the configured cap",
   "superseded_by": "Stage-1c (corrected cap, proven on the wire)"},

  {"stage": "Stage-1c", "date": "2026-09-02",
   "experiment_sha256": "2be5224f49142dab7641c0be4ca6455c12c203e000dc3dd89d8f6feee442ffb1",
   "models": ["google/gemini-3.7-flash"], "prompt": "m2-strict-v1",
   "case_set": "the same frozen smoke 8", "route": "openrouter, automatic",
   "intended": 8, "usable": "5/8",
   "failure_taxonomy": "3 content-filter refusals, 2 of them on handwritten crops; "
                       "60% handwritten filter rate on a denominator of 5",
   "cost_usd": 0.01099875,
   "route_forensics": route_forensics("dev__smoke__all__google-gemini-3.7-flash__45297cdd83"),
   "outcome": "B - a bounded larger seen-only comparison is warranted",
   "superseded_by": "paired seen32"},

  {"stage": "paired seen32", "date": "2026-09-02",
   "experiment_sha256": json.loads((R / "OCR_SEEN32_PAIRED_RESULT_2026-09-02.json").read_text(encoding="utf-8"))["experiment_sha256"],
   "models": ["google/gemini-3.7-flash", "anthropic/claude-sonnet-5"], "prompt": "m2-strict-v1",
   "case_set": "frozen seen46_ocr_dev 32 (100% handwritten)", "route": "openrouter, automatic",
   "intended": 64,
   "usable": {"gemini": "14/32", "sonnet": "27/32"},
   "failure_taxonomy": {"gemini": {"provider_content_filter": 10, "json_parse_failure": 6,
                                   "truncation": 2, "total_line_loss": 18},
                        "sonnet": {"model_text_refusal": 4, "total_line_loss": 5}},
   "quality": {"gemini_successful_cer": 0.1155, "gemini_failure_aware_cer": 0.6130,
               "sonnet_successful_cer": 0.4718, "sonnet_failure_aware_cer": 0.5544},
   "cost_usd": 0.12703875,
   "route_forensics": route_forensics("dev__seen46_ocr_dev__all__google-gemini-3.7-flash__c4ae61f634"),
   "outcome": "Gemini DROP as sole route; Sonnet MAYBE (coverage, not quality); "
              "Gemini->Sonnet composite NOT USEFUL and withdrawn",
   "superseded_by": "neutral-framing seen32 (for the Gemini prompt question only)"},

  {"stage": "neutral-framing seen32", "date": "2026-09-03",
   "experiment_sha256": json.loads((E / "OCR_PROMPT_V2_NEUTRAL_FRAMING_2026-09-02.json").read_text(encoding="utf-8"))["experiment_sha256"],
   "models": ["google/gemini-3.7-flash"], "prompt": "ocr-neutral-v2",
   "case_set": "the SAME frozen 32 - identical ids, order, crop bytes and references",
   "route": "openrouter, automatic",
   "intended": 32, "usable": "16/32",
   "failure_taxonomy": {"provider_content_filter": 14, "json_parse_failure": 1, "truncation": 1,
                        "total_line_loss": 16, "annotation_inclusion_errors": 0},
   "quality": {"successful_cer": 0.1608, "failure_aware_cer": 0.5804, "critical_errors": "2/16",
               "matched_pairs_delta_cer": 0.0257, "matched_pairs_sign_test_p": 0.7266},
   "cost_usd": 0.051831,
   "route_forensics": route_forensics("dev__seen46_ocr_dev__all__google-gemini-3.7-flash__61dd6641fb"),
   "outcome": "hypothesis REFUTED in the wrong direction (filters 10 -> 14); pre-registered drop "
              "rule fired at 16/32 hard failures -> DROP_AS_PRIMARY_ROUTE",
   "one_variable_caveat_recorded_explicitly":
       "This was NOT a perfectly isolated 'remove instructor framing' experiment. The cell prompt "
       "also swapped its annotation clause: 'Ignore any red instructor ink' became 'Ignore any "
       "marks written in a different colour of ink from the main handwriting'. On this population "
       "the first clause is a NO-OP (0/32 crops carry red ink) while the second is LIVE on 19/32 "
       "crops (blue handwriting over printed black rules). The line prompt changed only the "
       "framing sentence. The arm therefore compares two framing PACKAGES, and the cell and line "
       "halves received different edits.",
   "superseded_by": None}],

 "aggregate_confirmations": {"held_out_executions": 0, "audited_references_modified": 0,
                             "active_grades_changed": 0,
                             "total_ocr_campaign_spend_usd": 0.703232}
})
p1 = R / "OCR_EXPERIMENT_LINEAGE_2026-09-03.json"
p1.write_text(json.dumps(lineage, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

# ------------------------------------------------- provider-route forensics --
gem_rows = [r for r in led if r.get("model") == "google/gemini-3.7-flash"]
allf = [r for r in gem_rows if r.get("finish_reason") == "content_filter"]
forensics = seal({
 "artifact": "ocr_provider_route_forensics",
 "created_at": ts, "git_commit": commit, "provider_calls": 0,
 "method": "existing run artifacts ONLY - gateway ledger rows, preserved response metadata, run "
           "configs and the request cache. No generation request was made.",
 "requested_model_slug": "google/gemini-3.7-flash",
 "backend": "openrouter (effective_provider 'openrouter' on all 145 Gemini ledger rows)",
 "routing_was_pinned_or_automatic": "AUTOMATIC in every Gemini experiment: every run config "
                                    "carried extra_generation {} and RunSpec.provider None, so no "
                                    "provider routing object was ever sent.",
 "distinct_serving_providers_observed": {
   "Google": 92, "Google AI Studio": 4, "not recorded (null)": 49,
   "note": "these are the values OpenRouter returned in the response body's `provider` field and "
           "the gateway preserved verbatim."},
 "did_all_filter_outcomes_come_from_the_same_provider": "UNKNOWN",
 "did_successful_and_failed_outputs_use_the_same_provider": "UNKNOWN",
 "why_unknown": {
   "finding": f"ALL {len(allf)} content-filtered Gemini rows across every experiment record "
              "provider = null. Not one filtered generation is attributable to a serving provider.",
   "mechanism": "the gateway parses `provider` from the response body (openrouter.py "
                "_usage_from_response). A content-filtered generation returns HTTP 200 with "
                "inference_reached true but no usage block (usage_returned false) and no provider "
                "field, so there is nothing to record.",
   "not_recoverable_from_cache": "the request cache stores only the PARSED, schema-validated value "
                                 "(a {'transcription': ...} object) plus a small meta dict - never "
                                 "the raw HTTP body. 627 cache entries carry no provider field. "
                                 "Filtered responses are not cached at all.",
   "consequence": "provider-specificity of the filter is UNMEASURED. It is not established, and it "
                  "is not ruled out."},
 "route_variation_between_the_two_32_crop_arms": {
   "control_m2-strict-v1": {"Google": 11, "Google AI Studio": 3, "null (filtered/errored)": 18},
   "neutral_ocr-neutral-v2": {"Google": 17, "Google AI Studio": 0, "null (filtered/errored)": 15},
   "finding": "automatic routing sent 3 control-arm requests to Google AI Studio and 0 neutral-arm "
              "requests there. The serving-provider mix therefore DIFFERED between the two arms of "
              "a paired experiment, uncontrolled.",
   "could_route_variation_explain_part_of_the_behaviour":
       "PARTLY, AND IT IS AN UNCONTROLLED CONFOUND. The filter count rose 10 -> 14 in the same "
       "comparison where AI Studio rows fell 3 -> 0. That is consistent with a provider effect, "
       "but it is equally consistent with chance on n=32 (the usable-count change itself was not "
       "significant, exact McNemar p=0.7266) or with a pure prompt effect. Because no filtered row "
       "names its provider, the two explanations cannot be separated from these artifacts. This is "
       "reported as a confound to be resolved by measurement, NOT as a finding.",
   "honest_status": "the paired neutral-vs-control conclusion (filtering did not improve) stands, "
                    "because it does not depend on provider attribution; but the MAGNITUDE of the "
                    "10 -> 14 change should not be read as a pure prompt effect."},
 "does_openrouter_offer_multiple_routes_for_this_exact_model": {
   "answer": "YES - verified against the live public catalog",
   "endpoints": 6,
   "distinct_providers": ["Google (google-vertex)", "Google AI Studio (google-ai-studio)"],
   "endpoint_tags": ["google-vertex/global", "google-vertex/global/flex",
                     "google-vertex/global/priority", "google-ai-studio",
                     "google-ai-studio/flex", "google-ai-studio/priority"],
   "materially_distinct": "YES - Vertex and AI Studio are different serving surfaces from the same "
                          "vendor with different safety-configuration surfaces, not aliases of one "
                          "endpoint. The three tags WITHIN each provider (flex/priority/plain) are "
                          "service tiers of the SAME provider and are NOT treated as independent.",
   "openrouter_model_level_moderation": "is_moderated = false for this slug, so the content_filter "
                                        "outcomes come from Google's own safety layer rather than "
                                        "an OpenRouter moderation stage."},
 "can_the_project_pin_a_provider_deterministically": {
   "answer": "YES",
   "mechanism": "RunSpec.provider -> route knobs -> BackendConfig.extra_generation['provider'] -> "
                "OpenRouter payload['provider'] (openrouter.py _build_payload). The field already "
                "existed and was already honoured; only the CLI never passed it through, which "
                "this task fixed by adding --provider.",
   "enters_the_config_hash": "YES - 'provider' is in the hashed route knobs, so a pinned arm "
                             "resolves a different run id and is a distinct frozen experiment.",
   "determinism_requires": "a single entry in `order` AND allow_fallbacks=false; anything else can "
                           "silently change the serving provider mid-run (enforced by "
                           "ocr_decisions.provider_pin_of, which reports such a route as unpinned)"},
 "classification": "A provider-pinned Gemini run IS a distinct route experiment and is classified "
                   "as such in the decision registry."
})
p2 = R / "OCR_PROVIDER_ROUTE_FORENSICS_2026-09-03.json"
p2.write_text(json.dumps(forensics, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

print("wrote", p1)
print("wrote", p2)
print("content_sha256 lineage :", lineage["content_sha256"][:16])
print("content_sha256 forensics:", forensics["content_sha256"][:16])
print("total content-filtered gemini rows with a provider recorded:",
      sum(1 for r in allf if r.get("provider")), "/", len(allf))
