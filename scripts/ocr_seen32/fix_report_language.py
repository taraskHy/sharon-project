"""Task 4: correct report language WITHOUT changing any measurement.
Every edit below is a wording/claim-strength change; no number moves."""
import json, hashlib, pathlib

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
E = pathlib.Path("evaluation/model_selection/experiments")


def reseal(path, field="content_sha256"):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    d.pop(field, None)
    body = json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    d[field] = hashlib.sha256(body.encode()).hexdigest()
    pathlib.Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8", newline="\n")
    return d[field]


ATTRIB = ("is_moderated = false for this slug on OpenRouter, so these outcomes are NOT a declared "
          "OpenRouter moderation stage. They are consistent with an upstream model-side or "
          "provider-side content filter. Which of those it was, and on which endpoint, remains "
          "UNKNOWN: no content-filtered response in any historical arm records its provider.")

# ---- forensics -------------------------------------------------------------
f = R / "OCR_PROVIDER_ROUTE_FORENSICS_2026-09-03.json"
d = json.loads(f.read_text(encoding="utf-8"))
d["does_openrouter_offer_multiple_routes_for_this_exact_model"]["openrouter_model_level_moderation"] = ATTRIB
d["historical_filter_attribution"] = {
    "claim": ATTRIB,
    "what_is_established": "the outcomes carried finish_reason=content_filter on HTTP 200, and "
                           "OpenRouter does not declare model-level moderation for this slug",
    "what_is_NOT_established": "the mechanism (model-side vs provider-side) and the exact serving "
                               "endpoint. Both are UNKNOWN and unrecoverable from these artifacts.",
    "language_correction": "an earlier draft attributed these definitively to 'Google's own safety "
                           "layer'. That overstated what the artifacts support and has been "
                           "corrected here. No measurement changed.",
}
f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("forensics reseal:", reseal(f)[:16])

# ---- architectures ---------------------------------------------------------
a = R / "OCR_ALTERNATIVE_ARCHITECTURES_2026-09-03.json"
d = json.loads(a.read_text(encoding="utf-8"))
for opt in d["options"]:
    if opt["id"] == 3:
        opt["expected_benefit"] = ("NONE DEMONSTRATED - none of the tested preprocessing "
                                   "configurations was competitive")
        opt["verdict"] = ("NOT COMPETITIVE IN THE CONFIGURATIONS TESTED - four preprocessing "
                          "strategies across 7 configurations; do not re-attempt without a new idea")
    if opt["id"] == 4:
        opt["expected_benefit"] = ("NONE DEMONSTRATED - none of the 14 tested local configurations "
                                   "was competitive")
        opt["verdict"] = ("NOT COMPETITIVE IN THE CONFIGURATIONS TESTED. None of the 14 tested "
                          "local configurations was competitive at the model scales tried. A "
                          "materially larger local VLM on the RTX 2000 Ada is the only untested "
                          "variant and is bounded by VRAM.")
d["language_correction"] = ("'refuted' overstated the scope of a finite set of measurements: 14 "
                            "configurations were tested and none was competitive, which is not the "
                            "same as refuting local OCR as an architecture. Corrected throughout; "
                            "no measurement changed.")
a.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("architectures reseal:", reseal(a)[:16])

# ---- discovery -------------------------------------------------------------
c = R / "OCR_CANDIDATE_DISCOVERY_2026-09-03.json"
d = json.loads(c.read_text(encoding="utf-8"))
lp = d["local_prior_art_that_constrains_this_choice"]
lp["reading"] = lp["reading"].replace(
    "every local option - ", "none of the 14 tested local configurations was competitive - ")
lp["scope_of_the_claim"] = ("This is 14 measured configurations, not a proof about local OCR in "
                            "general. None was competitive; larger local VLMs remain untested and "
                            "are bounded by available VRAM.")
for s in d["shortlist"]:
    if "gemini" in s["candidate"]:
        s["arm_type"] = "PROVIDER-ROUTE ATTRIBUTION ARM"
        s["arm_type_note"] = ("This arm exists to attribute the content-filter outcome to a serving "
                              "provider, not to re-test the model. Gemini under automatic routing "
                              "is already dropped.")
    else:
        s["arm_type"] = "CROSS-FAMILY CANDIDATE ARM"
c.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("discovery reseal:", reseal(c)[:16])

# ---- the frozen screen -----------------------------------------------------
s = E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1_2026-09-03.json"
d = json.loads(s.read_text(encoding="utf-8"))
for arm in d["candidates"]:
    if arm["model"] == "google/gemini-3.7-flash":
        arm["arm_type"] = "PROVIDER-ROUTE ATTRIBUTION ARM"
        arm["arm_type_note"] = ("Measures WHICH SERVING PROVIDER produced the outcome, not whether "
                                "the model is acceptable. Gemini under automatic routing is "
                                "already DROP_AS_PRIMARY_ROUTE and this arm does not reopen that.")
    else:
        arm["arm_type"] = "CROSS-FAMILY CANDIDATE ARM"
g = d["advancement_and_drop_rules_stated_in_advance"]
g["passing_authorizes"] = ("ADVANCE_TO_SEEN32 for that candidate only - the 32-crop seen "
                           "experiment. Passing this screen CANNOT select a production winner.")
g["passing_does_not_establish"] = ("production readiness, a production winner, a route selection, "
                                   "or any claim about HELD_OUT. The only outcome a pass produces "
                                   "is ADVANCE_TO_SEEN32.")
g["outcome_vocabulary"] = {"pass": "ADVANCE_TO_SEEN32", "fail": "DROP",
                           "between": "REPORT_ONLY (reported, not resolved)"}
b = d["budget"]["screen"]
b["campaign_warning_increment_usd"] = 0.08
b["campaign_hard_increment_usd"] = 0.12
b["increment_semantics"] = ("$0.08 is the campaign WARNING increment and $0.12 the campaign HARD "
                            "increment, both applied ONCE to an immutable starting ledger L0 and "
                            "shared by all three arms - not a per-arm allowance. See "
                            "policies/OCR_ALTSCREEN_CAMPAIGN_BUDGET.json.")
b.pop("proposed_predicted_ceiling_usd", None)
b.pop("proposed_actual_ceiling_usd", None)
d["campaign_budget_manifest"] = "evaluation/model_selection/policies/OCR_ALTSCREEN_CAMPAIGN_BUDGET.json"
d["hardening_applied_2026-09-04"] = {
    "raw_response_preservation": "every provider reply of a live arm is archived sanitized to "
                                 "run_dir/raw_responses.jsonl BEFORE parsing, including HTTP 200 "
                                 "with no usage block and no provider field",
    "campaign_budget": "one immutable L0 with absolute thresholds shared by all three arms",
    "route_enforcement": "an EXPLICITLY different serving provider stops the arm; UNKNOWN "
                         "attribution does not",
}
# the freeze's own hash must be recomputed under its own field name
d.pop("experiment_sha256", None)
body = json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True, default=str)
d["experiment_sha256"] = hashlib.sha256(body.encode()).hexdigest()
s.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("screen re-hash:", d["experiment_sha256"])

# ---- registry --------------------------------------------------------------
reg = pathlib.Path("evaluation/model_selection/policies/ocr_decision_registry.json")
d = json.loads(reg.read_text(encoding="utf-8"))
for r in d["distinct_routes_explicitly_not_dropped"]:
    if r["model"] == "google/gemini-3.7-flash":
        r["arm_type"] = "PROVIDER-ROUTE ATTRIBUTION ARM"
    else:
        r["arm_type"] = "CROSS-FAMILY CANDIDATE ARM"
d["historical_filter_attribution"] = ATTRIB
reg.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("registry reseal:", reseal(reg)[:16])
