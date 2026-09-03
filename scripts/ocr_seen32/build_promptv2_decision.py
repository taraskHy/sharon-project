import json, hashlib, pathlib, subprocess, time

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
E = pathlib.Path("evaluation/model_selection/experiments")
res = json.loads((R / "OCR_PROMPT_V2_PAIRED_RESULT_2026-09-03.json").read_text(encoding="utf-8"))
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ts = time.strftime("%Y-%m-%d %H:%M:%S")


def seal(doc, field="content_sha256"):
    body = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    doc[field] = hashlib.sha256(body.encode()).hexdigest()
    return doc


# ---------------------------------------------------------------- decision ---
dec = seal({
 "artifact": "ocr_primary_route_decision_record",
 "created_at": ts, "git_commit": commit, "provider_calls": 0,
 "supersedes": "the OCR-route portion of OCR_SHIPMENT_READINESS_2026-09-02.json; that artifact "
               "stays as written and is not edited",
 "decision": "DROP google/gemini-3.7-flash as the primary OCR route on this OpenRouter path",
 "authority": "the drop rule pre-registered in OCR_PROMPT_V2_NEUTRAL_FRAMING_2026-09-02.json "
              "(experiment_sha256 05185839e204ca61...), stated BEFORE any request and not changed "
              "after seeing results: 'hard failures >= 10/32 -> the prompt is NOT the cause; "
              "Gemini's filter behaviour on this content is intrinsic at this price point'",
 "evidence": {
   "control_m2-strict-v1": {"usable": "14/32", "provider_content_filter": 10, "hard_failures": 18,
                            "successful_cer": 0.1155, "failure_aware_cer": 0.6130,
                            "critical_errors": "2/14", "cost_usd": 0.04492875},
   "treatment_ocr-neutral-v2": {"usable": "16/32", "provider_content_filter": 14, "hard_failures": 16,
                                "successful_cer": 0.1608, "failure_aware_cer": 0.5804,
                                "critical_errors": "2/16", "cost_usd": 0.051831},
   "hypothesis": "neutralising exam/grading framing materially reduces provider-filter outcomes",
   "verdict": "REFUTED, and in the wrong direction - filters rose 10 -> 14 of 32",
   "usable_delta_not_significant": {"exact_mcnemar_p": res["paired_test"]["p_value"],
                                    "discordant_pairs": res["paired_test"]["discordant"]},
   "quality_like_for_like": res["matched_pairs_quality"],
   "coverage_gain_source": "not fewer filters - JSON parse failures fell 6 -> 1 and truncation 2 -> 1",
   "annotation_contamination": "0 in BOTH arms; the Phase-2 guard held and there is nothing to fix"
 },
 "what_is_NOT_concluded": [
   "no claim that a different provider will do better - that is the next experiment, not this result",
   "no production-readiness claim for any OCR configuration",
   "no claim about a true failure rate below ~9%: n=32 cannot support one",
   "the cell-vs-line swap (cell 5->9 usable, line 9->7, line filter 2->8) is a HYPOTHESIS for a "
   "future freeze: each half is n=16 and the two categories received different prompt edits"
 ],
 "standing_prohibitions_reaffirmed": [
   "the Gemini->Sonnet composite fallback stays WITHDRAWN (strictly dominated; converts loud "
   "machine-detectable failures into plausible wrong text)",
   "AI prefill remains rejected by the gate",
   "instructor grade = reference truth; audits are flags-only"
 ],
 "open_blocker_unchanged": "wrong digit / sign / operator / negation in a usable transcription is "
                           "NOT detectable in production (no reference exists there). This remains "
                           "the blocking failure mode for unreviewed use, independent of route.",
 "ocr_primary_role_status": "UNSELECTED (unchanged) - dropping a candidate does not select one"
})
p1 = R / "OCR_PRIMARY_ROUTE_DECISION_2026-09-03.json"
p1.write_text(json.dumps(dec, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

# ------------------------------------------------------- next experiment -----
nxt = seal({
 "experiment": "OCR_ALTERNATE_PROVIDER_SEEN32",
 "status": "DRAFT - NOT FROZEN, NOT AUTHORIZED, NOT EXECUTED",
 "created_at": ts, "git_commit": commit, "provider_calls": 0,
 "why_this_experiment": "the pre-registered drop rule for gemini-3.7-flash fired (16/32 hard "
                        "failures). Its option D names the successor: evaluate a genuinely "
                        "DIFFERENT vision/OCR provider under a new freeze.",
 "BLOCKED_ON_OWNER": {
   "what_is_needed": "the owner must name and register the candidate model. This draft deliberately "
                     "does NOT pick one.",
   "why_the_model_cannot_be_chosen_here": "the registered ocr_primary candidates are exhausted: "
                                          "google/gemini-3.7-flash is now DROPPED; openai/gpt-5.6-luna "
                                          "was DROPPED at Stage-1 and its base slug is unpriced; "
                                          "anthropic/claude-sonnet-5 is already measured on this exact "
                                          "population (27/32 usable, CER 0.4718, 9/27 critical) and is "
                                          "not a genuinely different provider path from the paired arm.",
   "requirements_for_the_named_candidate": [
     "a vision model on a DIFFERENT provider path, not another OpenRouter Google/Anthropic route",
     "priced in models.toml - an unpriced slug cannot be budget-gated and will not run",
     "registered in candidates.toml under roles.ocr_primary",
     "reasoning/max_tokens overrides derived and recorded BEFORE the first call, as Stage-1b did"
   ]
 },
 "design_when_authorized": {
   "population": "the SAME frozen 32 seen-DEV handwritten crops, same order, same crop bytes, same "
                 "audited references (case_order_sha256 "
                 "7aeb6cffa515b2f296a7c944c8622040779ce2c06365be57842808b56c2dd4d1)",
   "prompt": "m2-strict-v1 - the CONTROL prompt. ocr-neutral-v2 is not carried forward: it did not "
             "reduce filtering and its clause is live on 19/32 crops in this population.",
   "arms": 1, "split": "DEV", "CALIBRATION": 0, "HELD_OUT": 0,
   "metrics": "unchanged - the same taxonomy, CER/WER, critical-error family and "
              "annotation_inclusion_error used by both existing arms",
   "comparison": "paired, case by case, against BOTH existing Gemini arms and the Sonnet arm"
 },
 "drop_rules_to_be_stated_before_execution": {
   "note": "these are PROPOSED and must be fixed in the frozen artifact before any request",
   "DROP": "hard failures >= 10/32 - same threshold as the Gemini rule, for comparability",
   "ADVANCE": "hard failures <= 4/32 AND successful-only CER <= 0.20",
   "REPORT_ONLY": "5 to 9 hard failures",
   "quality_veto": "successful-only CER > 0.20 rejects the arm regardless of coverage"
 },
 "budget_note": {
   "lesson_from_this_experiment": "set the PREDICTED ceiling against the dry-run worst case (all "
                                  "responses filling max_tokens), and set the ACTUAL ceiling above "
                                  "the all-succeed projection - otherwise success itself aborts the "
                                  "run mid-arm and yields a truncated result the drop rule cannot "
                                  "evaluate.",
   "observed_ratio_this_arm": {"predicted_worst_case_usd": 0.150003, "actual_usd": 0.051831,
                               "actual_over_predicted": 0.3455}
 },
 "prohibitions": ["no HELD_OUT", "no CALIBRATION", "no grading on OCR output", "no RAG",
                  "no OCR verification", "no fallback composite", "no reference edits",
                  "no prompt tuning after observing results"]
})
p2 = E / "OCR_ALTERNATE_PROVIDER_SEEN32_DRAFT_2026-09-03.json"
p2.write_text(json.dumps(nxt, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

print("wrote", p1)
print("wrote", p2)
