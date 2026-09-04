import json, hashlib, pathlib, subprocess, time

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
RD = pathlib.Path("evaluation/model_selection/runs_altscreen/ocr_primary/"
                  "dev__smoke__all__google-gemini-3.7-flash__45297cdd83")
screen = json.loads(pathlib.Path("evaluation/model_selection/experiments/"
                                 "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1_2026-09-03.json"
                                 ).read_text(encoding="utf-8"))
raw = [json.loads(l) for l in (RD / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
outs = [json.loads(l) for l in (RD / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
led = [json.loads(l) for l in pathlib.Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl"
                                           ).read_text(encoding="utf-8").splitlines() if l.strip()]
mine = [x for x in led if x.get("job_id") == RD.name and (x.get("ts") or "") >= "2026-09-04"]
cum = sum(float(r.get("reported_cost") or 0) for r in led if r.get("cloud") and not r.get("cache_hit"))

doc = {
 "artifact": "ocr_altscreen_campaign_mechanical_stop",
 "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
 "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
 "campaign": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1",
 "experiment_sha256": screen["experiment_sha256"],
 "outcome": "INCONCLUSIVE_MECHANICAL_STOP",
 "arms_attempted": 1, "arms_completed_validly": 0, "arms_not_started": 2,
 "stopped_before_arm": ["gemini_pinned_vertex", "qwen3_vl_235b_pinned_alibaba"],

 "spend": {
   "L0": 0.70323229,
   "ending_ledger_usd": round(cum, 8),
   "additional_spend_usd": round(cum - 0.70323229, 8),
   "authorization_usd": 0.12,
   "within_authorization": (cum - 0.70323229) <= 0.12,
   "hard_threshold_usd": 0.82323229,
   "headroom_remaining_usd": round(0.82323229 - cum, 8),
   "ledger_rows_before": 798, "ledger_rows_after": len(led),
 },

 "stop_conditions_triggered": [
  {"condition": "missing or inconsistent attempt linkage",
   "evidence": "an UnboundLocalError in the runner's raw-capture setup (`campaign` referenced "
               "before assignment) made the runner treat capture as unavailable and set "
               "live_backend = None. Consequences: the PRE-SEND BUDGET HOOK was never installed, "
               "per-attempt route aggregation never ran, and NO logical row carries "
               "attempt_records. Archiving continued only incidentally, because raw_archive had "
               "already been assigned on the backend object before the exception fired - so the "
               "3 archived rows have campaign_id, arm_id and case_id all null.",
   "surfaced_as": "run warning: 'raw response capture unavailable: UnboundLocalError'",
   "severity": "the per-physical-attempt enforcement built and reported at 87e79bf DID NOT RUN "
               "in this arm"},

  {"condition": "unexpected provider/run configuration",
   "evidence": "TaskRoute.fingerprint_fields() does NOT include `provider`. The pinned arm "
               "therefore resolved config_hash 45297cdd83 - IDENTICAL to the historical UNPINNED "
               "Stage-1b/1c Gemini runs - so a pinned arm has no distinct run identity, and "
               "run.json (which records fingerprint_fields) cannot record which provider was "
               "requested.",
   "consequence_1": "5 of the 8 cases were served from the REQUEST CACHE, whose fingerprint also "
                    "omits provider. Those 5 replays were produced by an earlier UNPINNED run and "
                    "are not measurements of the pinned route at all.",
   "consequence_2": "a previous report of mine claimed 'provider enters the config hash, so a "
                    "pinned arm is a distinct frozen experiment'. That claim was FALSE and is "
                    "retracted here.",
   "severity": "invalidates the arm as a provider-attribution measurement"}
 ],

 "what_the_three_live_attempts_do_show": {
   "caveat": "recorded for completeness only. This is NOT a gate evaluation: the arm is invalid, "
             "and 3 live attempts on a partially cache-served 8-case arm cannot evaluate any "
             "pre-registered gate.",
   "live_attempts": len(raw),
   "cache_replays": sum(1 for r in outs if r.get("cache_hit")),
   "pin_reached_the_wire": all(r.get("requested_provider_order") == ["google-ai-studio"]
                               and r.get("allow_fallbacks") is False for r in raw),
   "observed_provider": sorted({r.get("observed_provider") for r in raw}),
   "provider_attribution_status": sorted({r.get("provider_attribution_status") for r in raw}),
   "route_violations": sum(1 for r in raw if (r.get("route_check") or {}).get("violation")),
   "note": "on all three LIVE attempts the payload carried order=['google-ai-studio'] with "
           "allow_fallbacks=false and the response EXPLICITLY reported 'Google AI Studio'. The "
           "pin mechanism itself worked on the wire. That is the one positive finding here, and "
           "it does not rescue the arm."
 },

 "attribution_discipline_held": {
   "requested_and_observed_kept_separate": True,
   "unknown_treated_as_unknown": True,
   "no_provider_inferred_from_slug_or_requested_route": True,
   "explicit_mismatch_would_have_been_a_violation": True,
   "violations_observed": 0
 },

 "integrity_of_what_was_written": {
   "raw_response_rows": len(raw),
   "raw_body_sha256_present": all(r.get("raw_body_sha256") for r in raw),
   "ledger_rows_for_this_arm": len(mine),
   "billable_live_calls": sum(1 for r in mine if not r.get("cache_hit")),
   "cache_hit_rows": sum(1 for r in mine if r.get("cache_hit")),
   "archive_failure_marker": (RD / "raw_responses.jsonl.ARCHIVE_FAILURE").exists(),
   "held_out_access": 0,
   "audited_references_modified": 0,
   "active_grades_changed": 0
 },

 "gates_not_evaluated": ("No pre-registered gate is evaluated. The arm did not produce a valid "
                         "8-case pinned measurement, and evaluating gates on a partially "
                         "cache-served arm would be reading a number that does not mean what its "
                         "label says."),

 "required_before_any_re_run": [
  "fix TaskRoute.fingerprint_fields() to include `provider`, so a pinned arm has a distinct "
  "config_hash AND a distinct request-cache fingerprint (this is the defect that let unpinned "
  "cached responses serve a pinned arm)",
  "fix the UnboundLocalError so the pre-send budget hook and per-attempt route aggregation "
  "actually install",
  "record the requested provider routing in run.json, which fingerprint_fields currently omits",
  "re-freeze the screen (the config hashes and therefore the run identities will change) and "
  "obtain a fresh authorization - the $0.12 grant was for the frozen campaign as specified",
  "decide explicitly whether the 5 cached Stage-1c responses may be reused at all, or whether the "
  "arm must run cache-cold"
 ],

 "recommendation": ("INCONCLUSIVE_MECHANICAL_STOP. Do not run arms 2 and 3 under the present "
                    "freeze: they would inherit the same fingerprint collision and the same "
                    "cache contamination, and arm 2 (vertex) would additionally be at risk of "
                    "replaying the very ai-studio responses arm 1 just produced, since the two "
                    "pins share a fingerprint. Fix the four defects, re-freeze, re-authorize."),

 "authorization_respected": True,
 "no_seen32": True,
 "ocr_primary_role_status": "UNSELECTED (unchanged)",
}
body = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True, default=str)
doc["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p = R / "OCR_ALTSCREEN_MECHANICAL_STOP_2026-09-04.json"
p.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("wrote", p)
print("content_sha256", doc["content_sha256"][:16])
print("additional spend $%.8f" % doc["spend"]["additional_spend_usd"])
