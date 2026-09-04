"""Close V1 immutably and freeze V2. ZERO provider calls."""
import json, hashlib, pathlib, subprocess, time

from autograder.gateway import TaskRoute
from autograder.routeidentity import (CACHE_IDENTITY_VERSION, EXCLUDED_FIELDS,
                                      EXPERIMENT_ONLY_FIELDS, ROUTE_LEVEL_FIELDS,
                                      SEMANTIC_FIELDS, experiment_identity, identity_report)

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
E = pathlib.Path("evaluation/model_selection/experiments")
P = pathlib.Path("evaluation/model_selection/policies")
RD = pathlib.Path("evaluation/model_selection/runs_altscreen/ocr_primary/"
                  "dev__smoke__all__google-gemini-3.7-flash__45297cdd83")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ts = time.strftime("%Y-%m-%d %H:%M:%S")
V1 = json.loads((E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1_2026-09-03.json").read_text(encoding="utf-8"))
raw = [json.loads(l) for l in (RD / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
outs = [json.loads(l) for l in (RD / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
led = [json.loads(l) for l in pathlib.Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl"
                                           ).read_text(encoding="utf-8").splitlines() if l.strip()]
cum = round(sum(float(r.get("reported_cost") or 0) for r in led
                if r.get("cloud") and not r.get("cache_hit")), 8)


def seal(doc, field="content_sha256"):
    body = json.dumps({k: v for k, v in doc.items() if k != field},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    doc[field] = hashlib.sha256(body.encode()).hexdigest()
    return doc


# ------------------------------------------------------------------ V1 close --
v1 = seal({
 "artifact": "ocr_altscreen_v1_terminal_closure",
 "created_at": ts, "git_commit": commit, "provider_calls_in_this_artifact": 0,
 "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1",
 "experiment_sha256": V1["experiment_sha256"],
 "terminal_outcome": "INCONCLUSIVE_MECHANICAL_STOP",
 "immutability": ("V1 has been PARTIALLY EXECUTED. It must never be edited or re-hashed again. "
                  "Any correction lives in a NEW experiment (V2), never as an amendment to this "
                  "one. Its prior amendment chain (39305a62 -> 579b1f60 -> 67a716fc) is closed."),
 "gates_not_evaluated": True,
 "gates_not_evaluated_reason": ("no arm produced a valid 8-case pinned measurement; evaluating "
                                "pre-registered gates on a partially cache-served arm would "
                                "report a number that does not mean what its label says"),
 "arms": {"attempted": 1, "completed_validly": 0,
          "not_started": ["gemini_pinned_vertex", "qwen3_vl_235b_pinned_alibaba"]},

 "preserved_evidence": {
   "five_historical_unpinned_cache_hits": {
     "count": sum(1 for r in outs if r.get("cache_hit")),
     "case_ids": [r["case_id"] for r in outs if r.get("cache_hit")],
     "what_they_are": ("responses produced by an EARLIER UNPINNED run and replayed because the "
                       "v1 cache key omitted `provider`. They are not measurements of the pinned "
                       "route and are excluded from every metric."),
     "status": "PRESERVED ON DISK, EXCLUDED FROM ALL EVALUATION"},
   "three_live_ai_studio_attempts": {
     "count": len(raw),
     "attempt_ids": [r["attempt_id"] for r in raw],
     "raw_body_sha256": {r["attempt_id"]: r["raw_body_sha256"] for r in raw},
     "raw_body_bytes": {r["attempt_id"]: r["raw_body_bytes"] for r in raw},
     "requested_provider_order": sorted({tuple(r["requested_provider_order"]) for r in raw})[0],
     "allow_fallbacks": False,
     "observed_provider": sorted({r["observed_provider"] for r in raw}),
     "provider_attribution_status": sorted({r["provider_attribution_status"] for r in raw}),
     "route_violations": 0,
     "THE_ONLY_STATEMENT_THESE_SUPPORT": ("the requested pin reached the wire and the responses "
                                          "explicitly identified Google AI Studio"),
     "explicitly_not_reusable": ("must NOT be reused in V2 metrics, V2 denominators or any "
                                 "advancement decision")},
   "spend": {"additional_spend_usd": 0.00252675,
             "ledger_before": 0.70323229, "ledger_after": 0.70575904,
             "ledger_rows_before": 798, "ledger_rows_after": 806,
             "authorization_usd": 0.12, "within_authorization": True},
   "missing_linkage_fields": {
     "affected": "every logical row and all three archived attempts",
     "null_fields": ["campaign_id", "arm_id", "case_id"],
     "attempt_records_on_logical_rows": 0,
     "cause": "the runner treated capture as unavailable, so nothing populated them"},
 },

 "mechanical_defects": [
  {"id": "V1-D1", "title": "route identity omitted the provider pin",
   "detail": ("TaskRoute.fingerprint_fields() was a hand-maintained list that did not mention "
              "`provider`. The pinned arm therefore resolved config_hash 45297cdd83 - identical "
              "to the historical UNPINNED Stage-1b/1c runs - and the request cache, keyed from "
              "the same list, served 5 of 8 cases from an earlier unpinned run."),
   "fixed_in": "autograder/routeidentity.py (identity DERIVED from the effective backend config)",
   "retracted_claim": ("a report at 5f75438 stated 'provider enters the config hash, so a pinned "
                       "arm is a distinct frozen experiment'. That was false and is retracted.")},
  {"id": "V1-D2", "title": "campaign setup failed open",
   "detail": ("an UnboundLocalError (`campaign` referenced before assignment) was caught by a "
              "broad handler that downgraded it to 'raw response capture unavailable'. The "
              "pre-send budget hook and per-attempt route aggregation were never installed, so "
              "the per-physical-attempt enforcement built at 87e79bf did not run."),
   "fixed_in": "autograder/benchmark/runner.py (_install_attempt_protocol + CampaignSetupError)"},
 ],

 "ocr_primary_role_status": "UNSELECTED (unchanged)",
 "held_out_access": 0, "audited_references_modified": 0, "active_grades_changed": 0,
 "successor": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V2",
})
p1 = R / "OCR_ALTSCREEN_V1_CLOSURE_2026-09-04.json"
p1.write_text(json.dumps(v1, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

# --------------------------------------------------------------------- V2 -----
def route_for(model, pin, reasoning):
    return TaskRoute(task="ocr_primary", backend="openrouter", model=model,
                     base_url="https://openrouter.ai/api/v1", structured_mode="json_schema",
                     max_tokens=1000, temperature=0.0, reasoning=reasoning,
                     provider={"order": [pin], "allow_fallbacks": False},
                     prompt_version="m2-strict-v1")


GEM, QWEN = "google/gemini-3.7-flash", "qwen/qwen3-vl-235b-a22b-instruct"
ARMS = [("gemini_pinned_ai_studio", GEM, "google-ai-studio", {"effort": "low"}, 0.038328),
        ("gemini_pinned_vertex", GEM, "google-vertex", {"effort": "low"}, 0.038328),
        ("qwen3_vl_235b_pinned_alibaba", QWEN, "alibaba", None, 0.020240)]

cands = []
for arm_id, model, pin, reasoning, worst in ARMS:
    r = route_for(model, pin, reasoning)
    v1arm = next(a for a in V1["candidates"] if a["arm_id"] == arm_id)
    cands.append({
        "arm_id": arm_id, "model": model, "provider_pin": pin,
        "provider_routing": {"order": [pin], "allow_fallbacks": False},
        "arm_type": v1arm["arm_type"],
        "route": {"task": "ocr_primary", "backend": "openrouter", "model": model,
                  "structured_mode": "json_schema", "max_tokens": 1000, "temperature": 0.0,
                  "reasoning": reasoning, "prompt_version": "m2-strict-v1"},
        "reasoning_requirement": v1arm["reasoning_requirement"],
        "estimated_cost": v1arm["estimated_cost"],
        "why_genuinely_different": v1arm["why_genuinely_different"],
        "main_risk": v1arm["main_risk"],
        "experiment_identity": experiment_identity(r),
        "identity_report": identity_report(r),
    })

ids = [c["experiment_identity"] for c in cands]
assert len(set(ids)) == 3, "the three arms must have distinct identities"

L0 = cum
WARN_ABS, HARD_ABS = 0.78323229, 0.82323229
warn_inc = round(WARN_ABS - L0, 8)
hard_inc = round(HARD_ABS - L0, 8)
worst_total = round(sum(w for *_, w in ARMS), 6)
assert L0 + worst_total <= HARD_ABS, (
    f"V2 worst case {worst_total} does not fit under the remaining absolute hard limit")

v2 = {
 "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V2",
 "status": "FROZEN - NOT EXECUTED - NOT AUTHORIZED",
 "created_at": ts, "git_commit": commit, "provider_calls_made_preparing_this": 0,
 "supersedes": {"experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1",
                "experiment_sha256": V1["experiment_sha256"],
                "terminal_outcome": "INCONCLUSIVE_MECHANICAL_STOP",
                "incident_artifact": "OCR_ALTSCREEN_V1_CLOSURE_2026-09-04.json",
                "v1_outputs_excluded_from_v2_evaluation": True,
                "why_a_new_experiment": ("V1 was partially executed; a corrected V1 would be an "
                                         "edit to an executed freeze. V2 is a new identity.")},

 "question": V1["question"],
 "population": V1["population"],
 "prompt": V1["prompt"],
 "schema": V1["schema"],
 "adapter_version": V1["adapter_version"],
 "candidates": cands, "candidate_count": len(cands),
 "live_pricing_snapshot": V1["live_pricing_snapshot"],
 "advancement_and_drop_rules_stated_in_advance": V1["advancement_and_drop_rules_stated_in_advance"],
 "prohibitions": V1["prohibitions"],

 "identity_and_cache_policy": {
   "identity_version": CACHE_IDENTITY_VERSION,
   "derivation": ("both identities are DERIVED from TaskRoute.to_backend_config() - the effective "
                  "configuration that actually reaches the wire - not from a hand-maintained "
                  "field list. That list is what omitted `provider` in V1."),
   "semantic_request_identity_fields": list(SEMANTIC_FIELDS) + list(ROUTE_LEVEL_FIELDS),
   "experiment_only_fields": list(EXPERIMENT_ONLY_FIELDS),
   "excluded_from_all_identities": sorted(EXCLUDED_FIELDS),
   "secret_free": "the API key is never read; api_key_env (a variable NAME) is dropped too",
   "three_arms_hash_differently": True,
   "arm_identities": {c["arm_id"]: c["experiment_identity"] for c in cands},
   "cache_policy": "refresh",
   "cache_policy_meaning": ("bypass cache READS, perform the live request, still WRITE the "
                            "correctly-versioned entry. A new --runs-root is NOT a substitute: "
                            "the request cache is shared campaign state."),
   "cache_hits_allowed": 0,
   "cache_hit_consequence": "INCONCLUSIVE_MECHANICAL_STOP",
   "historical_entries": "preserved on disk and unreachable by v2 keys; never deleted or rewritten",
 },

 "execution_requirements": {
   "all_intended_logical_requests_live": 24,
   "required_attempt_linkage_fields": ["campaign_id", "arm_id", "case_id",
                                       "logical_request_id", "attempt_id", "retry_index"],
   "missing_linkage_consequence": "mechanical stop; the row is never usable",
   "campaign_setup_failure": "FATAL - zero cache lookups and zero transport activity",
   "retry_policy": {"transport_retries": 2, "validation_retries": 0,
                    "note": "unchanged from V1; every physical retry is separately budget-authorized"},
   "budget_enforcement": "per PHYSICAL HTTP attempt at the send boundary, worst-case reservation",
 },

 "budget": {
   "L0_verified_from_disk": L0,
   "ledger_rows_at_freeze": len(led),
   "campaign_family_absolute_limits_preserved": {"warning": WARN_ABS, "hard": HARD_ABS},
   "prospective_warning_increment": warn_inc,
   "prospective_hard_increment": hard_inc,
   "predicted_worst_case_usd": worst_total,
   "fits_under_remaining_hard_limit": True,
   "headroom_after_worst_case_usd": round(HARD_ABS - (L0 + worst_total), 8),
   "NOT_AUTHORIZED": ("these are PROSPECTIVE limits. This task does not authorize using them and "
                      "no campaign budget manifest is created here."),
 },
}
body = json.dumps(v2, ensure_ascii=False, indent=1, sort_keys=True, default=str)
v2["experiment_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p2 = E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V2_2026-09-04.json"
p2.write_text(json.dumps(v2, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

print("wrote", p1)
print("wrote", p2)
print("V1 closure sha:", v1["content_sha256"][:16])
print("V2 experiment_sha256:", v2["experiment_sha256"])
print(f"L0 (from disk) = {L0}  warn_inc = {warn_inc}  hard_inc = {hard_inc}")
print(f"V2 worst case = {worst_total}  headroom = {v2['budget']['headroom_after_worst_case_usd']}")
for c in cands:
    print(f"  {c['arm_id']:32s} identity {c['experiment_identity'][:16]}")
