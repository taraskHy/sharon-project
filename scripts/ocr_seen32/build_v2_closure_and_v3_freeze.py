"""Close V2 as SUPERSEDED_BEFORE_EXECUTION and freeze V3. ZERO provider calls."""
import json, hashlib, pathlib, subprocess, time

from autograder.gateway import TaskRoute
from autograder.routeidentity import (CACHE_IDENTITY_VERSION, EXCLUDED_FIELDS,
                                      EXPERIMENT_ONLY_FIELDS, ROUTE_LEVEL_FIELDS,
                                      SEMANTIC_FIELDS, experiment_identity, identity_report)

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
E = pathlib.Path("evaluation/model_selection/experiments")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ts = time.strftime("%Y-%m-%d %H:%M:%S")
V2 = json.loads((E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V2_2026-09-04.json").read_text(encoding="utf-8"))
led = [json.loads(l) for l in pathlib.Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl"
                                           ).read_text(encoding="utf-8").splitlines() if l.strip()]
cum = round(sum(float(r.get("reported_cost") or 0) for r in led
                if r.get("cloud") and not r.get("cache_hit")), 8)
L0, WARN_ABS, HARD_ABS = cum, 0.78323229, 0.82323229
HEADROOM = round(HARD_ABS - L0, 8)


def seal(doc, field="content_sha256"):
    body = json.dumps({k: v for k, v in doc.items() if k != field},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    doc[field] = hashlib.sha256(body.encode()).hexdigest()
    return doc


# per-attempt reserved cost, from the V2 per-arm single-attempt figures
PER_ATT = {a["arm_id"]: round(a["estimated_cost"]["worst_case_usd"] / 8, 8) for a in V2["candidates"]}
ONE_ATT = {a["arm_id"]: a["estimated_cost"]["worst_case_usd"] for a in V2["candidates"]}
NOMINAL = {a["arm_id"]: a["estimated_cost"]["expected_all_succeed_usd"] for a in V2["candidates"]}
single_total = round(sum(ONE_ATT.values()), 6)
nominal_total = round(sum(NOMINAL.values()), 6)
retry2_total = round(single_total * 3, 6)

# ------------------------------------------------------------- V2 closure ----
v2c = seal({
 "artifact": "ocr_altscreen_v2_closure",
 "created_at": ts, "git_commit": commit, "provider_calls": 0,
 "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V2",
 "experiment_sha256": V2["experiment_sha256"],
 "terminal_outcome": "SUPERSEDED_BEFORE_EXECUTION",
 "executed": False, "provider_requests_ever_made_under_v2": 0,
 "immutability": ("V2 is preserved byte-for-byte with its hash intact. It was NOT edited, NOT "
                  "re-hashed and NOT executed. The correction lives in V3."),

 "exact_reason": ("V2's frozen cost field is materially false. `predicted_worst_case_usd` = "
                  "0.096896 is the sum of ONE attempt per logical request, and the per-arm fields "
                  "are literally named `worst_case_usd`, but the frozen retry policy is "
                  "transport_retries=2 — up to 3 physical attempts per logical request, 72 for "
                  "the campaign. Treating every transmitted attempt as potentially billable (see "
                  "billing_evidence), the true completion-guarantee bound is $0.290688, which is "
                  "2.47x the remaining hard headroom of $0.11747325. V2 additionally asserts "
                  "`fits_under_remaining_hard_limit: true`, a claim of guaranteed completion "
                  "under a budget that cannot fund it."),

 "conditions_evaluated": {
   "schema_covered_by_semantic_identity": {
     "verdict": "MARGINAL AT FREEZE TIME, NOW CORRECTED",
     "detail": ("V2 froze identity_version 2, which digested the RAW model_json_schema() rather "
                "than the response_format block actually transmitted (the strict transform plus "
                "the schema name). Schema CONTENT did move the identity, so this was a proxy "
                "rather than a hole — but a change to the strict transform would have left the "
                "digest unmoved while changing the wire payload. Corrected in identity_version 3.")},
   "frozen_cost_field_accurately_defined": {"verdict": "FAIL", "detail": "see exact_reason"},
   "retry_inclusive_bound_fits_or_is_preregistered": {
     "verdict": "FAIL",
     "detail": ("$0.290688 does not fit under $0.11747325 and V2 contains NO preregistration that "
                "the hard cap may mechanically stop the campaign.")},
   "no_claim_of_guaranteed_completion_under_insufficient_budget": {
     "verdict": "FAIL", "detail": "`fits_under_remaining_hard_limit: true` is exactly such a claim."},
 },

 "billing_evidence": {
   "retryable_attempts_observed_in_806_ledger_rows": 0,
   "http_statuses_ever_seen": ["200", "400", "null (pre-transport)"],
   "conclusion": ("there is NO empirical evidence and NO authoritative contract establishing that "
                  "a 408/409/429/5xx attempt is free. Per the audit rule, every transmitted "
                  "attempt is therefore treated as potentially billable."),
   "counter_evidence_that_failed_attempts_can_bill": ("5 of 5 rows with finish_reason=length "
                                                      "carried a non-zero reported_cost — a "
                                                      "response useless to us that was still "
                                                      "billed. Failure does not imply free."),
   "contrast": "27 content_filter rows billed nothing, so billing on failure is genuinely mixed.",
 },

 "what_v2_got_right_and_is_carried_forward": [
   "derived (not hand-listed) route identity",
   "three arms with distinct experiment identities",
   "refresh cache policy and cache_hits_allowed=0",
   "required attempt-linkage fields",
   "fail-closed campaign setup",
   "the eight frozen Stage-1 cases, prompt, schema and advancement gates",
 ],
 "successor": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3",
 "ocr_primary_role_status": "UNSELECTED (unchanged)",
})
p1 = R / "OCR_ALTSCREEN_V2_CLOSURE_2026-09-05.json"
p1.write_text(json.dumps(v2c, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

# ------------------------------------------------------------------- V3 ------
# MINIMUM NECESSARY CORRECTION: transport_retries 2 -> 0. This is the ONLY
# setting at which the COMPLETE three-arm campaign is conservatively fundable
# under the unchanged family hard limit. It weakens no gate and raises no limit.
RETRIES = 0
MAXATT = RETRIES + 1
retry_incl_total = round(single_total * MAXATT, 6)


def route_for(model, pin, reasoning):
    # transport_retries is FROZEN AT 0 and is identity-bearing, so the arm
    # identities below are the identities of the arms as they will actually run.
    return TaskRoute(task="ocr_primary", backend="openrouter", model=model,
                     base_url="https://openrouter.ai/api/v1", structured_mode="json_schema",
                     max_tokens=1000, temperature=0.0, reasoning=reasoning,
                     transport_retries=RETRIES,
                     provider={"order": [pin], "allow_fallbacks": False},
                     prompt_version="m2-strict-v1")


REASONING = {"gemini_pinned_ai_studio": {"effort": "low"},
             "gemini_pinned_vertex": {"effort": "low"},
             "qwen3_vl_235b_pinned_alibaba": None}
cands = []
for a in V2["candidates"]:
    aid = a["arm_id"]
    r = route_for(a["model"], a["provider_pin"], REASONING[aid])
    c = dict(a)
    c["experiment_identity"] = experiment_identity(r)
    c["identity_report"] = identity_report(r)
    c["cost_bounds"] = {
        "logical_requests": 8,
        "max_physical_attempts_per_logical_request": MAXATT,
        "max_physical_attempts_for_the_arm": 8 * MAXATT,
        "max_reserved_cost_per_attempt_usd": PER_ATT[aid],
        "nominal_expected_usd": NOMINAL[aid],
        "one_attempt_per_case_maximum_usd": ONE_ATT[aid],
        "retry_inclusive_completion_bound_usd": round(ONE_ATT[aid] * MAXATT, 6),
    }
    c.pop("estimated_cost", None)
    cands.append(c)
assert len({c["experiment_identity"] for c in cands}) == 3

fits = (L0 + retry_incl_total) <= HARD_ABS

v3 = {
 "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3",
 "status": "FROZEN - NOT EXECUTED - NOT AUTHORIZED",
 "created_at": ts, "git_commit": commit, "provider_calls_made_preparing_this": 0,

 "supersedes": {
   "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V2",
   "experiment_sha256": V2["experiment_sha256"],
   "terminal_outcome": "SUPERSEDED_BEFORE_EXECUTION",
   "closure_artifact": "OCR_ALTSCREEN_V2_CLOSURE_2026-09-05.json",
   "v2_was_never_executed": True,
   "minimum_necessary_correction": (
       "(1) the cost model now states FOUR separate bounds instead of one mislabelled number, "
       "and (2) transport_retries is 2 -> 0, which is the ONLY setting at which the COMPLETE "
       "three-arm campaign is conservatively fundable under the unchanged family hard limit. "
       "No gate is weakened and no limit is raised."),
   "also_inherits": "the V1 incident lineage via OCR_ALTSCREEN_V1_CLOSURE_2026-09-04.json",
 },

 "question": V2["question"],
 "population": V2["population"],
 "prompt": V2["prompt"],
 "schema": V2["schema"],
 "adapter_version": V2["adapter_version"],
 "candidates": cands, "candidate_count": len(cands),
 "live_pricing_snapshot": V2["live_pricing_snapshot"],
 "advancement_and_drop_rules_stated_in_advance": V2["advancement_and_drop_rules_stated_in_advance"],
 "prohibitions": V2["prohibitions"],

 "identity_and_cache_policy": {
   **{k: v for k, v in V2["identity_and_cache_policy"].items()
      if k not in ("identity_version", "arm_identities", "semantic_request_identity_fields")},
   "identity_version": CACHE_IDENTITY_VERSION,
   "semantic_request_identity_fields": list(SEMANTIC_FIELDS) + list(ROUTE_LEVEL_FIELDS),
   "experiment_only_fields": list(EXPERIMENT_ONLY_FIELDS),
   "excluded_from_all_identities": sorted(EXCLUDED_FIELDS),
   "arm_identities": {c["arm_id"]: c["experiment_identity"] for c in cands},
   "response_schema_coverage": (
       "the CANONICAL WIRE SCHEMA enters the semantic/cache identity: the response_format block "
       "as transmitted, i.e. the strict-transformed schema plus the schema name, not the raw "
       "model_json_schema(). Proven against the backend's own _build_payload output."),
 },

 "execution_requirements": {
   **V2["execution_requirements"],
   "retry_policy": {
     "transport_retries": RETRIES,
     "validation_retries": 0,
     "max_physical_attempts_per_logical_request": MAXATT,
     "max_physical_attempts_for_the_campaign": 24 * MAXATT,
     "changed_from_v2": "transport_retries 2 -> 0",
     "why": ("at transport_retries=2 the completion-guarantee bound is $%.6f, which is %.2fx the "
             "remaining hard headroom of $%.8f. The campaign could not be funded to completion "
             "and would risk a second mid-campaign mechanical stop, which is how V1 ended. "
             "Retries are therefore disabled so the COMPLETE campaign fits."
             % (retry2_total, retry2_total / HEADROOM, HEADROOM)),
     "accepted_cost_of_this_choice": (
         "a transient transport failure (408/409/429/5xx) now FAILS that case instead of being "
         "retried. If that happens, the arm's coverage gate may fail for a TRANSPORT reason "
         "rather than a model reason. Such a failure MUST be reported as a transport failure and "
         "MUST NOT be counted as evidence about the model or the provider route."),
     "every_physical_attempt_is_still_separately_budget_authorized": True,
   },
 },

 "cost_model": {
   "billing_assumption": (
       "every transmitted physical attempt is treated as POTENTIALLY BILLABLE. No empirical "
       "evidence and no authoritative contract establishes that a retryable attempt is free; 0 "
       "such attempts appear in 806 ledger rows, and 5 of 5 finish_reason=length rows DID bill."),
   "nominal_expected_usd": nominal_total,
   "single_attempt_maximum_usd": single_total,
   "single_attempt_maximum_definition": "24 logical requests, exactly one physical attempt each",
   "retry_inclusive_completion_bound_usd": retry_incl_total,
   "retry_inclusive_definition": (
       "24 logical requests x %d physical attempts x the per-attempt reserved maximum = %d "
       "physical attempts" % (MAXATT, 24 * MAXATT)),
   "enforced_absolute_hard_limit_usd": HARD_ABS,
   "remaining_hard_headroom_usd": HEADROOM,
   "complete_campaign_fits_under_the_hard_limit": fits,
   "headroom_after_the_completion_bound_usd": round(HARD_ABS - (L0 + retry_incl_total), 8),
   "what_the_hard_limiter_does_NOT_excuse": (
       "the limiter stopping a campaign partway is NOT a reason to call an unaffordable campaign "
       "'fitting'. V3 is funded to COMPLETION at its frozen retry policy; that is why the policy "
       "changed rather than the claim."),
   "for_reference_only_at_transport_retries_2": {
     "retry_inclusive_completion_bound_usd": retry2_total,
     "fits": (L0 + retry2_total) <= HARD_ABS,
     "over_by_usd": round((L0 + retry2_total) - HARD_ABS, 6),
     "note": "NOT the frozen policy; recorded so the tradeoff is visible.",
   },
 },

 "budget": {
   "L0_verified_from_disk": L0,
   "ledger_rows_at_freeze": len(led),
   "campaign_family_absolute_limits_preserved": {"warning": WARN_ABS, "hard": HARD_ABS},
   "prospective_warning_increment": round(WARN_ABS - L0, 8),
   "prospective_hard_increment": HEADROOM,
   "NOT_AUTHORIZED": ("PROSPECTIVE limits. No campaign budget manifest is created here and this "
                      "freeze authorizes nothing."),
 },
}
body = json.dumps(v3, ensure_ascii=False, indent=1, sort_keys=True, default=str)
v3["experiment_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p2 = E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3_2026-09-05.json"
p2.write_text(json.dumps(v3, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

print("wrote", p1)
print("wrote", p2)
print("V2 closure sha :", v2c["content_sha256"])
print("V3 experiment  :", v3["experiment_sha256"])
print()
print(f"L0 {L0}  headroom {HEADROOM}")
print(f"nominal {nominal_total} | single-attempt {single_total} | retry-incl(0) {retry_incl_total}")
print(f"complete campaign fits: {fits}  headroom after: {v3['cost_model']['headroom_after_the_completion_bound_usd']}")
print(f"(reference) at retries=2: {retry2_total}  fits: {(L0+retry2_total)<=HARD_ABS}")
