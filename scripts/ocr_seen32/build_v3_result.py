"""V3 campaign result: INCONCLUSIVE_MECHANICAL_STOP. ZERO further provider calls."""
import json, hashlib, pathlib, subprocess, time, statistics, collections

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.ocr_outcomes import classify_row
from autograder.benchmark.ocr_writer_metrics import pair_metrics

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
BASE = pathlib.Path("evaluation/model_selection/runs_altscreen_v3/ocr_primary")
V3 = json.loads(pathlib.Path("evaluation/model_selection/experiments/"
                             "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3_2026-09-05.json"
                             ).read_text(encoding="utf-8"))
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ts = time.strftime("%Y-%m-%d %H:%M:%S")
man = load_manifest("ocr_primary"); by = {c.case_id: c for c in man.cases}
CASES = {c["case_id"]: c for c in V3["population"]["cases"]}
HW = {c for c, m in CASES.items() if m["category"].startswith("handwritten")}
led = [json.loads(l) for l in pathlib.Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl"
                                           ).read_text(encoding="utf-8").splitlines() if l.strip()]
cum = round(sum(float(r.get("reported_cost") or 0) for r in led
                if r.get("cloud") and not r.get("cache_hit")), 8)
L0 = 0.70575904


def load(dirname):
    d = BASE / dirname
    o = [json.loads(l) for l in (d / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    r = [json.loads(l) for l in (d / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return d, o, r


ARM1 = "dev__smoke__all__google-gemini-3.7-flash__7a31feab11"
ARM2 = "dev__smoke__all__google-gemini-3.7-flash__09cab786a7"
d1, o1, r1 = load(ARM1)
d2, o2, r2 = load(ARM2)

rows = []
for r in o1:
    cid = r["case_id"]; ref = by[cid].label["reference"]
    t = classify_row(r, ref)
    hyp = (r.get("output") or {}).get("transcription") if r.get("output") else None
    rows.append((cid, cid in HW, t, pair_metrics(ref, hyp).get("cer")
                 if t["usable_transcription_returned"] else None, r))
us = [x for x in rows if x[2]["usable_transcription_returned"]]
hw = [x for x in rows if x[1]]; pr = [x for x in rows if not x[1]]
hwu = [x for x in hw if x[2]["usable_transcription_returned"]]
lat = [r.get("latency_s") for *_, r in rows if r.get("latency_s") is not None]

arm1 = {
 "arm_id": "gemini_pinned_ai_studio", "run_id": ARM1,
 "arm_type": "PROVIDER-ROUTE ATTRIBUTION ARM",
 "completed": True,
 "intended": 8, "attempted": 8, "usable": len(us),
 "handwritten_coverage": f"{len(hwu)}/{len(hw)}",
 "printed_coverage": f"{sum(1 for x in pr if x[2]['usable_transcription_returned'])}/{len(pr)}",
 "transport_failures": 0,
 "provider_content_filter": sum(1 for x in rows if x[2]["provider_content_filter_failure"]),
 "other_provider_http_failures": sum(1 for x in rows if x[2]["provider_other_http_failure"]),
 "model_text_refusals": sum(1 for x in rows if x[2]["model_text_refusal"]),
 "truncation": sum(1 for x in rows if x[2]["truncation"]),
 "json_parse_failures": sum(1 for x in rows if x[2]["json_parse_failure"]),
 "schema_failures": sum(1 for x in rows if x[2]["schema_failure"]),
 "fabrication": sum(1 for x in rows if x[2]["fabrication_detected"]),
 "total_line_loss": sum(1 for x in rows if x[2]["total_line_loss"]),
 "critical_errors": "NOT COMPUTED — the campaign stopped mechanically; no gate is evaluated",
 "successful_output_handwritten_cer": round(statistics.mean([x[3] for x in hwu]), 4),
 "successful_output_handwritten_cer_denominator": f"{len(hwu)}/{len(hw)}",
 "failure_aware_handwritten_cer": round(statistics.mean(
     [(x[3] if x[3] is not None else 1.0) for x in hw]), 4),
 "mean_latency_s": round(statistics.mean(lat), 3),
 "physical_attempts": len(r1),
 "physical_attempts_per_logical_request": 1,
 "all_retry_index_zero": all(x["retry_index"] == 0 for x in r1),
 "cache_hits": sum(1 for r in o1 if r.get("cache_hit")),
 "requested_provider": sorted({tuple(x["requested_provider_order"]) for x in r1})[0][0],
 "observed_provider_counts": dict(collections.Counter(x["observed_provider"] for x in r1)),
 "unknown_attribution_count": sum(1 for x in r1 if x["provider_attribution_status"] == "UNKNOWN"),
 "route_violations": sum(1 for x in r1 if x["route_check"]["violation"]),
 "arm_cost_usd": 0.009792,
 "gates": "NOT EVALUATED — see gates_not_evaluated",
}

arm2 = {
 "arm_id": "gemini_pinned_vertex", "run_id": ARM2,
 "arm_type": "PROVIDER-ROUTE ATTRIBUTION ARM",
 "completed": False,
 "intended": 8, "attempted": 1, "usable": 0,
 "stopped_on": "RouteViolation, first case, before any further send",
 "physical_attempts": len(r2),
 "cache_hits": 0,
 "requested_provider": "google-vertex",
 "observed_provider_counts": dict(collections.Counter(x["observed_provider"] for x in r2)),
 "arm_cost_usd": 0.0022815,
 "gates": "NOT EVALUATED — arm did not run",
}

arm3 = {"arm_id": "qwen3_vl_235b_pinned_alibaba", "arm_type": "CROSS-FAMILY CANDIDATE ARM",
        "completed": False, "attempted": 0, "arm_cost_usd": 0.0,
        "status": "NEVER STARTED — the campaign stopped after arm 2",
        "gates": "NOT EVALUATED — arm did not run"}

doc = {
 "artifact": "ocr_altscreen_v3_campaign_result",
 "created_at": ts, "git_commit": commit,
 "campaign": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3",
 "experiment_sha256": V3["experiment_sha256"],
 "budget_manifest_commit": "d07dd20b6f783502db141896a2cad95cbbdd3f95",
 "starting_code_commit": "1259f3adfe6245fef431057080ae1b2daf091bd0",

 "terminal_outcome": "INCONCLUSIVE_MECHANICAL_STOP",
 "gates_not_evaluated": True,
 "gates_not_evaluated_reason": (
     "the campaign stopped on a route-violation stop condition after arm 2's first case. Arm 1 "
     "completed, but the paired Gemini route comparison it exists to support is impossible "
     "without arm 2, and arm 3 never started. Evaluating a preregistered gate on one unpaired arm "
     "would answer a question the screen was not designed to ask."),

 "stop_condition": {
   "type": "explicit route violation",
   "arm": "gemini_pinned_vertex", "case": "hl_e003_q1_r1__l1",
   "attempt_id": "9be63653fb744a6aa4b23db2180fb145",
   "requested_provider_order": ["google-vertex"], "allow_fallbacks": False,
   "observed_provider": "Google", "provider_attribution_status": "EXPLICIT",
   "raw_body_sha256": "9cc23da8495b812418553ca19f89bd68bfe6ef46d23379d53b3d44494fb0be61",
   "enforcement_behaved_as_designed": (
       "the check fired on the FIRST case and stopped the arm before any further send, rather "
       "than accepting output from a provider other than the pinned one"),
 },

 "ROOT_CAUSE_THE_VIOLATION_IS_A_FALSE_POSITIVE": {
   "finding": (
       "OpenRouter's provider SLUG for Vertex is `google-vertex` and its DISPLAY NAME is "
       "`Google`. The response reporting `Google` therefore means the pin WAS honoured. The "
       "violation was raised by rawcapture._norm(), which strips non-alphanumerics and lowercases "
       "both sides and then compares a SLUG against a DISPLAY NAME."),
   "evidence": ("OCR_PROVIDER_ROUTE_FORENSICS_2026-09-03.json records the mapping read from "
                "OpenRouter's /providers endpoint: 'Google (google-vertex)' and 'Google AI Studio "
                "(google-ai-studio)'."),
   "why_arm_1_passed": ("_norm('google-ai-studio') == _norm('Google AI Studio') == "
                        "'googleaistudio' — the slug and the display name normalise identically "
                        "BY COINCIDENCE. _norm('google-vertex')='googlevertex' vs "
                        "_norm('Google')='google' do not."),
   "consequence": ("arm 1's clean route record is therefore weaker evidence than it looks: it "
                   "passed because of a coincidence in string form, not because the comparison "
                   "is sound."),
   "not_fixed_here": ("fixing the comparison and re-running is outside this authorization, which "
                      "excludes retries, reruns and changed configuration and does not permit a "
                      "V3 amendment or a V4. The defect is reported, not repaired."),
   "fail_direction": "the bug fails SAFE — it stops rather than accepting possible fallback output",
 },

 "arms": [arm1, arm2, arm3],

 "paired_gemini_comparison": {
   "possible": False,
   "reason": "arm 2 produced one stopped case; there are no explicitly paired cases to compare.",
   "no_significance_claimed": "eight cases could not support a significance claim in any event",
 },

 "accounting": {
   "L0": L0, "ledger_rows_before": 806, "ledger_rows_after": len(led),
   "ending_ledger_usd": cum,
   "additional_spend_usd": round(cum - L0, 8),
   "authorized_increment_usd": 0.11747325,
   "within_authorization": round(cum - L0, 8) <= 0.11747325,
   "frozen_complete_campaign_maximum_usd": 0.096896,
   "within_frozen_maximum": round(cum - L0, 8) <= 0.096896,
   "warning_absolute_usd": 0.78323229, "hard_absolute_usd": 0.82323229,
   "warning_threshold_crossed": cum >= 0.78323229,
   "hard_threshold_crossed": cum >= 0.82323229,
   "arm_costs_usd": {"gemini_pinned_ai_studio": 0.009792,
                     "gemini_pinned_vertex": 0.0022815,
                     "qwen3_vl_235b_pinned_alibaba": 0.0},
 },

 "execution_contract_held": {
   "logical_requests_attempted": 9, "logical_requests_intended_for_the_campaign": 24,
   "physical_attempts": len(r1) + len(r2),
   "physical_attempts_per_logical_request": 1,
   "transport_retries": 0, "cache_policy": "refresh",
   "cache_hits": 0, "cache_hits_allowed": 0,
   "archive_failures": 0, "secret_leakage": 0,
   "held_out_access": 0, "grading_calls": 0, "rag_calls": 0,
   "every_attempt_carried_full_linkage": True,
 },

 "IDENTITY_MISMATCH_RECORDED": {
   "what": ("the executed arms' derived experiment_identity does not equal the value frozen in "
            "V3, for BOTH Gemini arms."),
   "arm1_frozen": V3["candidates"][0]["experiment_identity"],
   "arm1_executed": json.loads((d1 / "run.json").read_text(encoding="utf-8"))["config"]["experiment_identity"],
   "sole_differing_field": "base_url",
   "detail": ("V3's identities were computed from a hand-built TaskRoute carrying "
              "base_url='https://openrouter.ai/api/v1'. build_route() leaves base_url None and the "
              "OpenRouter backend supplies its own endpoint, so the effective config differs in "
              "that one field. Every other field — model, provider pin, allow_fallbacks, "
              "reasoning, max_tokens, temperature, structured_mode, strict_schema, prompt_version, "
              "transport_retries, validation_retries, timeout_s — is identical."),
   "was_this_a_wire_deviation": ("NO. The transmitted request matched the freeze on every "
                                 "dimension: model, prompt version, schema hash, case ids and "
                                 "order, provider order and allow_fallbacks, all verified from the "
                                 "raw archive."),
   "classification": "freeze bookkeeping defect, not a configuration deviation",
   "fix_for_the_next_freeze": ("compute frozen identities from a route produced by build_route() "
                               "— the actual construction path — instead of a hand-built one."),
 },

 "ocr_primary_role_status": "UNSELECTED (unchanged)",
 "seen32_not_run": True,
 "v1_and_v2_outputs_excluded": True,
}
body = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True, default=str)
doc["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p = R / "OCR_ALTSCREEN_V3_RESULT_2026-09-05.json"
p.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("wrote", p)
print("content_sha256:", doc["content_sha256"])
print("terminal outcome:", doc["terminal_outcome"])
print("additional spend: $%.8f" % doc["accounting"]["additional_spend_usd"])
print("within authorization:", doc["accounting"]["within_authorization"],
      "| within frozen max:", doc["accounting"]["within_frozen_maximum"])
