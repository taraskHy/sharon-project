"""Freeze OCR_ALTERNATIVE_CANDIDATE_SCREEN_V5. ZERO calls of any kind."""
import json, hashlib, pathlib, subprocess, time

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.roles import OcrPrimaryAdapter
from autograder.routeidentity import CACHE_IDENTITY_VERSION, identities_from_argv

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
E = pathlib.Path("evaluation/model_selection/experiments")
V4 = json.loads((E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V4_2026-09-05.json").read_text(encoding="utf-8"))
V4C = json.loads((R / "OCR_ALTSCREEN_V4_CLOSURE_2026-09-06.json").read_text(encoding="utf-8"))
PROTO = json.loads((E / "OCR_PROVIDER_METADATA_CAPTURE_PROTOCOL_V1_2026-09-06.json").read_text(encoding="utf-8"))
SNAP = json.loads((R / "OCR_METADATA_SNAPSHOT_2026-09-06.json").read_text(encoding="utf-8"))
acc = SNAP["acceptance"]
assert acc["result"] == "ACCEPTED", acc["result"]
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ts = time.strftime("%Y-%m-%d %H:%M:%S")

led = [json.loads(l) for l in pathlib.Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl"
                                           ).read_text(encoding="utf-8").splitlines() if l.strip()]
L0 = round(sum(float(r.get("reported_cost") or 0) for r in led
               if r.get("cloud") and not r.get("cache_hit")), 8)
assert L0 == 0.71783254 and len(led) == 815, (L0, len(led))
WARN_ABS, HARD_ABS = 0.78323229, 0.82323229

man = load_manifest("ocr_primary"); by = {c.case_id: c for c in man.cases}
adapter = OcrPrimaryAdapter(prompt_version="m2-strict-v1")
order = V4["population"]["ordered_case_ids"]

cands = []
for a4 in V4["candidates"]:
    argv = a4["cli_argv"]
    req0 = adapter.build_request(dict(by[order[0]].inputs), man.root)
    ident = identities_from_argv(argv, output_model=req0.output_model, system=req0.system,
                                 content_blocks=req0.content_blocks, max_tokens=1000)
    sem = {}
    for cid in order:
        rq = adapter.build_request(dict(by[cid].inputs), man.root)
        sem[cid] = identities_from_argv(argv, output_model=rq.output_model, system=rq.system,
                                        content_blocks=rq.content_blocks, max_tokens=1000
                                        )["semantic_request_identity"]
    pin = a4["provider_pin"]
    c = dict(a4)
    c["experiment_identity"] = ident["experiment_identity"]
    c["semantic_request_identity_by_case"] = sem
    c["effective_config"] = ident["effective_config"]
    c["wire_response_format"] = ident["wire_response_format"]
    c["provider_mapping"] = {
        "requested_slug": pin,
        "resolved_display_name": acc["resolved_mapping"][pin],
        "slug_mapping_status": "VERIFIED",
        "evidence": (f"captured under {PROTO['protocol']} ({PROTO['content_sha256'][:16]}...) and "
                     f"accepted with 0 failures; see OCR_METADATA_SNAPSHOT_2026-09-06.json"),
    }
    c["conservative_cost_usd"] = acc["conservative_arm_costs_usd"][a4["arm_id"]]
    cands.append(c)

assert len({c["experiment_identity"] for c in cands}) == 3
assert all(c["experiment_identity"] == a["experiment_identity"]
           for c, a in zip(cands, V4["candidates"])), "identities must be unchanged from V4"

maxi = acc["conservative_campaign_maximum_usd"]
v5 = {
 "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V5",
 "status": "FROZEN - NOT EXECUTED - NOT AUTHORIZED",
 "created_at": ts, "git_commit": commit,
 "provider_calls_made_preparing_this": 0,
 "metadata_calls_made_preparing_this": 0,

 "supersedes": {
   "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V4",
   "experiment_sha256": V4["experiment_sha256"],
   "outcome": "SUPERSEDED_BEFORE_EXECUTION",
   "closure_artifact": "OCR_ALTSCREEN_V4_CLOSURE_2026-09-06.json",
   "closure_sha256": V4C["content_sha256"],
   "v1_v2_v3_v4_outputs_excluded_from_v5_evaluation": True,
   "sole_substantive_correction": (
       "the unverified `alibaba` provider slug is now RESOLVED from a captured catalogue under a "
       "protocol frozen before the first request. Nothing else about the experiment changed: the "
       "arms, cases, order, crops, references, prompt, schema, gates, pins, identities, retry "
       "policy and cache policy are byte-identical to V4."),
 },

 "question": V4["question"],
 "population": V4["population"],
 "prompt": V4["prompt"],
 "schema": V4["schema"],
 "adapter_version": V4["adapter_version"],
 "candidates": cands, "candidate_count": len(cands),
 "live_pricing_snapshot": V4["live_pricing_snapshot"],
 "advancement_and_drop_rules_stated_in_advance": V4["advancement_and_drop_rules_stated_in_advance"],
 "prohibitions": V4["prohibitions"],
 "identity_and_cache_policy": {**V4["identity_and_cache_policy"],
                               "identity_version": CACHE_IDENTITY_VERSION},
 "execution_requirements": V4["execution_requirements"],

 "metadata_prerequisite": {
   "protocol": PROTO["protocol"],
   "protocol_sha256": PROTO["content_sha256"],
   "snapshot_artifact": "OCR_METADATA_SNAPSHOT_2026-09-06.json",
   "snapshot_sha256": SNAP["content_sha256"],
   "raw_body_sha256": {k: v["raw_body_sha256"] for k, v in SNAP["raw_bodies"].items()},
   "captured_at": SNAP["requests"]["providers"]["timestamp"],
   "result": acc["result"],
   "failure_count": acc["failure_count"],
   "accepted_provider_mapping": acc["resolved_mapping"],
   "endpoint_availability": {
     m: sorted({e["provider_name"] for e in v["endpoints"]})
     for m, v in SNAP["parsed"]["model_endpoints"].items()},
   "captured_prices_per_m": SNAP["parsed"]["prices"],
   "effective_prices_per_m": {k: {"input_per_m": v["input_per_m"],
                                  "output_per_m": v["output_per_m"],
                                  "retained_frozen_because_live_is_lower":
                                      v["retained_frozen_because_live_is_lower"]}
                              for k, v in acc["effective_prices_per_m"].items()},
   "staleness_rule": (
       "this snapshot is evidence as of its capture timestamp. Before any PAID execution the "
       "capture MUST be repeated under the same protocol hash and must re-accept with 0 failures. "
       "A snapshot older than 14 days at execution time, or any re-capture that does not accept, "
       "is a METADATA_PREREQUISITE_FAILED and no arm runs. The snapshot is never 'refreshed' by "
       "editing this artifact."),
 },

 "cost_model": {
   **{k: v for k, v in V4["cost_model"].items()
      if k not in ("remaining_hard_headroom_usd", "headroom_after_the_completion_bound_usd",
                   "recomputed_from_ledger_at")},
   "recomputed_from_captured_metadata_at": ts,
   "conservative_arm_costs_usd": acc["conservative_arm_costs_usd"],
   "conservative_campaign_maximum_usd": maxi,
   "matches_frozen_maximum": maxi == 0.096896,
   "remaining_hard_headroom_usd": round(HARD_ABS - L0, 8),
   "headroom_after_the_completion_bound_usd": round(HARD_ABS - (L0 + maxi), 8),
   "complete_campaign_fits_under_the_hard_limit": (L0 + maxi) <= HARD_ABS,
 },

 "budget": {
   "L0_verified_from_disk": L0,
   "ledger_rows_at_freeze": len(led),
   "campaign_family_absolute_limits_preserved": {"warning": WARN_ABS, "hard": HARD_ABS},
   "prospective_warning_increment": round(WARN_ABS - L0, 8),
   "prospective_hard_increment": round(HARD_ABS - L0, 8),
   "v4_authorization_unused_and_non_transferable": True,
   "no_v5_budget_manifest_created": True,
   "NOT_AUTHORIZED": ("PROSPECTIVE limits. No V5 campaign budget manifest exists and this freeze "
                      "authorizes nothing. Paid execution requires a new explicit authorization."),
 },

 "qwen_arm_prior_risk": {
   "status": "RETAINED — all three arms are in V5",
   "evidence": ("this project measured the same FAMILY on this corpus locally in 2026-07: "
                "qwen3-vl 8B (Q4_K_M and q8_0) and 30B-a3b, across 14 configurations including "
                "blue-channel isolation, text subtraction, line pre-segmentation and contrast, at "
                "mean CER 0.94-1.63 with fluent hallucination."),
   "what_that_evidence_IS": "prior risk that the arm may fail, recorded so the result is not a surprise",
   "what_that_evidence_IS_NOT": (
       "proof that the remote 235B-a22b arm is dominated. Those were heavily quantised 8B and "
       "30B-a3b (3B active) models running locally; this arm is a 22B-active model served "
       "unquantised. The screen exists to measure it, not to confirm a prior."),
 },
}
body = json.dumps(v5, ensure_ascii=False, indent=1, sort_keys=True, default=str)
v5["experiment_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p = E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V5_2026-09-06.json"
p.write_text(json.dumps(v5, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("wrote", p)
print("V5 experiment_sha256:", v5["experiment_sha256"])
print(f"L0={L0} warn_inc={round(WARN_ABS-L0,8)} hard_inc={round(HARD_ABS-L0,8)}")
print(f"conservative maximum={maxi} headroom_after={v5['cost_model']['headroom_after_the_completion_bound_usd']}")
for c in cands:
    print(f"  {c['arm_id']:32s} {c['experiment_identity']}  pin={c['provider_pin']} -> "
          f"{c['provider_mapping']['resolved_display_name']!r} ({c['provider_mapping']['slug_mapping_status']})")
print("semantic identities per arm:", {c["arm_id"]: len(c["semantic_request_identity_by_case"]) for c in cands})
