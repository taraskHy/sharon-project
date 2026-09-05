"""Close V3 immutably and freeze V4. ZERO provider calls."""
import json, hashlib, pathlib, subprocess, time

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.roles import OcrPrimaryAdapter
from autograder.providermap import load_provider_map, source_digest
from autograder.routeidentity import CACHE_IDENTITY_VERSION, identities_from_argv

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
E = pathlib.Path("evaluation/model_selection/experiments")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ts = time.strftime("%Y-%m-%d %H:%M:%S")
V3 = json.loads((E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3_2026-09-05.json").read_text(encoding="utf-8"))
V3RES = json.loads((R / "OCR_ALTSCREEN_V3_RESULT_2026-09-05.json").read_text(encoding="utf-8"))
MANIFEST = pathlib.Path("evaluation/model_selection/policies/OCR_ALTSCREEN_V3_CAMPAIGN_BUDGET.json")
led = [json.loads(l) for l in pathlib.Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl"
                                           ).read_text(encoding="utf-8").splitlines() if l.strip()]
cum = round(sum(float(r.get("reported_cost") or 0) for r in led
                if r.get("cloud") and not r.get("cache_hit")), 8)
assert cum == 0.71783254, cum
assert len(led) == 815, len(led)


def seal(doc, field="content_sha256"):
    body = json.dumps({k: v for k, v in doc.items() if k != field},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    doc[field] = hashlib.sha256(body.encode()).hexdigest()
    return doc


# ------------------------------------------------------------- V3 closure ----
v3c = seal({
 "artifact": "ocr_altscreen_v3_closure",
 "created_at": ts, "git_commit": commit, "provider_calls": 0,
 "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3",
 "experiment_sha256": V3["experiment_sha256"],
 "outcome": "INCONCLUSIVE_MECHANICAL_STOP",
 "gates_not_evaluated": True,

 "provenance": {
   "original_code_head": "1259f3adfe6245fef431057080ae1b2daf091bd0",
   "budget_manifest_commit": "d07dd20b6f783502db141896a2cad95cbbdd3f95",
   "budget_manifest_content_sha256":
       json.loads(MANIFEST.read_text(encoding="utf-8"))["content_sha256"],
   "results_commit": "17438df5b552a3893ec7d08db0249e85ac517db4",
 },

 "execution": {
   "gemini_pinned_ai_studio_completed_attempts": 8,
   "gemini_pinned_vertex_completed_attempts": 1,
   "qwen3_vl_235b_pinned_alibaba": "NOT STARTED",
   "additional_spend_usd": 0.01207350,
   "ledger_before": {"rows": 806, "usd": 0.70575904},
   "ledger_after": {"rows": 815, "usd": 0.71783254},
 },

 "false_positive_route_violation": {
   "arm": "gemini_pinned_vertex", "case": "hl_e003_q1_r1__l1",
   "requested_slug": "google-vertex", "observed_provider": "Google",
   "verdict_at_the_time": "VIOLATION (campaign halted)",
   "verdict_under_the_canonical_mapping": "COMPLIANT",
   "cause": ("the check normalised a provider SLUG and a DISPLAY NAME to the same string space. "
             "OpenRouter reports slug `google-vertex` as display name `Google`, so the pin was "
             "honoured. `google-ai-studio`/`Google AI Studio` normalise identically by "
             "coincidence, which is the only reason arm 1 passed."),
   "fixed_in": "autograder/providermap.py (canonical mapping from a preserved artifact)",
 },

 "frozen_versus_runtime_identity_mismatch": {
   "arms_affected": ["gemini_pinned_ai_studio", "gemini_pinned_vertex"],
   "sole_differing_field": "base_url",
   "cause": ("V3's identities were computed from a hand-built TaskRoute carrying the explicit "
             "OpenRouter URL; build_route() leaves base_url None and the backend supplies its "
             "own endpoint."),
   "also_discovered_during_the_v4_repair": (
       "the Qwen arm was mis-frozen the SAME way and more severely: the runtime path gives it "
       "max_tokens=400 and reasoning={'effort':'none'} from the production route, while V3 froze "
       "1000 with no reasoning. It was never detected because arm 3 never ran, and sending a "
       "reasoning parameter to a model with no reasoning support is what destroyed the Stage-1 "
       "Gemini arm with a pre-inference HTTP 400."),
   "fixed_in": ("routeidentity.canonical_base_url + identities_from_argv (frozen identities are "
                "now derived through the real CLI path), and a declared Qwen candidate_override"),
 },

 "PROTOCOL_DEVIATION": {
   "what": ("after arm 1's frozen-versus-runtime identity mismatch was detected, the campaign "
            "continued and issued the first Vertex request."),
   "assessment": ("this remained inside the financial authorization and no unauthorized "
                  "configuration was transmitted, but it VIOLATED the experiment-integrity rule. "
                  "A detected identity mismatch means the arm being measured is not demonstrably "
                  "the arm that was frozen, and the correct action was to stop the campaign at "
                  "that point rather than to judge the differing field harmless."),
   "accountability": ("the judgement was mine and it was wrong. 'The differing field looks "
                      "harmless' is precisely the reasoning an integrity gate exists to "
                      "override."),
   "remedy": ("identity equality is now a MANDATORY hard stop checked before any cache read and "
              "before any send (routeidentity.assert_identity_matches, RunSpec.expect_identity, "
              "--expect-identity). No field-level judgement is permitted."),
 },

 "evidence_status": {
   "all_nine_outputs_excluded_from_v4_metrics_and_advancement": True,
   "ai_studio_measurements": ("DIAGNOSTIC EVIDENCE ONLY. They may be cited to describe what was "
                              "observed, and for nothing else. They may NOT receive a formal DROP "
                              "decision and may NOT be combined with future V4 outputs for a "
                              "confirmatory paired comparison."),
   "vertex_attempt": "one attempt, stopped; diagnostic only",
   "immutability": ("V3, its budget manifest, its raw responses and its result artifact are "
                    "preserved byte-for-byte and are neither edited nor re-hashed."),
 },
 "ocr_primary_role_status": "UNSELECTED (unchanged)",
 "successor": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V4",
})
p1 = R / "OCR_ALTSCREEN_V3_CLOSURE_2026-09-05.json"
p1.write_text(json.dumps(v3c, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

# ------------------------------------------------------------------- V4 -----
GEM, QWEN = "google/gemini-3.7-flash", "qwen/qwen3-vl-235b-a22b-instruct"
ARMS = [("gemini_pinned_ai_studio", GEM, "google-ai-studio"),
        ("gemini_pinned_vertex", GEM, "google-vertex"),
        ("qwen3_vl_235b_pinned_alibaba", QWEN, "alibaba")]
RUNS_ROOT = "evaluation/model_selection/runs_altscreen_v4"


def argv_for(model, pin):
    return ("bench run --role ocr_primary --split dev --candidate " + model +
            " --subset smoke --prompt-version m2-strict-v1 --research"
            " --models-config models.toml --cache-policy refresh --transport-retries 0"
            ' --provider {"order":["' + pin + '"],"allow_fallbacks":false}'
            " --i-understand-this-spends-money").split()


man = load_manifest("ocr_primary"); by = {c.case_id: c for c in man.cases}
adapter = OcrPrimaryAdapter(prompt_version="m2-strict-v1")
order = V3["population"]["ordered_case_ids"]
pmap = load_provider_map()

cands = []
for arm_id, model, pin in ARMS:
    v3arm = next(a for a in V3["candidates"] if a["arm_id"] == arm_id)
    argv = argv_for(model, pin)
    req0 = adapter.build_request(dict(by[order[0]].inputs), man.root)
    ident = identities_from_argv(argv, output_model=req0.output_model,
                                 system=req0.system, content_blocks=req0.content_blocks,
                                 max_tokens=1000)
    sem = {}
    for cid in order:
        rq = adapter.build_request(dict(by[cid].inputs), man.root)
        sem[cid] = identities_from_argv(argv, output_model=rq.output_model, system=rq.system,
                                        content_blocks=rq.content_blocks, max_tokens=1000
                                        )["semantic_request_identity"]
    e = pmap.get(pin)
    cands.append({
        "arm_id": arm_id, "model": model, "provider_pin": pin,
        "provider_routing": {"order": [pin], "allow_fallbacks": False},
        "arm_type": v3arm["arm_type"],
        "why_genuinely_different": v3arm["why_genuinely_different"],
        "main_risk": v3arm["main_risk"],
        "cost_bounds": v3arm["cost_bounds"],
        "cli_argv": argv,
        "experiment_identity": ident["experiment_identity"],
        "semantic_request_identity_by_case": sem,
        "effective_config": ident["effective_config"],
        "wire_response_format": ident["wire_response_format"],
        "provider_mapping": {
            "requested_slug": pin,
            "expected_display_names": list(e.display_names) if e else [],
            "slug_mapping_status": e.status if e else "UNKNOWN_UNRECOGNISED",
            "evidence": e.evidence if e else None,
        },
    })

assert len({c["experiment_identity"] for c in cands}) == 3
for c in cands:
    assert c["effective_config"]["base_url"] == "https://openrouter.ai/api/v1"
    assert c["effective_config"]["transport_retries"] == 0
    assert c["effective_config"]["max_tokens"] == 1000

L0 = cum
WARN_ABS, HARD_ABS = 0.78323229, 0.82323229
warn_inc, hard_inc = round(WARN_ABS - L0, 8), round(HARD_ABS - L0, 8)
nominal = V3["cost_model"]["nominal_expected_usd"]
single = V3["cost_model"]["single_attempt_maximum_usd"]
headroom_after = round(HARD_ABS - (L0 + single), 8)
fits = (L0 + single) <= HARD_ABS

v4 = {
 "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V4",
 "status": "FROZEN - NOT EXECUTED - NOT AUTHORIZED",
 "created_at": ts, "git_commit": commit, "provider_calls_made_preparing_this": 0,

 "supersedes": {
   "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3",
   "experiment_sha256": V3["experiment_sha256"],
   "outcome": "INCONCLUSIVE_MECHANICAL_STOP",
   "closure_artifact": "OCR_ALTSCREEN_V3_CLOSURE_2026-09-05.json",
   "v1_v2_v3_outputs_excluded_from_v4_evaluation": True,
   "substantive_corrections": [
     "canonical provider slug/display-name matching from a preserved artifact",
     "frozen identities derived through the real CLI -> build_route runtime path",
     "base_url canonicalised to the effective endpoint",
     "mandatory pre-send identity equality (zero cache reads, zero sends on mismatch)",
     "a DECLARED Qwen candidate_override, without which the runtime path would send "
     "max_tokens=400 and an unsupported reasoning parameter — a divergence discovered "
     "only while eliminating hand-built identities",
   ],
 },

 "question": V3["question"],
 "population": V3["population"],
 "prompt": V3["prompt"],
 "schema": V3["schema"],
 "adapter_version": V3["adapter_version"],
 "candidates": cands, "candidate_count": len(cands),
 "live_pricing_snapshot": V3["live_pricing_snapshot"],
 "advancement_and_drop_rules_stated_in_advance": V3["advancement_and_drop_rules_stated_in_advance"],
 "prohibitions": V3["prohibitions"],

 "provider_mapping_source": source_digest(),
 "provider_mapping_caveat": (
     "the `alibaba` slug has NO preserved slug->display-name evidence; the 2026-09-03 discovery "
     "read it from OpenRouter's /providers endpoint but never persisted the response. Its "
     "attribution will therefore report UNKNOWN_UNVERIFIED_SLUG — never silently compliant and "
     "never a violation. Capturing and committing a provider catalogue (public metadata, no "
     "inference) is a PREREQUISITE for the Qwen arm's route attribution to mean anything."),

 "identity_and_cache_policy": {
   **{k: v for k, v in V3["identity_and_cache_policy"].items()
      if k not in ("identity_version", "arm_identities")},
   "identity_version": CACHE_IDENTITY_VERSION,
   "arm_identities": {c["arm_id"]: c["experiment_identity"] for c in cands},
   "derivation": ("identities are produced by identities_from_argv(), which runs the FROZEN CLI "
                  "argv through build_parser -> _spec_from_args -> load_registry -> build_route "
                  "-> to_backend_config. No parallel hand-built TaskRoute exists."),
   "base_url_canonicalisation": ("None and the explicit default resolve to the same effective "
                                 "endpoint and hash identically; a different endpoint does not."),
 },

 "execution_requirements": {
   **V3["execution_requirements"],
   "identity_equality_is_mandatory": (
       "before ANY cache lookup and before ANY send, the runtime route's experiment identity must "
       "equal the frozen arm value (--expect-identity). A mismatch yields zero cache reads, zero "
       "requests, INCONCLUSIVE_MECHANICAL_STOP for the whole campaign, and a persisted "
       "field-level safe diff. No field may be judged harmless."),
 },

 "cost_model": {
   **{k: v for k, v in V3["cost_model"].items() if k not in
      ("remaining_hard_headroom_usd", "headroom_after_the_completion_bound_usd",
       "complete_campaign_fits_under_the_hard_limit")},
   "remaining_hard_headroom_usd": hard_inc,
   "complete_campaign_fits_under_the_hard_limit": fits,
   "headroom_after_the_completion_bound_usd": headroom_after,
   "recomputed_from_ledger_at": ts,
 },

 "budget": {
   "L0_verified_from_disk": L0,
   "ledger_rows_at_freeze": len(led),
   "campaign_family_absolute_limits_preserved": {"warning": WARN_ABS, "hard": HARD_ABS},
   "prospective_warning_increment": warn_inc,
   "prospective_hard_increment": hard_inc,
   "v3_authorization_does_not_carry_over": True,
   "v3_budget_manifest_not_reused": True,
   "NOT_AUTHORIZED": ("PROSPECTIVE limits. No V4 campaign budget manifest is created here and "
                      "this freeze authorizes nothing."),
 },
}
body = json.dumps(v4, ensure_ascii=False, indent=1, sort_keys=True, default=str)
v4["experiment_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p2 = E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V4_2026-09-05.json"
p2.write_text(json.dumps(v4, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")

print("wrote", p1); print("wrote", p2)
print("V3 closure sha:", v3c["content_sha256"])
print("V4 experiment :", v4["experiment_sha256"])
print(f"L0={L0} warn_inc={warn_inc} hard_inc={hard_inc}")
print(f"nominal={nominal} single={single} fits={fits} headroom_after={headroom_after}")
for c in cands:
    print(f"  {c['arm_id']:32s} exp={c['experiment_identity']}")
