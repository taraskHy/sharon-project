"""Create the AUTHORIZED V3 campaign budget manifest. ZERO provider calls."""
import json, hashlib, pathlib, subprocess, time

from autograder.campaignbudget import create_campaign_budget, load_campaign_budget

E = pathlib.Path("evaluation/model_selection/experiments")
P = pathlib.Path("evaluation/model_selection/policies")
V3 = json.loads((E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3_2026-09-05.json").read_text(encoding="utf-8"))
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert commit == "1259f3adfe6245fef431057080ae1b2daf091bd0", commit
assert V3["experiment_sha256"] == "4b84a0644064c95498f6091a22c7580f5ad1895411f1e0e9dba2efce582e4257"

led = [json.loads(l) for l in pathlib.Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl"
                                           ).read_text(encoding="utf-8").splitlines() if l.strip()]
L0 = round(sum(float(r.get("reported_cost") or 0) for r in led
               if r.get("cloud") and not r.get("cache_hit")), 8)
assert L0 == 0.70575904, L0
WARN_ABS, HARD_ABS = 0.78323229, 0.82323229
MAX_INC = 0.11747325
assert round(L0 + MAX_INC, 8) == HARD_ABS

path = P / "OCR_ALTSCREEN_V3_CAMPAIGN_BUDGET.json"
arm_costs = {a["arm_id"]: a["cost_bounds"]["one_attempt_per_case_maximum_usd"]
             for a in V3["candidates"]}

b = create_campaign_budget(
    campaign="OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3",
    experiment_sha256=V3["experiment_sha256"],
    starting_ledger_usd=L0,
    warning_increment_usd=round(WARN_ABS - L0, 8),
    hard_increment_usd=MAX_INC,
    predicted_arm_costs=arm_costs,
    path=path)

# Bind the authorization terms immutably alongside the envelope.
doc = json.loads(path.read_text(encoding="utf-8"))
doc.pop("content_sha256", None)
doc["authorization"] = {
    "granted_for": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3 only",
    "starting_code_commit": commit,
    "experiment_sha256": V3["experiment_sha256"],
    "cache_identity_version": V3["identity_and_cache_policy"]["identity_version"],
    "maximum_authorized_increment_usd": MAX_INC,
    "nominal_expected_increment_usd": V3["cost_model"]["nominal_expected_usd"],
    "frozen_complete_single_attempt_maximum_usd": V3["cost_model"]["single_attempt_maximum_usd"],
    "warning_absolute_usd": WARN_ABS,
    "hard_absolute_usd": HARD_ABS,
    "logical_requests": 24,
    "permitted_physical_attempts_per_logical_request": 1,
    "transport_retries": 0,
    "cache_policy": "refresh",
    "cache_hits_allowed": 0,
    "arm_order": [a["arm_id"] for a in V3["candidates"]],
    "arm_experiment_identities": {a["arm_id"]: a["experiment_identity"] for a in V3["candidates"]},
    "v1_and_v2_outputs_excluded": True,
    "held_out_forbidden": True,
    "not_authorized": ["retries", "reruns", "Seen-32", "replacement cases",
                       "changed configuration", "grading", "RAG", "HELD_OUT",
                       "any other inference"],
    "frozen_cost_stop_rule": (
        "if actual accounting invalidates the frozen complete-campaign maximum of "
        "$0.096896, stop mechanically EVEN IF the $0.11747325 authorization still has "
        "headroom. The frozen figure binds independently of the grant."),
}
body = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True, default=str)
doc["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
path.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str),
                encoding="utf-8", newline="\n")

r = load_campaign_budget(path)
print("wrote", path)
print("content_sha256      :", doc["content_sha256"])
print("L0                  :", r.starting_ledger_usd)
print("warn absolute       :", r.warn_usd)
print("hard absolute       :", r.hard_usd)
print("hard increment      :", r.hard_increment_usd)
print("arm costs           :", r.predicted_arm_costs)
print("campaign worst case :", r.predicted_campaign_worst_case_usd)
print("self-hash verifies on load: True")
assert round(r.warn_usd, 8) == WARN_ABS and round(r.hard_usd, 8) == HARD_ABS
assert r.predicted_campaign_worst_case_usd == V3["cost_model"]["single_attempt_maximum_usd"]
print("bound to experiment :", r.experiment_sha256 == V3["experiment_sha256"])
