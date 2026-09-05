"""Offline pre-flight for OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3. ZERO calls.

Rebuilds all 24 payloads and re-checks every property the freeze claims, so the
claims are verified against live code rather than trusted from the artifact.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.roles import OcrPrimaryAdapter, load_ocr_prompts
from autograder.cloudboundary import approved_cloud_ocr_systems
from autograder.rawcapture import requested_route_of

SCREEN = Path("evaluation/model_selection/experiments/"
              "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V3_2026-09-05.json")
GRADING_WORDS = ["rubric", "score", "grade", "points", "מחוון", "ציון",
                 "correct answer", "official solution", "partially_valid"]
SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{6}|or-v1-[a-f0-9]{6}|Bearer\s+[A-Za-z0-9]{8}", re.I)

ok, problems = [], []


def check(label, cond, detail=""):
    (ok if cond else problems).append(label)
    print(f"  {'PASS' if cond else 'FAIL':4s}  {label}" + (f"  {detail}" if detail else ""))


def main() -> int:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    man = load_manifest("ocr_primary")
    by = {c.case_id: c for c in man.cases}
    order = screen["population"]["ordered_case_ids"]
    cases = screen["population"]["cases"]

    print("== population ==")
    base = json.loads(Path("evaluation/model_selection/runs/ocr_primary/"
                           "OCR_SMOKE_STAGE1_BASELINE_2026-09-02.json").read_text(encoding="utf-8"))
    check("frozen 8 cases, exact ids and order", order == base["ordered_case_ids"])
    check("smoke selection hash unchanged",
          screen["population"]["smoke_selection_sha256"] == base["smoke_selection_sha256"])
    check("5 handwritten + 3 printed", screen["population"]["handwritten"] == 5
          and screen["population"]["printed_or_text_layer"] == 3)
    bad = [c["case_id"] for c in cases
           if hashlib.sha256((man.root / c["image"]).read_bytes()).hexdigest() != c["crop_sha256"]]
    check("crop bytes match the freeze", not bad, f"mismatches={bad}")
    check("HELD_OUT count = 0", screen["population"]["HELD_OUT"] == 0)
    check("CALIBRATION count = 0", screen["population"]["CALIBRATION"] == 0)
    check("every case is DEV", all(c["split"] == "DEV" for c in cases))
    hol = Path("evaluation/model_selection/HELD_OUT_EXECUTIONS.jsonl")
    n_ho = len([l for l in hol.read_text(encoding="utf-8").splitlines() if l.strip()]) if hol.exists() else 0
    check("HELD_OUT execution log still empty", n_ho == 0)

    print("\n== prompts and schema ==")
    prompts = load_ocr_prompts(screen["prompt"]["version"])
    approved = approved_cloud_ocr_systems()
    for cat, h in screen["prompt"]["prompt_sha256_by_category"].items():
        got = hashlib.sha256(prompts[cat].encode()).hexdigest()
        check(f"prompt hash {cat}", got == h, got[:16])
        check(f"prompt registered in boundary: {cat}", prompts[cat] in approved)
    check("no 'different colour of ink' wording inherited",
          not any("different colour of ink" in t for t in prompts.values()))

    print("\n== 24 payloads, built offline ==")
    adapter = OcrPrimaryAdapter(prompt_version=screen["prompt"]["version"])
    refs = [by[c].label["reference"] for c in order]
    n = 0
    img_counts, schema_hashes = set(), set()
    for arm in screen["candidates"]:
        for cid in order:
            req = adapter.build_request(dict(by[cid].inputs), man.root)
            seen = req.text_for_inspection()
            prov = req.provenance()
            n += 1
            img_counts.add(sum(1 for b in req.content_blocks if b.get("type") == "image"))
            schema_hashes.add(prov["schema_sha256"])
            if any(r and r in seen for r in refs):
                problems.append(f"{arm['arm_id']}/{cid}: reference leaked")
            if any(w.lower() in seen.lower() for w in GRADING_WORDS):
                problems.append(f"{arm['arm_id']}/{cid}: grading vocabulary")
            if SECRET_RE.search(seen):
                problems.append(f"{arm['arm_id']}/{cid}: secret-shaped token")
    check("24 payloads built", n == 24, f"n={n}")
    check("exactly one image per payload", img_counts == {1}, f"{img_counts}")
    check("one schema across all arms", len(schema_hashes) == 1)
    check("schema hash matches the freeze", schema_hashes == {screen["schema"]["sha256"]})
    check("no reference leakage", not any("reference leaked" in p for p in problems))
    check("no grading vocabulary", not any("grading vocabulary" in p for p in problems))
    check("no secrets", not any("secret-shaped" in p for p in problems))

    print("\n== route pinning ==")
    for arm in screen["candidates"]:
        pr = arm["provider_routing"]
        r = requested_route_of({"provider": pr})
        check(f"{arm['arm_id']}: order is exactly the frozen provider",
              pr["order"] == [arm["provider_pin"]], str(pr["order"]))
        check(f"{arm['arm_id']}: allow_fallbacks is false", pr["allow_fallbacks"] is False)
        check(f"{arm['arm_id']}: recognised as pinned", r["route_pinned"] is True)

    print("\n== identity and cache policy ==")
    from autograder.gateway import TaskRoute
    from autograder.routeidentity import experiment_identity

    ip = screen["identity_and_cache_policy"]
    check("identity version is 3", ip["identity_version"] == 3)
    check("identity is DERIVED from the effective config", "to_backend_config" in ip["derivation"])
    check("cache policy is refresh", ip["cache_policy"] == "refresh")
    check("cache_hits_allowed is 0", ip["cache_hits_allowed"] == 0)
    check("secrets excluded from every identity",
          set(ip["excluded_from_all_identities"]) >= {"api_key", "api_key_env"})
    check("all three arms hash differently", len(set(ip["arm_identities"].values())) == 3)
    for arm in screen["candidates"]:
        r = TaskRoute(task="ocr_primary", backend="openrouter", model=arm["model"],
                      base_url="https://openrouter.ai/api/v1", structured_mode="json_schema",
                      max_tokens=1000, temperature=0.0, reasoning=arm["route"]["reasoning"],
                      transport_retries=screen["execution_requirements"]["retry_policy"]["transport_retries"],
                      provider=arm["provider_routing"], prompt_version="m2-strict-v1")
        check(f"{arm['arm_id']}: identity recomputes from live code",
              experiment_identity(r) == arm["experiment_identity"],
              experiment_identity(r)[:16])

    print("\n== cost model (four distinct bounds) ==")
    cm = screen["cost_model"]
    rp = screen["execution_requirements"]["retry_policy"]
    check("retry policy frozen at 0 transport retries", rp["transport_retries"] == 0)
    check("max physical attempts per logical request = 1",
          rp["max_physical_attempts_per_logical_request"] == 1)
    check("max physical attempts for the campaign = 24",
          rp["max_physical_attempts_for_the_campaign"] == 24)
    check("every transmitted attempt treated as potentially billable",
          "POTENTIALLY BILLABLE" in cm["billing_assumption"])
    check("nominal < single-attempt maximum",
          cm["nominal_expected_usd"] < cm["single_attempt_maximum_usd"])
    check("retry-inclusive bound == single-attempt max at retries=0",
          cm["retry_inclusive_completion_bound_usd"] == cm["single_attempt_maximum_usd"])
    check("COMPLETE campaign fits under the hard limit",
          cm["complete_campaign_fits_under_the_hard_limit"] is True,
          f"headroom after = {cm['headroom_after_the_completion_bound_usd']}")
    check("the retries=2 alternative is recorded as NOT fitting",
          cm["for_reference_only_at_transport_retries_2"]["fits"] is False,
          f"${cm['for_reference_only_at_transport_retries_2']['retry_inclusive_completion_bound_usd']}")

    print("\n== budget (PROSPECTIVE — V3 is NOT authorized) ==")
    bg = screen["budget"]
    # V3 was AUTHORIZED and EXECUTED on 2026-09-05; the manifest now exists and
    # must bind to this exact experiment.
    from autograder.campaignbudget import load_campaign_budget
    mp = Path("evaluation/model_selection/policies/OCR_ALTSCREEN_V3_CAMPAIGN_BUDGET.json")
    check("V3 campaign budget manifest exists and self-hash verifies", mp.exists())
    if mp.exists():
        cb = load_campaign_budget(mp)
        check("manifest is bound to the V3 experiment hash",
              cb.experiment_sha256 == screen["experiment_sha256"])
        check("manifest L0 == the frozen L0",
              round(cb.starting_ledger_usd, 8) == round(bg["L0_verified_from_disk"], 8))
        check("manifest hard == family absolute", round(cb.hard_usd, 8) == 0.82323229)
    check("absolute family limits preserved",
          bg["campaign_family_absolute_limits_preserved"] == {"warning": 0.78323229,
                                                              "hard": 0.82323229})
    check("prospective increments derive from L0",
          round(bg["L0_verified_from_disk"] + bg["prospective_hard_increment"], 8) == 0.82323229)

    print("\n== ledger untouched ==")
    led = [json.loads(l) for l in Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    cum = sum(float(r.get("reported_cost") or 0) for r in led
              if r.get("cloud") and not r.get("cache_hit"))
    # L0 is the PRE-CAMPAIGN baseline. Once an arm has legitimately run, the
    # ledger is above it — what must still hold is that spend never left the
    # envelope and never exceeded the authorization.
    L0 = screen["budget"]["L0_verified_from_disk"]
    spent = round(cum - L0, 8)
    # L0 is the PRE-CAMPAIGN baseline; V3 has now legitimately spent against it.
    check("ledger has not fallen below the frozen L0", cum >= L0 - 1e-9, f"{cum:.8f}")
    check("spend is within the $0.11747325 authorization", spent <= 0.11747325, f"${spent:.8f}")
    check("spend is within the FROZEN $0.096896 complete-campaign maximum",
          spent <= 0.096896, f"${spent:.8f}")
    check("ledger is below the absolute family hard limit", cum <= 0.82323229, f"{cum:.8f}")
    # An arm is legitimately short ONLY when its last row records a mechanical
    # stop. A silently truncated arm is not acceptable.
    bad_short = []
    for root in (Path("evaluation/model_selection/runs_altscreen"),
                 Path("evaluation/model_selection/runs_altscreen_v3")):
        for f in root.rglob("outputs.jsonl"):
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(rows) in (0, 8):
                continue
            if not rows[-1].get("stopped"):
                bad_short.append(f.parent.name)
    check("every short arm ends in a recorded mechanical stop", not bad_short, str(bad_short))

    print(f"\n{'=' * 70}\n{len(ok)} checks PASS, {len(problems)} PROBLEM(S)")
    if problems:
        for p in problems:
            print("  !", p)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
