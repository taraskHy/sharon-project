"""Mechanical integrity check for one executed arm of
OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1. ZERO provider calls.

Deliberately MECHANICAL ONLY. It does not look at transcription quality: the
checkpoint between arms exists to catch safety failures (linkage, archiving,
route, budget, secrets), and reacting to early OCR quality would be reading the
result before the campaign finishes.

Usage:  python -m scripts.ocr_seen32.altscreen_integrity_check <run_dir>
"""
import json
import re
import sys
from pathlib import Path

from autograder.campaignbudget import load_campaign_budget

SCREEN = Path("evaluation/model_selection/experiments/"
              "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1_2026-09-03.json")
LEDGER = Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl")
SECRET_RE = re.compile(r"sk-or-v1-[A-Za-z0-9]{8}|or-v1-[a-f0-9]{12}|Bearer\s+[A-Za-z0-9._\-]{12}", re.I)

ok, bad = [], []


def ck(label, cond, detail=""):
    (ok if cond else bad).append(label)
    print(f"  {'PASS' if cond else 'FAIL':4s} {label}" + (f"  {detail}" if detail else ""))


def _read(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(run_dir: str) -> int:
    d = Path(run_dir)
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    budget = load_campaign_budget(screen["campaign_budget_manifest"])
    outputs = _read(d / "outputs.jsonl")
    raw = _read(d / "raw_responses.jsonl") if (d / "raw_responses.jsonl").exists() else []
    ledger = _read(LEDGER)
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    run_id = run["run_id"]
    arm_rows = [r for r in ledger if r.get("job_id") == run_id]

    print(f"== arm {run_id} ==")
    print(f"   logical rows {len(outputs)} | raw responses {len(raw)} | ledger rows {len(arm_rows)}")

    # ---- 1. every physical send has a unique attempt_id ----------------------
    print("\n== attempt identity ==")
    raw_ids = [r.get("attempt_id") for r in raw]
    ck("every raw response carries an attempt_id", all(raw_ids) and bool(raw_ids),
       f"{len(raw_ids)} attempts")
    ck("attempt_ids are unique", len(set(raw_ids)) == len(raw_ids),
       f"{len(set(raw_ids))} distinct")

    # ---- 2. linkage: raw <-> route record <-> logical row <-> ledger ---------
    print("\n== linkage ==")
    row_attempt_ids, rows_with_records = [], 0
    for r in outputs:
        recs = r.get("attempt_records") or []
        if recs:
            rows_with_records += 1
            row_attempt_ids += [a.get("attempt_id") for a in recs]
    ledger_ids = [r.get("attempt_id") for r in arm_rows if r.get("attempt_id")]
    ck("logical rows carry per-attempt route records", rows_with_records > 0,
       f"{rows_with_records}/{len(outputs)} rows")
    ck("route-record ids are a subset of archived ids",
       set(row_attempt_ids) <= set(raw_ids),
       f"{len(set(row_attempt_ids) - set(raw_ids))} orphan(s)")
    ck("ledger attempt_ids are a subset of archived ids",
       set(ledger_ids) <= set(raw_ids),
       f"{len(set(ledger_ids) - set(raw_ids))} orphan(s)")
    ck("every archived attempt reaches the ledger",
       set(raw_ids) <= set(ledger_ids) or not ledger_ids,
       f"{len(set(raw_ids) - set(ledger_ids))} unledgered")
    ck("archive carries the campaign and arm id",
       all(r.get("campaign_id") and r.get("arm_id") for r in raw))
    ck("archive carries a logical_request_id",
       all(r.get("logical_request_id") for r in raw))

    # ---- 3. no archive failure, no route violation ---------------------------
    print("\n== safety conditions ==")
    marker = d / "raw_responses.jsonl.ARCHIVE_FAILURE"
    ck("no archive failure marker", not marker.exists())
    ck("no row reports an archive failure",
       not any(r.get("archive_failure") for r in outputs))
    viol = [r for r in raw if (r.get("route_check") or {}).get("violation")]
    ck("no explicit route violation in any archived attempt", not viol,
       f"{len(viol)} violation(s)")
    ck("no row stopped on a route violation",
       not any(r.get("error_type") == "RouteViolation" for r in outputs))
    ck("no row stopped on a campaign-budget refusal",
       not any(r.get("error_type") == "CampaignBudgetExceeded" for r in outputs))
    ck("no HELD_OUT case present",
       not any(str(r.get("split", "")).upper() == "HELD_OUT" for r in outputs))

    # ---- 4. configuration is the frozen one ---------------------------------
    print("\n== frozen configuration ==")
    arm = next((a for a in screen["candidates"]
                if a["model"] == run["config"]["route"]["model"]
                and (a["provider_routing"] == (run["config"]["route"]
                     .get("extra_generation", {}) or {}).get("provider"))), None)
    ck("route matches a frozen arm", arm is not None,
       run["config"]["route"]["model"])
    ck("prompt version is m2-strict-v1",
       run["config"]["prompt_version"] == "m2-strict-v1")
    ck("schema hash matches the freeze",
       run["config"]["schema_sha256"] == screen["schema"]["sha256"])
    ck("case ids and order match the freeze",
       [r["case_id"] for r in outputs] == screen["population"]["ordered_case_ids"][:len(outputs)])
    if arm:
        pr = arm["provider_routing"]
        ck("requested order is exactly the frozen provider",
           all(r.get("requested_provider_order") == pr["order"] for r in raw))
        ck("allow_fallbacks false on every attempt",
           all(r.get("allow_fallbacks") is False for r in raw))

    # ---- 5. accounting -------------------------------------------------------
    print("\n== campaign accounting ==")
    cum = sum(float(r.get("reported_cost") or 0) for r in ledger
              if r.get("cloud") and not r.get("cache_hit"))
    arm_cost = sum(float(r.get("reported_cost") or 0) for r in arm_rows)
    spent = round(cum - budget.starting_ledger_usd, 8)
    ck("cumulative ledger is below the fixed hard threshold",
       cum <= budget.hard_usd, f"{cum:.8f} <= {budget.hard_usd:.8f}")
    ck("campaign spend so far is within the $0.12 authorization",
       spent <= 0.12, f"${spent:.6f}")
    remaining = round(budget.hard_usd - cum, 8)
    print(f"   arm cost ${arm_cost:.6f} | campaign spent ${spent:.6f} | remaining ${remaining:.6f}")

    # ---- 6. remaining arms still fit ----------------------------------------
    done = {a for a in budget.predicted_arm_costs if a == (arm or {}).get("arm_id")}
    todo = {k: v for k, v in budget.predicted_arm_costs.items() if k not in done}
    ck("remaining arms' worst case fits below the hard threshold",
       sum(todo.values()) <= remaining + 1e-9,
       f"need ${sum(todo.values()):.6f}, have ${remaining:.6f}")

    # ---- 7. no secrets -------------------------------------------------------
    print("\n== secrets ==")
    leaks = []
    for f in (d / "raw_responses.jsonl", d / "outputs.jsonl", d / "run.json"):
        if f.exists() and SECRET_RE.search(f.read_text(encoding="utf-8")):
            leaks.append(f.name)
    ck("no secret-shaped token in arm artifacts", not leaks, str(leaks))
    ck("no authorization header archived",
       not any("authorization" in json.dumps(r.get("headers") or {}).lower() for r in raw))

    print("\n" + "=" * 68)
    print(f"{len(ok)}/{len(ok)+len(bad)} mechanical checks PASS")
    if bad:
        print("STOP CONDITIONS TRIGGERED:")
        for b in bad:
            print("  !", b)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
