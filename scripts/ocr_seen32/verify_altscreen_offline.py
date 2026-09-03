"""Offline pre-flight for OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1. ZERO calls.

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
from autograder.campaignbudget import load_campaign_budget
from autograder.cloudboundary import approved_cloud_ocr_systems
from autograder.rawcapture import requested_route_of

SCREEN = Path("evaluation/model_selection/experiments/"
              "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1_2026-09-03.json")
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

    print("\n== one shared campaign budget ==")
    b = load_campaign_budget(screen["campaign_budget_manifest"])
    check("budget bound to this exact experiment", b.experiment_sha256 == screen["experiment_sha256"])
    check("warning increment is $0.08", b.warning_increment_usd == 0.08)
    check("hard increment is $0.12", b.hard_increment_usd == 0.12)
    check("thresholds are absolute (L0 + increment)",
          round(b.starting_ledger_usd + 0.12, 8) == round(b.hard_usd, 8),
          f"L0={b.starting_ledger_usd} hard={b.hard_usd}")
    check("envelope covers all three arms at worst case",
          b.hard_usd >= b.starting_ledger_usd + b.predicted_campaign_worst_case_usd,
          f"worst={b.predicted_campaign_worst_case_usd}")
    check("all three arms priced in the manifest", len(b.predicted_arm_costs) == 3)

    print("\n== ledger untouched ==")
    led = [json.loads(l) for l in Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    cum = sum(float(r.get("reported_cost") or 0) for r in led
              if r.get("cloud") and not r.get("cache_hit"))
    check("ledger cumulative == L0", round(cum, 8) == round(b.starting_ledger_usd, 8),
          f"{cum:.8f}")
    check("no run directory for the screen",
          not Path("evaluation/model_selection/runs_altscreen").exists())

    print(f"\n{'=' * 70}\n{len(ok)} checks PASS, {len(problems)} PROBLEM(S)")
    if problems:
        for p in problems:
            print("  !", p)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
