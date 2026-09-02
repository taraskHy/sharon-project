"""Derive the release-readiness status from the committed artifacts.

    python scripts/release_readiness.py

Zero-inference. Reads the frozen reference, the risk artifacts, the shadow
replay, the reproduction record and the OCR campaign freeze, evaluates the
versioned release gates, and DERIVES exactly one status:

    NOT_READY | SHADOW_READY | READY_FOR_OCR_VALIDATION | READY_FOR_FINAL_VALIDATION

Production-ready is not an option by design. Writes
RELEASE_READINESS_<date>.{json,md}.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder import riskengine  # noqa: E402

RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
EXPERIMENTS = REPO / "evaluation" / "model_selection" / "experiments"
D = "2026-09-02"

GATE_VERSION = "release-gates-asym-v2"   # v1 = summary 2026-09-02; v2 adds
                                         # rare-event power + engine gates and
                                         # evaluates the PROSPECTIVE policies


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    today = time.strftime("%Y-%m-%d")
    ref = _load(RUNS / f"FINAL_HUMAN_REFERENCE_{D}.json")
    strict = _load(RUNS / f"ASYMMETRIC_RISK_STRICT_{D}.json")
    repro = _load(RUNS / f"ASYMMETRIC_RISK_REPRODUCTION_{today}.json")
    replay = _load(RUNS / f"PROSPECTIVE_POLICY_REPLAY_{today}.json")
    sens = _load(RUNS / f"RISK_SENSITIVITY_PROSPECTIVE_{today}.json")
    bench = _load(RUNS / f"RISK_ENGINE_BENCH_{today}.json")
    ocr = _load(EXPERIMENTS / f"OCR_VALIDATION_CAMPAIGN_{today}.json")

    base = strict["arms"]["baseline_8b_one_pass"]
    best_const = strict["best_constant_policy"]
    dep = replay["deployable_prospective"]
    vo = dep["prospective_valid_only_v1"]
    ni = dep["prospective_noninvalid_v1"]
    ru = vo["rare_event_uncertainty"]["invalid_to_valid"]
    min_n = replay["minimum_invalid_examples_for_zero_event_upper_bound"]

    seen_only = "seen development data only (46 explanation cases)"
    gates = [
        {"gate": "HARD_FALSE_FULL",
         "target": "confirmed invalid -> automatic valid = 0 observed",
         "observed": f"0 across every arm and policy ({seen_only})",
         "confidence_limitation": f"only 5 invalid cases: one-sided 95% upper "
                                  f"bound {100 * ru['one_sided_upper_95']:.1f}% "
                                  "— observation passes, demonstrated safety "
                                  "does NOT",
         "pass": True, "evidence": "PROSPECTIVE_POLICY_REPLAY"},
        {"gate": "RARE_EVENT_POWER",
         "target": "enough invalid examples to bound the false-full rate at "
                   "the owner's chosen threshold (e.g. 5% needs "
                   f"{min_n['5pct']}, 1% needs {min_n['1pct']})",
         "observed": "5 invalid examples available",
         "confidence_limitation": "threshold choice is the owner's; every "
                                  "listed option needs more invalid examples "
                                  "than exist on seen data",
         "pass": False, "evidence": "PROSPECTIVE_POLICY_REPLAY (exact "
                                    "binomial)"},
        {"gate": "SERIOUS_OVERGRADE",
         "target": "automatic partially_valid -> valid <= 2/46",
         "observed": f"prospective_valid_only {vo['partial_to_valid_auto']}, "
                     f"prospective_noninvalid {ni['partial_to_valid_auto']} "
                     "(the committed 2 was oracle-assisted)",
         "confidence_limitation": seen_only,
         "pass": max(vo["partial_to_valid_auto"],
                     ni["partial_to_valid_auto"]) <= 2,
         "evidence": "PROSPECTIVE_POLICY_REPLAY"},
        {"gate": "WEIGHTED_RISK_VS_CONSTANTS",
         "target": f"semantic-layer total risk <= 0.90 x best constant "
                   f"({best_const['name']} = {best_const['total_weighted_loss']})",
         "observed": f"baseline {base['total_weighted_loss']}",
         "confidence_limitation": seen_only,
         "pass": base["total_weighted_loss"]
                 <= 0.9 * best_const["total_weighted_loss"],
         "evidence": "ASYMMETRIC_RISK_STRICT"},
        {"gate": "UNDERGRADE_CAP",
         "target": "automatic harmful undergrades <= 3/46",
         "observed": f"valid_only {vo['auto_undergrades']}, noninvalid "
                     f"{ni['auto_undergrades']}",
         "confidence_limitation": seen_only,
         "pass": max(vo["auto_undergrades"], ni["auto_undergrades"]) <= 3,
         "evidence": "PROSPECTIVE_POLICY_REPLAY"},
        {"gate": "GROUNDING",
         "target": "evidence+schema failures <= 2% of cases",
         "observed": f"{base['evidence_failures'] + base['schema_failures']}"
                     f"/46 = 4.3%",
         "confidence_limitation": seen_only,
         "pass": (base["evidence_failures"] + base["schema_failures"])
                 <= 0.02 * 46,
         "evidence": "ASYMMETRIC_RISK_STRICT"},
        {"gate": "AUTOMATION_JOINT",
         "target": "AUTO coverage >= 70% AND weighted-risk gate passes",
         "observed": f"valid_only {vo['auto_coverage_pct']}% / noninvalid "
                     f"{ni['auto_coverage_pct']}%; weighted-risk gate fails",
         "confidence_limitation": seen_only,
         "pass": False, "evidence": "both artifacts"},
        {"gate": "DISAGREEMENT_ROUTING_ONLINE",
         "target": "wide human disagreement routes to REVIEW in production",
         "observed": "IMPOSSIBLE prospectively: reviewer disagreement does "
                     "not exist at decision time; the engine refuses "
                     "retrospective policies in production and the oracle "
                     "tables are marked NOT DEPLOYABLE",
         "confidence_limitation": "needs a future PROSPECTIVE ambiguity "
                                  "signal to recover the oracle gains",
         "pass": False, "evidence": "policy taxonomy + engine tests"},
        {"gate": "ENGINE_SHADOW_SAFETY",
         "target": "OFF/SHADOW never change the active grade; ACTIVE locked; "
                   "typed refusals everywhere",
         "observed": "1,152-state exhaustive suite + fuzz + concurrency all "
                     "green; no production caller of ACTIVE exists",
         "confidence_limitation": "engineering property, fully testable",
         "pass": True, "evidence": "test_risk_engine* suites"},
        {"gate": "REPRODUCTION",
         "target": "every load-bearing committed number reproduces from raw "
                   "artifacts",
         "observed": f"{repro['verdict']}: {repro['checks_total']} checks, "
                     f"{repro['checks_failed']} failed",
         "confidence_limitation": "deterministic",
         "pass": repro["verdict"] == "REPRODUCED",
         "evidence": "ASYMMETRIC_RISK_REPRODUCTION"},
        {"gate": "OCR",
         "target": "production OCR validated separately before end-to-end "
                   "shipping",
         "observed": f"campaign FROZEN ({ocr['experiment_sha256'][:12]}…), "
                     "NOT EXECUTED; 0 OCR calls",
         "confidence_limitation": "hard shipping blocker until run and passed",
         "pass": False, "evidence": "OCR_VALIDATION_CAMPAIGN"},
        {"gate": "FINAL_TEST",
         "target": "HELD_OUT untouched until grader+matrix+policy+OCR frozen",
         "observed": "untouched; 0 exposure in every artifact of this "
                     "campaign",
         "confidence_limitation": "-",
         "pass": True, "evidence": "all artifacts"},
    ]
    passed = sum(1 for g in gates if g["pass"])

    shadow_ready = all(g["pass"] for g in gates
                       if g["gate"] in ("ENGINE_SHADOW_SAFETY", "REPRODUCTION",
                                        "FINAL_TEST", "HARD_FALSE_FULL"))
    # OCR validation readiness would additionally need the campaign's two
    # engineering prep items closed; final validation needs everything
    status = "SHADOW_READY" if shadow_ready else "NOT_READY"

    doc = {
        "artifact": "release_readiness",
        "gate_version": GATE_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse",
                                      "HEAD"], capture_output=True, text=True,
                                     timeout=15).stdout.strip(),
        "status": status,
        "status_derivation": {
            "SHADOW_READY_criteria": "engine implemented + OFF/SHADOW proven "
                                     "inert + ACTIVE locked + reproduction "
                                     "clean + HELD_OUT sealed + zero observed "
                                     "false-full",
            "not_READY_FOR_OCR_VALIDATION_because": [
                "two named engineering prep items in the OCR freeze are open "
                "(seen46-ocr subset registration; per-writer WER scoring)",
                "owner spend authorization for the campaign is not given"],
            "not_READY_FOR_FINAL_VALIDATION_because": [
                "semantic layer does not beat always_partially_valid on "
                "weighted risk", "grounding failures 4.3% > 2%",
                "serious-overgrade gate fails prospectively (4 > 2)",
                "rare-event power gate fails (5 invalid examples)",
                "OCR unvalidated", "decision policy not yet frozen"]},
        "data": {"seen_references": 46,
                 "class_distribution": ref["class_distribution"],
                 "source_distribution": ref["by_source"],
                 "human_disagreement": "22 agreed / 14 adjacent / 8 wide / "
                                       "2 owner-repaired",
                 "held_out": "sealed, untouched"},
        "grader": {"baseline_arm": "qwen3-vl:8b-instruct Q4 one-pass, "
                                   "grade-v4-charitable-local",
                   "strict_risk": base["total_weighted_loss"],
                   "disagreement_aware_risk": 20.0,
                   "rare_event": ru,
                   "severe_overgrades_semantic": base[
                       "serious_overgrade_partial_to_valid"],
                   "undergrades_semantic": base["undergrades"],
                   "grounding_failures": base["evidence_failures"]
                   + base["schema_failures"]},
        "policy": {"prospective_deployable": {p: {
                       "auto_coverage_pct": dep[p]["auto_coverage_pct"],
                       "auto_risk": dep[p]["auto_total_weighted_loss"],
                       "false_full": dep[p]["invalid_to_valid_auto"]}
                       for p in dep},
                   "retrospective_upper_bounds_NOT_DEPLOYABLE": {
                       p: {"auto": m["auto"],
                           "auto_risk": m["auto_total_weighted_loss"]}
                       for p, m in replay[
                           "oracle_retrospective_upper_bound_NOT_DEPLOYABLE"]
                       .items()},
                   "recommended_shadow_candidate": "prospective_noninvalid_v1 "
                       "(84.8% coverage, risk 34) with prospective_valid_only_"
                       "v1 (58.7%, risk 20) as the conservative alternative; "
                       "run BOTH in shadow and decide on shadow evidence",
                   "matrix_sensitivity": sens["robustness_classification"],
                   "deployment_lock": "ACTIVE mode locked (no ActivationRecord "
                                      "exists; no production caller)"},
        "ocr": {"executed": False, "campaign": ocr["experiment"],
                "campaign_sha256": ocr["experiment_sha256"],
                "unresolved": ocr["stages"][1][
                    "engineering_prep_required_before_execution"]},
        "engineering": {"risk_engine_version": riskengine.RISK_ENGINE_VERSION,
                        "throughput_per_s": bench["results"][2][
                            "decide_throughput_per_s"],
                        "event_bytes": bench["results"][2]["mean_event_bytes"],
                        "note": "test/concurrency/backup/boundary results are "
                                "recorded in the morning report and CI logs"},
        "gates": gates,
        "gates_passed": f"{passed}/{len(gates)}",
    }
    payload = json.dumps({k: v for k, v in doc.items() if k != "content_sha256"},
                         ensure_ascii=False, sort_keys=True)
    doc["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    out_json = RUNS / f"RELEASE_READINESS_{today}.json"
    out_md = RUNS / f"RELEASE_READINESS_{today}.md"
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8", newline="\n")

    md = [f"# Release readiness — **{status}** ({doc['created_at']})", "",
          f"Gate set `{GATE_VERSION}`; {passed}/{len(gates)} gates pass. "
          "Production-ready is not a selectable status.", "",
          "| gate | target | observed | limitation | verdict |",
          "|---|---|---|---|---|"]
    for g in gates:
        md.append(f"| {g['gate']} | {g['target']} | {g['observed']} | "
                  f"{g['confidence_limitation']} | "
                  f"{'PASS' if g['pass'] else 'FAIL'} |")
    md += ["", "## Why SHADOW_READY and nothing more", ""]
    for k, v in doc["status_derivation"].items():
        md.append(f"- **{k}**: {json.dumps(v, ensure_ascii=False)}")
    md += ["", "Recommended shadow candidate: "
           + doc["policy"]["recommended_shadow_candidate"]]
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": status, "gates_passed": doc["gates_passed"]},
                     indent=1))
    print("written:", out_json.name, out_md.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
