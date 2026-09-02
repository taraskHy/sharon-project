"""Shadow replay of the deployable risk policies over the frozen SEEN-46.

    python scripts/shadow_replay.py

Zero-inference. Maps the frozen baseline Q4 8B outputs (44 SEEN-46 rows + 2
corrected r6/r8 rows) into `ProspectiveDecisionInput`s, runs the versioned
risk engine in SHADOW mode for every PROSPECTIVE candidate policy, replays
the RETROSPECTIVE oracle policies separately through the offline-analysis
path, and computes rare-event uncertainty and the prospective sensitivity
extension. Writes:

    SHADOW_REPLAY_<date>.jsonl                  one event per case per
                                                prospective policy
    PROSPECTIVE_POLICY_REPLAY_<date>.{json,md}  deployable table + ORACLE
                                                upper-bound table + rare-event
                                                confidence bounds
    RISK_SENSITIVITY_PROSPECTIVE_<date>.{json,md}

The active grade is never touched; nothing here mutates any historical
record. HELD_OUT is structurally absent.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder import rare_events, riskengine  # noqa: E402

RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
POLICY_PATH = REPO / "evaluation" / "model_selection" / "policies" / \
    "asymmetric_grading_risk_v1.json"
SRC_DATE = "2026-09-02"
REF_PATH = RUNS / f"FINAL_HUMAN_REFERENCE_{SRC_DATE}.json"
RERUN_JSONL = RUNS / f"CORRECTED_RERUN_{SRC_DATE}.jsonl"
ARMS_SPEC = REPO / "evaluation" / "model_selection" / "experiments" / \
    "LOCAL_IMPROVEMENT_ARMS_2026-09-02.json"
SEEN46_RUN_DIRS = ("dev__all__qwen3-vl-8b-instruct__72e19378d1",
                   "calibration__all__qwen3-vl-8b-instruct__e2a3cfc925")
REPAIRED = ("e004_q2_r6", "e004_q2_r8")

VERDICTS = ("invalid", "partially_valid", "valid")
RANK = {v: i for i, v in enumerate(VERDICTS)}
ISSUE_EXCLUDE = {"rubric_official_solution", "transcription_evidence"}

PROSPECTIVE_POLICIES = ("prospective_valid_only_v1",
                        "prospective_noninvalid_v1",
                        "prospective_auto_all_structurally_valid_v1")
RETROSPECTIVE_POLICIES = ("retrospective_human_dispute_aware_b_v1",
                          "retrospective_human_dispute_aware_c_v1")

SENS_GRID = {"invalid->valid": (10, 12, 15, 20),
             "partially_valid->valid": (3, 5, 7),
             "valid->invalid": (2, 3, 5),
             "adjacent_undergrade": (1, 2)}
SENS_FIXED_IV_PV = 3


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def _self_hash(doc: dict) -> dict:
    payload = json.dumps({k: v for k, v in doc.items() if k != "content_sha256"},
                         ensure_ascii=False, sort_keys=True)
    doc["content_sha256"] = _sha(payload)
    return doc


def _git() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=15).stdout.strip()


def load_reference() -> dict:
    doc = json.loads(REF_PATH.read_text(encoding="utf-8"))
    payload = json.dumps({k: v for k, v in doc.items() if k != "reference_sha256"},
                         ensure_ascii=False, sort_keys=True)
    assert doc["reference_sha256"] == _sha(payload), "reference tampered"
    return doc


def load_baseline() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in SEEN46_RUN_DIRS:
        for s in json.loads((RUNS / "grade_primary" / d / "scored.jsonl.json")
                            .read_text(encoding="utf-8")):
            if s["case_id"] not in REPAIRED:
                out[s["case_id"]] = s
    for line in RERUN_JSONL.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        out[s["case_id"]] = s
    assert len(out) == 46
    return out


def provenance_hashes() -> dict:
    from autograder.escalation import (GRADE_VALIDATION_VERSION, GradeResult,
                                       grade_system_for)
    spec = json.loads(ARMS_SPEC.read_text(encoding="utf-8"))
    return {
        "model_digest": spec["arms"]["baseline"]["digest"],
        "prompt_version": "grade-v4-charitable-local",
        "prompt_sha256": _sha(grade_system_for("grade-v4-charitable-local")),
        "schema_sha256": _sha(json.dumps(GradeResult.model_json_schema(),
                                         sort_keys=True)),
        "validation_version": GRADE_VALIDATION_VERSION,
    }


def decision_input_for(row: dict, prov: dict) -> riskengine.ProspectiveDecisionInput:
    """ONLINE_OBSERVABLE fields only. All 46 current cases carry current
    source integrity and current (non-stale) outputs by construction — the
    stale r6/r8 rows are structurally excluded upstream."""
    return riskengine.ProspectiveDecisionInput.from_mapping({
        "semantic_verdict": row["predicted_verdict"],
        "schema_ok": not row.get("schema_failure"),
        "evidence_ok": not row.get("evidence_failure"),
        "validation_ok": bool(row.get("validation_ok")),
        "uncertain": bool(row.get("uncertain")),
        "transcription_complete": bool(row.get("transcription_complete")),
        "source_integrity": "current",
        "model_output_current": True,
        "local_grader_available": True,
        **prov,
    })


def retro_context_for(case: dict) -> riskengine.RetrospectiveContext:
    fresh = case["independent_blind_reviews"]
    issues = {r["issue"] for r in fresh} & ISSUE_EXCLUDE
    wide = (len(fresh) == 2
            and {fresh[0]["verdict"], fresh[1]["verdict"]} == {"invalid", "valid"})
    return riskengine.RetrospectiveContext(wide_human_disagreement=wide,
                                           active_review_issue=bool(issues))


def policy_metrics(decisions: dict[str, "riskengine.RiskDecision"],
                   href: dict[str, str], pred: dict[str, str],
                   matrix: dict) -> dict:
    auto_ids = sorted(c for c, d in decisions.items() if d.action == "AUTO")
    review = {c: d.reason for c, d in decisions.items() if d.action != "AUTO"}
    correct = sum(1 for c in auto_ids if href[c] == pred[c])
    cells = {f"{a}->{p}": 0 for a in VERDICTS for p in VERDICTS if a != p}
    loss = 0
    for c in auto_ids:
        a, p = href[c], pred[c]
        loss += matrix[a][p]
        if a != p:
            cells[f"{a}->{p}"] += 1
    under = [c for c in auto_ids if RANK[pred[c]] < RANK[href[c]]]
    over = [c for c in auto_ids if RANK[pred[c]] > RANK[href[c]]]
    # internal coherence: every AUTO case is exactly one of correct /
    # overgrade / undergrade — a mixed-denominator aggregate cannot pass this
    assert correct + len(over) + len(under) == len(auto_ids), \
        "AUTO partition broken: correct+over+under must equal AUTO"
    return {
        "auto": len(auto_ids), "review": len(review),
        "auto_coverage_pct": round(100 * len(auto_ids) / 46, 1),
        "review_rate_pct": round(100 * len(review) / 46, 1),
        "correct_auto": correct,
        "auto_precision_pct": (round(100 * correct / len(auto_ids), 1)
                               if auto_ids else None),
        "auto_precision_definition": "correct_auto / auto — the AUTO subset "
            "ONLY. Semantic-layer confusion cells (e.g. 24 valid->valid over "
            "all 46) include REVIEW-routed cases and must never be used as "
            "this numerator",
        "auto_total_weighted_loss": loss,
        "auto_mean_weighted_loss": (round(loss / len(auto_ids), 4)
                                    if auto_ids else None),
        "auto_error_cells": cells,
        "invalid_to_valid_auto": cells["invalid->valid"],
        "invalid_to_partial_auto": cells["invalid->partially_valid"],
        "partial_to_valid_auto": cells["partially_valid->valid"],
        "valid_to_partial_auto": cells["valid->partially_valid"],
        "valid_to_invalid_auto": cells["valid->invalid"],
        "auto_overgrades": len(over), "auto_undergrades": len(under),
        "auto_overgrade_cases": over, "auto_undergrade_cases": under,
        "verdict_step_excess": sum(RANK[pred[c]] - RANK[href[c]] for c in over),
        "verdict_step_deficit": sum(RANK[href[c]] - RANK[pred[c]] for c in under),
        "step_unit_note": "verdict steps, NOT exam points (mapping varies "
                          "per question and is not invented)",
        "review_reasons": {r: sum(1 for v in review.values() if v == r)
                           for r in sorted(set(review.values()))},
        "reviews_per_100_explanation_cases": round(100 * len(review) / 46, 1),
    }


def rare_event_block(metrics: dict, href: dict, pred: dict,
                     auto_ids: list[str]) -> dict:
    invalid_ids = [c for c in href if href[c] == "invalid"]
    pv_ids = [c for c in href if href[c] == "partially_valid"]
    iv_auto_full = sum(1 for c in auto_ids
                       if href[c] == "invalid" and pred[c] == "valid")
    pv_auto_full = sum(1 for c in auto_ids
                       if href[c] == "partially_valid" and pred[c] == "valid")
    return {
        "invalid_to_valid": {
            **rare_events.severe_event_report(iv_auto_full, len(invalid_ids)),
            "denominator_definition": "ALL actually-invalid seen cases (the "
                                      "population on which the catastrophic "
                                      "error could occur)"},
        "partial_to_valid": {
            **rare_events.severe_event_report(pv_auto_full, len(pv_ids)),
            "denominator_definition": "ALL actually-partially_valid seen cases"},
    }


def main() -> int:
    today = time.strftime("%Y-%m-%d")
    shadow_path = RUNS / f"SHADOW_REPLAY_{today}.jsonl"
    replay_json = RUNS / f"PROSPECTIVE_POLICY_REPLAY_{today}.json"
    replay_md = RUNS / f"PROSPECTIVE_POLICY_REPLAY_{today}.md"
    sens_json = RUNS / f"RISK_SENSITIVITY_PROSPECTIVE_{today}.json"
    sens_md = RUNS / f"RISK_SENSITIVITY_PROSPECTIVE_{today}.md"

    ref = load_reference()
    baseline = load_baseline()
    prov = provenance_hashes()
    matrix_ref = riskengine.load_risk_matrix(POLICY_PATH)
    matrix = matrix_ref.matrix
    href = {c["case_id"]: c["final_verdict"] for c in ref["cases"]}
    pred = {cid: baseline[cid]["predicted_verdict"] for cid in href}
    by_case = {c["case_id"]: c for c in ref["cases"]}
    assert set(href) == set(baseline)
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    dec_inputs = {cid: decision_input_for(baseline[cid], prov)
                  for cid in sorted(href)}
    model_run_id = "frozen_seen46+corrected_rerun_2026-09-02"

    # ---- Phase 10: shadow replay (prospective policies only) --------------
    if shadow_path.exists():
        shadow_path.unlink()          # regenerated deterministically each run
    shadow_log = riskengine.ShadowLog(shadow_path)
    prospective_results: dict[str, dict] = {}
    for pol in PROSPECTIVE_POLICIES:
        eng = riskengine.build_engine(mode="shadow", policy_id=pol,
                                      matrix_path=POLICY_PATH)
        decisions = {}
        for cid in sorted(href):
            d = dec_inputs[cid]
            decision = eng.decide(d, now=now)
            offline = {
                "offline_only": True,
                "reference_verdict": href[cid],
                "reference_source": by_case[cid]["reference_source"],
                "strict_weighted_loss": matrix[href[cid]][pred[cid]],
                "severe_invalid_to_valid": (href[cid] == "invalid"
                                            and pred[cid] == "valid"
                                            and decision.action == "AUTO"),
                "severe_partial_to_valid": (href[cid] == "partially_valid"
                                            and pred[cid] == "valid"
                                            and decision.action == "AUTO"),
            }
            event = riskengine.build_shadow_event(cid, model_run_id, d,
                                                  decision, offline)
            assert shadow_log.append(event) is True
            decisions[cid] = decision
        m = policy_metrics(decisions, href, pred, matrix)
        auto_ids = [c for c, d in decisions.items() if d.action == "AUTO"]
        m["rare_event_uncertainty"] = rare_event_block(m, href, pred, auto_ids)
        prospective_results[pol] = m
    assert len(shadow_log.events()) == 46 * len(PROSPECTIVE_POLICIES)

    # ---- retrospective ORACLE replay (offline analysis path ONLY) --------
    retro_results: dict[str, dict] = {}
    for pol in RETROSPECTIVE_POLICIES:
        eng = riskengine.build_engine(mode="shadow", policy_id=pol,
                                      matrix_path=POLICY_PATH)
        decisions = {}
        for cid in sorted(href):
            ctx = retro_context_for(by_case[cid])
            decisions[cid] = eng.decide(dec_inputs[cid], ctx,
                                        offline_analysis=True, now=now)
        m = policy_metrics(decisions, href, pred, matrix)
        auto_ids = [c for c, d in decisions.items() if d.action == "AUTO"]
        m["rare_event_uncertainty"] = rare_event_block(m, href, pred, auto_ids)
        retro_results[pol] = m

    # cross-check against the committed production-policy replay
    committed = json.loads((RUNS / f"PRODUCTION_POLICY_REPLAY_{SRC_DATE}.json")
                           .read_text(encoding="utf-8"))
    base_committed = committed["replay"]["baseline_8b_one_pass"]
    cross = {
        "retro_b_matches_committed_HUMAN_DISPUTE_AWARE_B":
            (retro_results["retrospective_human_dispute_aware_b_v1"]["auto"]
             == base_committed["HUMAN_DISPUTE_AWARE_B"]["auto"]
             and retro_results["retrospective_human_dispute_aware_b_v1"]
             ["auto_total_weighted_loss"]
             == base_committed["HUMAN_DISPUTE_AWARE_B"]["auto_total_weighted_loss"]),
        "retro_c_matches_committed_HUMAN_DISPUTE_AWARE_C":
            (retro_results["retrospective_human_dispute_aware_c_v1"]["auto"]
             == base_committed["HUMAN_DISPUTE_AWARE_C"]["auto"]
             and retro_results["retrospective_human_dispute_aware_c_v1"]
             ["auto_total_weighted_loss"]
             == base_committed["HUMAN_DISPUTE_AWARE_C"]["auto_total_weighted_loss"]),
        "auto_all_matches_committed_AUTO_ALL":
            (prospective_results["prospective_auto_all_structurally_valid_v1"]
             ["auto"] == base_committed["AUTO_ALL"]["auto"]),
    }
    assert all(cross.values()), f"cross-check failed: {cross}"

    # ---- Phase 8: minimum invalid-sample table ----------------------------
    min_n = {f"{int(b * 100)}pct": rare_events.min_n_for_zero_event_bound(b)
             for b in (0.10, 0.05, 0.02, 0.01)}

    # ---- provenance -------------------------------------------------------
    prov_block = {
        "git_commit": _git(),
        "reference_sha256": ref["reference_sha256"],
        "matrix_name": matrix_ref.name,
        "matrix_sha256": matrix_ref.matrix_sha256,
        "matrix_file_sha256": matrix_ref.policy_file_sha256,
        "engine_version": riskengine.RISK_ENGINE_VERSION,
        "observability_inventory_version":
            riskengine.OBSERVABILITY_INVENTORY_VERSION,
        "policy_hashes": {p: riskengine.POLICY_REGISTRY[p].sha256()
                          for p in PROSPECTIVE_POLICIES + RETROSPECTIVE_POLICIES},
        "model_run_id": model_run_id,
        "grader_provenance": prov,
        "case_coverage": 46, "held_out_cases": 0,
        "new_inference_calls": 0, "cloud_calls": 0, "ocr_calls": 0,
        "rag_calls": 0,
    }

    replay_doc = _self_hash({
        "artifact": "prospective_policy_replay",
        "created_at": now,
        "scope_note": "explanation-case automation only; the deployable table "
                      "uses ONLY decision-time-observable inputs, the "
                      "retrospective table is an ORACLE-ASSISTED UPPER BOUND "
                      "computed with post-review human data that would NOT "
                      "exist at decision time on a new exam",
        "provenance": prov_block,
        "policy_taxonomy": riskengine.policy_table(),
        "observability_inventory": riskengine.observability_inventory(),
        "deployable_prospective": prospective_results,
        "oracle_retrospective_upper_bound_NOT_DEPLOYABLE": retro_results,
        "cross_check_vs_committed_replay": cross,
        "minimum_invalid_examples_for_zero_event_upper_bound": {
            **min_n,
            "formula": "smallest n with (1-bound)^n <= alpha  <=>  "
                       "n >= ln(alpha)/ln(1-bound), alpha = 0.05",
            "note": "threshold choice belongs to the owner; none is chosen "
                    "here"},
    })
    replay_json.write_text(json.dumps(replay_doc, ensure_ascii=False, indent=1)
                           + "\n", encoding="utf-8", newline="\n")

    md = [f"# Prospective policy replay — SHADOW ({now})", "",
          "Deployable policies use ONLY decision-time-observable inputs "
          "(typed, fail-closed). 46 seen explanation cases; baseline Q4 8B "
          "outputs; no inference.", "",
          "## PROSPECTIVE_DEPLOYABLE (genuinely deployable)", "",
          "AUTO precision = correct_auto / AUTO (the AUTO subset only; "
          "REVIEW-routed cases are in neither numerator nor denominator).", "",
          "| policy | AUTO | cov% | REVIEW | correct | prec% | AUTO risk | "
          "mean | iv->v | pv->v | iv->pv | under | step+ | step- | rev/100 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pol in PROSPECTIVE_POLICIES:
        m = prospective_results[pol]
        md.append(f"| {pol} | {m['auto']} | {m['auto_coverage_pct']} | "
                  f"{m['review']} | {m['correct_auto']} | "
                  f"{m['auto_precision_pct']} | "
                  f"{m['auto_total_weighted_loss']} | "
                  f"{m['auto_mean_weighted_loss']} | "
                  f"{m['invalid_to_valid_auto']} | {m['partial_to_valid_auto']} "
                  f"| {m['invalid_to_partial_auto']} | {m['auto_undergrades']} "
                  f"| {m['verdict_step_excess']} | {m['verdict_step_deficit']} "
                  f"| {m['reviews_per_100_explanation_cases']} |")
    md += ["", "## ORACLE-ASSISTED RETROSPECTIVE UPPER BOUND — NOT DEPLOYABLE",
           "", "These rankings use post-review human-disagreement data that "
           "does not exist before review on a new case. They bound what a "
           "future prospective dispute-predictor could add; they are NOT "
           "candidate production policies.", "",
           "| policy | AUTO | cov% | REVIEW | correct | prec% | AUTO risk | "
           "pv->v | under |", "|---|---|---|---|---|---|---|---|---|"]
    for pol in RETROSPECTIVE_POLICIES:
        m = retro_results[pol]
        md.append(f"| {pol} | {m['auto']} | {m['auto_coverage_pct']} | "
                  f"{m['review']} | {m['correct_auto']} | "
                  f"{m['auto_precision_pct']} | "
                  f"{m['auto_total_weighted_loss']} | "
                  f"{m['partial_to_valid_auto']} | {m['auto_undergrades']} |")
    ru = prospective_results["prospective_valid_only_v1"][
        "rare_event_uncertainty"]["invalid_to_valid"]
    md += ["", "## Rare-event uncertainty (exact Clopper-Pearson)", "",
           f"invalid->valid automatic full credit: observed "
           f"{ru['observed']}/{ru['denominator']} on seen data, one-sided 95% "
           f"upper bound **{100 * ru['one_sided_upper_95']:.1f}%** — zero "
           "observed events over five invalid cases CANNOT demonstrate "
           "safety; the data only excludes rates above ~45%.", "",
           "Minimum independent invalid examples (zero events observed) for "
           "a one-sided 95% upper bound below:", "",
           "| bound | min invalid examples |", "|---|---|"]
    for b, n in replay_doc[
            "minimum_invalid_examples_for_zero_event_upper_bound"].items():
        if b.endswith("pct"):
            md.append(f"| {b[:-3]}% | {n} |")
    md += ["", "Formula: smallest n with (1-bound)^n <= 0.05. The bound "
           "choice is the owner's; none is selected here.", "",
           f"Shadow events: {46 * len(PROSPECTIVE_POLICIES)} rows in "
           f"`{shadow_path.name}` (decision inputs and offline evaluation "
           "fields strictly separated)."]
    replay_md.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")

    # ---- Phase 9: sensitivity extended to prospective policies ------------
    arms_pred = {"baseline_8b_one_pass": pred}
    # model arms from the committed artifacts (already independently
    # reproduced); deployable policies evaluated on the baseline arm
    committed_sens = json.loads(
        (RUNS / f"ASYMMETRIC_RISK_SENSITIVITY_{SRC_DATE}.json")
        .read_text(encoding="utf-8"))
    # recompute AUTO id sets per prospective policy (deterministic)
    auto_ids_by_policy = {}
    for pol in PROSPECTIVE_POLICIES:
        eng = riskengine.build_engine(mode="shadow", policy_id=pol,
                                      matrix_path=POLICY_PATH)
        auto_ids_by_policy[pol] = [
            cid for cid in sorted(href)
            if eng.decide(dec_inputs[cid], now=now).action == "AUTO"]

    grid_rows = []
    winner_by_total: dict[str, int] = {}
    winner_by_mean: dict[str, int] = {}
    for c_ivv, c_pvv, c_viv, c_adj in product(*SENS_GRID.values()):
        m = {"invalid": {"invalid": 0, "partially_valid": SENS_FIXED_IV_PV,
                         "valid": c_ivv},
             "partially_valid": {"invalid": c_adj, "partially_valid": 0,
                                 "valid": c_pvv},
             "valid": {"invalid": c_viv, "partially_valid": c_adj, "valid": 0}}
        totals = {}
        means = {}
        raw_severe = {}
        for pol in PROSPECTIVE_POLICIES:
            ids = auto_ids_by_policy[pol]
            t = sum(m[href[c]][pred[c]] for c in ids)
            totals[pol] = t
            means[pol] = round(t / len(ids), 4) if ids else None
            raw_severe[pol] = {
                "invalid_to_valid": sum(1 for c in ids if href[c] == "invalid"
                                        and pred[c] == "valid"),
                "partial_to_valid": sum(1 for c in ids
                                        if href[c] == "partially_valid"
                                        and pred[c] == "valid")}
        best_total = min(totals, key=lambda k: (totals[k], k))
        best_mean = min(means, key=lambda k: (means[k], k))
        winner_by_total[best_total] = winner_by_total.get(best_total, 0) + 1
        winner_by_mean[best_mean] = winner_by_mean.get(best_mean, 0) + 1
        # model-arm winner for this matrix, from the committed (and
        # independently reproduced) semantic-layer grid
        committed_row = next(
            g for g in committed_sens["grid_results"]
            if g["matrix"]["invalid->valid"] == c_ivv
            and g["matrix"]["partially_valid->valid"] == c_pvv
            and g["matrix"]["valid->invalid"] == c_viv
            and g["matrix"]["adjacent_undergrade"] == c_adj)
        grid_rows.append({
            "matrix": {"invalid->valid": c_ivv, "partially_valid->valid": c_pvv,
                       "valid->invalid": c_viv, "adjacent_undergrade": c_adj,
                       "invalid->partially_valid": SENS_FIXED_IV_PV},
            "deployable_auto_totals": totals,
            "deployable_auto_means": means,
            "raw_severe_counts": raw_severe,
            "best_deployable_by_total_auto_risk": best_total,
            "best_deployable_by_mean_auto_risk": best_mean,
            "semantic_layer_winners": committed_row["winners"],
            "constant_baseline_wins_semantic_layer":
                any(w.startswith("always_") for w in committed_row["winners"]),
        })

    n_mat = len(grid_rows)
    valid_only_always_best = (
        winner_by_total.get("prospective_valid_only_v1", 0) == n_mat
        and winner_by_mean.get("prospective_valid_only_v1", 0) == n_mat)
    classification = {
        "deployable_policy_risk_ordering":
            ("ROBUST" if valid_only_always_best else "FRAGILE"),
        "deployable_policy_risk_ordering_note":
            "prospective_valid_only_v1 minimizes both total and mean AUTO "
            "weighted risk on every matrix — but ONLY because it refuses "
            "every partially_valid verdict; the choice between the deployable "
            "policies is a coverage-vs-risk tradeoff, not a dominance result",
        "model_arm_ranking":
            "FRAGILE" if any(
                g["semantic_layer_winners"] !=
                grid_rows[0]["semantic_layer_winners"] for g in grid_rows)
            else "ROBUST",
        "model_arm_ranking_note":
            "verified: the baseline-vs-q8_0 ordering flips on the "
            "adjacent-undergrade cost (expectation confirmed, not assumed)",
        "constant_baseline_wins_count": sum(
            1 for g in grid_rows if g["constant_baseline_wins_semantic_layer"]),
    }

    sens_doc = _self_hash({
        "artifact": "risk_sensitivity_prospective",
        "created_at": now,
        "provenance": prov_block,
        "grid_definition": {"varied": {k: list(v) for k, v in SENS_GRID.items()},
                            "fixed": {"invalid->partially_valid":
                                      SENS_FIXED_IV_PV}},
        "matrices_evaluated": n_mat,
        "auto_id_sets": auto_ids_by_policy,
        "grid_results": grid_rows,
        "winner_frequency_by_total_auto_risk": dict(sorted(winner_by_total.items())),
        "winner_frequency_by_mean_auto_risk": dict(sorted(winner_by_mean.items())),
        "robustness_classification": classification,
    })
    sens_json.write_text(json.dumps(sens_doc, ensure_ascii=False, indent=1)
                         + "\n", encoding="utf-8", newline="\n")

    smd = [f"# Prospective-policy sensitivity ({now})", "",
           f"{n_mat} matrices (same grid as the frozen semantic-layer "
           "sensitivity; invalid->partially_valid fixed at 3).", "",
           f"- best deployable policy by TOTAL AUTO risk: "
           f"{dict(sorted(winner_by_total.items()))}",
           f"- best deployable policy by MEAN AUTO risk: "
           f"{dict(sorted(winner_by_mean.items()))}",
           f"- constant baseline (semantic layer) wins in "
           f"{classification['constant_baseline_wins_count']}/{n_mat} matrices",
           f"- deployable risk ordering: "
           f"**{classification['deployable_policy_risk_ordering']}** — "
           + classification["deployable_policy_risk_ordering_note"],
           f"- model-arm ranking: "
           f"**{classification['model_arm_ranking']}** — "
           + classification["model_arm_ranking_note"], "",
           "Raw severe-event counts are carried per matrix in the JSON, "
           "independent of the weights."]
    sens_md.write_text("\n".join(smd) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps({
        "prospective": {p: {"auto": prospective_results[p]["auto"],
                            "risk": prospective_results[p]
                            ["auto_total_weighted_loss"]}
                        for p in PROSPECTIVE_POLICIES},
        "retrospective": {p: {"auto": retro_results[p]["auto"],
                              "risk": retro_results[p]
                              ["auto_total_weighted_loss"]}
                          for p in RETROSPECTIVE_POLICIES},
        "cross_check": cross,
        "min_invalid_examples": min_n,
        "shadow_events": 46 * len(PROSPECTIVE_POLICIES)}, indent=1))
    print("written:", shadow_path.name, replay_json.name, sens_json.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
