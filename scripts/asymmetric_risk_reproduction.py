"""Independent reproduction of the committed asymmetric-risk analysis.

    python scripts/asymmetric_risk_reproduction.py

Recomputes every load-bearing number of the 2026-09-02 asymmetric-risk
campaign DIRECTLY from the raw source artifacts (reference freeze, frozen
scored run files, arm JSONLs, frozen policy JSON) with self-contained logic —
deliberately NOT importing the metric code in scripts/asymmetric_risk.py —
and compares each value against what the committed analysis artifacts report.

Writes ASYMMETRIC_RISK_REPRODUCTION_<date>.{json,md} with, per metric:
reported value, recomputed value, source artifact, source sha256, pass/fail.

Zero-inference: no model / cloud / OCR / RAG call, no HELD_OUT access,
nothing modified anywhere.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
POLICY_PATH = REPO / "evaluation" / "model_selection" / "policies" / \
    "asymmetric_grading_risk_v1.json"
DATE = "2026-09-02"
REF_PATH = RUNS / f"FINAL_HUMAN_REFERENCE_{DATE}.json"
RERUN_JSONL = RUNS / f"CORRECTED_RERUN_{DATE}.jsonl"
ARM_A_JSONL = RUNS / f"ARM_A_Q8_{DATE}.jsonl"
ARM_B_JSONL = RUNS / f"ARM_B_VERIFY_{DATE}.jsonl"
STRICT = RUNS / f"ASYMMETRIC_RISK_STRICT_{DATE}.json"
DIS = RUNS / f"ASYMMETRIC_RISK_DISAGREEMENT_AWARE_{DATE}.json"
SENS = RUNS / f"ASYMMETRIC_RISK_SENSITIVITY_{DATE}.json"
REPLAY = RUNS / f"PRODUCTION_POLICY_REPLAY_{DATE}.json"
SUMMARY = RUNS / f"ASYMMETRIC_RISK_SUMMARY_{DATE}.md"
IMPROVEMENT = RUNS / f"LOCAL_IMPROVEMENT_REPORT_{DATE}.json"
SEEN46_RUN_DIRS = ("dev__all__qwen3-vl-8b-instruct__72e19378d1",
                   "calibration__all__qwen3-vl-8b-instruct__e2a3cfc925")
REPAIRED = ("e004_q2_r6", "e004_q2_r8")

VERDICTS = ("invalid", "partially_valid", "valid")
RANK = {v: i for i, v in enumerate(VERDICTS)}
ISSUE_EXCLUDE = {"rubric_official_solution", "transcription_evidence"}

OUT_JSON = RUNS / "ASYMMETRIC_RISK_REPRODUCTION_{d}.json"
OUT_MD = RUNS / "ASYMMETRIC_RISK_REPRODUCTION_{d}.md"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def self_hash_ok(doc: dict, field: str) -> bool:
    payload = json.dumps({k: v for k, v in doc.items() if k != field},
                         ensure_ascii=False, sort_keys=True)
    return doc.get(field) == sha_text(payload)


class Checks:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, name: str, reported, recomputed, source: str, source_sha: str):
        self.rows.append({"check": name, "reported": reported,
                          "recomputed": recomputed, "source_artifact": source,
                          "source_sha256": source_sha,
                          "pass": reported == recomputed})

    def add_bool(self, name: str, ok: bool, detail, source: str, source_sha: str):
        self.rows.append({"check": name, "reported": "required", "recomputed": detail,
                          "source_artifact": source, "source_sha256": source_sha,
                          "pass": bool(ok)})

    @property
    def failed(self):
        return [r for r in self.rows if not r["pass"]]


def main() -> int:
    today = time.strftime("%Y-%m-%d")
    out_json = RUNS / f"ASYMMETRIC_RISK_REPRODUCTION_{today}.json"
    out_md = RUNS / f"ASYMMETRIC_RISK_REPRODUCTION_{today}.md"
    c = Checks()

    # ---- raw sources ------------------------------------------------------
    ref = load_json(REF_PATH)
    ref_sha = sha_file(REF_PATH)
    c.add_bool("reference self-hash verifies", self_hash_ok(ref, "reference_sha256"),
               ref.get("reference_sha256", "")[:16], REF_PATH.name, ref_sha)

    cases = ref["cases"]
    ids = [x["case_id"] for x in cases]
    c.add("1. case count", 46, len(cases), REF_PATH.name, ref_sha)
    c.add_bool("7a. no duplicate ids", len(set(ids)) == 46, len(set(ids)),
               REF_PATH.name, ref_sha)
    dist = {v: sum(1 for x in cases if x["final_verdict"] == v) for v in VERDICTS}
    c.add("2. class distribution",
          {"invalid": 5, "partially_valid": 13, "valid": 28}, dist,
          REF_PATH.name, ref_sha)
    src = {}
    for x in cases:
        src[x["reference_source"]] = src.get(x["reference_source"], 0) + 1
    c.add("3. source distribution",
          {"two_reviewer_consensus": 22, "adjudicated_human_reference": 22,
           "owner_adjudicated_after_source_repair": 2}, src, REF_PATH.name, ref_sha)

    # corrected r6/r8 active, stale preserved+excluded
    rerun = {json.loads(l)["case_id"]: json.loads(l)
             for l in RERUN_JSONL.read_text(encoding="utf-8").splitlines()}
    baseline: dict[str, dict] = {}
    stale_rows: dict[str, dict] = {}
    for d in SEEN46_RUN_DIRS:
        for s in load_json(RUNS / "grade_primary" / d / "scored.jsonl.json"):
            (stale_rows if s["case_id"] in REPAIRED else baseline)[s["case_id"]] = s
    for cid, s in rerun.items():
        baseline[cid] = s
    c.add("4. corrected r6/r8 active (reference pointer)",
          ["corrected_rerun_2026-09-02"] * 2,
          [x["baseline_model_output"]["output_source"] for x in cases
           if x["case_id"] in REPAIRED], REF_PATH.name, ref_sha)
    c.add_bool("5. stale r6/r8 rows preserved in run dirs but not used",
               set(stale_rows) == set(REPAIRED)
               and all(baseline[cid] == rerun[cid] and baseline[cid] != stale_rows[cid]
                       for cid in REPAIRED),
               sorted(stale_rows), "grade_primary run dirs + CORRECTED_RERUN",
               sha_file(RERUN_JSONL))

    arm_a = {json.loads(l)["case_id"]: json.loads(l)
             for l in ARM_A_JSONL.read_text(encoding="utf-8").splitlines()}
    arm_b = {json.loads(l)["case_id"]: json.loads(l)
             for l in ARM_B_JSONL.read_text(encoding="utf-8").splitlines()}
    c.add_bool("6/7. all arms cover exactly the 46 reference ids",
               set(ids) == set(baseline) == set(arm_a) == set(arm_b),
               {"baseline": len(baseline), "arm_a": len(arm_a), "arm_b": len(arm_b)},
               "arm files", sha_file(ARM_A_JSONL)[:16] + "/" + sha_file(ARM_B_JSONL)[:16])
    c.add_bool("15. no HELD_OUT writer id anywhere",
               not any(re.search(r"(?<![0-9a-fA-F])(e005|e006)(?![0-9a-fA-F])", i)
                       for i in ids), sorted({i.split("_")[0] for i in ids}),
               REF_PATH.name, ref_sha)

    # ---- policy matrix ----------------------------------------------------
    pol = load_json(POLICY_PATH)
    pol_sha_file = sha_file(POLICY_PATH)
    expected_matrix = {"invalid": {"invalid": 0, "partially_valid": 3, "valid": 12},
                       "partially_valid": {"invalid": 1, "partially_valid": 0, "valid": 5},
                       "valid": {"invalid": 3, "partially_valid": 1, "valid": 0}}
    c.add_bool("policy self-hash verifies + expected prefix",
               self_hash_ok(pol, "policy_sha256")
               and pol["policy_sha256"].startswith("11e65e79e0f36cf6"),
               pol["policy_sha256"][:16], POLICY_PATH.name, pol_sha_file)
    c.add("policy matrix values", expected_matrix, pol["cost_matrix"],
          POLICY_PATH.name, pol_sha_file)
    L = pol["cost_matrix"]

    # ---- verdict maps per arm --------------------------------------------
    href = {x["case_id"]: x["final_verdict"] for x in cases}
    pred = {
        "baseline_8b_one_pass": {i: baseline[i]["predicted_verdict"] for i in ids},
        "arm_a_q8_0": {i: arm_a[i]["predicted_verdict"] for i in ids},
        "arm_b_two_pass": {i: arm_b[i]["combined"]["final_verdict"] for i in ids},
    }
    decision = {
        "baseline_8b_one_pass": {i: baseline[i]["decision"] for i in ids},
        "arm_a_q8_0": {i: arm_a[i]["decision"] for i in ids},
        "arm_b_two_pass": {i: arm_b[i]["combined"]["decision"] for i in ids},
    }

    def confusion(p):
        m = {a: {b: 0 for b in VERDICTS} for a in VERDICTS}
        for i in ids:
            m[href[i]][p[i]] += 1
        return m

    def total_loss(p, costs):
        return sum(costs[href[i]][p[i]] for i in ids)

    strict = load_json(STRICT)
    strict_sha = sha_file(STRICT)
    c.add_bool("strict artifact self-hash verifies",
               self_hash_ok(strict, "content_sha256"),
               strict.get("content_sha256", "")[:16], STRICT.name, strict_sha)

    # 8. confusion matrices; 9. strict weighted losses
    expected_totals = {"baseline_8b_one_pass": 43, "arm_a_q8_0": 40,
                       "arm_b_two_pass": 43}
    for arm in pred:
        conf = confusion(pred[arm])
        reported_cells = strict["arms"][arm]["error_cells"]
        recomputed_cells = {f"{a}->{b}": conf[a][b]
                           for a in VERDICTS for b in VERDICTS if a != b}
        c.add(f"8. confusion cells {arm}", reported_cells, recomputed_cells,
              STRICT.name, strict_sha)
        t = total_loss(pred[arm], L)
        c.add(f"9. strict total loss {arm}",
              strict["arms"][arm]["total_weighted_loss"], t, STRICT.name, strict_sha)
        c.add(f"9b. strict total loss {arm} (mission expectation)",
              expected_totals[arm], t, "mission spec", "-")
        exact = sum(1 for i in ids if href[i] == pred[arm][i])
        c.add(f"exact agreement {arm}", strict["arms"][arm]["exact_agreement"],
              exact, STRICT.name, strict_sha)
    c.add("baseline mean weighted loss", strict["arms"]["baseline_8b_one_pass"]
          ["mean_weighted_loss"], round(43 / 46, 4), STRICT.name, strict_sha)
    b_conf = confusion(pred["baseline_8b_one_pass"])
    c.add("baseline invalid->valid / partial->valid",
          [0, 4], [b_conf["invalid"]["valid"], b_conf["partially_valid"]["valid"]],
          STRICT.name, strict_sha)
    over_cost = sum(L[a][b] * b_conf[a][b] for a in VERDICTS for b in VERDICTS
                    if RANK[b] > RANK[a])
    under_cost = sum(L[a][b] * b_conf[a][b] for a in VERDICTS for b in VERDICTS
                     if RANK[b] < RANK[a])
    c.add("baseline overgrade/undergrade cost", [32, 11], [over_cost, under_cost],
          STRICT.name, strict_sha)

    # 11. constant baselines
    for name, v, want in (("always_invalid", "invalid", 97),
                          ("always_partially_valid", "partially_valid", 43),
                          ("always_valid", "valid", 125)):
        t = sum(L[href[i]][v] for i in ids)
        c.add(f"11. constant {name}",
              strict["constant_baselines"][name]["total_weighted_loss"], t,
              STRICT.name, strict_sha)
        c.add(f"11b. constant {name} (mission expectation)", want, t,
              "mission spec", "-")

    # 10. disagreement-aware losses (independent re-derivation)
    dis = load_json(DIS)
    dis_sha = sha_file(DIS)
    c.add_bool("disagreement artifact self-hash verifies",
               self_hash_ok(dis, "content_sha256"),
               dis.get("content_sha256", "")[:16], DIS.name, dis_sha)
    clean_losses = {}
    included_n = 0
    for arm in pred:
        total = 0.0
        included_n = 0
        for x in cases:
            if x["reference_source"] == "owner_adjudicated_after_source_repair":
                continue
            fresh = x["independent_blind_reviews"]
            issues = {r["issue"] for r in fresh} & ISSUE_EXCLUDE
            pair = {fresh[0]["verdict"], fresh[1]["verdict"]}
            if issues or pair == {"invalid", "valid"}:
                continue
            weight = 1.0 if len(pair) == 1 else 0.5
            total += weight * min(L[a][pred[arm][x["case_id"]]] for a in pair)
            included_n += 1
        clean_losses[arm] = round(total, 4)
        c.add(f"10. disagreement-aware clean loss {arm}",
              dis["arms"][arm]["total_weighted_loss"], clean_losses[arm],
              DIS.name, dis_sha)
    c.add("10b. disagreement-aware included cases",
          dis["counts"]["included_clean"], included_n, DIS.name, dis_sha)

    # 12. sensitivity grid (full independent recomputation)
    sens = load_json(SENS)
    sens_sha = sha_file(SENS)
    c.add_bool("sensitivity artifact self-hash verifies",
               self_hash_ok(sens, "content_sha256"),
               sens.get("content_sha256", "")[:16], SENS.name, sens_sha)
    const_pred = {"always_invalid": {i: "invalid" for i in ids},
                  "always_partially_valid": {i: "partially_valid" for i in ids},
                  "always_valid": {i: "valid" for i in ids}}
    all_pred = {**pred, **const_pred}
    grid_ok = True
    winner_freq: dict[str, int] = {}
    for g in sens["grid_results"]:
        m = g["matrix"]
        costs = {"invalid": {"invalid": 0, "partially_valid":
                             m["invalid->partially_valid"], "valid": m["invalid->valid"]},
                 "partially_valid": {"invalid": m["adjacent_undergrade"],
                                     "partially_valid": 0,
                                     "valid": m["partially_valid->valid"]},
                 "valid": {"invalid": m["valid->invalid"],
                           "partially_valid": m["adjacent_undergrade"], "valid": 0}}
        totals = {name: total_loss(p, costs) for name, p in all_pred.items()}
        if totals != g["totals"]:
            grid_ok = False
        best = min(totals.values())
        for w in sorted(n for n, t in totals.items() if t == best):
            winner_freq[w] = winner_freq.get(w, 0) + 1
    c.add_bool("12. sensitivity: all 72 matrices' totals reproduce",
               grid_ok and len(sens["grid_results"]) == 72,
               len(sens["grid_results"]), SENS.name, sens_sha)
    c.add("12b. sensitivity winner frequency", sens["winner_frequency"],
          dict(sorted(winner_freq.items())), SENS.name, sens_sha)
    expected_grid = [dict(zip(("invalid->valid", "partially_valid->valid",
                               "valid->invalid", "adjacent_undergrade"), combo))
                     for combo in product((10, 12, 15, 20), (3, 5, 7), (2, 3, 5), (1, 2))]
    seen_grid = [{k: g["matrix"][k] for k in expected_grid[0]}
                 for g in sens["grid_results"]]
    c.add_bool("12c. sensitivity grid complete (4x3x3x2)",
               seen_grid == expected_grid, len(seen_grid), SENS.name, sens_sha)

    # 13. production-policy replay counts (independent routing re-derivation)
    replay = load_json(REPLAY)
    replay_sha = sha_file(REPLAY)
    c.add_bool("replay artifact self-hash verifies",
               self_hash_ok(replay, "content_sha256"),
               replay.get("content_sha256", "")[:16], REPLAY.name, replay_sha)

    def case_flags(x):
        fresh = x["independent_blind_reviews"]
        issues = sorted({r["issue"] for r in fresh} & ISSUE_EXCLUDE)
        wide = (len(fresh) == 2
                and {fresh[0]["verdict"], fresh[1]["verdict"]} == {"invalid", "valid"})
        return issues, wide

    struct_rows = {"baseline_8b_one_pass": baseline, "arm_a_q8_0": arm_a}

    def route(policy, arm, cid, x):
        d = decision[arm][cid]
        if d != "AUTO":
            return "REVIEW"
        if policy == "AUTO_ALL":
            return "AUTO"
        s = struct_rows.get(arm, baseline)[cid] if arm != "arm_b_two_pass" \
            else baseline[cid]
        ok = (not s.get("schema_failure") and not s.get("evidence_failure")
              and not s.get("uncertain") and s.get("transcription_complete")
              and s.get("validation_ok"))
        if not ok:
            return "REVIEW"
        issues, wide = case_flags(x)
        if issues:
            return "REVIEW"
        allowed = ("valid",) if policy in ("AUTO_VALID_ONLY", "HUMAN_DISPUTE_AWARE_B") \
            else ("valid", "partially_valid")
        if pred[arm][cid] not in allowed:
            return "REVIEW"
        if policy.startswith("HUMAN_DISPUTE_AWARE") and wide:
            return "REVIEW"
        return "AUTO"

    expected_replay = {  # mission-stated expectations, baseline arm
        "AUTO_ALL": {"cov": 95.7, "risk": 42, "pv_val": 4, "under": 6},
        "AUTO_VALID_ONLY": {"cov": 54.3, "risk": 15, "pv_val": 3, "under": 0,
                            "review": 21, "prec": 88.0},
        "AUTO_VALID_AND_PARTIAL": {"cov": 76.1, "risk": 29, "pv_val": 3,
                                   "under": 2, "review": 11, "prec": 74.3},
        "HUMAN_DISPUTE_AWARE_B": {"cov": 47.8, "risk": 10},
        "HUMAN_DISPUTE_AWARE_C": {"cov": 67.4, "risk": 23},
    }
    by_case = {x["case_id"]: x for x in cases}
    for policy_name, want in expected_replay.items():
        auto_ids = [i for i in ids
                    if route(policy_name, "baseline_8b_one_pass", i, by_case[i]) == "AUTO"]
        risk = sum(L[href[i]][pred["baseline_8b_one_pass"][i]] for i in auto_ids)
        cov = round(100 * len(auto_ids) / 46, 1)
        pv_val = sum(1 for i in auto_ids if href[i] == "partially_valid"
                     and pred["baseline_8b_one_pass"][i] == "valid")
        under = sum(1 for i in auto_ids
                    if RANK[pred["baseline_8b_one_pass"][i]] < RANK[href[i]])
        rep = replay["replay"]["baseline_8b_one_pass"][policy_name]
        c.add(f"13. replay {policy_name} (artifact)",
              {"auto": rep["auto"], "risk": rep["auto_total_weighted_loss"],
               "cov": rep["auto_coverage_pct"]},
              {"auto": len(auto_ids), "risk": risk, "cov": cov},
              REPLAY.name, replay_sha)
        got = {"cov": cov, "risk": risk}
        if "pv_val" in want:
            got["pv_val"] = pv_val
        if "under" in want:
            got["under"] = under
        if "review" in want:
            got["review"] = 46 - len(auto_ids)
        if "prec" in want:
            correct = sum(1 for i in auto_ids if href[i] == pred["baseline_8b_one_pass"][i])
            got["prec"] = round(100 * correct / len(auto_ids), 1)
        c.add(f"13b. replay {policy_name} (mission expectation)", want, got,
              "mission spec", "-")

    # 14. release-gate values from the summary MD table
    md_text = SUMMARY.read_text(encoding="utf-8")
    gate_rows = re.findall(r"^\| (\w+) \| .* \| (PASS|FAIL) \|", md_text, re.M)
    gates_reported = dict(gate_rows)
    gates_recomputed = {
        "HARD_FALSE_FULL": "PASS" if b_conf["invalid"]["valid"] == 0 else "FAIL",
        "WEIGHTED_RISK": "PASS" if 43 <= 0.90 * 43 else "FAIL",
        "GROUNDING": "PASS" if 2 <= 0.02 * 46 else "FAIL",
    }
    for g, want in gates_recomputed.items():
        c.add(f"14. gate {g}", gates_reported.get(g), want, SUMMARY.name,
              sha_file(SUMMARY))
    c.add("14b. gates passed count", "5/9",
          f"{sum(1 for _, v in gate_rows if v == 'PASS')}/{len(gate_rows)}",
          SUMMARY.name, sha_file(SUMMARY))

    # cross-check vs the earlier improvement report (independent artifact)
    imp = load_json(IMPROVEMENT)
    c.add("cross-check exact agreement vs improvement report",
          {"baseline": 31, "arm_a": 28, "arm_b": 31},
          {k: imp["arms"][k]["metrics"]["exact_agreement"]
           for k in ("baseline", "arm_a", "arm_b")},
          IMPROVEMENT.name, sha_file(IMPROVEMENT))

    # ---- verdict ----------------------------------------------------------
    ok = not c.failed
    git_commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True,
                                timeout=15).stdout.strip()
    doc = {"artifact": "asymmetric_risk_reproduction",
           "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "git_commit": git_commit,
           "verdict": "REPRODUCED" if ok else "INVALID_ANALYSIS",
           "checks_total": len(c.rows),
           "checks_failed": len(c.failed),
           "failed_checks": c.failed,
           "checks": c.rows,
           "confirmations": {"new_inference_calls": 0, "cloud_calls": 0,
                             "ocr_calls": 0, "rag_calls": 0,
                             "held_out_exposure": 0, "records_modified": 0}}
    payload = json.dumps({k: v for k, v in doc.items() if k != "content_sha256"},
                         ensure_ascii=False, sort_keys=True)
    doc["content_sha256"] = sha_text(payload)
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8", newline="\n")

    md = [f"# Asymmetric-risk independent reproduction ({doc['created_at']})", "",
          f"**Verdict: {doc['verdict']}** — {len(c.rows)} checks, "
          f"{len(c.failed)} failed. Recomputed from raw artifacts with "
          "self-contained logic (no import of scripts/asymmetric_risk.py "
          "metric code).", "",
          "| check | reported | recomputed | pass |", "|---|---|---|---|"]
    for r in c.rows:
        rep_s = json.dumps(r["reported"], ensure_ascii=False)
        rec_s = json.dumps(r["recomputed"], ensure_ascii=False)
        md.append(f"| {r['check']} | {rep_s[:60]} | {rec_s[:60]} | "
                  f"{'PASS' if r['pass'] else 'FAIL'} |")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": doc["verdict"], "checks": len(c.rows),
                      "failed": c.failed}, ensure_ascii=False, indent=1))
    print("written:", out_json.name, out_md.name)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
