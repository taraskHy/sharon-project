"""Asymmetric, cost-sensitive evaluation of the local grading arms (2026-09-02).

    python scripts/asymmetric_risk.py freeze-policy   # Phase 1 (refuses drift)
    python scripts/asymmetric_risk.py strict          # Phase 2 + 4
    python scripts/asymmetric_risk.py disagreement    # Phase 3
    python scripts/asymmetric_risk.py sensitivity     # Phase 5
    python scripts/asymmetric_risk.py replay          # Phase 6 + 7
    python scripts/asymmetric_risk.py summary         # Phase 8 + 9
    python scripts/asymmetric_risk.py all             # everything, in order

ZERO-INFERENCE analysis over existing persisted artifacts only: the frozen
final 46-case human reference, the frozen baseline outputs (44 SEEN-46 + 2
corrected r6/r8), the arm A (q8_0) and arm B (two-pass) run files, and the
independent blind-review history. No model / cloud / OCR / RAG call, no
HELD_OUT id or content, no mutation of any historical record. Semantic
grading (the model's verdict) and downstream risk policy (the cost matrix +
AUTO/REVIEW routing) are kept as SEPARATE layers: nothing here re-prompts or
re-tunes the grader.

The risk policy `asymmetric_grading_risk_v1` encodes the owner's objective:
an invalid explanation awarded full credit is the most expensive mistake
(awarded points are hard to retract); conservative undergrades are cheaper
but remain nonzero cost (appeals are a correction channel, not an excuse).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
POLICIES = REPO / "evaluation" / "model_selection" / "policies"
REF_PATH = RUNS / "FINAL_HUMAN_REFERENCE_2026-09-02.json"
RERUN_JSONL = RUNS / "CORRECTED_RERUN_2026-09-02.jsonl"
ARM_A_JSONL = RUNS / "ARM_A_Q8_2026-09-02.jsonl"
ARM_B_JSONL = RUNS / "ARM_B_VERIFY_2026-09-02.jsonl"
SEEN46_RUN_DIRS = ("dev__all__qwen3-vl-8b-instruct__72e19378d1",
                   "calibration__all__qwen3-vl-8b-instruct__e2a3cfc925")
REPAIRED = ("e004_q2_r6", "e004_q2_r8")

POLICY_PATH = POLICIES / "asymmetric_grading_risk_v1.json"
DATE = "2026-09-02"
OUT_STRICT = RUNS / f"ASYMMETRIC_RISK_STRICT_{DATE}.json"
OUT_STRICT_MD = RUNS / f"ASYMMETRIC_RISK_STRICT_{DATE}.md"
OUT_DIS = RUNS / f"ASYMMETRIC_RISK_DISAGREEMENT_AWARE_{DATE}.json"
OUT_DIS_MD = RUNS / f"ASYMMETRIC_RISK_DISAGREEMENT_AWARE_{DATE}.md"
OUT_SENS = RUNS / f"ASYMMETRIC_RISK_SENSITIVITY_{DATE}.json"
OUT_SENS_MD = RUNS / f"ASYMMETRIC_RISK_SENSITIVITY_{DATE}.md"
OUT_REPLAY = RUNS / f"PRODUCTION_POLICY_REPLAY_{DATE}.json"
OUT_REPLAY_MD = RUNS / f"PRODUCTION_POLICY_REPLAY_{DATE}.md"
OUT_SUMMARY = RUNS / f"ASYMMETRIC_RISK_SUMMARY_{DATE}.md"

VERDICTS = ("invalid", "partially_valid", "valid")
RANK = {v: i for i, v in enumerate(VERDICTS)}

# ---- Phase 1: the frozen cost matrix (rows = ACTUAL, cols = PREDICTED) ------
COSTS_V1 = {
    "invalid": {"invalid": 0, "partially_valid": 3, "valid": 12},
    "partially_valid": {"invalid": 1, "partially_valid": 0, "valid": 5},
    "valid": {"invalid": 3, "partially_valid": 1, "valid": 0},
}
POLICY_NAME = "asymmetric_grading_risk_v1"
POLICY_SCHEMA_VERSION = 1
# normalization denominator: cases x the worst single-case cost in the matrix
MAX_CELL_COST = 12

# Human-review issue flags that mark an ACTIVE evidence/transcription/source
# problem (excluded from the clean disagreement-aware aggregate; routed to
# REVIEW by the structural policies). `genuinely_ambiguous` is deliberately
# NOT in this set: ambiguity is already expressed through reviewer
# disagreement weighting, not through data-integrity exclusion.
ISSUE_EXCLUDE = ("rubric_official_solution", "transcription_evidence")

ARM_ORDER = ("baseline_8b_one_pass", "arm_a_q8_0", "arm_b_two_pass")
CONST_ORDER = ("always_invalid", "always_partially_valid", "always_valid")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _self_hash(doc: dict, field: str = "content_sha256") -> dict:
    payload = json.dumps({k: v for k, v in doc.items() if k != field},
                         ensure_ascii=False, sort_keys=True)
    doc[field] = _sha(payload)
    return doc


def _verify_self_hash(doc: dict, field: str = "content_sha256") -> None:
    payload = json.dumps({k: v for k, v in doc.items() if k != field},
                         ensure_ascii=False, sort_keys=True)
    assert doc.get(field) == _sha(payload), "artifact self-hash mismatch"


def _write_frozen(path: Path, doc: dict) -> bool:
    """Append-only artifact discipline: identical re-derivation is a no-op,
    any different content is REFUSED (historical artifacts stay immutable)."""
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("content_sha256") == doc["content_sha256"]:
            return False
        raise SystemExit(f"REFUSED: {path.name} exists with different content "
                         f"({old.get('content_sha256', '?')[:12]} != "
                         f"{doc['content_sha256'][:12]}); artifacts are immutable")
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8", newline="\n")
    return True


def _git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=15).stdout.strip()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- loading ----

def load_reference() -> dict:
    doc = json.loads(REF_PATH.read_text(encoding="utf-8"))
    payload = json.dumps({k: v for k, v in doc.items() if k != "reference_sha256"},
                         ensure_ascii=False, sort_keys=True)
    assert doc["reference_sha256"] == _sha(payload), "reference freeze tampered"
    assert len(doc["cases"]) == 46
    assert not any(c["case_id"].startswith(("e005", "e006")) for c in doc["cases"])
    return doc


def load_policy() -> dict:
    doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    _verify_self_hash(doc, "policy_sha256")
    assert doc["policy_name"] == POLICY_NAME
    assert doc["cost_matrix"] == COSTS_V1, "policy matrix drifted from source"
    return doc


def load_baseline() -> dict[str, dict]:
    """The frozen baseline outputs: 44 SEEN-46 rows (stale r6/r8 excluded)
    + the 2 corrected rerun rows. Never mutated."""
    out: dict[str, dict] = {}
    for d in SEEN46_RUN_DIRS:
        p = RUNS / "grade_primary" / d / "scored.jsonl.json"
        for s in json.loads(p.read_text(encoding="utf-8")):
            if s["case_id"] not in REPAIRED:
                out[s["case_id"]] = s
    for line in RERUN_JSONL.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        assert s["case_id"] in REPAIRED
        out[s["case_id"]] = s
    assert len(out) == 46
    return out


def load_arm_a() -> dict[str, dict]:
    out = {json.loads(l)["case_id"]: json.loads(l)
           for l in ARM_A_JSONL.read_text(encoding="utf-8").splitlines()}
    assert len(out) == 46
    return out


def load_arm_b() -> dict[str, dict]:
    out = {json.loads(l)["case_id"]: json.loads(l)
           for l in ARM_B_JSONL.read_text(encoding="utf-8").splitlines()}
    assert len(out) == 46
    return out


def arm_rows() -> dict[str, dict[str, dict]]:
    """Per arm, per case: predicted verdict, decision, and the STRUCTURAL
    flags the replay policies may see (never the reference verdict).

    Arm B's verdict/decision come from its pre-registered combination; its
    structural flags come from pass-1 (the frozen baseline row) because pass-2
    uncertainty is already folded into the combined decision (verifier_unusable
    -> REVIEW)."""
    baseline, a, b = load_baseline(), load_arm_a(), load_arm_b()
    ref_ids = {c["case_id"] for c in load_reference()["cases"]}
    assert set(baseline) == set(a) == set(b) == ref_ids

    def flags(s: dict) -> dict:
        return {"schema_failure": bool(s.get("schema_failure")),
                "evidence_failure": bool(s.get("evidence_failure")),
                "uncertain": bool(s.get("uncertain")),
                "transcription_complete": bool(s.get("transcription_complete")),
                "validation_ok": bool(s.get("validation_ok"))}

    rows: dict[str, dict[str, dict]] = {arm: {} for arm in ARM_ORDER}
    for cid in sorted(ref_ids):
        rows["baseline_8b_one_pass"][cid] = {
            "predicted": baseline[cid]["predicted_verdict"],
            "decision": baseline[cid]["decision"], **flags(baseline[cid])}
        rows["arm_a_q8_0"][cid] = {
            "predicted": a[cid]["predicted_verdict"],
            "decision": a[cid]["decision"], **flags(a[cid])}
        rows["arm_b_two_pass"][cid] = {
            "predicted": b[cid]["combined"]["final_verdict"],
            "decision": b[cid]["combined"]["decision"], **flags(baseline[cid])}
    return rows


def classify_cases(ref: dict) -> dict[str, dict]:
    """Per case: the reviewer-pair relationship + active issue flags.

    Bucket precedence (deterministic, documented):
        owner_repaired > evidence_source_issue > wide_disagreement
        > adjacent_disagreement > agreed
    Raw relationship flags are kept alongside so nothing hides in the
    precedence (e.g. a wide case that is ALSO transcription-flagged)."""
    out = {}
    for c in ref["cases"]:
        fresh = c["independent_blind_reviews"]
        issues = sorted({r["issue"] for r in fresh if r["issue"] in ISSUE_EXCLUDE})
        ambiguity = sorted({r["issue"] for r in fresh if r["issue"] == "genuinely_ambiguous"})
        if c["reference_source"] == "owner_adjudicated_after_source_repair":
            kind, rel, acceptable = "owner_repaired", "owner_repaired", [c["final_verdict"]]
        else:
            assert len(fresh) == 2, c["case_id"]
            pair = {fresh[0]["verdict"], fresh[1]["verdict"]}
            if len(pair) == 1:
                rel, acceptable = "agreed", sorted(pair)
                assert acceptable == [c["final_verdict"]], c["case_id"]
            elif pair == {"invalid", "valid"}:
                rel, acceptable = "wide", sorted(pair, key=RANK.get)
            else:
                rel, acceptable = "adjacent", sorted(pair, key=RANK.get)
            if issues:
                kind = "evidence_source_issue"
            elif rel == "wide":
                kind = "wide_disagreement"
            elif rel == "adjacent":
                kind = "adjacent_disagreement"
            else:
                kind = "agreed"
        out[c["case_id"]] = {
            "bucket": kind, "relationship": rel,
            "acceptable_set": acceptable, "final_verdict": c["final_verdict"],
            "reference_source": c["reference_source"],
            "active_issue_flags": issues, "ambiguity_flags": ambiguity,
            "reviews": [{"reviewer": r["reviewer"], "verdict": r["verdict"],
                         "confidence": r["confidence"], "issue": r["issue"]}
                        for r in fresh]}
    return out


# ----------------------------------------------------------- Phase 1 ---------

RATIONALE = {
    "invalid->valid (12)": "catastrophic false full-credit: an invalid "
        "explanation is awarded full credit; awarded points are difficult to "
        "retract, and the error is invisible to the student (no appeal comes)",
    "partially_valid->valid (5)": "serious overgrade: full credit for a "
        "partially valid answer; also hard to retract, but the magnitude of "
        "the unearned award is smaller",
    "invalid->partially_valid (3)": "moderate overgrade: partial credit for "
        "an invalid answer; unearned award, smaller magnitude",
    "valid->invalid (3)": "serious undergrade: a fully valid answer stripped "
        "to zero; correctable through review/appeal, but a real, measured "
        "harm to the student and to trust in the system",
    "partially_valid->invalid (1)": "mild conservative error: adjacent "
        "undergrade, the most appeal-correctable mistake",
    "valid->partially_valid (1)": "mild conservative error: adjacent "
        "undergrade, the most appeal-correctable mistake",
    "correct (0)": "no cost",
}


def build_policy() -> dict:
    doc = {
        "policy_name": POLICY_NAME,
        "schema_version": POLICY_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "cost_matrix": COSTS_V1,
        "matrix_orientation": "rows = ACTUAL (human reference), "
                              "cols = PREDICTED (model)",
        "rationale": RATIONALE,
        "required_orderings": [
            "loss(invalid, valid) > loss(partially_valid, valid)",
            "loss(partially_valid, valid) > loss(partially_valid, invalid)",
            "every undergrade cost is nonzero (undergrading is an error, "
            "not a free action)"],
        "separation_of_layers": "the semantic grader "
            "(grade-v4-charitable-local) is UNCHANGED; this matrix and any "
            "AUTO/REVIEW routing built on it are a separate downstream risk "
            "layer — asymmetry is never implemented by telling the model to "
            "be stricter",
        "anti_gaming_statement": "matrix values were set from the owner's "
            "stated harm ordering BEFORE computing any arm's weighted loss "
            "and are not tuned to make any particular model win; robustness "
            "to nearby values is measured by the pre-registered sensitivity "
            "grid, never by editing this file",
        "frozen_before_held_out": "FROZEN before any HELD_OUT run; HELD_OUT "
            "remains sealed and untouched by the entire asymmetric-risk "
            "analysis",
        "normalization": {"denominator": "cases x max off-diagonal cost "
                                         f"({MAX_CELL_COST})",
                          "note": "normalized loss = total / (n_cases * "
                                  f"{MAX_CELL_COST}); 0 = perfect, 1 = every "
                                  "case at the catastrophic cell"},
    }
    return _self_hash(doc, "policy_sha256")


def freeze_policy() -> int:
    POLICIES.mkdir(parents=True, exist_ok=True)
    doc = build_policy()
    if POLICY_PATH.exists():
        old = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        _verify_self_hash(old, "policy_sha256")
        if old["cost_matrix"] == doc["cost_matrix"] and \
                old["policy_name"] == doc["policy_name"]:
            print(f"policy already frozen: {old['policy_sha256'][:12]} (kept)")
            return 0
        print("REFUSED: a different frozen policy already exists")
        return 3
    POLICY_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"froze {POLICY_PATH.name}, policy {doc['policy_sha256'][:12]}")
    return 0


# ------------------------------------------------------------ metrics --------

def weighted_loss(pairs: list[tuple[str, str]], costs: dict) -> dict:
    cells = {f"{a}->{p}": 0 for a in VERDICTS for p in VERDICTS if a != p}
    total = 0
    over_loss = under_loss = 0
    for a, p in pairs:
        c = costs[a][p]
        total += c
        if a != p:
            cells[f"{a}->{p}"] += 1
            if RANK[p] > RANK[a]:
                over_loss += c
            else:
                under_loss += c
    n = len(pairs)
    return {"cases": n, "total_weighted_loss": total,
            "mean_weighted_loss": round(total / n, 4) if n else None,
            "normalized_weighted_loss": round(total / (n * MAX_CELL_COST), 4) if n else None,
            "normalization_denominator": n * MAX_CELL_COST,
            "error_cells": cells,
            "false_full_invalid_to_valid": cells["invalid->valid"],
            "serious_overgrade_partial_to_valid": cells["partially_valid->valid"],
            "total_overgrade_loss": over_loss, "total_undergrade_loss": under_loss,
            "overgrades": sum(v for k, v in cells.items()
                              if RANK[k.split("->")[1]] > RANK[k.split("->")[0]]),
            "undergrades": sum(v for k, v in cells.items()
                               if RANK[k.split("->")[1]] < RANK[k.split("->")[0]])}


def exact_metrics(pairs: list[tuple[str, str]]) -> dict:
    """Same definitions as improvement_arms._metrics (kept numerically
    identical so the strict artifact cross-checks against the frozen report)."""
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    conf: dict[str, dict[str, int]] = {}
    for a, b in pairs:
        conf.setdefault(a, {}).setdefault(b, 0)
        conf[a][b] += 1
    per_class = {}
    for cls in VERDICTS:
        support = sum(1 for a, _ in pairs if a == cls)
        predicted = sum(1 for _, b in pairs if b == cls)
        tp = sum(1 for a, b in pairs if a == b == cls)
        rec = tp / support if support else None
        prec = tp / predicted if predicted else None
        f1 = (2 * prec * rec / (prec + rec)
              if prec is not None and rec is not None and (prec + rec) else 0.0)
        per_class[cls] = {"support": support, "tp": tp,
                          "recall": round(rec, 4) if rec is not None else None,
                          "precision": round(prec, 4) if prec is not None else None,
                          "f1": round(f1, 4)}
    recalls = [per_class[c]["tp"] / per_class[c]["support"] for c in VERDICTS
               if per_class[c]["support"]]
    f1s = [per_class[c]["f1"] for c in VERDICTS
           if per_class[c]["support"] or any(b == c for _, b in pairs)]
    return {"exact_agreement": agree,
            "exact_agreement_pct": round(100 * agree / n, 1) if n else None,
            "confusion_rows_reference": conf, "per_class": per_class,
            "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "balanced_accuracy": round(sum(recalls) / len(recalls), 4) if recalls else None}


def provenance(policy: dict, ref: dict) -> dict:
    return {
        "git_commit": _git_commit(),
        "policy_name": policy["policy_name"],
        "policy_sha256": policy["policy_sha256"],
        "reference_sha256": ref["reference_sha256"],
        "model_run_files_sha256": {
            "baseline_dev_scored": _file_sha(RUNS / "grade_primary" / SEEN46_RUN_DIRS[0]
                                             / "scored.jsonl.json"),
            "baseline_calibration_scored": _file_sha(RUNS / "grade_primary"
                                                     / SEEN46_RUN_DIRS[1]
                                                     / "scored.jsonl.json"),
            "corrected_rerun": _file_sha(RERUN_JSONL),
            "arm_a": _file_sha(ARM_A_JSONL),
            "arm_b": _file_sha(ARM_B_JSONL)},
        "case_coverage": 46,
        "held_out_cases": 0,
        "stale_outputs_excluded": "the 14 rows registered in "
                                  "STALE_MODEL_OUTPUTS_2026-09-01.json; corrected "
                                  "r6/r8 rows used instead",
        "new_inference_calls": 0, "cloud_calls": 0, "ocr_calls": 0, "rag_calls": 0,
    }


# ------------------------------------------------------------ Phase 2 + 4 ----

def constant_pairs(ref: dict, verdict: str) -> list[tuple[str, str]]:
    return [(c["final_verdict"], verdict) for c in
            sorted(ref["cases"], key=lambda c: c["case_id"])]


def run_strict() -> dict:
    policy = load_policy()
    ref = load_reference()
    rows = arm_rows()
    href = {c["case_id"]: c["final_verdict"] for c in ref["cases"]}
    arms = {}
    for arm in ARM_ORDER:
        pairs = [(href[cid], rows[arm][cid]["predicted"]) for cid in sorted(href)]
        auto = sum(1 for cid in href if rows[arm][cid]["decision"] == "AUTO")
        arms[arm] = {
            **weighted_loss(pairs, policy["cost_matrix"]),
            **exact_metrics(pairs),
            "evidence_failures": sum(1 for cid in href
                                     if rows[arm][cid]["evidence_failure"]),
            "schema_failures": sum(1 for cid in href
                                   if rows[arm][cid]["schema_failure"]),
            "auto": auto, "review": 46 - auto,
            "auto_pct": round(100 * auto / 46, 1)}
    constants = {}
    for name, v in zip(CONST_ORDER, VERDICTS):
        pairs = constant_pairs(ref, v)
        constants[name] = {**weighted_loss(pairs, policy["cost_matrix"]),
                           **exact_metrics(pairs)}
    best_const = min(constants, key=lambda k: constants[k]["total_weighted_loss"])
    doc = {
        "artifact": "asymmetric_risk_strict",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "final current 46-case human reference (strict, "
                  "reference_sha256 below); instructor grades, audit flags, "
                  "model votes and HELD_OUT are NOT targets",
        "provenance": provenance(policy, ref),
        "arms": arms,
        "constant_baselines": constants,
        "best_constant_policy": {
            "name": best_const,
            "total_weighted_loss": constants[best_const]["total_weighted_loss"],
            "note": "a model arm should not be recommended if its weighted "
                    "risk does not materially beat this trivial policy"},
        "comparison_vs_best_constant": {
            arm: {"arm_total": arms[arm]["total_weighted_loss"],
                  "best_constant_total": constants[best_const]["total_weighted_loss"],
                  "margin": constants[best_const]["total_weighted_loss"]
                            - arms[arm]["total_weighted_loss"],
                  "beats_best_constant": arms[arm]["total_weighted_loss"]
                            < constants[best_const]["total_weighted_loss"]}
            for arm in ARM_ORDER},
    }
    return _self_hash(doc)


def strict_md(doc: dict) -> str:
    md = [f"# Asymmetric risk — STRICT view ({doc['created_at']})", "",
          f"Policy `{doc['provenance']['policy_name']}` "
          f"`{doc['provenance']['policy_sha256'][:12]}…`; reference "
          f"`{doc['provenance']['reference_sha256'][:12]}…`; 46 seen cases; "
          "descriptive development numbers, NOT independent validation.", "",
          "| arm | total risk | mean | norm. | inv->val | pv->val | inv->pv | "
          "over-loss | under-loss | exact | macro-F1 | AUTO% |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARM_ORDER:
        a = doc["arms"][arm]
        md.append(f"| {arm} | {a['total_weighted_loss']} | {a['mean_weighted_loss']} "
                  f"| {a['normalized_weighted_loss']} | "
                  f"{a['false_full_invalid_to_valid']} | "
                  f"{a['serious_overgrade_partial_to_valid']} | "
                  f"{a['error_cells']['invalid->partially_valid']} | "
                  f"{a['total_overgrade_loss']} | {a['total_undergrade_loss']} | "
                  f"{a['exact_agreement']}/46 | {a['macro_f1']} | {a['auto_pct']}% |")
    md += ["", "Constant baselines (same 46-case denominator):", "",
           "| policy | total risk | mean | inv->val | pv->val | over-loss | "
           "under-loss | exact |", "|---|---|---|---|---|---|---|---|"]
    for name in CONST_ORDER:
        c = doc["constant_baselines"][name]
        md.append(f"| {name} | {c['total_weighted_loss']} | {c['mean_weighted_loss']} "
                  f"| {c['false_full_invalid_to_valid']} | "
                  f"{c['serious_overgrade_partial_to_valid']} | "
                  f"{c['total_overgrade_loss']} | {c['total_undergrade_loss']} | "
                  f"{c['exact_agreement']}/46 |")
    b = doc["best_constant_policy"]
    md += ["", f"**Best constant policy: `{b['name']}` at total risk "
           f"{b['total_weighted_loss']}.**"]
    for arm in ARM_ORDER:
        cmp_ = doc["comparison_vs_best_constant"][arm]
        verdict = ("beats it" if cmp_["beats_best_constant"] else
                   "does NOT beat it")
        md.append(f"- {arm}: {cmp_['arm_total']} — {verdict} "
                  f"(margin {cmp_['margin']}).")
    md += ["", "No arm produced invalid->valid (the catastrophic cell) on seen "
           "data; the weighted comparison is therefore driven by the smaller "
           "overgrade cells and the undergrade profile. The invalid class has "
           "only 5 seen cases — low statistical power, seen-data only."]
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- Phase 3 ----

def min_loss_over(acceptable: list[str], predicted: str, costs: dict) -> float:
    """Disagreement-aware case loss: the minimum loss against any
    human-supported verdict in the acceptable set."""
    return min(costs[a][predicted] for a in acceptable)


def run_disagreement() -> dict:
    policy = load_policy()
    ref = load_reference()
    rows = arm_rows()
    cls = classify_cases(ref)
    costs = policy["cost_matrix"]
    included = [cid for cid, c in sorted(cls.items())
                if c["bucket"] in ("agreed", "adjacent_disagreement")]
    excluded = {cid: c for cid, c in sorted(cls.items())
                if c["bucket"] in ("evidence_source_issue", "wide_disagreement")}
    owner_block = {cid: c for cid, c in sorted(cls.items())
                   if c["bucket"] == "owner_repaired"}
    weights = {cid: (1.0 if cls[cid]["relationship"] == "agreed" else 0.5)
               for cid in included}
    total_weight = sum(weights.values())

    arms = {}
    for arm in ARM_ORDER:
        per_case = {}
        total = 0.0
        strict_vs_aware_diffs = []
        for cid in included:
            pred = rows[arm][cid]["predicted"]
            acceptable = cls[cid]["acceptable_set"]
            min_loss = min_loss_over(acceptable, pred, costs)
            strict_loss = costs[cls[cid]["final_verdict"]][pred]
            contribution = round(weights[cid] * min_loss, 4)
            total += contribution
            per_case[cid] = {"predicted": pred, "acceptable_set": acceptable,
                             "weight": weights[cid], "min_loss": min_loss,
                             "strict_loss": strict_loss,
                             "contribution": contribution}
            if (min_loss == 0) != (strict_loss == 0):
                strict_vs_aware_diffs.append(cid)
        # separate blocks — never merged into the clean aggregate
        wide_rows = {cid: {"predicted": rows[arm][cid]["predicted"],
                           "reviewer_verdicts": [r["verdict"] for r in c["reviews"]],
                           "adjudicated_final": c["final_verdict"],
                           "production_recommendation": "REVIEW"}
                     for cid, c in excluded.items()
                     if c["bucket"] == "wide_disagreement"}
        issue_rows = {cid: {"predicted": rows[arm][cid]["predicted"],
                            "active_issue_flags": c["active_issue_flags"],
                            "relationship": c["relationship"],
                            "adjudicated_final": c["final_verdict"],
                            "strict_loss": costs[c["final_verdict"]][rows[arm][cid]["predicted"]]}
                      for cid, c in excluded.items()
                      if c["bucket"] == "evidence_source_issue"}
        owner_rows = {cid: {"predicted": rows[arm][cid]["predicted"],
                            "owner_reference": c["final_verdict"],
                            "strict_loss": costs[c["final_verdict"]][rows[arm][cid]["predicted"]],
                            "note": "owner-adjudicated after source repair — "
                                    "NOT two-reviewer consensus"}
                      for cid, c in owner_block.items()}
        arms[arm] = {
            "included_cases": len(included),
            "total_weight": total_weight,
            "total_weighted_loss": round(total, 4),
            "mean_weighted_loss_per_weight_unit": round(total / total_weight, 4),
            "mean_weighted_loss_per_included_case": round(total / len(included), 4),
            "cases_where_strict_and_aware_disagree_on_error": strict_vs_aware_diffs,
            "per_case": per_case,
            "wide_disagreement_block": wide_rows,
            "evidence_issue_block": issue_rows,
            "owner_repaired_block": owner_rows}

    doc = {
        "artifact": "asymmetric_risk_disagreement_aware",
        "view": "disagreement_aware_weighted_risk_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(policy, ref),
        "rules": [
            "agreed pair: weight 1.0, acceptable set = the agreed verdict",
            "adjacent disagreement: weight 0.5, acceptable set = both "
            "reviewer-supported verdicts, model loss = min over the set",
            "wide (invalid vs valid) disagreement: WIDE_HUMAN_DISAGREEMENT — "
            "excluded from the clean aggregate, reported separately, "
            "production recommendation REVIEW",
            "active evidence/transcription/source issue "
            f"({', '.join(ISSUE_EXCLUDE)}): excluded from the clean aggregate "
            "until resolved, reported separately",
            "owner-adjudicated source-repair cases: separate owner-reference "
            "block, never described as two-reviewer consensus",
            "bucket precedence: owner_repaired > evidence_source_issue > "
            "wide_disagreement > adjacent_disagreement > agreed; raw "
            "relationship counts reported independently so the overlap "
            "(1 wide case is also transcription-flagged) stays visible",
            "genuinely_ambiguous review flags do NOT exclude a case; ambiguity "
            "is captured by the disagreement weighting itself"],
        "case_classification": {cid: {k: v for k, v in c.items() if k != "reviews"}
                                for cid, c in sorted(cls.items())},
        "counts": {
            "included_clean": len(included),
            "included_agreed": sum(1 for cid in included
                                   if cls[cid]["relationship"] == "agreed"),
            "included_adjacent": sum(1 for cid in included
                                     if cls[cid]["relationship"] == "adjacent"),
            "excluded_evidence_source_issue": sum(
                1 for c in excluded.values() if c["bucket"] == "evidence_source_issue"),
            "excluded_wide_disagreement": sum(
                1 for c in excluded.values() if c["bucket"] == "wide_disagreement"),
            "owner_repaired_block": len(owner_block),
            "raw_relationship_counts": {
                "agreed": sum(1 for c in cls.values() if c["relationship"] == "agreed"),
                "adjacent": sum(1 for c in cls.values() if c["relationship"] == "adjacent"),
                "wide": sum(1 for c in cls.values() if c["relationship"] == "wide"),
                "owner_repaired": len(owner_block)},
            "overlap_wide_and_issue": sum(
                1 for c in cls.values()
                if c["relationship"] == "wide" and c["active_issue_flags"])},
        "arms": arms,
    }
    return _self_hash(doc)


def disagreement_md(doc: dict) -> str:
    n = doc["counts"]
    md = [f"# Asymmetric risk — DISAGREEMENT-AWARE view ({doc['created_at']})", "",
          f"`disagreement_aware_weighted_risk_v1`; policy "
          f"`{doc['provenance']['policy_sha256'][:12]}…`; reference "
          f"`{doc['provenance']['reference_sha256'][:12]}…`.", "",
          f"Included clean: **{n['included_clean']}** ({n['included_agreed']} "
          f"agreed @ weight 1.0 + {n['included_adjacent']} adjacent @ weight "
          f"0.5). Excluded: {n['excluded_evidence_source_issue']} active "
          f"evidence/source issues, {n['excluded_wide_disagreement']} wide "
          f"disagreements (raw wide = {n['raw_relationship_counts']['wide']}; "
          f"{n['overlap_wide_and_issue']} wide case is also issue-flagged and "
          f"is counted in the issue bucket). Owner-repaired block: "
          f"{n['owner_repaired_block']} (separate, never consensus).", "",
          "| arm | clean weighted loss | per weight unit | per included case | "
          "strict-vs-aware flips |", "|---|---|---|---|---|"]
    for arm in ARM_ORDER:
        a = doc["arms"][arm]
        md.append(f"| {arm} | {a['total_weighted_loss']} "
                  f"(of weight {a['total_weight']}) | "
                  f"{a['mean_weighted_loss_per_weight_unit']} | "
                  f"{a['mean_weighted_loss_per_included_case']} | "
                  f"{len(a['cases_where_strict_and_aware_disagree_on_error'])} |")
    md += ["", "Strict-vs-aware flips are cases scored as errors against the "
           "adjudicated verdict that match one of the two reviewer-supported "
           "verdicts (or vice versa) — boundary calls, not clear mistakes.", ""]
    base = doc["arms"]["baseline_8b_one_pass"]
    md += ["Wide-disagreement block (production recommendation: REVIEW):", ""]
    for cid, r in base["wide_disagreement_block"].items():
        md.append(f"- {cid}: reviewers {r['reviewer_verdicts']}, adjudicated "
                  f"{r['adjudicated_final']}, baseline model {r['predicted']}")
    md += ["", "Evidence/source-issue block (excluded until resolved):", ""]
    for cid, r in base["evidence_issue_block"].items():
        md.append(f"- {cid}: flags {r['active_issue_flags']}, adjudicated "
                  f"{r['adjudicated_final']}, baseline model {r['predicted']}")
    md += ["", "Owner-repaired block (separate reference source):", ""]
    for cid, r in base["owner_repaired_block"].items():
        md.append(f"- {cid}: owner reference {r['owner_reference']}, baseline "
                  f"model {r['predicted']} (strict loss {r['strict_loss']})")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- Phase 5 ----

SENS_GRID = {
    "invalid->valid": (10, 12, 15, 20),
    "partially_valid->valid": (3, 5, 7),
    "valid->invalid": (2, 3, 5),
    "adjacent_undergrade": (1, 2),   # partially_valid->invalid AND valid->partially_valid
}
SENS_FIXED = {"invalid->partially_valid": 3}


def sens_matrix(c_ivv: int, c_pvv: int, c_viv: int, c_adj: int) -> dict:
    return {"invalid": {"invalid": 0,
                        "partially_valid": SENS_FIXED["invalid->partially_valid"],
                        "valid": c_ivv},
            "partially_valid": {"invalid": c_adj, "partially_valid": 0,
                                "valid": c_pvv},
            "valid": {"invalid": c_viv, "partially_valid": c_adj, "valid": 0}}


def run_sensitivity() -> dict:
    policy = load_policy()
    ref = load_reference()
    rows = arm_rows()
    href = {c["case_id"]: c["final_verdict"] for c in ref["cases"]}
    pairs_by_policy = {arm: [(href[cid], rows[arm][cid]["predicted"])
                             for cid in sorted(href)] for arm in ARM_ORDER}
    for name, v in zip(CONST_ORDER, VERDICTS):
        pairs_by_policy[name] = constant_pairs(ref, v)
    order = list(ARM_ORDER) + list(CONST_ORDER)

    grid_results = []
    winner_freq: dict[str, int] = {}
    constant_win_matrices = []
    arm_a_beats_baseline = []
    for c_ivv, c_pvv, c_viv, c_adj in product(*SENS_GRID.values()):
        costs = sens_matrix(c_ivv, c_pvv, c_viv, c_adj)
        # coherence: catastrophic > serious overgrade > mild undergrade;
        # undergrades nonzero; serious undergrade >= adjacent undergrade
        assert costs["invalid"]["valid"] > costs["partially_valid"]["valid"] \
            > costs["partially_valid"]["invalid"] > 0
        assert costs["valid"]["invalid"] >= costs["valid"]["partially_valid"] > 0
        totals = {name: sum(costs[a][p] for a, p in pairs_by_policy[name])
                  for name in order}
        best = min(totals.values())
        winners = sorted(name for name in order if totals[name] == best)
        ranking = sorted(order, key=lambda n: (totals[n], n))
        for w in winners:
            winner_freq[w] = winner_freq.get(w, 0) + 1
        mid = {"matrix": {"invalid->valid": c_ivv, "partially_valid->valid": c_pvv,
                          "valid->invalid": c_viv, "adjacent_undergrade": c_adj,
                          "invalid->partially_valid": SENS_FIXED["invalid->partially_valid"]},
               "totals": totals, "winners": winners, "ranking": ranking}
        grid_results.append(mid)
        if any(w in CONST_ORDER for w in winners):
            constant_win_matrices.append(mid["matrix"])
        if totals["arm_a_q8_0"] < totals["baseline_8b_one_pass"]:
            arm_a_beats_baseline.append(mid["matrix"])

    n_matrices = len(grid_results)
    pair_stability = {}
    for i, x in enumerate(order):
        for y in order[i + 1:]:
            x_wins = sum(1 for g in grid_results if g["totals"][x] < g["totals"][y])
            y_wins = sum(1 for g in grid_results if g["totals"][y] < g["totals"][x])
            ties = n_matrices - x_wins - y_wins
            pair_stability[f"{x} vs {y}"] = {
                f"{x}_better": x_wins, f"{y}_better": y_wins, "ties": ties,
                # stable = the direction never reverses (all-win, all-loss,
                # or identical totals on every matrix)
                "stable": x_wins == n_matrices or y_wins == n_matrices
                          or ties == n_matrices}
    raw_counts = {name: weighted_loss(pairs_by_policy[name],
                                      policy["cost_matrix"])["error_cells"]
                  for name in order}
    doc = {
        "artifact": "asymmetric_risk_sensitivity",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(policy, ref),
        "grid_definition": {"varied": {k: list(v) for k, v in SENS_GRID.items()},
                            "fixed": SENS_FIXED,
                            "note": "adjacent_undergrade sets BOTH "
                                    "partially_valid->invalid and "
                                    "valid->partially_valid; every matrix "
                                    "satisfies the required cost ordering"},
        "matrices_evaluated": n_matrices,
        "raw_error_counts_per_policy": raw_counts,
        "grid_results": grid_results,
        "winner_frequency": dict(sorted(winner_freq.items())),
        "constant_policy_wins": {"count": len(constant_win_matrices),
                                 "matrices": constant_win_matrices},
        "arm_a_beats_baseline": {"count": len(arm_a_beats_baseline),
                                 "of": n_matrices,
                                 "driver": "adjacent_undergrade cost: arm A "
                                           "trades overgrades for adjacent "
                                           "undergrades (baseline-arm_a = "
                                           "3 + c_pv_to_valid - 5*c_adjacent)"},
        "pairwise_rank_stability": pair_stability,
    }
    return _self_hash(doc)


def sensitivity_md(doc: dict) -> str:
    md = [f"# Asymmetric risk — SENSITIVITY grid ({doc['created_at']})", "",
          f"{doc['matrices_evaluated']} matrices (deterministic full grid; "
          "fixed invalid->partially_valid = 3). Winner frequency (ties count "
          "for both):", ""]
    for name, k in sorted(doc["winner_frequency"].items()):
        md.append(f"- {name}: wins/ties best in {k}/{doc['matrices_evaluated']}")
    cw = doc["constant_policy_wins"]
    cw_desc = ""
    if cw["count"]:
        which = sorted({w for g in doc["grid_results"]
                        for w in g["winners"] if w in CONST_ORDER})
        adjs = sorted({m["adjacent_undergrade"] for m in cw["matrices"]})
        pvvs = sorted({m["partially_valid->valid"] for m in cw["matrices"]})
        cw_desc = (f" — {'/'.join(which)}, in matrices with "
                   f"adjacent_undergrade in {adjs} and "
                   f"partially_valid->valid in {pvvs} (full list in JSON)")
    md += ["", f"A CONSTANT policy is (co-)winner in **{cw['count']}"
           f"/{doc['matrices_evaluated']}** matrices{cw_desc}.",
           "", f"arm_a beats baseline in "
           f"{doc['arm_a_beats_baseline']['count']}/{doc['matrices_evaluated']} "
           "matrices; the flip is driven entirely by the adjacent-undergrade "
           "cost (arm A converts overgrades into adjacent undergrades).", "",
           "Key pairwise stability (stable = same direction on ALL matrices):", ""]
    for pair, s in doc["pairwise_rank_stability"].items():
        if not s["stable"]:
            md.append(f"- UNSTABLE {pair}: {s}")
    stable_pairs = [p for p, s in doc["pairwise_rank_stability"].items() if s["stable"]]
    md += ["", f"Stable pairs ({len(stable_pairs)}): " + "; ".join(stable_pairs), "",
           "Conclusion: the model-vs-model ranking (baseline vs q8_0) is NOT "
           "stable under plausible cost perturbations, and no arm robustly "
           "separates from `always_partially_valid`. Raw error counts (frozen "
           "v1 matrix) are reported alongside so weighted totals never hide "
           "count differences."]
    return "\n".join(md) + "\n"


# ------------------------------------------------------------ Phase 6 + 7 ----

REPLAY_POLICIES = ("AUTO_ALL", "AUTO_VALID_ONLY", "AUTO_VALID_AND_PARTIAL",
                   "HUMAN_DISPUTE_AWARE_B", "HUMAN_DISPUTE_AWARE_C")


def route_case(policy_name: str, row: dict, case_flags: dict) -> tuple[str, str]:
    """AUTO/REVIEW routing. Sees ONLY the model row's structural fields and
    the case's human-review dispute/issue flags — NEVER the reference verdict.
    Returns (decision, reason)."""
    if row["decision"] != "AUTO":
        return "REVIEW", "model_validator_review"
    if policy_name == "AUTO_ALL":
        return "AUTO", "structurally_valid"
    structural = (not row["schema_failure"] and not row["evidence_failure"]
                  and not row["uncertain"] and row["transcription_complete"]
                  and row["validation_ok"])
    if not structural:
        return "REVIEW", "structural_flag"
    if case_flags["active_issue_flags"]:
        return "REVIEW", "active_evidence_source_issue"
    allowed = (("valid",) if policy_name in ("AUTO_VALID_ONLY", "HUMAN_DISPUTE_AWARE_B")
               else ("valid", "partially_valid"))
    if row["predicted"] not in allowed:
        return "REVIEW", "verdict_not_in_auto_set"
    if policy_name.startswith("HUMAN_DISPUTE_AWARE") and case_flags["wide"]:
        return "REVIEW", "wide_human_disagreement"
    return "AUTO", "auto_conditions_met"


def run_replay() -> dict:
    policy = load_policy()
    ref = load_reference()
    rows = arm_rows()
    cls = classify_cases(ref)
    costs = policy["cost_matrix"]
    href = {c["case_id"]: c["final_verdict"] for c in ref["cases"]}
    case_flags = {cid: {"active_issue_flags": c["active_issue_flags"],
                        "wide": c["relationship"] == "wide",
                        "owner_repaired": c["relationship"] == "owner_repaired"}
                  for cid, c in cls.items()}

    replay = {}
    for arm in ARM_ORDER:
        replay[arm] = {}
        for pol in REPLAY_POLICIES:
            routed = {cid: route_case(pol, rows[arm][cid], case_flags[cid])
                      for cid in sorted(href)}
            auto_ids = [cid for cid, (d, _) in routed.items() if d == "AUTO"]
            review = {cid: reason for cid, (d, reason) in routed.items()
                      if d == "REVIEW"}
            auto_pairs = [(href[cid], rows[arm][cid]["predicted"]) for cid in auto_ids]
            wl = weighted_loss(auto_pairs, costs) if auto_pairs else None
            correct = sum(1 for a, p in auto_pairs if a == p)
            under_ids = [cid for cid in auto_ids
                         if RANK[rows[arm][cid]["predicted"]] < RANK[href[cid]]]
            over_ids = [cid for cid in auto_ids
                        if RANK[rows[arm][cid]["predicted"]] > RANK[href[cid]]]
            replay[arm][pol] = {
                "auto": len(auto_ids), "review": len(review),
                "auto_coverage_pct": round(100 * len(auto_ids) / 46, 1),
                "review_rate_pct": round(100 * len(review) / 46, 1),
                "correct_auto": correct,
                "auto_precision_pct": (round(100 * correct / len(auto_ids), 1)
                                       if auto_ids else None),
                "auto_total_weighted_loss": wl["total_weighted_loss"] if wl else 0,
                "auto_mean_weighted_loss": wl["mean_weighted_loss"] if wl else None,
                "auto_false_full_invalid_to_valid": (wl["error_cells"]["invalid->valid"]
                                                     if wl else 0),
                "auto_partial_to_valid": (wl["error_cells"]["partially_valid->valid"]
                                          if wl else 0),
                "auto_invalid_to_partial": (wl["error_cells"]["invalid->partially_valid"]
                                            if wl else 0),
                "auto_undergrades": len(under_ids),
                "auto_undergrade_cases": under_ids,
                "auto_overgrades": len(over_ids),
                "auto_overgrade_cases": over_ids,
                "review_reasons": {r: sum(1 for v in review.values() if v == r)
                                   for r in sorted(set(review.values()))},
                "review_cases": review,
                "owner_repaired_in_auto": [cid for cid in auto_ids
                                           if case_flags[cid]["owner_repaired"]],
                "est_review_per_100_explanation_cases": round(100 * len(review) / 46, 1),
                # Phase 7 — appeal-aware, counts only, no fabricated probabilities
                "appeal_aware": {
                    "auto_undergrade_count": len(under_ids),
                    "expected_appeal_candidates_upper_bound": len(under_ids),
                    "note": "upper bound = every automatically undergraded "
                            "student appeals; no measured appeal rate exists, "
                            "so no probability is fabricated",
                    "undergrade_verdict_step_deficit": sum(
                        RANK[href[cid]] - RANK[rows[arm][cid]["predicted"]]
                        for cid in under_ids),
                    "auto_overgrade_count": len(over_ids),
                    "overgrade_verdict_step_excess": sum(
                        RANK[rows[arm][cid]["predicted"]] - RANK[href[cid]]
                        for cid in over_ids),
                    "step_unit_note": "verdict steps (invalid<->partially_valid"
                                      "<->valid), NOT exam points: point "
                                      "mapping varies per question and is not "
                                      "invented here"}}
    doc = {
        "artifact": "production_policy_replay",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(policy, ref),
        "scope_note": "explanation-case automation ONLY: this 46-case "
                      "reference contains no OCR-quality or deterministic-MC "
                      "population, so NO full-exam automation claim is made",
        "non_stale_note": "all replayed outputs are current (stale r6/r8 "
                          "outputs structurally excluded; corrected reruns "
                          "used); owner-repaired cases are listed per policy "
                          "when they reach AUTO",
        "policies": {
            "AUTO_ALL": "every structurally valid model verdict is applied "
                        "(the arm's own validator decision)",
            "AUTO_VALID_ONLY": "AUTO iff verdict=valid AND schema/evidence "
                               "valid AND not uncertain AND transcription "
                               "complete AND no active human-flagged "
                               "evidence/source/rubric issue AND non-stale",
            "AUTO_VALID_AND_PARTIAL": "same structure, verdict in "
                                      "{valid, partially_valid}",
            "HUMAN_DISPUTE_AWARE_B": "AUTO_VALID_ONLY + wide human "
                                     "disagreement -> REVIEW",
            "HUMAN_DISPUTE_AWARE_C": "AUTO_VALID_AND_PARTIAL + wide human "
                                     "disagreement -> REVIEW"},
        "replay": replay,
    }
    return _self_hash(doc)


def replay_md(doc: dict) -> str:
    md = [f"# Production policy replay ({doc['created_at']})", "",
          doc["scope_note"] + ".", ""]
    for arm in ARM_ORDER:
        md += [f"## {arm}", "",
               "| policy | AUTO | cov% | REVIEW | AUTO prec% | AUTO risk | "
               "mean | false-full | pv->val | inv->pv | auto under | "
               "review/100 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for pol in REPLAY_POLICIES:
            r = doc["replay"][arm][pol]
            md.append(f"| {pol} | {r['auto']} | {r['auto_coverage_pct']} | "
                      f"{r['review']} | {r['auto_precision_pct']} | "
                      f"{r['auto_total_weighted_loss']} | "
                      f"{r['auto_mean_weighted_loss']} | "
                      f"{r['auto_false_full_invalid_to_valid']} | "
                      f"{r['auto_partial_to_valid']} | "
                      f"{r['auto_invalid_to_partial']} | "
                      f"{r['auto_undergrades']} | "
                      f"{r['est_review_per_100_explanation_cases']} |")
        md.append("")
    b = doc["replay"]["baseline_8b_one_pass"]["HUMAN_DISPUTE_AWARE_C"]
    md += ["Appeal-aware view (baseline, HUMAN_DISPUTE_AWARE_C): "
           f"{b['appeal_aware']['auto_undergrade_count']} automatic "
           f"undergrade(s) (upper-bound appeal candidates), verdict-step "
           f"deficit {b['appeal_aware']['undergrade_verdict_step_deficit']}; "
           f"{b['appeal_aware']['auto_overgrade_count']} automatic overgrades, "
           f"step excess {b['appeal_aware']['overgrade_verdict_step_excess']}. "
           "Counts only — no fabricated appeal probabilities.", "",
           "Raw semantic verdicts, structural evidence fields, risk-policy "
           "version and review reasons are preserved per case in the JSON."]
    return "\n".join(md) + "\n"


# ------------------------------------------------------------ Phase 8 + 9 ----

def run_summary() -> str:
    policy = load_policy()
    strict = json.loads(OUT_STRICT.read_text(encoding="utf-8"))
    dis = json.loads(OUT_DIS.read_text(encoding="utf-8"))
    sens = json.loads(OUT_SENS.read_text(encoding="utf-8"))
    replay = json.loads(OUT_REPLAY.read_text(encoding="utf-8"))
    for d in (strict, dis, sens, replay):
        _verify_self_hash(d)

    best_const = strict["best_constant_policy"]
    rec_arm = "baseline_8b_one_pass"
    rec_pol = "HUMAN_DISPUTE_AWARE_C"
    r = replay["replay"][rec_arm][rec_pol]
    base_all = replay["replay"][rec_arm]["AUTO_ALL"]
    s = strict["arms"][rec_arm]

    gates = [
        ("HARD_FALSE_FULL", "confirmed invalid -> automatic valid = 0",
         f"{r['auto_false_full_invalid_to_valid']} observed (all arms/policies "
         "0; only 5 seen invalid cases — low power)",
         r["auto_false_full_invalid_to_valid"] == 0, "PRODUCTION_POLICY_REPLAY"),
        ("SERIOUS_OVERGRADE", "automatic partially_valid -> valid <= 2/46",
         f"{r['auto_partial_to_valid']} (AUTO_ALL: {base_all['auto_partial_to_valid']})",
         r["auto_partial_to_valid"] <= 2, "PRODUCTION_POLICY_REPLAY"),
        ("WEIGHTED_RISK", "semantic-layer total risk <= 0.90 x best constant "
         f"({best_const['name']} = {best_const['total_weighted_loss']})",
         f"baseline {s['total_weighted_loss']} (needs <= "
         f"{round(0.90 * best_const['total_weighted_loss'], 1)})",
         s["total_weighted_loss"] <= 0.90 * best_const["total_weighted_loss"],
         "ASYMMETRIC_RISK_STRICT"),
        ("UNDERGRADE_CAP", "automatic harmful undergrades <= 3/46",
         f"{r['auto_undergrades']} under {rec_pol} (AUTO_ALL: "
         f"{base_all['auto_undergrades']})",
         r["auto_undergrades"] <= 3, "PRODUCTION_POLICY_REPLAY"),
        ("GROUNDING", "evidence+schema failures <= 2% of cases",
         f"{s['evidence_failures'] + s['schema_failures']}/46 = "
         f"{round(100 * (s['evidence_failures'] + s['schema_failures']) / 46, 1)}%",
         (s["evidence_failures"] + s["schema_failures"]) <= 0.02 * 46,
         "ASYMMETRIC_RISK_STRICT"),
        ("AUTOMATION_JOINT", "AUTO coverage >= 70% AND weighted-risk gate passes",
         f"coverage {r['auto_coverage_pct']}%; weighted-risk gate "
         f"{'passes' if s['total_weighted_loss'] <= 0.90 * best_const['total_weighted_loss'] else 'fails'}",
         r["auto_coverage_pct"] >= 70
         and s["total_weighted_loss"] <= 0.90 * best_const["total_weighted_loss"],
         "both artifacts"),
        ("DISAGREEMENT_ROUTING", "wide disagreement + active issues -> REVIEW",
         f"{rec_pol} routes all "
         f"{dis['counts']['raw_relationship_counts']['wide']} wide + "
         f"{dis['counts']['excluded_evidence_source_issue']} issue cases to "
         "REVIEW by construction",
         True, "PRODUCTION_POLICY_REPLAY (policy property)"),
        ("OCR", "production OCR validated separately before end-to-end shipping",
         "not validated (OCR_VALIDATION_PLAN_2026-09-02.md pending)",
         False, "OCR_VALIDATION_PLAN"),
        ("FINAL_TEST", "HELD_OUT untouched until grader+matrix+policy+OCR frozen",
         "HELD_OUT untouched (0 exposure in this task); matrix frozen; "
         "grader unchanged; decision policy NOT yet frozen; OCR NOT frozen",
         True, "this analysis"),
    ]

    md = [f"# Asymmetric-risk summary — {DATE}", "",
          f"Policy `{policy['policy_name']}` v{policy['schema_version']} "
          f"`{policy['policy_sha256'][:16]}…`; reference "
          f"`{strict['provenance']['reference_sha256'][:16]}…`; git "
          f"`{strict['provenance']['git_commit'][:12]}`. Zero-inference "
          "analysis; seen development data only.", "",
          "## Strict weighted risk (46 cases)", "",
          "| arm | total | mean | inv->val | pv->val | over-loss | under-loss "
          "| exact |", "|---|---|---|---|---|---|---|---|"]
    for arm in ARM_ORDER:
        a = strict["arms"][arm]
        md.append(f"| {arm} | {a['total_weighted_loss']} | "
                  f"{a['mean_weighted_loss']} | "
                  f"{a['false_full_invalid_to_valid']} | "
                  f"{a['serious_overgrade_partial_to_valid']} | "
                  f"{a['total_overgrade_loss']} | {a['total_undergrade_loss']} "
                  f"| {a['exact_agreement']}/46 |")
    for name in CONST_ORDER:
        c = strict["constant_baselines"][name]
        md.append(f"| {name} | {c['total_weighted_loss']} | "
                  f"{c['mean_weighted_loss']} | "
                  f"{c['false_full_invalid_to_valid']} | "
                  f"{c['serious_overgrade_partial_to_valid']} | "
                  f"{c['total_overgrade_loss']} | {c['total_undergrade_loss']} "
                  f"| {c['exact_agreement']}/46 |")
    margins = {arm: strict["comparison_vs_best_constant"][arm]["margin"]
               for arm in ARM_ORDER}
    md += ["",
           f"**No arm materially beats the best constant policy "
           f"(`{best_const['name']}`, {best_const['total_weighted_loss']}).** "
           f"Margins vs it: {margins} — the largest is "
           f"{round(100 * max(margins.values()) / best_const['total_weighted_loss'], 1)}% "
           "— below the 10% material threshold and not robust "
           "(see sensitivity).", "",
           f"## Disagreement-aware (clean {dis['counts']['included_clean']} "
           f"cases, weight "
           f"{dis['arms']['baseline_8b_one_pass']['total_weight']})", "",
           "| arm | clean loss | per weight unit |", "|---|---|---|"]
    for arm in ARM_ORDER:
        a = dis["arms"][arm]
        md.append(f"| {arm} | {a['total_weighted_loss']} | "
                  f"{a['mean_weighted_loss_per_weight_unit']} |")
    md += ["",
           "## Sensitivity",
           "",
           f"{sens['matrices_evaluated']} matrices; winner frequency "
           f"{sens['winner_frequency']}; a constant policy co-wins in "
           f"{sens['constant_policy_wins']['count']} matrices; arm_a beats "
           f"baseline in only {sens['arm_a_beats_baseline']['count']}"
           f"/{sens['matrices_evaluated']} (flips on the adjacent-undergrade "
           "cost). **Model ranking is NOT stable under the matrix.**", "",
           "## Release gates (proposed release_gates_asym_v1 — evaluated on "
           f"{rec_arm} + {rec_pol} unless noted; ALL estimates seen-data only)",
           "",
           "| gate | target | observed | verdict | evidence |",
           "|---|---|---|---|---|"]
    for name, target, observed, ok, ev in gates:
        md.append(f"| {name} | {target} | {observed} | "
                  f"{'PASS' if ok else 'FAIL'} | {ev} |")
    passed = sum(1 for *_, ok, _ev in gates if ok)
    md += ["", f"{passed}/{len(gates)} gates pass. The WEIGHTED_RISK, "
           "GROUNDING, AUTOMATION_JOINT and OCR gates block production.", "",
           "## Recommendation (NOT deployed — no models.toml, prompt, or "
           "policy change was made)", "",
           f"- semantic layer: keep `{rec_arm}` (qwen3-vl:8b-instruct Q4, "
           "one-pass, grade-v4-charitable-local). arm_a (q8_0) trades "
           "overgrades for undergrades with no robust risk win and lower "
           "exact agreement. arm_b (two-pass) changes no verdict; under the "
           "asymmetric objective its verifier DOES concentrate some serious "
           "overgrades into REVIEW (AUTO pv->valid "
           f"{replay['replay']['arm_b_two_pass']['HUMAN_DISPUTE_AWARE_C']['auto_partial_to_valid']}"
           f" vs {r['auto_partial_to_valid']}, AUTO risk "
           f"{replay['replay']['arm_b_two_pass']['HUMAN_DISPUTE_AWARE_C']['auto_total_weighted_loss']}"
           f" vs {r['auto_total_weighted_loss']}), but at 2x inference and "
           f"{replay['replay']['arm_b_two_pass']['HUMAN_DISPUTE_AWARE_C']['est_review_per_100_explanation_cases']}"
           "/100 review workload — a documented lower-risk/lower-coverage "
           "alternative, not the primary recommendation.",
           f"- risk layer candidate: `{rec_pol}` — AUTO {r['auto']}/46 "
           f"({r['auto_coverage_pct']}%), REVIEW {r['review']} "
           f"({r['est_review_per_100_explanation_cases']}/100 explanation "
           f"cases), AUTO precision {r['auto_precision_pct']}%, AUTO risk "
           f"{r['auto_total_weighted_loss']} (mean "
           f"{r['auto_mean_weighted_loss']}), false-full 0, pv->valid "
           f"{r['auto_partial_to_valid']}, automatic undergrades "
           f"{r['auto_undergrades']} — vs AUTO_ALL risk "
           f"{base_all['auto_total_weighted_loss']} at "
           f"{base_all['auto_coverage_pct']}% coverage.",
           "- NOT production-ready: the semantic layer does not beat "
           "`always_partially_valid` on weighted risk, invalid-recall is "
           "1/5, grounding failures exceed 2%, and OCR is unvalidated. "
           "Deploying the risk layer cannot fix the semantic layer.",
           "", "## Before HELD_OUT", "",
           "1. improve the semantic layer's invalid/partial discrimination "
           "(model or evidence improvements — NOT prompt-tightening as a "
           "risk proxy);",
           "2. freeze the AUTO/REVIEW decision policy version;",
           "3. validate the OCR route separately;",
           "4. re-pass gates; only then run HELD_OUT once, with the already-"
           "frozen matrix.", "",
           "Confirmations: new local inference 0; cloud 0; OCR 0; RAG 0; "
           "HELD_OUT exposure 0; human references modified 0; instructor "
           "grades modified 0; spend $0."]
    return "\n".join(md) + "\n"


# -------------------------------------------------------------------- cli ----

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["freeze-policy", "strict", "disagreement",
                                    "sensitivity", "replay", "summary", "all"])
    args = ap.parse_args(argv)
    steps = ([args.cmd] if args.cmd != "all" else
             ["freeze-policy", "strict", "disagreement", "sensitivity",
              "replay", "summary"])
    for step in steps:
        if step == "freeze-policy":
            rc = freeze_policy()
            if rc:
                return rc
        elif step == "strict":
            doc = run_strict()
            wrote = _write_frozen(OUT_STRICT, doc)
            OUT_STRICT_MD.write_text(strict_md(doc), encoding="utf-8", newline="\n")
            print(f"strict: total risk "
                  f"{ {a: doc['arms'][a]['total_weighted_loss'] for a in ARM_ORDER} } "
                  f"best constant {doc['best_constant_policy']['name']}="
                  f"{doc['best_constant_policy']['total_weighted_loss']} "
                  f"({'written' if wrote else 'unchanged'})")
        elif step == "disagreement":
            doc = run_disagreement()
            wrote = _write_frozen(OUT_DIS, doc)
            OUT_DIS_MD.write_text(disagreement_md(doc), encoding="utf-8", newline="\n")
            print(f"disagreement: clean loss "
                  f"{ {a: doc['arms'][a]['total_weighted_loss'] for a in ARM_ORDER} } "
                  f"({'written' if wrote else 'unchanged'})")
        elif step == "sensitivity":
            doc = run_sensitivity()
            wrote = _write_frozen(OUT_SENS, doc)
            OUT_SENS_MD.write_text(sensitivity_md(doc), encoding="utf-8", newline="\n")
            print(f"sensitivity: {doc['matrices_evaluated']} matrices, winners "
                  f"{doc['winner_frequency']} ({'written' if wrote else 'unchanged'})")
        elif step == "replay":
            doc = run_replay()
            wrote = _write_frozen(OUT_REPLAY, doc)
            OUT_REPLAY_MD.write_text(replay_md(doc), encoding="utf-8", newline="\n")
            b = doc["replay"]["baseline_8b_one_pass"]
            print("replay (baseline): " +
                  ", ".join(f"{p}: AUTO {b[p]['auto']} risk "
                            f"{b[p]['auto_total_weighted_loss']}"
                            for p in REPLAY_POLICIES) +
                  f" ({'written' if wrote else 'unchanged'})")
        elif step == "summary":
            OUT_SUMMARY.write_text(run_summary(), encoding="utf-8", newline="\n")
            print(f"summary written: {OUT_SUMMARY.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
