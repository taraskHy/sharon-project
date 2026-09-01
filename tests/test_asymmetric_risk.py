"""The frozen asymmetric risk policy + the zero-inference risk analyses.

Covers: matrix freeze integrity and required cost orderings, deterministic
strict / disagreement-aware / sensitivity / replay computation, exclusion and
provenance rules (wide disagreement, evidence issues, owner-repaired cases,
stale r6/r8 outputs), constant-baseline denominators, AUTO/REVIEW routing
without target leakage, record immutability, HELD_OUT absence, and the
no-model/no-cloud/no-OCR/no-RAG property of the analysis code itself.
No model / provider / network call anywhere in this file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from itertools import product
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
POLICY_PATH = REPO / "evaluation" / "model_selection" / "policies" / \
    "asymmetric_grading_risk_v1.json"
DATE = "2026-09-02"

# The frozen policy is pinned by its full content hash: any edit to the file
# (matrix, rationale, timestamps — anything) fails this suite.
FROZEN_POLICY_SHA256 = \
    "11e65e79e0f36cf6d1b4c12b1c2f8898b97244b402731091f894f462a000ebdd"


def _load():
    spec = importlib.util.spec_from_file_location(
        "asymmetric_risk", REPO / "scripts" / "asymmetric_risk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ar = _load()
needs_policy = pytest.mark.skipif(not POLICY_PATH.exists(),
                                  reason="policy not frozen")
needs_artifacts = pytest.mark.skipif(
    not (RUNS / f"ASYMMETRIC_RISK_STRICT_{DATE}.json").exists(),
    reason="strict artifact not present")


@pytest.fixture(scope="module")
def policy():
    return ar.load_policy()


@pytest.fixture(scope="module")
def ref():
    return ar.load_reference()


@pytest.fixture(scope="module")
def strict_doc():
    doc = json.loads((RUNS / f"ASYMMETRIC_RISK_STRICT_{DATE}.json")
                     .read_text(encoding="utf-8"))
    ar._verify_self_hash(doc)
    return doc


@pytest.fixture(scope="module")
def dis_doc():
    doc = json.loads((RUNS / f"ASYMMETRIC_RISK_DISAGREEMENT_AWARE_{DATE}.json")
                     .read_text(encoding="utf-8"))
    ar._verify_self_hash(doc)
    return doc


@pytest.fixture(scope="module")
def sens_doc():
    doc = json.loads((RUNS / f"ASYMMETRIC_RISK_SENSITIVITY_{DATE}.json")
                     .read_text(encoding="utf-8"))
    ar._verify_self_hash(doc)
    return doc


@pytest.fixture(scope="module")
def replay_doc():
    doc = json.loads((RUNS / f"PRODUCTION_POLICY_REPLAY_{DATE}.json")
                     .read_text(encoding="utf-8"))
    ar._verify_self_hash(doc)
    return doc


# ------------------------------------------------- 1-4: the frozen matrix ----


@needs_policy
def test_1_policy_name_version_and_content_hash_are_fixed(policy):
    assert policy["policy_name"] == "asymmetric_grading_risk_v1"
    assert policy["schema_version"] == 1
    assert policy["policy_sha256"] == FROZEN_POLICY_SHA256
    assert policy["cost_matrix"] == ar.COSTS_V1


@needs_policy
def test_1b_tampered_policy_is_refused(tmp_path, monkeypatch):
    doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    doc["cost_matrix"]["invalid"]["valid"] = 99
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ar, "POLICY_PATH", p)
    with pytest.raises(AssertionError):
        ar.load_policy()


def test_2_invalid_to_valid_is_the_largest_cost():
    cells = [ar.COSTS_V1[a][p] for a in ar.VERDICTS for p in ar.VERDICTS if a != p]
    assert ar.COSTS_V1["invalid"]["valid"] == max(cells) == 12
    assert all(ar.COSTS_V1["invalid"]["valid"] > c for c in cells
               if c != ar.COSTS_V1["invalid"]["valid"])


def test_3_partial_to_invalid_costs_less_than_partial_to_valid():
    assert ar.COSTS_V1["partially_valid"]["invalid"] \
        < ar.COSTS_V1["partially_valid"]["valid"]
    # and undergrading is never free
    assert ar.COSTS_V1["partially_valid"]["invalid"] > 0
    assert ar.COSTS_V1["valid"]["partially_valid"] > 0
    assert ar.COSTS_V1["valid"]["invalid"] > 0


def test_4_correct_predictions_cost_zero():
    for v in ar.VERDICTS:
        assert ar.COSTS_V1[v][v] == 0


# ------------------------------------------------ 5-6: deterministic loss ----


def test_5_strict_weighted_loss_is_deterministic_and_exact():
    pairs = [("invalid", "valid"), ("partially_valid", "valid"),
             ("valid", "invalid"), ("valid", "valid"),
             ("partially_valid", "invalid")]
    a = ar.weighted_loss(list(pairs), ar.COSTS_V1)
    b = ar.weighted_loss(list(reversed(pairs)), ar.COSTS_V1)
    assert a == b
    assert a["total_weighted_loss"] == 12 + 5 + 3 + 0 + 1
    assert a["false_full_invalid_to_valid"] == 1
    assert a["serious_overgrade_partial_to_valid"] == 1
    assert a["total_overgrade_loss"] == 17 and a["total_undergrade_loss"] == 4


@needs_artifacts
def test_5b_strict_totals_match_frozen_artifact_and_improvement_report(strict_doc):
    doc = ar.run_strict()
    for arm in ar.ARM_ORDER:
        for k in ("total_weighted_loss", "exact_agreement", "macro_f1",
                  "error_cells", "auto", "review"):
            assert doc["arms"][arm][k] == strict_doc["arms"][arm][k]
    rep = json.loads((RUNS / "LOCAL_IMPROVEMENT_REPORT_2026-09-02.json")
                     .read_text(encoding="utf-8"))
    for arm, key in (("baseline_8b_one_pass", "baseline"),
                     ("arm_a_q8_0", "arm_a"), ("arm_b_two_pass", "arm_b")):
        assert strict_doc["arms"][arm]["exact_agreement"] \
            == rep["arms"][key]["metrics"]["exact_agreement"]
        assert strict_doc["arms"][arm]["macro_f1"] \
            == rep["arms"][key]["metrics"]["macro_f1"]


@needs_artifacts
def test_6_disagreement_aware_loss_is_deterministic(dis_doc):
    doc = ar.run_disagreement()
    for arm in ar.ARM_ORDER:
        assert doc["arms"][arm]["total_weighted_loss"] \
            == dis_doc["arms"][arm]["total_weighted_loss"]
        assert doc["arms"][arm]["per_case"] == dis_doc["arms"][arm]["per_case"]
    assert doc["counts"] == dis_doc["counts"]


# --------------------------------------- 7-10: disagreement-aware rules -----


def test_7_adjacent_disagreement_takes_the_minimum_supported_loss():
    costs = ar.COSTS_V1
    # {partially_valid, valid} support: predicting either costs 0
    assert ar.min_loss_over(["partially_valid", "valid"], "valid", costs) == 0
    assert ar.min_loss_over(["partially_valid", "valid"], "partially_valid", costs) == 0
    # predicting invalid against that set = the milder undergrade (1, not 3)
    assert ar.min_loss_over(["partially_valid", "valid"], "invalid", costs) == 1
    # {invalid, partially_valid} support, prediction valid = min(12, 5) = 5
    assert ar.min_loss_over(["invalid", "partially_valid"], "valid", costs) == 5


@needs_artifacts
def test_7b_real_adjacent_case_uses_min_loss(dis_doc):
    # e002_q2_r5: reviewers valid/partially_valid, adjudicated valid,
    # baseline predicted partially_valid -> strict loss 1, aware loss 0
    pc = dis_doc["arms"]["baseline_8b_one_pass"]["per_case"]["e002_q2_r5"]
    assert pc["strict_loss"] == 1 and pc["min_loss"] == 0
    assert pc["weight"] == 0.5
    assert "e002_q2_r5" in dis_doc["arms"]["baseline_8b_one_pass"][
        "cases_where_strict_and_aware_disagree_on_error"]


@needs_artifacts
def test_8_wide_disagreement_is_excluded_and_flagged(dis_doc, ref):
    cls = ar.classify_cases(ref)
    wide = {cid for cid, c in cls.items() if c["relationship"] == "wide"}
    assert len(wide) == 8
    for arm in ar.ARM_ORDER:
        included = set(dis_doc["arms"][arm]["per_case"])
        assert not (wide & included)
        block = dis_doc["arms"][arm]["wide_disagreement_block"]
        for cid, row in block.items():
            assert row["production_recommendation"] == "REVIEW"
            assert set(row["reviewer_verdicts"]) == {"invalid", "valid"}
    # raw count 8 stays visible even though 1 sits in the issue bucket
    assert dis_doc["counts"]["raw_relationship_counts"]["wide"] == 8
    assert dis_doc["counts"]["excluded_wide_disagreement"] \
        + dis_doc["counts"]["overlap_wide_and_issue"] == 8


@needs_artifacts
def test_9_evidence_issues_are_excluded_from_clean_aggregate(dis_doc, ref):
    cls = ar.classify_cases(ref)
    flagged = {cid for cid, c in cls.items() if c["active_issue_flags"]}
    assert flagged, "expected active issue flags in the review history"
    for arm in ar.ARM_ORDER:
        assert not (flagged & set(dis_doc["arms"][arm]["per_case"]))
    # genuinely_ambiguous alone does NOT exclude
    amb_only = {cid for cid, c in cls.items()
                if c["ambiguity_flags"] and not c["active_issue_flags"]
                and c["relationship"] == "agreed"}
    for cid in amb_only:
        assert cid in dis_doc["arms"]["baseline_8b_one_pass"]["per_case"]


@needs_artifacts
def test_10_owner_repaired_cases_never_described_as_consensus(dis_doc, ref):
    cls = ar.classify_cases(ref)
    for cid in ("e004_q2_r6", "e004_q2_r8"):
        assert cls[cid]["bucket"] == "owner_repaired"
        assert cls[cid]["reference_source"] == "owner_adjudicated_after_source_repair"
        for arm in ar.ARM_ORDER:
            assert cid not in dis_doc["arms"][arm]["per_case"]
            row = dis_doc["arms"][arm]["owner_repaired_block"][cid]
            assert "NOT two-reviewer consensus" in row["note"]
    assert dis_doc["counts"]["owner_repaired_block"] == 2


# ----------------------------------------- 11-12: constants + sensitivity ---


@needs_artifacts
def test_11_constant_baselines_use_identical_denominators(strict_doc):
    for name in ar.CONST_ORDER:
        c = strict_doc["constant_baselines"][name]
        assert c["cases"] == 46
        assert c["normalization_denominator"] == 46 * ar.MAX_CELL_COST
    for arm in ar.ARM_ORDER:
        assert strict_doc["arms"][arm]["cases"] == 46


@needs_artifacts
def test_12_sensitivity_grid_is_deterministic_and_complete(sens_doc):
    doc = ar.run_sensitivity()
    assert doc["grid_results"] == sens_doc["grid_results"]
    assert doc["winner_frequency"] == sens_doc["winner_frequency"]
    expected = list(product(*ar.SENS_GRID.values()))
    assert len(sens_doc["grid_results"]) == len(expected) == 72
    seen = [(g["matrix"]["invalid->valid"], g["matrix"]["partially_valid->valid"],
             g["matrix"]["valid->invalid"], g["matrix"]["adjacent_undergrade"])
            for g in sens_doc["grid_results"]]
    assert seen == expected
    for g in sens_doc["grid_results"]:
        m = g["matrix"]
        assert m["invalid->valid"] > m["partially_valid->valid"] \
            > m["adjacent_undergrade"] > 0
        assert set(g["totals"]) == set(ar.ARM_ORDER) | set(ar.CONST_ORDER)


# ------------------------------------------------- 13-14: policy replay -----


@needs_artifacts
def test_13_replay_counts_auto_review_exactly(replay_doc):
    doc = ar.run_replay()
    for arm in ar.ARM_ORDER:
        for pol in ar.REPLAY_POLICIES:
            r = doc["replay"][arm][pol]
            assert r == replay_doc["replay"][arm][pol]
            assert r["auto"] + r["review"] == 46
            assert r["auto"] == 46 - len(r["review_cases"])
    base = replay_doc["replay"]["baseline_8b_one_pass"]
    assert base["AUTO_ALL"]["auto"] == 44          # the validator's 2 REVIEWs stick
    assert base["HUMAN_DISPUTE_AWARE_C"]["review_reasons"][
        "wide_human_disagreement"] == 4


def test_14_auto_valid_only_refuses_structurally_invalid_or_uncertain():
    clean = {"predicted": "valid", "decision": "AUTO", "schema_failure": False,
             "evidence_failure": False, "uncertain": False,
             "transcription_complete": True, "validation_ok": True}
    flags = {"active_issue_flags": [], "wide": False, "owner_repaired": False}
    assert ar.route_case("AUTO_VALID_ONLY", clean, flags) == \
        ("AUTO", "auto_conditions_met")
    for bad in ({"uncertain": True}, {"schema_failure": True},
                {"evidence_failure": True}, {"transcription_complete": False},
                {"validation_ok": False}, {"decision": "REVIEW"}):
        decision, _ = ar.route_case("AUTO_VALID_ONLY", {**clean, **bad}, flags)
        assert decision == "REVIEW", bad
    # non-valid verdicts refused by B; invalid refused by C
    for v in ("partially_valid", "invalid"):
        assert ar.route_case("AUTO_VALID_ONLY", {**clean, "predicted": v},
                             flags)[0] == "REVIEW"
    assert ar.route_case("AUTO_VALID_AND_PARTIAL",
                         {**clean, "predicted": "invalid"}, flags)[0] == "REVIEW"
    # active issue and wide disagreement routing
    assert ar.route_case("AUTO_VALID_ONLY", clean,
                         {**flags, "active_issue_flags": ["transcription_evidence"]}
                         )[0] == "REVIEW"
    assert ar.route_case("HUMAN_DISPUTE_AWARE_B", clean,
                         {**flags, "wide": True})[0] == "REVIEW"
    # AUTO_ALL applies any structurally valid verdict, even invalid
    assert ar.route_case("AUTO_ALL", {**clean, "predicted": "invalid"},
                         flags)[0] == "AUTO"


# ------------------------------------------- 15-17: immutability + stale ----


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@needs_artifacts
def test_15_source_records_remain_immutable_under_analysis():
    watched = [ar.REF_PATH, ar.RERUN_JSONL, ar.ARM_A_JSONL, ar.ARM_B_JSONL,
               POLICY_PATH] + \
        [RUNS / "grade_primary" / d / "scored.jsonl.json"
         for d in ar.SEEN46_RUN_DIRS]
    before = {p: _sha_file(p) for p in watched}
    ar.run_strict()
    ar.run_disagreement()
    ar.run_sensitivity()
    ar.run_replay()
    assert {p: _sha_file(p) for p in watched} == before


def test_15b_frozen_artifacts_refuse_different_content(tmp_path):
    doc1 = ar._self_hash({"artifact": "x", "value": 1})
    doc2 = ar._self_hash({"artifact": "x", "value": 2})
    p = tmp_path / "frozen.json"
    assert ar._write_frozen(p, doc1) is True
    assert ar._write_frozen(p, dict(doc1)) is False       # identical: no-op
    with pytest.raises(SystemExit, match="immutable"):
        ar._write_frozen(p, doc2)


def test_16_stale_r6_r8_outputs_are_excluded():
    baseline = ar.load_baseline()
    rerun = {json.loads(l)["case_id"]: json.loads(l)
             for l in ar.RERUN_JSONL.read_text(encoding="utf-8").splitlines()}
    stale = {s["case_id"]: s for d in ar.SEEN46_RUN_DIRS
             for s in json.loads((RUNS / "grade_primary" / d /
                                  "scored.jsonl.json").read_text(encoding="utf-8"))
             if s["case_id"] in ar.REPAIRED}
    assert set(stale) == set(ar.REPAIRED)      # stale rows exist, preserved
    for cid in ar.REPAIRED:
        assert baseline[cid] == rerun[cid]     # 17: corrected rows included
        assert baseline[cid] != stale[cid]     # 16: stale rows NOT used


@needs_artifacts
def test_17_provenance_pins_corrected_rerun(strict_doc):
    prov = strict_doc["provenance"]
    assert prov["model_run_files_sha256"]["corrected_rerun"] \
        == _sha_file(ar.RERUN_JSONL)
    assert "corrected" in prov["stale_outputs_excluded"]


# --------------------------------------------- 18-20: leakage + isolation ---


def test_18_routing_never_sees_the_reference_verdict(ref):
    rows = ar.arm_rows()
    cls = ar.classify_cases(ref)
    case_flags = {cid: {"active_issue_flags": c["active_issue_flags"],
                        "wide": c["relationship"] == "wide",
                        "owner_repaired": c["relationship"] == "owner_repaired"}
                  for cid, c in cls.items()}
    # the flags dict carries no reference/label field at all
    for flags in case_flags.values():
        assert set(flags) == {"active_issue_flags", "wide", "owner_repaired"}
    for row in rows["baseline_8b_one_pass"].values():
        assert "final_verdict" not in row and "label" not in str(sorted(row))
    # routing output is a pure function of (policy, row, flags): permuting the
    # reference verdicts cannot change it because it is never an input
    for arm in ar.ARM_ORDER:
        for pol in ar.REPLAY_POLICIES:
            for cid in list(case_flags)[:5]:
                a = ar.route_case(pol, rows[arm][cid], case_flags[cid])
                b = ar.route_case(pol, rows[arm][cid], case_flags[cid])
                assert a == b


@needs_artifacts
def test_19_no_held_out_ids_or_content_in_any_new_artifact():
    pat = re.compile(r"(?<![0-9a-fA-F])(e005|e006)(?![0-9a-fA-F])")
    files = [POLICY_PATH,
             RUNS / f"ASYMMETRIC_RISK_STRICT_{DATE}.json",
             RUNS / f"ASYMMETRIC_RISK_STRICT_{DATE}.md",
             RUNS / f"ASYMMETRIC_RISK_DISAGREEMENT_AWARE_{DATE}.json",
             RUNS / f"ASYMMETRIC_RISK_DISAGREEMENT_AWARE_{DATE}.md",
             RUNS / f"ASYMMETRIC_RISK_SENSITIVITY_{DATE}.json",
             RUNS / f"ASYMMETRIC_RISK_SENSITIVITY_{DATE}.md",
             RUNS / f"PRODUCTION_POLICY_REPLAY_{DATE}.json",
             RUNS / f"PRODUCTION_POLICY_REPLAY_{DATE}.md",
             RUNS / f"ASYMMETRIC_RISK_SUMMARY_{DATE}.md",
             RUNS / f"ASYMMETRIC_RISK_SOURCE_VERIFICATION_{DATE}.md"]
    for p in files:
        text = p.read_text(encoding="utf-8")
        assert not pat.search(text), p.name
    for name in (f"ASYMMETRIC_RISK_STRICT_{DATE}.json",
                 f"ASYMMETRIC_RISK_DISAGREEMENT_AWARE_{DATE}.json",
                 f"ASYMMETRIC_RISK_SENSITIVITY_{DATE}.json",
                 f"PRODUCTION_POLICY_REPLAY_{DATE}.json"):
        doc = json.loads((RUNS / name).read_text(encoding="utf-8"))
        assert doc["provenance"]["held_out_cases"] == 0
        assert doc["provenance"]["case_coverage"] == 46


def test_20_analysis_code_makes_no_model_cloud_ocr_rag_call():
    src = (REPO / "scripts" / "asymmetric_risk.py").read_text(encoding="utf-8")
    for banned in ("ModelGateway", "openrouter", "OpenRouter", "requests.",
                   "httpx", "urllib.request", "socket", "11434", "base_url",
                   "ollama", "anthropic", "openai", "easyocr", "tesseract",
                   "embedding", "faiss"):
        assert banned not in src, banned
    # subprocess is used ONLY for `git rev-parse HEAD`
    for m in re.finditer(r"subprocess\.run\(\[([^\]]*)\]", src):
        assert '"git"' in m.group(1)
    # and every analysis entry point declares zero calls in provenance
    prov = ar.provenance(ar.load_policy(), ar.load_reference())
    assert prov["new_inference_calls"] == 0 and prov["cloud_calls"] == 0
    assert prov["ocr_calls"] == 0 and prov["rag_calls"] == 0
