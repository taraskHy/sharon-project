"""The frozen final human reference + the two pre-registered improvement arms.

Covers: freeze integrity and source separation, the deterministic pre-registered
combination rule, target-leakage prevention, writer folds, local-only execution,
HELD_OUT absence, metric determinism, and append-only artifact discipline.
No model / provider / network call anywhere in this file.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
REF_PATH = RUNS / "FINAL_HUMAN_REFERENCE_2026-09-02.json"
SPEC_PATH = REPO / "evaluation" / "model_selection" / "experiments" / \
    "LOCAL_IMPROVEMENT_ARMS_2026-09-02.json"


def _load_arms():
    spec = importlib.util.spec_from_file_location(
        "improvement_arms", REPO / "scripts" / "improvement_arms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


arms = _load_arms()
needs_ref = pytest.mark.skipif(not REF_PATH.exists(), reason="reference freeze not present")
needs_spec = pytest.mark.skipif(not SPEC_PATH.exists(), reason="experiment spec not frozen")


@pytest.fixture(scope="module")
def ref():
    return arms.load_reference()          # verifies the sha256 itself


# ------------------------------------------------------ freeze integrity ----


@needs_ref
def test_reference_freeze_hash_verifies(ref):
    assert len(ref["cases"]) == 46


@needs_ref
def test_a_tampered_reference_is_refused(tmp_path, monkeypatch):
    doc = json.loads(REF_PATH.read_text(encoding="utf-8"))
    doc["cases"][0]["final_verdict"] = "valid" if doc["cases"][0]["final_verdict"] != "valid" \
        else "invalid"
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(arms, "REF_PATH", p)
    with pytest.raises(AssertionError, match="tampered"):
        arms.load_reference()


@needs_ref
def test_sources_are_distinct_and_never_collapsed(ref):
    counts = {}
    for c in ref["cases"]:
        counts[c["reference_source"]] = counts.get(c["reference_source"], 0) + 1
    assert counts == {"two_reviewer_consensus": 22, "adjudicated_human_reference": 22,
                      "owner_adjudicated_after_source_repair": 2}
    assert set(counts) == set(("two_reviewer_consensus", "adjudicated_human_reference",
                               "owner_adjudicated_after_source_repair"))


@needs_ref
def test_class_distribution_is_28_13_5(ref):
    assert ref["class_distribution"] == {"invalid": 5, "partially_valid": 13, "valid": 28}
    assert len(ref["invalid_class_cases"]) == 5   # invalid is now MEASURED on seen data


@needs_ref
def test_every_case_preserves_full_provenance(ref):
    for c in ref["cases"]:
        assert c["original_instructor"]["ground_truth_source"] == "original_instructor_grade"
        assert "baseline_model_output" in c and "adjudication_record" in c
        if c["case_id"] in ("e004_q2_r6", "e004_q2_r8"):
            assert c["reference_source"] == "owner_adjudicated_after_source_repair"
            assert len(c["stale_historical_reviews"]) == 2
            assert not c["independent_blind_reviews"]
            assert c["corrected_provenance"]
        else:
            assert len(c["independent_blind_reviews"]) == 2
            assert not c["stale_historical_reviews"]


@needs_ref
def test_redundant_final_analysis_is_provenance_only(ref):
    r2 = ref["redundant_final_analysis"]
    assert r2["case_id"] == "e004_q2_r2"
    assert r2["effective_reference_verdict"] == r2["consensus_verdict"] \
        == r2["final_verdict_in_db"]
    assert "no verdict" in r2["reopen_effect"] or "Numeric change: none" in r2["reopen_effect"]
    assert r2["action_taken"].startswith("none")


@needs_ref
def test_no_held_out_writer_in_the_population(ref):
    pat = re.compile(r"(?<![0-9a-fA-F])(e005|e006)(?![0-9a-fA-F])")
    for c in ref["cases"]:
        assert not pat.search(c["case_id"])
    assert len({c["writer"] for c in ref["cases"]} & {"e005", "e006"}) == 0


@needs_ref
def test_writer_folds_cover_the_population(ref):
    by_writer = {}
    for c in ref["cases"]:
        by_writer[c["writer"]] = by_writer.get(c["writer"], 0) + 1
    assert by_writer == {"e002": 16, "e003": 15, "e004": 14, "e007": 1}


# --------------------------------------- the pre-registered combination -----


def _v(rec="valid", central=True, directional=False, uncertain=False, quotes=("שלום עולם",)):
    return {"supported": True, "central_idea_present": central,
            "directionally_correct_but_incomplete": directional,
            "proposed_too_strict": False, "proposed_too_generous": False,
            "recommended_verdict": rec, "evidence": [{"quote": q} for q in quotes],
            "uncertain": uncertain}


TXT = "הסבר: שלום עולם וגם עוד רעיון מרכזי"


def test_rule_0_pass1_review_sticks():
    out = arms.combine("valid", "REVIEW", _v("valid"), TXT)
    assert out == {"final_verdict": "valid", "decision": "REVIEW",
                   "rule": "pass1_review_sticks"}


def test_rule_1_unusable_verifier_never_changes_anything():
    for bad in (None, _v(uncertain=True), _v(quotes=("not in text",)),
                _v(quotes=("של",)),                      # below the 3-char minimum
                _v("valid", quotes=())):                 # credit without evidence
        out = arms.combine("partially_valid", "AUTO", bad, TXT)
        assert out["final_verdict"] == "partially_valid"
        assert out["decision"] == "REVIEW"
        assert out["rule"] == "verifier_unusable"


def test_rule_2_agreement_is_auto():
    out = arms.combine("valid", "AUTO", _v("valid"), TXT)
    assert out == {"final_verdict": "valid", "decision": "AUTO", "rule": "agreed"}


def test_rule_3_one_step_upgrade_needs_grounded_central_idea():
    up = arms.combine("partially_valid", "AUTO", _v("valid", central=True), TXT)
    assert up == {"final_verdict": "valid", "decision": "AUTO", "rule": "verifier_upgrade"}
    no_gate = arms.combine("partially_valid", "AUTO", _v("valid", central=False), TXT)
    assert no_gate["final_verdict"] == "partially_valid" and no_gate["decision"] == "REVIEW"
    # invalid -> partially_valid may pass on directional correctness instead
    up2 = arms.combine("invalid", "AUTO", _v("partially_valid", central=False,
                                             directional=True), TXT)
    assert up2 == {"final_verdict": "partially_valid", "decision": "AUTO",
                   "rule": "verifier_upgrade"}


def test_rule_4_two_step_upgrade_is_never_automated():
    out = arms.combine("invalid", "AUTO", _v("valid"), TXT)
    assert out == {"final_verdict": "invalid", "decision": "REVIEW",
                   "rule": "two_step_disagreement"}


def test_rule_5_downgrades_are_never_automated():
    for p1, p2 in (("valid", "partially_valid"), ("valid", "invalid"),
                   ("partially_valid", "invalid")):
        out = arms.combine(p1, "AUTO", _v(p2), TXT)
        assert out == {"final_verdict": p1, "decision": "REVIEW",
                       "rule": "verifier_flags_generosity"}


def test_rule_is_deterministic():
    for _ in range(3):
        assert arms.combine("partially_valid", "AUTO", _v("valid"), TXT) \
            == arms.combine("partially_valid", "AUTO", _v("valid"), TXT)


@needs_spec
def test_rule_hash_matches_the_frozen_spec():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert arms.rule_source_sha() == spec["combination_rule_sha256"]


# --------------------------------------------------- leakage prevention -----


def test_request_scan_catches_every_forbidden_token():
    for banned in arms.FORBIDDEN_IN_REQUEST:
        with pytest.raises(AssertionError, match="leakage"):
            arms._scan_request(f"prefix {banned} suffix")
    arms._scan_request("clean question text with rubric and transcription")


def test_verifier_prompt_and_blocks_carry_no_target_fields():
    arms._scan_request(arms.VERIFY_SYSTEM)
    from autograder.benchmark.manifests import load_manifest
    from autograder.benchmark.roles import pack_from_inputs
    m = load_manifest("grade_primary")
    case = next(c for c in m.cases if c.split in ("DEV", "CALIBRATION"))
    pack = pack_from_inputs(case.inputs["pack"])
    blocks = arms.verify_blocks(pack, case.inputs["transcription"] or "", "partially_valid",
                                [{"id": "R1", "met": True, "student_evidence": "x"}])
    arms._scan_request(blocks[0]["text"])
    for banned in ("actual_instructor_score", "ground_truth", "final_labels"):
        assert banned not in blocks[0]["text"]


@needs_spec
def test_spec_is_local_only_and_rag_disabled():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["backend"]["base_url"].startswith("http://localhost")
    assert spec["backend"]["rag_policy"] == "RAG_DISABLED"
    assert spec["backend"]["cacheable"] is False
    assert spec["writer_folds"]["held_writer_for_rule"] == "e004"
    assert spec["writer_folds"]["rule_derivation_writers"] == ["e002", "e003"]
    ids = spec["population"]["case_ids_in_order"]
    assert len(ids) == 46 and ids == sorted(ids)
    pat = re.compile(r"(?<![0-9a-fA-F])(e005|e006)(?![0-9a-fA-F])")
    assert not any(pat.search(i) for i in ids)


@needs_spec
def test_freezing_refuses_to_overwrite():
    assert arms.main(["freeze"]) == 3


# ------------------------------------------------- deterministic metrics ----


def test_metrics_are_deterministic_and_exact():
    pairs = [("valid", "valid"), ("valid", "invalid"), ("partially_valid", "valid"),
             ("invalid", "invalid"), ("partially_valid", "partially_valid")]
    m1, m2 = arms._metrics(list(pairs)), arms._metrics(list(reversed(pairs)))
    assert m1 == m2
    assert m1["exact_agreement"] == 3 and m1["cases"] == 5
    assert m1["harmful_overgrades"] == 1 and m1["harmful_undergrades"] == 1
    assert m1["per_class"]["valid"]["recall_exact"] == "1/2"
    assert m1["per_class"]["invalid"]["recall_exact"] == "1/1"
    # hand-computed: recalls 0.5, 0.5, 1.0 -> balanced accuracy 2/3
    assert m1["balanced_accuracy"] == round((0.5 + 0.5 + 1.0) / 3, 4)


@needs_ref
def test_baseline_metrics_artifact_uses_exact_rows():
    doc = json.loads((RUNS / "BASELINE_CLASS_METRICS_2026-09-02.json").read_text(encoding="utf-8"))
    pc = doc["per_class"]
    assert pc["valid"]["recall_exact"] == "24/28"
    assert pc["partially_valid"]["recall_exact"] == "6/13"
    assert pc["invalid"]["recall_exact"] == "1/5"
    assert doc["exact_agreement"] == 31
    assert doc["harmful_overgrades"] == 8 and doc["harmful_undergrades"] == 7


# ------------------------------------------------------- append-only jsonl --


@needs_spec
def test_runs_refuse_to_exceed_the_registered_budget(tmp_path, monkeypatch):
    """A run directory that already holds 46 rows must never take another call:
    the runner exits 3 (over budget) before touching any gateway."""
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    fake = tmp_path / "arm_a.jsonl"
    rows = [json.dumps({"case_id": f"fake_{i}"}) for i in range(46)]
    fake.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(arms, "ARM_A_JSONL", fake)
    assert arms.run_a() == 3
    assert len(fake.read_text(encoding="utf-8").splitlines()) == 46  # untouched
    assert spec["arms"]["arm_a"]["max_local_evaluations"] == 46
