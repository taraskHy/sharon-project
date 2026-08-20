"""Package-level preflight (§18): one setup warning instead of hundreds of
per-student reviews. Deterministic; no model, no student exams."""

from __future__ import annotations

import copy

import pytest

from autograder.preflight import (READY, SETUP_REQUIRED, preflight_package, reviews_avoided)
from tests.test_grade import make_key

VARIANTS = ["A1", "A2", "A3"]


def full_alignment(key, variants=VARIANTS):
    return {v: {q.id: {s.id: s.id for s in q.sub_items} for q in key.questions}
            for v in variants}


def good(**over):
    key = over.pop("key", None) or make_key()
    kw = dict(key=key, variants=VARIANTS, alignment=full_alignment(key),
              policies={q.id: "choice_and_explanation_independent" for q in key.questions},
              rubric_question_ids=[q.id for q in key.questions],
              template={"answer_sheet_pages": [11, 12, 13]},
              required_crops={"answer_table": True})
    kw.update(over)
    return preflight_package(**kw)


def codes(rep):
    return sorted({f.code for f in rep.blocking})


# ------------------------------------------------------------------ valid ----


def test_a_valid_package_is_ready():
    rep = good()
    assert rep.ok and rep.status == READY and not rep.blocking
    assert rep.total_possible_score == sum(q.max_points for q in make_key().questions)
    assert "alignment_complete" in rep.checked and "grading_policies" in rep.checked
    assert READY in rep.summary()


def test_variant_ids_are_generic_tokens_not_a_fixed_vocabulary():
    key = make_key()
    key.versions = ["variant_1", "variant_2", "variant_3"]
    for q in key.questions:
        for s in q.sub_items:
            s.correct_by_version = {v: ["A"] for v in key.versions}
    rep = good(key=key, variants=key.versions, alignment=full_alignment(key, key.versions))
    assert rep.ok


# -------------------------------------------------------------- alignment ----


def test_a_variant_without_alignment_blocks_the_package():
    key = make_key()
    a = full_alignment(key)
    a["A2"] = "unresolved"
    rep = good(key=key, alignment=a)
    assert rep.status == SETUP_REQUIRED and codes(rep) == ["ALIGNMENT_UNRESOLVED"]
    f = rep.blocking[0]
    assert f.subject == "variant" and f.subject_id == "A2" and f.needed


def test_a_non_bijective_mapping_is_blocking():
    key = make_key()
    a = full_alignment(key)
    a["A1"]["1"]["2"] = "1"          # printed 1 and 2 both -> canonical 1
    rep = good(key=key, alignment=a)
    assert "DUPLICATE_CANONICAL_ASSIGNMENT" in codes(rep)
    assert "ALIGNMENT_INCOMPLETE" in codes(rep)    # canonical 2 now uncovered


def test_alignment_referring_to_unknown_ids_is_blocking():
    key = make_key()
    a = full_alignment(key)
    a["A1"]["99"] = {"1": "1"}
    a["A3"]["1"]["1"] = "does-not-exist"
    rep = good(key=key, alignment=a)
    assert "ALIGNMENT_UNKNOWN_QUESTION" in codes(rep)
    assert "ALIGNMENT_UNKNOWN_SUB_ITEM" in codes(rep)


# -------------------------------------------------------- key/rubric/ids -----


def test_rubric_naming_a_question_the_key_lacks_is_blocking():
    rep = good(rubric_question_ids=["1", "3", "7"])      # the key defines 1 and 3
    assert codes(rep) == ["RUBRIC_UNKNOWN_QUESTION"]
    assert "question 7" in rep.blocking[0].message.replace("'", "")


def test_duplicate_question_and_sub_item_ids_are_blocking():
    key = make_key()
    key.questions.append(copy.deepcopy(key.questions[0]))
    key.questions[0].sub_items.append(copy.deepcopy(key.questions[0].sub_items[0]))
    rep = good(key=key, alignment=full_alignment(key),
               policies={q.id: "choice_only" for q in key.questions})
    assert "DUPLICATE_QUESTION_ID" in codes(rep) and "DUPLICATE_SUB_ITEM_ID" in codes(rep)


def test_a_missing_key_answer_for_one_variant_is_blocking():
    key = make_key()
    key.questions[0].sub_items[0].correct_by_version = {"A1": ["F"], "A2": ["G"]}   # A3 missing
    rep = good(key=key)
    assert "MISSING_KEY_ANSWER" in codes(rep)
    assert any("variant A3" in f.message for f in rep.blocking)


def test_a_deterministically_unverified_key_value_blocks_once_at_package_level():
    key = make_key()
    key.questions[0].sub_items[0].versions_unverified = ["A3"]
    rep = good(key=key)
    assert "KEY_ANSWER_UNVERIFIED" in codes(rep)


def test_missing_or_inconsistent_maxima():
    key = make_key()
    key.questions[0].max_points = 0
    rep = good(key=key)
    assert "MISSING_MAX_SCORE" in codes(rep)
    key2 = make_key()
    key2.questions[0].max_points = 9999
    rep2 = good(key=key2)
    assert any(f.code == "MAX_SCORE_INCONSISTENT" for f in rep2.warnings)


def test_total_score_is_deterministic_and_a_stated_mismatch_only_warns():
    key = make_key()
    key.total_points = 1234
    rep = good(key=key)
    assert rep.ok and any(f.code == "TOTAL_SCORE_MISMATCH" for f in rep.warnings)
    assert rep.total_possible_score == sum(q.max_points for q in key.questions)


# ----------------------------------------------------------- policies etc ----


def test_invalid_and_orphaned_policies_are_blocking():
    key = make_key()
    rep = good(key=key, policies={"1": "grade_it_somehow", "42": "choice_only"})
    assert "INVALID_POLICY" in codes(rep) and "POLICY_UNKNOWN_QUESTION" in codes(rep)


def test_a_question_without_a_policy_only_warns():
    rep = good(policies={"1": "choice_only"})
    assert rep.ok and any(f.code == "POLICY_MISSING" for f in rep.warnings)


def test_missing_required_crop_regions_are_blocking():
    rep = good(required_crops={"answer_table": True, "explanation_area": False})
    assert codes(rep) == ["MISSING_CROP_REGION"]
    assert rep.blocking[0].subject == "template"


def test_unresolved_discovery_facts_are_carried_through():
    rep = good(unresolved=["variants", "policy:2"])
    assert codes(rep) == ["DISCOVERY_UNRESOLVED"] and len(rep.blocking) == 2


def test_no_variants_at_all_is_blocking():
    key = make_key()
    key.versions = []
    rep = preflight_package(key=key, variants=[])
    assert "NO_VARIANTS" in codes(rep)


def test_variant_not_present_in_the_key_is_blocking():
    rep = good(variants=["A1", "A2", "A9"], alignment=full_alignment(make_key(), ["A1", "A2", "A9"]))
    assert "VARIANT_NOT_IN_KEY" in codes(rep)


def test_invalid_internal_variant_ids_are_blocking():
    rep = good(variants=["A1", ""], alignment={"A1": full_alignment(make_key())["A1"]})
    assert "INVALID_VARIANT_ID" in codes(rep)


# ------------------------------------------------------------ the payoff -----


def test_one_package_warning_replaces_a_review_per_student():
    key = make_key()
    a = full_alignment(key)
    a["A2"] = "unresolved"
    rep = good(key=key, alignment=a)
    assert len(rep.blocking) == 1
    assert reviews_avoided(rep, n_exams=180) == 180
    assert SETUP_REQUIRED in rep.summary() and "needed:" in rep.summary()


def test_report_serialises_for_the_ui():
    rep = good(rubric_question_ids=["1", "3", "9"])
    d = rep.as_dict()
    assert d["status"] == SETUP_REQUIRED and d["ok"] is False
    assert d["blocking"][0]["code"] == "RUBRIC_UNKNOWN_QUESTION"
    assert d["blocking"][0]["needed"]


# ----------------------------------------- integration with the orchestrator --


def test_prepare_exam_package_runs_the_preflight(tmp_path):
    from autograder.orchestrator import prepare_exam_package
    from autograder.preflight import alignment_from_discovery

    key = make_key()
    out = prepare_exam_package(None, key=key, key_bytes=b"KEY", key_path=tmp_path / "key.pdf",
                               packages_root=tmp_path / "pkg", write_missing_sidecars=False)
    pre = out["preflight"]
    assert out["package_status"] in (READY, SETUP_REQUIRED)
    assert pre.total_possible_score == sum(q.max_points for q in key.questions)
    assert isinstance(out["setup_required"], list)
    # discovery's identity alignment is complete by construction
    norm = alignment_from_discovery({"A1": {"identity": True}}, ["A1"], key)
    assert norm["A1"]["1"]["1"] == "1"
    assert alignment_from_discovery({}, ["A1"], key)["A1"]["1"]["1"] == "1"
    assert alignment_from_discovery({"A1": {"identity": True}}, ["A1", "A2"], key)["A2"] == "unresolved"
