"""Review reason codes (§12), prioritisation (§11) and mechanical grouping
(§13). Deterministic; no model, no I/O."""

from __future__ import annotations

import copy

import pytest

from autograder.reviewqueue import (REASONS, ReviewCase, apply_scope, classify_reason,
                                    group_cases, mechanical_fingerprint, prioritize,
                                    queue_summary, render_explanation)


def case(**kw) -> ReviewCase:
    base = dict(exam_id="exam001", question_id="1", sub_item_id="3",
                reason_code="GRADE_UNCERTAIN", points_affected=1.0, max_points=4.0)
    base.update(kw)
    return ReviewCase(**base)


# ---------------------------------------------------------------- §12 codes --


def test_every_required_reason_code_exists():
    required = {"MC_UNRESOLVED", "MC_CONFLICT", "OCR_UNRESOLVED", "OCR_PROVIDER_DISAGREEMENT",
                "GRADE_INVALID", "GRADE_UNCERTAIN", "GRADE_DISAGREEMENT", "EVIDENCE_INVALID",
                "VARIANT_UNRESOLVED", "ALIGNMENT_UNRESOLVED", "PACKAGE_ANOMALY",
                "BUDGET_PAUSED", "PROVIDER_FAILED"}
    assert required <= set(REASONS)


def test_mc_conflict_explanation_shows_the_competing_readings():
    text = render_explanation("MC_CONFLICT", {"deterministic": ["B", "C"], "local": "B",
                                              "cloud": "C", "resolution": "unresolved"})
    assert "MC_CONFLICT" in text and "Local resolver: B" in text and "Cloud resolver: C" in text
    assert "Resolution: unresolved" in text
    assert len(text) < 400          # concise: no generated narrative


def test_grade_disagreement_explanation_shows_scores_and_disputed_items():
    text = render_explanation("GRADE_DISAGREEMENT", {
        "primary_score": 7, "escalation_score": 4, "max_score": 10,
        "disputed_rubric_items": ["R2", "R3"]})
    assert "Primary: 7/10" in text and "Escalation: 4/10" in text
    assert "Disputed rubric items: R2, R3" in text


def test_evidence_invalid_explanation_names_the_items():
    text = render_explanation("EVIDENCE_INVALID", {"fabricated_items": ["R1"], "max_score": 4})
    assert "Unsupported evidence for: R1" in text


def test_legacy_free_text_reasons_get_a_code():
    assert classify_reason("exam version detection is uncertain: ...") == "VARIANT_UNRESOLVED"
    assert classify_reason("the explanation could not be read reliably") == "OCR_UNRESOLVED"
    assert classify_reason("final intention could not be determined (candidates: B, C)") == "MC_UNRESOLVED"
    assert classify_reason("rubric item R1 cites evidence absent from the student") == "EVIDENCE_INVALID"
    assert classify_reason("something new entirely", kind="grading") == "GRADE_UNCERTAIN"


# ----------------------------------------------------------- §11 priority ----


def test_systemic_case_sorts_ahead_of_a_high_point_individual_case():
    systemic = case(exam_id="exam009", reason_code="VARIANT_UNRESOLVED", points_affected=2.0,
                    batch_warning_code="VARIANT_UNRESOLVED_RATE", batch_warning_students=14,
                    mechanical_kind="variant_marker", facts={"marker_seen": "faint icon"})
    high = case(exam_id="exam001", reason_code="GRADE_DISAGREEMENT", points_affected=10.0)
    mc = case(exam_id="exam002", reason_code="MC_UNRESOLVED", points_affected=2.0,
              wrong_choice_zero=True)
    ocr = case(exam_id="exam003", reason_code="OCR_PROVIDER_DISAGREEMENT", points_affected=6.0)
    low = case(exam_id="exam004", reason_code="GRADE_UNCERTAIN", points_affected=0.5)
    order = [c.reason_code for c in prioritize([low, ocr, mc, high, systemic])]
    assert order == ["VARIANT_UNRESOLVED", "GRADE_DISAGREEMENT", "MC_UNRESOLVED",
                     "OCR_PROVIDER_DISAGREEMENT", "GRADE_UNCERTAIN"]


def test_unresolved_mc_under_wrong_choice_zero_outranks_a_low_point_grade_doubt():
    mc = case(exam_id="e1", reason_code="MC_UNRESOLVED", points_affected=2.0, wrong_choice_zero=True)
    low = case(exam_id="e2", reason_code="GRADE_UNCERTAIN", points_affected=3.0)
    assert prioritize([low, mc])[0] is mc


def test_more_students_and_more_points_sort_first_within_a_tier():
    a = case(exam_id="e1", reason_code="GRADE_DISAGREEMENT", points_affected=5.0)
    b = case(exam_id="e2", reason_code="GRADE_DISAGREEMENT", points_affected=9.0)
    assert prioritize([a, b])[0] is b


def test_priority_is_deterministic_and_changes_no_grade():
    cases = [case(exam_id=f"e{i}", points_affected=float(i % 3), max_points=4.0) for i in range(8)]
    snapshot = [copy.deepcopy(c) for c in cases]
    first = [c.exam_id for c in prioritize(cases)]
    second = [c.exam_id for c in prioritize(list(reversed(cases)))]
    assert first == second
    assert all(a.points_affected == b.points_affected and a.max_points == b.max_points
               for a, b in zip(cases, snapshot))


# ----------------------------------------------------------- §13 grouping ----


def test_identical_mechanical_fingerprints_group_into_one_decision():
    cases = [case(exam_id=f"exam{i:03d}", reason_code="VARIANT_UNRESOLVED",
                  mechanical_kind="variant_marker",
                  facts={"marker_seen": "four-petal icon, bottom third"})
             for i in range(7)]
    groups = group_cases(cases)
    assert len(groups) == 1
    g = groups[0]
    assert g.size == 7 and g.apply_to_all_eligible and len(g.students) == 7
    scope = apply_scope(g)
    assert scope["mechanical_kind"] == "variant_marker" and len(scope["items"]) == 7
    assert scope["fingerprint"] == g.fingerprint


def test_different_mechanical_facts_do_not_group():
    a = case(reason_code="VARIANT_UNRESOLVED", mechanical_kind="variant_marker",
             facts={"marker_seen": "icon A"})
    b = case(exam_id="exam002", reason_code="VARIANT_UNRESOLVED", mechanical_kind="variant_marker",
             facts={"marker_seen": "icon B"})
    assert len({g.fingerprint for g in group_cases([a, b])}) == 2


def test_semantically_similar_answers_are_never_grouped():
    a = case(exam_id="exam001", reason_code="GRADE_UNCERTAIN",
             facts={"transcription": "התדרים הגבוהים נשמרים"})
    b = case(exam_id="exam002", reason_code="GRADE_UNCERTAIN",
             facts={"transcription": "התדרים הגבוהים נשמרים"})   # identical text!
    groups = group_cases([a, b])
    assert len(groups) == 2 and not any(g.apply_to_all_eligible for g in groups)
    with pytest.raises(ValueError):
        apply_scope(groups[0])


def test_mechanical_fingerprint_refuses_semantic_kinds():
    assert mechanical_fingerprint("student_answer_similarity", text="x") is None
    assert mechanical_fingerprint("alignment", printed="16", canonical="20") is not None


def test_grading_cases_are_never_reusable_even_with_a_fingerprint():
    c = case(reason_code="GRADE_DISAGREEMENT", mechanical_kind="alignment",
             facts={"printed": "16"})
    assert c.mechanical_fingerprint and not c.reusable


def test_queue_summary_reports_the_work_grouping_saves():
    cases = [case(exam_id=f"exam{i:03d}", reason_code="ALIGNMENT_UNRESOLVED",
                  mechanical_kind="alignment", facts={"printed": "16", "canonical": "?"})
             for i in range(10)]
    cases.append(case(exam_id="exam099", reason_code="GRADE_DISAGREEMENT", points_affected=8.0))
    s = queue_summary(cases)
    assert s["cases"] == 11 and s["decisions_required"] == 2
    assert s["cases_absorbed_by_grouping"] == 9
    assert s["by_reason"]["ALIGNMENT_UNRESOLVED"] == 10


def test_case_dict_exposes_reusability_and_explanation():
    c = case(reason_code="MC_UNRESOLVED", wrong_choice_zero=True, points_affected=2.0,
             facts={"deterministic": ["B", "C"], "state": "multiple_marks"})
    d = c.as_dict()
    assert d["reusable"] is False and "MC_UNRESOLVED" in d["explanation"]
    assert "scores 0 on a wrong selection" in d["explanation"]


# ------------------------------------------- integration with the review UI --


def test_review_ui_items_carry_codes_and_sort_by_priority():
    from autograder.anomaly import BatchWarning
    from autograder.reviewui import build_review_items, review_queue
    from tests.test_reviewui import EXTRACTION, RESULT

    warn = BatchWarning("GRADE_REVIEW_CLUSTER", "warning", "question", "1", 12, 12,
                        "review concentrated on question 1")
    items = build_review_items("exam-001", RESULT, EXTRACTION,
                               chain_traces={("3", "7"): {"local": "B", "cloud": "D"}},
                               warnings=[warn])
    codes = [i.reason_code for i in items]
    assert "MC_CONFLICT" in codes and "VARIANT_UNRESOLVED" in codes
    assert all(i.explanation.startswith(i.reason_code) for i in items)
    # the systemic (batch-warning backed) items come first, ordering only
    assert items[0].batch_warning_code == "GRADE_REVIEW_CLUSTER"
    assert [i.priority_tier for i in items] == sorted(i.priority_tier for i in items)
    q = review_queue(items)
    assert q["summary"]["decisions_required"] <= len(items)


def test_apply_to_all_persists_the_fingerprint_scope(tmp_path):
    from autograder.reviewui import ResolutionStore

    job = tmp_path / "job"
    for e in ("exam-001", "exam-002", "exam-003"):
        (job / "exams" / e).mkdir(parents=True)
    cases = [ReviewCase(exam_id=e, question_id="*", sub_item_id="*",
                        reason_code="VARIANT_UNRESOLVED", mechanical_kind="variant_marker",
                        facts={"marker_seen": "icon A"})
             for e in ("exam-001", "exam-002")]
    group = group_cases(cases)[0]
    scope = apply_scope(group)
    n = ResolutionStore(job / "exams" / "exam-001").apply_to_all(
        job, "variant", "*", "*", decision="variant_2", scope=scope)
    assert n == 2                                       # exam-003 was NOT in the group
    assert not (job / "exams" / "exam-003" / "review_resolutions.json").exists()
    log = (job / "apply_to_all.jsonl").read_text(encoding="utf-8")
    assert group.fingerprint in log and "exam-002" in log
