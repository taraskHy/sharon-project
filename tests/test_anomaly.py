"""Batch-level anomaly detection (§5). Synthetic batches only."""

from __future__ import annotations

import pytest

from autograder.anomaly import (AnomalyConfig, BatchWarning, ExamObservation, ItemObservation,
                                detect_batch_anomalies, explained_by)

QUESTIONS = ["1", "2", "3", "4"]


def batch(n_exams=12, **per_item):
    """A healthy batch: everything answered, nothing flagged."""
    items = []
    for e in range(n_exams):
        for q in QUESTIONS:
            items.append(ItemObservation(
                exam_id=f"exam{e:03d}", question_id=q, sub_item_id="1",
                variant=f"variant_{1 + e % 3}", template="tpl_a", page=10 + int(q),
                score=3.0, max_score=4.0, ocr_chars=90, **per_item))
    exams = [ExamObservation(exam_id=f"exam{e:03d}", variant=f"variant_{1 + e % 3}",
                             template="tpl_a", page_count=13) for e in range(n_exams)]
    return items, exams


def codes(ws):
    return sorted({w.code for w in ws})


def test_normal_batch_produces_no_warning():
    items, exams = batch()
    assert detect_batch_anomalies(items, exams) == []


def test_small_batches_never_fire():
    items, exams = batch(n_exams=3)
    for i in items:
        i.blank = i.question_id == "2"
    assert detect_batch_anomalies(items, exams) == []


# ------------------------------------------------------------ extraction -----


def test_one_question_blank_for_everyone_is_one_warning_not_fifty():
    items, exams = batch()
    for i in items:
        if i.question_id == "3":
            i.blank = True
            i.review = True
    ws = detect_batch_anomalies(items, exams)
    blanks = [w for w in ws if w.code == "QUESTION_BLANK_RATE_SPIKE"]
    assert len(blanks) == 1                       # ONE warning...
    assert blanks[0].scope == "question" and blanks[0].scope_id == "3"
    assert blanks[0].affected_students == 12      # ...covering all twelve students
    assert blanks[0].severity == "critical"


def test_crop_failures_cluster_on_a_page():
    items, exams = batch()
    for i in items:
        if i.page == 12:
            i.crop_failed = True
    ws = [w for w in detect_batch_anomalies(items, exams) if w.code == "CROP_FAILURE_CLUSTER"]
    assert len(ws) == 1 and ws[0].scope == "page" and ws[0].scope_id == "12"


def test_mc_ambiguity_spike_on_one_question():
    items, exams = batch()
    for i in items:
        if i.question_id == "1":
            i.ambiguous_mc = True
    assert "MC_AMBIGUITY_SPIKE" in codes(detect_batch_anomalies(items, exams))


# ------------------------------------------------------------------ OCR ------


def test_ocr_failure_cluster_on_one_question():
    items, exams = batch()
    for i in items:
        if i.question_id == "4":
            i.ocr_failed = True
    ws = [w for w in detect_batch_anomalies(items, exams) if w.code == "OCR_FAILURE_CLUSTER"]
    assert ws and ws[0].scope_id == "4"


def test_suspicious_output_length_distribution():
    items, exams = batch()
    for i in items:
        if i.question_id == "2":
            i.ocr_chars = 4
    ws = [w for w in detect_batch_anomalies(items, exams) if w.code == "OCR_LENGTH_ANOMALY"]
    assert len(ws) == 1 and ws[0].scope_id == "2"


# -------------------------------------------------------------- grading ------


def test_near_zero_scores_for_everyone_on_one_question():
    items, exams = batch()
    for i in items:
        if i.question_id == "2":
            i.score = 0.0
    ws = [w for w in detect_batch_anomalies(items, exams) if w.code == "QUESTION_SCORE_DEGENERATE"]
    assert len(ws) == 1 and ws[0].signals["kind"] == "zero" and ws[0].severity == "critical"


def test_near_full_scores_for_everyone_on_one_question():
    items, exams = batch()
    for i in items:
        if i.question_id == "2":
            i.score = i.max_score
    ws = [w for w in detect_batch_anomalies(items, exams) if w.code == "QUESTION_SCORE_DEGENERATE"]
    assert len(ws) == 1 and ws[0].signals["kind"] == "full"


def test_review_concentrated_on_a_single_question():
    items, exams = batch()
    for i in items:
        if i.question_id == "1":
            i.review = True
    ws = [w for w in detect_batch_anomalies(items, exams) if w.code == "GRADE_REVIEW_CLUSTER"]
    assert len(ws) == 1 and ws[0].scope_id == "1"


def test_grader_invalidity_concentrated_on_one_variant():
    items, exams = batch()
    for i in items:
        if i.variant == "variant_2":
            i.grade_invalid = True
    ws = [w for w in detect_batch_anomalies(items, exams) if w.code == "GRADE_INVALID_CLUSTER"]
    assert ws and any(w.scope == "variant" and w.scope_id == "variant_2" for w in ws)


# -------------------------------------------------------------- variants -----


def test_variant_distribution_anomaly_against_setup_expectations():
    items, exams = batch()
    for e in exams:
        e.variant = "variant_1"
    for i in items:
        i.variant = "variant_1"
    ws = detect_batch_anomalies(items, exams, expected_variant_distribution={
        "variant_1": 1 / 3, "variant_2": 1 / 3, "variant_3": 1 / 3})
    dist = [w for w in ws if w.code == "VARIANT_DISTRIBUTION_ANOMALY"]
    assert {w.scope_id for w in dist} == {"variant_1", "variant_2", "variant_3"}
    assert [w for w in dist if w.scope_id == "variant_2"][0].affected_students == 0


def test_one_variant_dominating_without_an_expectation():
    items, exams = batch()
    for e in exams[:-1]:
        e.variant = "variant_1"
    ws = [w for w in detect_batch_anomalies(items, exams) if w.code == "VARIANT_DISTRIBUTION_ANOMALY"]
    assert len(ws) == 1 and ws[0].scope_id == "variant_1"


def test_high_unknown_variant_rate_is_one_package_level_warning():
    items, exams = batch()
    for e in exams[:6]:
        e.variant_unknown = True
    ws = [w for w in detect_batch_anomalies(items, exams) if w.code == "VARIANT_UNRESOLVED_RATE"]
    assert len(ws) == 1 and ws[0].scope == "batch" and ws[0].affected_students == 6


# ---------------------------------------------------------- package level ----


def test_alignment_and_template_clusters():
    items, exams = batch()
    for e in exams[:-1]:            # one exam still matches a template...
        e.alignment_failed = True
        e.template = None
        e.page_count_mismatch = True
    got = codes(detect_batch_anomalies(items, exams))
    assert {"ALIGNMENT_FAILURE_CLUSTER", "TEMPLATE_MISMATCH_CLUSTER",
            "PAGE_COUNT_MISMATCH_CLUSTER"} <= set(got)


def test_a_batch_that_tracks_no_templates_at_all_is_not_a_mismatch():
    """Silence is not evidence: a pipeline that records no template for any
    exam must not be reported as a template mismatch for every exam."""
    items, exams = batch()
    for e in exams:
        e.template = None
    for i in items:
        i.template = None
    assert "TEMPLATE_MISMATCH_CLUSTER" not in codes(detect_batch_anomalies(items, exams))


def test_alignment_failures_on_one_question_are_grouped():
    items, exams = batch()
    for i in items:
        if i.question_id == "3":
            i.alignment_failed = True
            i.review = True
    ws = [w for w in detect_batch_anomalies(items, exams)
          if w.code == "ALIGNMENT_FAILURE_CLUSTER" and w.scope == "question"]
    assert len(ws) == 1 and ws[0].scope_id == "3"


# ------------------------------------------------------------- grouping ------


def test_warnings_are_ordered_and_explain_individual_reviews():
    items, exams = batch()
    for i in items:
        if i.question_id == "3":
            i.blank, i.review = True, True
    ws = detect_batch_anomalies(items, exams)
    assert ws[0].severity == "critical"
    reviewed = [i for i in items if i.review]
    assert all(explained_by(ws, i) is not None for i in reviewed)
    assert explained_by(ws, [i for i in items if i.question_id == "1"][0]) is None


def test_detection_never_changes_a_grade():
    items, exams = batch()
    for i in items:
        if i.question_id == "2":
            i.score = 0.0
    before = [(i.exam_id, i.question_id, i.score, i.review) for i in items]
    detect_batch_anomalies(items, exams)
    assert [(i.exam_id, i.question_id, i.score, i.review) for i in items] == before
