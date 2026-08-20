"""Guarantees the reliability integration must keep:

§3 decision traces are written AS THE ROUTE EXECUTES (and reconstructions say so)
§5 uncalibrated heuristics route and advise — they never change a score
§6 one blocking package fact stops the batch once; it never becomes N reviews
§7 the shadow comparison contract
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from autograder.anomaly import ExamObservation, ItemObservation, detect_batch_anomalies
from autograder.escalation import GradeResult, OCRVerifyResult
from autograder.gradingpack import build_all_packs
from autograder.imagequality import triage_crop
from autograder.preflight import (PackageSetupRequired, gate_package, package_report_for_key,
                                  preflight_package, reviews_avoided)
from autograder.reliability import (ReliabilityConfig, compare_shadow, run_reliability_judging)
from autograder.schema import ExamExtraction, QuestionExtraction, SubItemExtraction
from autograder.tablecrop import _encode_png_gray
from autograder.trace import DecisionTraceStore
from tests.test_grading_modes import FakeRuntime, _grade_responses
from tests.test_grade import make_key

TEXT_HE = "התדרים הגבוהים נשמרים בתמונה לאחר הסינון"
TEXT_SUSPICIOUS = "DC filter"          # short + latin: trips the uncalibrated signals


def _extraction(text=TEXT_HE, n=2, legibility="full"):
    """Question 1 answered with a written explanation; question 3 (MC only)
    left blank, so the whole key is covered for the deterministic scorer."""
    key = make_key()
    return ExamExtraction(questions=[
        QuestionExtraction(
            question_id="1", source_pages=[1], authoritative_source="sheet",
            sub_items=[SubItemExtraction(sub_item_id=s.id, status="answered", final_answer="F",
                                         explanation_transcription=text,
                                         explanation_legibility=legibility,
                                         interpretation_rationale="", confidence=1.0)
                       for s in key.questions[0].sub_items[:n]]),
        QuestionExtraction(
            question_id="3", source_pages=[2], authoritative_source="sheet",
            sub_items=[SubItemExtraction(sub_item_id=s.id, status="unanswered",
                                         interpretation_rationale="", confidence=1.0)
                       for s in key.questions[1].sub_items])])


def _run(tmp_path, extraction, responses=None, crops=None, store=None, key=None):
    key = key or make_key()
    rt = FakeRuntime(tmp_path, responses or _grade_responses())
    run = run_reliability_judging(
        key=key, extraction=extraction, version="A1",
        config=ReliabilityConfig(mode="reliability"), gateway=rt.gateway,
        packs=build_all_packs(key, {}), exam_id="exam-001", crops=crops or {},
        trace_store=store)
    return run, rt


def _png(value=40, blank=False) -> str:
    a = np.full((40, 200), 255, dtype=np.uint8)
    if not blank:
        a[10:30, 40:60] = value
    return base64.standard_b64encode(_encode_png_gray(a)).decode()


# --------------------------------------------------------------------------
# §3 traces are written by the route, not reconstructed afterwards
# --------------------------------------------------------------------------


def test_traces_are_persisted_as_each_item_is_decided(tmp_path):
    store = DecisionTraceStore(tmp_path / "decisions.jsonl")
    run, rt = _run(tmp_path, _extraction(n=3), store=store)
    rows = store.read()
    assert len(rows) == 3 == len(run.records)
    r = rows[0]
    for field in ("item_id", "question_id", "grading_policy", "rag_policy", "variant",
                  "ocr_status", "grade_status", "evidence", "invariants", "escalation",
                  "final_state", "reason_code", "stages", "model_calls", "signals"):
        assert field in r, field
    assert r["item_id"].startswith("item-") and "exam-001" not in r["item_id"]
    assert r["ocr_status"] == "OCR_OK" and r["grade_status"] == "GRADE_OK"
    assert r["final_state"] == "AUTO" and r["rag_policy"] == "RAG_DISABLED"
    assert [c["task"] for c in r["model_calls"]] == ["grade_primary"]
    assert r["proposed_score"] == 4.0


def test_traces_contain_no_student_text_or_identity(tmp_path):
    store = DecisionTraceStore(tmp_path / "decisions.jsonl")
    _run(tmp_path, _extraction(n=2), store=store)
    blob = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8")
    assert TEXT_HE not in blob and "student" not in blob.lower()


def test_a_recorded_trace_is_preferred_over_a_reconstruction(tmp_path):
    from autograder.reviewui import decision_trace_for

    exam_dir = tmp_path / "exams" / "exam-001"
    exam_dir.mkdir(parents=True)
    store = DecisionTraceStore(exam_dir / "decisions.jsonl")
    _run(tmp_path, _extraction(n=1), store=store)
    text = decision_trace_for(exam_dir, {"questions": []}, "1", "1")
    assert "RECONSTRUCTED" not in text and "ran grade_primary" in text
    assert "status: OCR OCR_OK" in text


# --------------------------------------------------------------------------
# §5 uncalibrated heuristics are advisory, never authoritative
# --------------------------------------------------------------------------


def test_an_uncalibrated_suspicion_signal_flags_but_does_not_cost_points(tmp_path):
    """The deterministic OCR-suspicion signals are not calibrated. On their own
    they may route an item to REVIEW; they may NOT change its score."""
    clean, _ = _run(tmp_path, _extraction(text=TEXT_HE, n=1))
    flagged, _ = _run(tmp_path, _extraction(text=TEXT_SUSPICIOUS, n=1),
                      responses={"grade_primary": [GradeResult(score=4.0)]})   # no verifier route
    assert clean.decisions[0].final_state == "AUTO"
    assert flagged.decisions[0].final_state == "REVIEW"          # routing changed...
    assert flagged.decisions[0].reason_code == "OCR_UNRESOLVED"
    # ...the awarded verdict (and therefore the points) did not
    assert (flagged.evaluations["1"]["1"].verdict
            == clean.evaluations["1"]["1"].verdict == "valid")
    assert flagged.decisions[0].proposed_score == clean.decisions[0].proposed_score


def test_an_uncalibrated_image_threshold_does_not_change_the_verdict(tmp_path):
    """A low-contrast/skew verdict is a threshold, not a fact: it advises."""
    faint = _png(value=150)
    assert triage_crop(base64.b64decode(faint)).status in ("LOW_CONTRAST", "OK")
    good, _ = _run(tmp_path, _extraction(n=1), crops={("1", "1"): _png()})
    dim, _ = _run(tmp_path, _extraction(n=1), crops={("1", "1"): faint})
    assert good.evaluations["1"]["1"].verdict == dim.evaluations["1"]["1"].verdict


def test_an_empty_crop_is_a_fact_not_a_threshold_and_withholds_judgement(tmp_path):
    """The one image finding that IS evidence: there is nothing to read. The
    item is withheld for a human — and no grading call is spent."""
    run, rt = _run(tmp_path, _extraction(n=1), crops={("1", "1"): _png(blank=True)})
    assert run.decisions[0].final_state == "REVIEW"
    assert run.decisions[0].reason_code == "OCR_UNRESOLVED"
    assert rt.calls.get("grade_primary") is None            # no model call on an empty crop


def test_batch_anomaly_detection_cannot_touch_a_grade():
    items = [ItemObservation(exam_id=f"e{i}", question_id=q, score=0.0, max_score=4.0,
                             variant="A1", review=True)
             for i in range(8) for q in ("1", "2", "3")]
    exams = [ExamObservation(exam_id=f"e{i}", variant="A1") for i in range(8)]
    before = json.dumps([i.__dict__ for i in items], sort_keys=True, default=str)
    warnings = detect_batch_anomalies(items, exams)
    assert warnings                                          # it did fire...
    assert json.dumps([i.__dict__ for i in items], sort_keys=True, default=str) == before
    assert all(not hasattr(w, "points") and not hasattr(w, "score") for w in warnings)


def test_the_final_number_is_always_the_deterministic_one(tmp_path):
    """The grader proposes; the deterministic scorer disposes."""
    from autograder.config import GraderConfig
    from autograder.grade import VersionDecision, grade_exam

    ext = _extraction(n=2)
    run, _ = _run(tmp_path, ext, responses={"grade_primary": [GradeResult(score=999.0)],
                                            "grade_escalate": [GradeResult(score=999.0)],
                                            "ocr_verify": [OCRVerifyResult(verdict="supported")]})
    key = make_key()
    result = grade_exam(key, ext, run.evaluations, VersionDecision("A1", "", False),
                        GraderConfig())
    # an out-of-range proposal is invalid -> REVIEW, and never becomes points
    assert all(d.final_state == "REVIEW" for d in run.decisions)
    assert result.total_awarded <= result.total_max


# --------------------------------------------------------------------------
# §6 blocking vs non-blocking package findings
# --------------------------------------------------------------------------


def test_one_blocking_fact_stops_the_batch_once_and_creates_no_student_reviews(tmp_path):
    key = make_key()
    key.questions[0].sub_items[0].correct_by_version = {"A1": ["F"]}   # A2/A3 missing
    report = package_report_for_key(key)
    assert not report.ok and report.status == "PACKAGE_SETUP_REQUIRED"
    with pytest.raises(PackageSetupRequired) as e:
        gate_package(report)
    assert "PACKAGE_SETUP_REQUIRED" in str(e.value)
    # the stop happens ONCE, before any exam is graded: no per-student review
    # item exists, and the report quantifies what that saved
    assert reviews_avoided(report, n_exams=180) >= 180
    assert all(f.subject in ("question", "variant", "package", "template")
               for f in report.blocking)


def test_non_blocking_findings_never_stop_grading(tmp_path):
    key = make_key()
    key.total_points = 9999                     # informational mismatch only
    report = preflight_package(key=key, variants=list(key.versions),
                               alignment={v: {q.id: {s.id: s.id for s in q.sub_items}
                                              for q in key.questions} for v in key.versions},
                               policies={"1": "choice_only"},
                               template={})
    assert report.ok and gate_package(report) is report
    codes = {f.code for f in report.warnings}
    assert "TOTAL_SCORE_MISMATCH" in codes and "POLICY_MISSING" in codes
    assert {f.category for f in report.warnings} >= {"informational", "policy"}


def test_findings_carry_a_persisted_category_and_severity(tmp_path):
    key = make_key()
    key.questions[0].sub_items[0].correct_by_version = {"A1": ["F"]}
    d = package_report_for_key(key).as_dict()
    assert d["blocking"][0]["category"] == "key"
    assert d["blocking"][0]["severity"] == "blocking"
    assert d["by_category"]["key"]["blocking"] >= 1


def test_the_cli_gate_raises_before_a_single_exam_is_graded(tmp_path, monkeypatch):
    import argparse

    from autograder import orchestrator as orch
    from autograder.cli import run_grade_pipeline
    from autograder.key_parser import save_answer_key
    from tests.conftest import make_pdf
    from tests.test_grading_modes import _backend

    key = make_key()
    key.questions[0].sub_items[0].correct_by_version = {"A1": ["F"]}
    key_path = tmp_path / "answer_key.json"
    save_answer_key(key, key_path)
    exam = make_pdf(tmp_path / "01_50.pdf")
    rt = FakeRuntime(tmp_path, _grade_responses())
    monkeypatch.setattr(orch, "setup_from_config", lambda *a, **kw: rt)
    ns = argparse.Namespace(key=str(key_path), rubric=None, resume=False, version="auto",
                            exam=str(exam), grading_mode="reliability",
                            rag_policy="RAG_DISABLED",
                            models_config=str(tmp_path / "models.toml"),
                            grading_policies=None)
    backend = _backend()
    with pytest.raises(PackageSetupRequired):
        run_grade_pipeline(ns, backend, tmp_path / "out", 800, exam_path=exam,
                           exam_label="exam-001")
    assert backend.calls == [] and rt.total_calls == 0     # nothing was graded at all
    assert not (tmp_path / "out" / "result.json").exists()


# --------------------------------------------------------------------------
# §7 shadow comparison contract
# --------------------------------------------------------------------------


def test_shadow_comparison_records_everything_a_migration_decision_needs(tmp_path):
    from autograder.config import GraderConfig
    from autograder.grade import VersionDecision, grade_exam, judge_all
    from autograder.backends import BackendConfig
    from autograder.backends.mock import MockBackend
    from tests.test_offline_pipeline import _fixture_judgement

    key = make_key()
    ext = _extraction(n=8)
    legacy_backend = MockBackend(config=BackendConfig(backend="mock", model="m"),
                                 responder=lambda model, s, b: _fixture_judgement())
    legacy = grade_exam(key, ext, judge_all(legacy_backend, key, ext, "A1"),
                        VersionDecision("A1", "", False), GraderConfig())
    run, _ = _run(tmp_path, ext, responses={"grade_primary": [GradeResult(score=0.0)],
                                            "grade_escalate": [GradeResult(score=0.0)],
                                            "ocr_verify": [OCRVerifyResult(verdict="supported")]})
    shadow = grade_exam(key, ext, run.evaluations, VersionDecision("A1", "", False),
                        GraderConfig())
    cmp = compare_shadow(legacy_result=legacy, shadow_result=shadow, run=run)

    assert cmp["authoritative"] == "legacy"
    item = cmp["items"][0]
    for field in ("legacy_points", "reliability_points", "reliability_proposal", "score_delta",
                  "legacy_review", "reliability_state", "legacy_reason_code",
                  "reliability_reason_code", "route_difference"):
        assert field in item, field
    assert item["legacy_points"] == 4.0 and item["reliability_points"] == 0.0
    assert item["score_delta"] == -4.0
    a = cmp["agreement"]
    # only sub-item 1 has a correct selection under A1, so it is the only item
    # where the disagreeing shadow grader changes the outcome
    assert a["items"] == 8 and a["exact_score_agreement"] == 87.5
    assert a["mean_abs_delta"] == 0.5
    assert "route_differences" in a and "reason_code_differences" in a
    assert cmp["totals"]["reliability_total"] < cmp["totals"]["legacy_total"]
    assert cmp["accounting"]["questions"] == 8 and cmp["states"] == {"AUTO": 8}


def test_shadow_decisions_can_never_be_applied_to_a_result(tmp_path):
    from autograder.reliability import GradingModeError, apply_review_items

    run, _ = _run(tmp_path, _extraction(n=1))
    run.mode = "shadow"
    with pytest.raises(GradingModeError):
        apply_review_items(object(), run)
