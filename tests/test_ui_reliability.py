"""UI-facing reliability surfaces (§20): batch warnings, review reason codes
and priority, decision traces, package setup and the pre-run estimate.

The assembly logic is tested directly; the Streamlit layer is exercised with
AppTest. Offline: no model, no network, no secrets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autograder import reviewui
from autograder.reviewui import (batch_overview, decision_trace_for, load_job_results,
                                 observations_from_results)
from autograder.trace import DecisionTrace, DecisionTraceStore

REPO = Path(__file__).resolve().parents[1]


def result_for(exam_id: str, *, blank_q3=False, review_q3=False, variant="A1", spread=0):
    def sub(qid, sid, pts, mx, status="answered", review=False, reason="", text="הסבר תקין"):
        return {"question_id": qid, "sub_item_id": sid, "status": status,
                "student_answer": "F", "accepted_answers": ["F"], "selection_correct": True,
                "explanation_transcription": text, "points_selection": pts,
                "points_explanation": 0.0, "points_total": pts, "points_max": mx,
                "reason": reason, "needs_review": review}
    pts = [4.0, 2.0, 0.0, 3.0][spread % 4]
    q3_sub = sub("3", "1", 0.0 if blank_q3 else pts, 4.0,
                 status="unanswered" if blank_q3 else "answered",
                 review=review_q3, reason="could not be read reliably" if review_q3 else "")
    return {
        "exam_file": f"{exam_id}.pdf", "graded_at": "", "model": "mock",
        "detected_version": variant, "version_detection": "marker",
        "total_awarded": 8.0, "total_max": 12.0,
        "questions": [
            {"question_id": "1", "question_type": "multiple_choice", "points_awarded": pts,
             "points_max": 4.0, "sub_results": [sub("1", "1", pts, 4.0)], "summary": ""},
            {"question_id": "2", "question_type": "multiple_choice",
             "points_awarded": 4.0 - pts, "points_max": 4.0,
             "sub_results": [sub("2", "1", 4.0 - pts, 4.0)], "summary": ""},
            {"question_id": "3", "question_type": "open", "points_awarded": q3_sub["points_total"],
             "points_max": 4.0, "sub_results": [q3_sub], "summary": ""},
        ],
        "needs_human_review": ([{"question_id": "3", "sub_item_id": "1",
                                 "reason": "the explanation could not be read reliably"}]
                               if review_q3 else []),
        "unanswered": [], "mark_interpretations": [],
    }


def broken_batch(n=10):
    """A batch where ONE question is broken for everyone."""
    return {f"exam-{i:03d}": result_for(f"exam-{i:03d}", blank_q3=True, review_q3=True, spread=i)
            for i in range(n)}


def healthy_batch(n=10):
    return {f"exam-{i:03d}": result_for(f"exam-{i:03d}", variant=f"A{1 + i % 3}", spread=i)
            for i in range(n)}


# ------------------------------------------------------- assembly backend ----


def test_observations_are_derived_from_persisted_artefacts():
    items, exams = observations_from_results(healthy_batch(3))
    assert len(items) == 9 and len(exams) == 3
    assert {i.question_id for i in items} == {"1", "2", "3"}
    assert all(i.variant for i in items) and not any(e.variant_unknown for e in exams)


def test_a_healthy_batch_shows_no_warnings_and_no_review_work():
    ov = batch_overview(healthy_batch())
    assert ov["warnings"] == [] and ov["summary"]["cases"] == 0
    assert ov["exams"] == 10


def test_a_systemic_failure_becomes_one_warning_plus_a_grouped_queue():
    ov = batch_overview(broken_batch())
    codes = {w["code"] for w in ov["warnings"]}
    assert "QUESTION_BLANK_RATE_SPIKE" in codes
    assert all(w["affected_students"] == 10 for w in ov["warnings"] if w["scope"] == "question")
    # every individual review points at the systemic cause instead of standing alone
    assert ov["summary"]["cases"] == 10
    assert all(i.batch_warning_code for i in ov["review_items"])
    assert all(i.reason_code == "OCR_UNRESOLVED" for i in ov["review_items"])
    assert all(i.priority_tier == 0 for i in ov["review_items"])   # systemic sorts first


def test_review_items_expose_a_code_points_and_a_structured_explanation():
    ov = batch_overview(broken_batch(6))
    it = ov["review_items"][0]
    assert it.reason_code and it.explanation.startswith(it.reason_code)
    assert it.points_affected >= 0 and "Batch warning" in it.explanation


# ------------------------------------------------------- decision traces ----


def test_decision_trace_prefers_a_recorded_route(tmp_path):
    exam_dir = tmp_path / "exams" / "exam-001"
    exam_dir.mkdir(parents=True)
    t = DecisionTrace("exam-001", "1", "1", grading_policy="choice_only")
    t.deterministic("single clean mark in column F")
    t.skipped("ocr_explanation", "choice_only", avoided={"ocr": 1, "cloud": 1})
    DecisionTraceStore(exam_dir / "decisions.jsonl").append(t.finish("AUTO", "AUTO", "policy"))
    text = decision_trace_for(exam_dir, result_for("exam-001"), "1", "1")
    assert "AUTO" in text and "skipped ocr_explanation: choice_only" in text
    assert "single clean mark" in text


def test_decision_trace_falls_back_honestly_without_a_recorded_route(tmp_path):
    exam_dir = tmp_path / "exams" / "exam-002"
    exam_dir.mkdir(parents=True)
    text = decision_trace_for(exam_dir, result_for("exam-002"), "1", "1")
    assert "reconstructed from the persisted result" in text and "AUTO" in text
    assert "score: 4.0/4.0" in text
    missing = decision_trace_for(exam_dir, result_for("exam-002"), "9", "9")
    assert "no decision record" in missing


def test_job_results_are_loaded_from_disk(tmp_path):
    for eid, res in healthy_batch(3).items():
        d = tmp_path / "exams" / eid
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
    (tmp_path / "exams" / "exam-099").mkdir(parents=True)          # started, no result yet
    loaded = load_job_results(tmp_path)
    assert sorted(loaded) == ["exam-000", "exam-001", "exam-002"]


# ------------------------------------------------------------- streamlit ----


def _job_with_results(tmp_path, results) -> Path:
    from autograder import jobs

    job_dir = tmp_path / "jobs" / "job-ui"
    (job_dir / "uploads").mkdir(parents=True)
    entries = [{"anon_id": eid, "original_name": f"{eid}.pdf", "file": f"exams/{eid}.pdf"}
               for eid in results]
    (job_dir / "job.json").write_text(json.dumps({
        "job_id": "job-ui", "created_at": "", "key": "uploads/answer_key.json", "rubric": None,
        "mask": True, "backend_args": {"--model": "mock"}, "grading_args": {},
        "intake_issues": [], "exams": entries}), encoding="utf-8")
    (job_dir / "state.json").write_text(json.dumps({
        "job_id": "job-ui", "status": "done", "created_at": "", "updated_at": "",
        "started_at": None, "finished_at": None, "current": None,
        "exams": {eid: {"status": "done", "original_name": f"{eid}.pdf",
                        "file": f"exams/{eid}.pdf", "predicted": 8.0,
                        "review_items": len(r.get("needs_human_review", [])),
                        "unanswered": 0, "variant": "A1", "runtime_s": 1.0, "error": None}
                  for eid, r in results.items()}}), encoding="utf-8")
    for eid, res in results.items():
        d = job_dir / "exams" / eid
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
    return job_dir


def test_ui_renders_the_new_panels_for_a_broken_batch(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GRADER_JOBS_DIR", str(tmp_path / "jobs"))
    _job_with_results(tmp_path, broken_batch(8))

    at = AppTest.from_file(str(REPO / "autograder" / "webui.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"UI raised: {at.exception}"
    headers = {str(s.value) for s in at.subheader}
    assert {"Package setup", "Estimated cloud usage", "Batch checks"} <= headers
    blob = json.dumps([str(e.value) for e in at.error] + [str(e.value) for e in at.caption]
                      + [str(e.value) for e in at.success], ensure_ascii=False)
    assert "batch-level warning" in blob
    assert "ESTIMATE" in blob or "answer key has been parsed" in blob


def test_ui_reports_a_clean_batch_as_clean(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GRADER_JOBS_DIR", str(tmp_path / "jobs"))
    _job_with_results(tmp_path, healthy_batch(8))

    at = AppTest.from_file(str(REPO / "autograder" / "webui.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"UI raised: {at.exception}"
    assert any("No batch-level anomaly" in str(s.value) for s in at.success)


def test_settings_view_still_exposes_no_secret(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRET")
    from autograder.backends.mock import MockBackend
    from autograder.gateway import ModelGateway

    gw = ModelGateway.from_dict({"models": {"grade_primary": {"backend": "mock", "model": "m"}},
                                 "pricing": {"m": {"input": 1.0, "output": 2.0}}},
                                backend_factory=lambda c: MockBackend(config=c))
    s = reviewui.settings_summary(gateway=gw, openrouter_key_present=True)
    assert "SECRET" not in json.dumps(s)
