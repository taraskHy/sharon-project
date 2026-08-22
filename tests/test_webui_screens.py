"""The six GUI screens under Streamlit AppTest — no model, no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEBUI = str(REPO / "autograder" / "webui.py")
SCREENS = ["🏠 Dashboard", "🧭 Exam setup", "⏳ Grading progress", "🔍 Review queue", "📄 Results / export",
           "🛠 Advanced / diagnostics"]


def _run(screen: str, timeout: int = 60):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(WEBUI, default_timeout=timeout)
    at.session_state["screen"] = screen
    at.run()
    assert not at.exception, f"{screen} raised: {[str(e.value) for e in at.exception]}"
    return at


def _blob(at) -> str:
    parts = []
    for coll in (at.markdown, at.caption, at.text, at.error, at.warning, at.success, at.info, at.subheader,
                 at.header, at.code):
        parts += [str(x.value) for x in coll]
    for t in at.table:
        parts.append(t.value.to_json(force_ascii=False))
    for d in at.dataframe:
        parts.append(d.value.to_json(force_ascii=False))
    for m in at.metric:
        parts.append(f"{m.label}={m.value}")
    return "\n".join(parts)


def _job_with_review(tmp_path: Path) -> Path:
    job_dir = tmp_path / "jobs" / "job-ui"
    (job_dir / "uploads").mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({
        "job_id": "job-ui", "created_at": "", "key": "uploads/answer_key.json", "rubric": None, "mask": True,
        "course_id": "cv101", "grading_mode": "reliability", "backend_args": {"--model": "mock"},
        "grading_args": {}, "intake_issues": [],
        "exams": [{"anon_id": "exam-000", "original_name": "a.pdf", "file": "exams/exam-000.pdf"},
                  {"anon_id": "exam-001", "original_name": "b.pdf", "file": "exams/exam-001.pdf"}]}),
        encoding="utf-8")
    (job_dir / "state.json").write_text(json.dumps({
        "job_id": "job-ui", "status": "finished", "created_at": "", "updated_at": "", "started_at": None,
        "finished_at": None, "current": None,
        "exams": {"exam-000": {"status": "done", "original_name": "a.pdf", "file": "exams/exam-000.pdf",
                               "predicted": 6.0, "review_items": 1, "unanswered": 0, "variant": "A1",
                               "runtime_s": 1.0, "error": None},
                  "exam-001": {"status": "done", "original_name": "b.pdf", "file": "exams/exam-001.pdf",
                               "predicted": 8.0, "review_items": 0, "unanswered": 0, "variant": "A1",
                               "runtime_s": 1.0, "error": None}}}), encoding="utf-8")
    base = {"exam_file": "x.pdf", "graded_at": "", "model": "mock:m", "detected_version": "A1",
            "version_detection": "ok", "total_max": 10.0,
            "backend_info": {"evidence_crops": {"status": "UNAVAILABLE", "reason": "no calibrated geometry"}}}
    r0 = {**base, "total_awarded": 6.0, "questions": [{
        "question_id": "1", "question_type": "multiple_choice", "points_awarded": 6.0, "points_max": 10.0,
        "summary": "q1", "sub_results": [{
            "question_id": "1", "sub_item_id": "1", "question_type": "multiple_choice", "status": "answered",
            "student_answer": "a", "accepted_answers": ["a"], "selection_correct": True,
            "explanation_transcription": "the frozen transcription", "explanation_evaluation": None,
            "points_selection": 2.0, "points_explanation": 4.0, "points_total": 6.0, "points_max": 10.0,
            "reason": "the student wrote an explanation but it could not be read reliably (legibility: partial)",
            "needs_review": True}]}],
        "needs_human_review": [{"question_id": "1", "sub_item_id": "1",
                                "reason": "[OCR_UNRESOLVED] the student wrote an explanation but it could not be "
                                          "read reliably (legibility: partial) (suspicious; no evidence crop "
                                          "available (crop producer unavailable))"}]}
    r1 = {**base, "total_awarded": 8.0, "questions": [], "needs_human_review": []}
    for eid, res in (("exam-000", r0), ("exam-001", r1)):
        d = job_dir / "exams" / eid
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
    # a batch ledger with one cloud call (spend truth for the dashboard)
    led = job_dir / "exams" / "gateway_ledger"
    led.mkdir(parents=True)
    (led / "usage.jsonl").write_text(json.dumps({
        "task": "grade_primary", "backend": "openrouter", "model": "vendor/x", "cloud": True, "cache_hit": False,
        "input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "reported_cost": 0.0123}) + "\n",
        encoding="utf-8")
    return job_dir


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADER_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return tmp_path


def test_every_screen_renders_without_a_batch(env):
    for s in SCREENS:
        at = _run(s)
        assert any("Exam Autograder" in str(t.value) for t in at.sidebar.title)


def test_dashboard_shows_course_exam_readiness_counts_and_spend(env):
    _job_with_review(env)
    at = _run(SCREENS[0])
    blob = _blob(at)
    assert "Course=cv101" in blob
    assert "Students=2" in blob and "Auto graded=1" in blob and "Needs review=1" in blob and "Failures=0" in blob
    assert "Package readiness=KEY_NOT_PARSED" in blob
    assert "OpenRouter=not configured" in blob
    assert "This batch (ledger)=$0.0123" in blob          # spend truth from the persistent ledger
    labels = [str(b.label) for b in at.button]
    assert any("Set up exam" in l for l in labels) and any("Start grading" in l for l in labels)
    assert any("Review" in l for l in labels) and any("Results" in l for l in labels)


def test_review_queue_states_typed_reason_and_missing_crop_honestly(env):
    _job_with_review(env)
    at = _run(SCREENS[3])
    blob = _blob(at)
    assert "OCR_UNRESOLVED — Handwriting could not be read" in blob      # typed reason + human title
    assert "AI uncertain" not in blob
    assert "the frozen transcription" in blob                          # immutable transcription shown
    assert "no image evidence available" in blob                        # never fabricated
    assert "Image evidence: not available" in blob
    opts = [str(b.label) for b in at.button]
    assert {"accept primary", "accept secondary", "mark unreadable"} <= set(opts)


def test_results_screen_shows_deterministic_grade_and_history(env):
    _job_with_review(env)
    at = _run(SCREENS[4])
    blob = _blob(at)
    assert "Final grade (deterministic)=6/10" in blob
    assert "Review history" in blob and "no human decisions recorded" in blob


def test_advanced_marks_unselected_roles_and_shows_budget_and_key_status(env, monkeypatch):
    _job_with_review(env)
    at = _run(SCREENS[5])
    blob = _blob(at)
    assert "UNSELECTED" in blob                     # role -> model table marks unselected cloud roles
    assert "ocr_primary" in blob and "grade_primary" in blob
    assert "Campaign warning=$8.00" in blob and "Campaign hard stop=$10.00" in blob
    assert "Credential in environment=missing" in blob and "Configured=NO" in blob
    assert "Key-usage endpoint=ready (not called)" in blob
    assert "UNAVAILABLE" in blob                    # verifier crop producer status
    assert "sk-" not in blob


def test_progress_screen_keeps_batch_panels_and_estimate_wording(env):
    _job_with_review(env)
    at = _run(SCREENS[2])
    headers = {str(s.value) for s in at.subheader}
    assert {"Package setup", "Estimated cloud usage", "Batch checks"} <= headers
    blob = _blob(at)
    assert "OCR calls=0" in blob and "Grader calls=1" in blob and "Cloud cost (ledger)=$0.0123" in blob
    assert "answer key has been parsed" in blob
