"""Mission-1 tests: exam templates/modes, prob-exam first-page authority,
variant-aware key selection, job runner (ZIP intake, interruption, resume),
grade-label isolation, combined reports, and UI smoke.

All offline: the model is a recording/fixture mock; grading subprocesses run
with ``--backend mock``.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
import zipfile
from pathlib import Path

import pytest

from autograder import jobs
from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.cli import run_grade_pipeline
from autograder.key_parser import load_answer_key
from autograder.schema import ExplanationJudgement, VariantDetection
from autograder.template import (
    ExamTemplate,
    apply_template_to_key,
    load_template,
    synthesized_survey,
)
from tests.conftest import make_pdf

REPO = Path(__file__).resolve().parents[1]
PROB_KEY = REPO / "prob_data" / "sol.answer_key.json"
FIXTURES = Path(__file__).parent / "fixtures_prob"

CLUB_ANSWERS = ["C", "A", "B", "D", "A", "A", "D", "D", "B", "A"]


def _fixture(name: str):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _prob_backend(marker: str = "club") -> MockBackend:
    """Recording mock that serves the prob flow: variant detection with the
    requested marker, table extraction from the fixture."""
    from autograder.schema import QuestionExtraction

    detection = VariantDetection.model_validate(_fixture("VariantDetection"))
    detection.matched_marker = marker
    detection.marker_seen = f"a {marker} symbol next to בהצלחה!"
    extraction = QuestionExtraction.model_validate(_fixture("QuestionExtraction"))

    def responder(model, system, blocks):
        if model is VariantDetection:
            return detection.model_copy(deep=True)
        if model is QuestionExtraction:
            return extraction.model_copy(deep=True)
        raise AssertionError(f"unexpected model call: {model.__name__}")

    return MockBackend(
        config=BackendConfig(backend="mock", model=f"prob-{marker}"),
        responder=responder,
    )


def _grade_prob(tmp_path: Path, backend: MockBackend, exam_name: str = "student.pdf"):
    exam = make_pdf(tmp_path / exam_name, pages=3)
    ns = argparse.Namespace(
        key=str(PROB_KEY), rubric=None, resume=False, version="auto",
        exam=str(exam), variant_map=None, alignment_map=None, template=None,
        no_key_cache=True, key_cache_dir=str(tmp_path / "kc"), mask=False,
    )
    return run_grade_pipeline(
        ns, backend, tmp_path / "out", 800,
        exam_path=exam, exam_label="exam-001", survey_image_edge=400,
    )


# --------------------------------------------------------------------------
# template mechanics
# --------------------------------------------------------------------------


def test_prob_template_loads_and_is_mc_fixed_page():
    tpl = load_template(PROB_KEY)
    assert tpl is not None
    assert tpl.mode == "multiple_choice"
    assert tpl.answer_sheet_rule == "fixed_pages"
    assert tpl.answer_sheet_pages == [1]
    assert tpl.booklet_answers_not_graded


def test_synthesized_survey_marks_only_fixed_pages_authoritative():
    key = load_answer_key(PROB_KEY)
    tpl = load_template(PROB_KEY)
    survey = synthesized_survey(tpl, key, n_pages=3)
    assert survey.answer_sheet_policy.authoritative_pages == [1]
    assert survey.answer_sheet_policy.booklet_answers_not_graded
    kinds = {p.page_number: p.page_kind for p in survey.pages}
    assert kinds == {1: "answer_sheet", 2: "question_or_instructions",
                     3: "question_or_instructions"}


def test_apply_template_modes_mixed_routing():
    key = load_answer_key(PROB_KEY)
    tpl = ExamTemplate(
        template_id="t", mode="mixed",
        question_modes={"1": "with_explanation"},
    )
    apply_template_to_key(key, tpl)
    assert key.questions[0].type == "selection_with_explanation"

    key2 = load_answer_key(PROB_KEY)
    tpl2 = ExamTemplate(template_id="t2", mode="multiple_choice")
    apply_template_to_key(key2, tpl2)
    assert key2.questions[0].type == "multiple_choice"
    assert not key2.questions[0].explanation_required


# --------------------------------------------------------------------------
# end-to-end prob flow (mock model)
# --------------------------------------------------------------------------


def test_mc_only_run_makes_no_explanation_calls_and_reads_page1_only(tmp_path):
    backend = _prob_backend("club")
    result = _grade_prob(tmp_path, backend)

    called_models = [c.output_model for c in backend.calls]
    assert "ExplanationJudgement" not in called_models, "MC-only must skip judging"
    assert "ExamSurvey" not in called_models, "fixed_pages template must skip the survey call"
    assert "VariantDetection" in called_models

    # Extraction calls must carry exactly ONE image: page 1 (the fixed answer
    # sheet). Question pages 2-3 (scratch work) must never be sent.
    extraction_calls = [c for c in backend.calls if c.output_model == "QuestionExtraction"]
    assert extraction_calls, "extraction must run"
    for call in extraction_calls:
        images = [b for b in call.content_blocks if b.get("type") == "image"]
        assert len(images) == 1, "only the fixed answer-sheet page may be sent"
        assert "--- Page 1 ---" in call.all_text()
        assert "--- Page 2 ---" not in call.all_text()
        assert "--- Page 3 ---" not in call.all_text()

    # Perfect club answers against the club column → full marks, no review.
    assert result.detected_version == "club"
    assert result.total_awarded == 100.0
    assert not result.needs_human_review


def test_variant_aware_key_selection_not_score_maximisation(tmp_path):
    """The SAME extracted answers score differently under different detected
    markers — proving the key column follows the marker, never the score."""
    r_club = _grade_prob(tmp_path / "club", _prob_backend("club"))
    r_heart = _grade_prob(tmp_path / "heart", _prob_backend("heart"))
    assert r_club.detected_version == "club"
    assert r_heart.detected_version == "heart"
    assert r_club.total_awarded == 100.0
    # club answers vs heart column: matches only on Q2,4,6,7,8 → 50
    assert r_heart.total_awarded == 50.0


def test_unknown_marker_flags_review_and_uses_deterministic_fallback(tmp_path):
    backend = _prob_backend("club")
    # Sabotage detection: unknown marker, not confident.
    original_responder = backend.responder

    def responder(model, system, blocks):
        out = original_responder(model, system, blocks)
        if model is VariantDetection:
            out.matched_marker = None
            out.confident = False
            out.marker_seen = "smudged symbol"
        return out

    backend.responder = responder
    result = _grade_prob(tmp_path, backend)
    # Deterministic fallback = first variant in the mapping (club, alphabetical
    # by marker name is heart/spade/diamond/club → mapping order is dict order:
    # heart, spade, diamond, club → sorted(mapping.values())[0] = "club").
    assert result.detected_version in {"club", "diamond", "heart", "spade"}
    assert result.variant_detection["confident"] is False
    assert any(
        "variant" in item.reason.lower() or "marker" in item.reason.lower()
        for item in result.needs_human_review
    ), "uncertain variant must be routed to human review"


def test_grade_labels_and_original_names_never_reach_model(tmp_path):
    """§F: capture the complete model request stream and assert no expected
    grade, no grades.csv content, and no original filename appear."""
    backend = _prob_backend("club")
    _grade_prob(tmp_path, backend, exam_name="02_60.pdf")  # worst case name
    label_rows = [
        line.strip()
        for line in (REPO / "prob_data" / "grades.csv").read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip().split(",")[-1].strip()  # rows carrying a grade
    ]
    assert label_rows, "sanity: grades.csv has grade-bearing rows"
    for call in backend.calls:
        text = call.all_text()
        assert "02_60" not in text, "original filename leaked"
        assert "grades.csv" not in text
        for row in label_rows:
            assert row not in text, f"grades.csv row leaked: {row}"


# --------------------------------------------------------------------------
# jobs: intake, ZIP handling, duplicates
# --------------------------------------------------------------------------


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_intake_zip_anonymizes_and_reports_issues(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", pages=1).read_bytes()
    zpath = tmp_path / "batch.zip"
    zpath.write_bytes(_zip_bytes({
        "05.pdf": pdf,
        "sub/dir/13.pdf": pdf,               # nested: flattened by basename
        "../evil.pdf": pdf,                  # traversal attempt: basename only
        "notes.txt": b"not an exam",         # unsupported
        "empty.pdf": b"",                    # empty
    }))
    job_dir = tmp_path / "job"
    (job_dir / "uploads").mkdir(parents=True)
    result = jobs.intake_exams(job_dir, [zpath])

    anon_ids = [e["anon_id"] for e in result.entries]
    assert len(anon_ids) == len(set(anon_ids))
    assert all(a.startswith("exam-") for a in anon_ids)
    originals = {e["original_name"] for e in result.entries}
    assert originals == {"05.pdf", "13.pdf", "evil.pdf"}
    # every copied file lives INSIDE the job dir
    for e in result.entries:
        p = (job_dir / e["file"]).resolve()
        assert job_dir.resolve() in p.parents
    joined = "\n".join(result.issues)
    assert "notes.txt" in joined and "empty.pdf" in joined


def test_intake_malformed_zip_and_duplicates(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip archive")
    pdf_path = make_pdf(tmp_path / "02.pdf", pages=1)
    dup_zip = tmp_path / "dup.zip"
    dup_zip.write_bytes(_zip_bytes({"02.pdf": pdf_path.read_bytes()}))

    job_dir = tmp_path / "job"
    (job_dir / "uploads").mkdir(parents=True)
    result = jobs.intake_exams(job_dir, [bad, pdf_path, dup_zip])
    assert any("malformed ZIP" in i for i in result.issues)
    assert any("duplicate" in i for i in result.issues)
    assert len(result.entries) == 1  # the duplicate was not double-ingested


# --------------------------------------------------------------------------
# jobs: run, interrupt, resume, combined reports
# --------------------------------------------------------------------------


def _make_job(tmp_path: Path, n_exams: int = 2) -> Path:
    exams = [make_pdf(tmp_path / f"src{i}.pdf", pages=3) for i in range(1, n_exams + 1)]
    return jobs.create_job(
        key=PROB_KEY,
        exams=exams,
        backend_args={"--backend": "mock", "--model": str(FIXTURES)},
        grading_args={"--max-image-edge": 600, "--survey-image-edge": 400,
                      "--no-key-cache": True},
        mask=False,
        job_root=tmp_path / "jobs",
    )


def test_run_job_offline_completes_and_writes_combined_reports(tmp_path):
    job_dir = _make_job(tmp_path, n_exams=2)
    rc = jobs.run_job(job_dir, poll_interval=0.1)
    assert rc == 0
    state = jobs.load_state(job_dir)
    assert state["status"] == "finished"
    assert all(e["status"] == "done" for e in state["exams"].values())
    assert all(e["predicted"] == 100.0 for e in state["exams"].values())
    assert all(e["variant"] == "club" for e in state["exams"].values())

    combined = json.loads((job_dir / "combined_results.json").read_text(encoding="utf-8"))
    assert combined["counts"]["done"] == 2
    csv_text = (job_dir / "combined_results.csv").read_text(encoding="utf-8")
    assert "exam-001" in csv_text and "exam-002" in csv_text
    assert (job_dir / "summary.md").exists()
    with zipfile.ZipFile(job_dir / "reports.zip") as zf:
        names = zf.namelist()
    assert "summary.md" in names
    assert "exam-001/result.json" in names and "exam-001/report.md" in names


def test_job_interrupt_terminates_child_and_resume_completes(tmp_path, monkeypatch):
    job_dir = _make_job(tmp_path, n_exams=1)

    import sys as _sys

    real_cmd = jobs._grade_command

    def slow_cmd(job, jd, entry):  # a child that would run 60s if not stopped
        return [_sys.executable, "-c", "import time; time.sleep(60)"]

    monkeypatch.setattr(jobs, "_grade_command", slow_cmd)
    t0 = time.monotonic()
    runner = threading.Thread(target=jobs.run_job, args=(job_dir,), kwargs={"poll_interval": 0.1})
    runner.start()
    # wait until the exam is marked running, then request stop
    for _ in range(100):
        if jobs.load_state(job_dir)["exams"]["exam-001"]["status"] == "running":
            break
        time.sleep(0.1)
    jobs.request_stop(job_dir)
    runner.join(timeout=30)
    assert not runner.is_alive(), "runner must exit after a stop request"
    assert time.monotonic() - t0 < 45, "stop must terminate the child, not wait it out"
    state = jobs.load_state(job_dir)
    assert state["status"] in ("stopped", "paused")
    assert state["exams"]["exam-001"]["status"] == "pending", (
        "interrupted exam returns to pending for safe resume"
    )

    # resume with the REAL command → completes offline via the fixture mock
    monkeypatch.setattr(jobs, "_grade_command", real_cmd)
    rc = jobs.run_job(job_dir, poll_interval=0.1)
    assert rc == 0
    state = jobs.load_state(job_dir)
    assert state["exams"]["exam-001"]["status"] == "done"
    assert state["exams"]["exam-001"]["predicted"] == 100.0


# --------------------------------------------------------------------------
# UI smoke (AppTest): renders against a finished job without exceptions
# --------------------------------------------------------------------------


def test_webui_smoke_with_finished_job(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GRADER_JOBS_DIR", str(tmp_path / "jobs"))
    job_dir = _make_job(tmp_path, n_exams=1)
    jobs.run_job(job_dir, poll_interval=0.1)

    at = AppTest.from_file(str(REPO / "autograder" / "webui.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"UI raised: {at.exception}"
    # the jobs tab should list our job and its completed exam
    assert any("Exam Autograder" in str(t.value) for t in at.sidebar.title)
