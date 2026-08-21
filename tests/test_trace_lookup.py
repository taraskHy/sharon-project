"""Decision-trace UI lookup uses stable identifiers, not basename equality.

Audit finding: records are written with exam_id = exam label or exam-file
stem, while the UI looked up by the OUTPUT DIRECTORY name — an ad-hoc CLI
run whose --out basename differs silently degraded to the reconstruction
fallback. These tests pin the fixed lookup and the persisted-RAG rendering.
"""

from __future__ import annotations

from autograder.reviewui import decision_trace_for
from autograder.trace import DecisionTrace, DecisionTraceStore


def _store_with_record(exam_dir, exam_id="custom-label"):
    exam_dir.mkdir(parents=True, exist_ok=True)
    t = DecisionTrace(exam_id, "2", "1", grading_policy="wrong_choice_zero",
                      rag_policy="RAG_ON_UNCERTAIN")
    t.deterministic("selection F (single_mark, confidence 0.93)")
    t.skipped("ocr_explanation", "wrong_choice_zero", detail="wrong MC -> 0",
              avoided={"ocr": 1, "cloud": 1})
    t.rag(policy="RAG_ON_UNCERTAIN", used=True, available=True,
          chunk_ids=["c0", "c1"], chars=311)
    rec = t.finish("AUTO", "AUTO", "wrong_choice_zero: wrong MC -> 0, no OCR",
                   points_max=4.0)
    DecisionTraceStore(exam_dir / "decisions.jsonl").append(rec)
    return rec


RESULT = {"exam_file": "custom-label.pdf", "questions": [
    {"question_id": "2", "sub_results": [
        {"sub_item_id": "1", "needs_review": False, "points_total": 0.0,
         "points_max": 4.0, "student_answer": "F", "accepted_answers": ["Z"],
         "reason": "wrong selection"}]}]}


def test_trace_found_when_out_dir_name_differs_from_exam_id(tmp_path):
    exam_dir = tmp_path / "some-unrelated-out-name"
    _store_with_record(exam_dir, exam_id="custom-label")
    text = decision_trace_for(exam_dir, RESULT, "2", "1")
    assert "RECONSTRUCTED" not in text                 # the REAL trace was found
    assert "wrong_choice_zero" in text and "AUTO" in text


def test_trace_found_via_unanimous_exam_id_without_result_metadata(tmp_path):
    exam_dir = tmp_path / "another-dir"
    _store_with_record(exam_dir, exam_id="totally-different-label")
    text = decision_trace_for(exam_dir, {"questions": RESULT["questions"]}, "2", "1")
    assert "RECONSTRUCTED" not in text                 # single-exam file: accepted


def test_trace_renders_the_persisted_rag_route(tmp_path):
    exam_dir = tmp_path / "out"
    _store_with_record(exam_dir)
    text = decision_trace_for(exam_dir, RESULT, "2", "1")
    assert "RAG: used (2 chunks, 311 chars)" in text
    assert "course evidence chunks: c0, c1" in text
    assert "skipped ocr_explanation: wrong_choice_zero" in text


def test_reconstruction_fallback_remains_loudly_labeled(tmp_path):
    exam_dir = tmp_path / "legacy-out"
    exam_dir.mkdir()
    text = decision_trace_for(exam_dir, RESULT, "2", "1")   # no decisions.jsonl at all
    assert "RECONSTRUCTED SUMMARY - NOT AN EXECUTION TRACE" in text


def test_shadow_traces_are_labeled_non_authoritative(tmp_path):
    """A trace recorded by the SHADOW route is a proposal — it must never be
    readable as the student's actual recorded grade."""
    exam_dir = tmp_path / "shadow-out"
    _store_with_record(exam_dir, exam_id="custom-label")
    (exam_dir / "shadow_comparison.json").write_text("{}", encoding="utf-8")
    text = decision_trace_for(exam_dir, RESULT, "2", "1")
    assert text.startswith("SHADOW / NON-AUTHORITATIVE")
    assert "actual recorded grade is the legacy result" in text
    assert "wrong_choice_zero" in text                     # the real trace still renders


def test_reliability_traces_carry_no_shadow_label(tmp_path):
    exam_dir = tmp_path / "reliability-out"
    _store_with_record(exam_dir, exam_id="custom-label")   # no shadow_comparison.json
    text = decision_trace_for(exam_dir, RESULT, "2", "1")
    assert "SHADOW / NON-AUTHORITATIVE" not in text
    assert "RECONSTRUCTED" not in text
