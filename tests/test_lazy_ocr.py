"""Lazy explanation OCR in reliability mode, at the REAL orchestration seam.

The audit found the policy gate could only skip judging/RAG/escalation —
explanation transcription had already happened during extraction, defeating
the token-saving goal once OCR is cloud-backed. These tests prove OCR is now
deferred until AFTER MC extraction + resolution + the grading-policy gate,
with ACTUAL call counts (a raising mock, not post-hoc 'skipped' records).
Mock backend + mock gateway; no provider, no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autograder import orchestrator as orch
from autograder.cli import run_grade_pipeline
from autograder.escalation import GradeResult, OCRVerifyResult
from autograder.key_parser import save_answer_key
from autograder.schema import ExplanationTranscription
from tests.conftest import make_pdf
from tests.test_grade import make_key
from tests.test_grading_modes import FakeRuntime, _backend, _grade_responses


def _responses(*, ocr: list | None = None, score: float = 4.0) -> dict:
    """Gateway task queues INCLUDING an ocr_primary route. An empty ocr list
    means any OCR call raises ('unexpected extra call to ocr_primary')."""
    return {"ocr_primary": ocr if ocr is not None else [],
            **_grade_responses(score=score)}


def _run(tmp_path, monkeypatch, *, key, policies, responses, version="A1", out_name="out"):
    key_path = tmp_path / "answer_key.json"
    save_answer_key(key, key_path)
    (tmp_path / "policies.json").write_text(json.dumps(policies), encoding="utf-8")
    rt = FakeRuntime(tmp_path, responses)
    monkeypatch.setattr(orch, "setup_from_config", lambda *a, **kw: rt)
    exam = make_pdf(tmp_path / "01_50.pdf")
    ns = argparse.Namespace(
        key=str(key_path), rubric=None, resume=False, version=version, exam=str(exam),
        grading_mode="reliability", rag_policy="RAG_DISABLED", course=None,
        packs_root=str(tmp_path / "packs"),
        models_config=str(tmp_path / "models.toml"),
        grading_policies=str(tmp_path / "policies.json"))
    out = tmp_path / out_name
    result = run_grade_pipeline(ns, _backend(), out, 800, exam_path=exam,
                                exam_label="exam-001")
    return result, rt, out


def _records(out: Path) -> list[dict]:
    return [json.loads(l) for l in
            (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]


def _wrong_key():
    key = make_key()
    for q in key.questions:
        for s in q.sub_items:
            s.correct_by_version = {v: ["Z"] for v in key.versions}
    return key


def test_wrong_choice_zero_truly_prevents_explanation_ocr(tmp_path, monkeypatch):
    """Confident wrong MC + wrong_choice_zero: score 0, and the OCR mock (which
    RAISES if called) proves explanation OCR actually never ran — along with
    RAG, grader, escalation, and REVIEW."""
    key = _wrong_key()
    policies = {q.id: "wrong_choice_zero" for q in key.questions}
    result, rt, out = _run(tmp_path, monkeypatch, key=key, policies=policies,
                           responses=_responses(ocr=[]))     # OCR would raise
    assert rt.total_calls == 0                               # ACTUAL calls: none at all
    assert result.total_awarded == 0.0
    assert not [r for r in result.needs_human_review if r.question_id == "1"]
    for r in _records(out):
        assert r["final_state"] == "AUTO"
        assert not [s for s in r["stages"] if s["status"] == "executed"]
        skipped = {s["stage"]: s["skip_reason"] for s in r["stages"] if s["status"] == "skipped"}
        assert skipped["ocr_explanation"] == "wrong_choice_zero"
    # the persisted extraction holds NO transcription — nothing was ever read.
    # Gradeable-explanation sub-items stay "deferred"; selection-only ones "none".
    ext = json.loads((out / "extraction.json").read_text(encoding="utf-8"))
    legibilities = [s["explanation_legibility"]
                    for q in ext["questions"] for s in q["sub_items"]]
    assert all(s["explanation_transcription"] is None
               for q in ext["questions"] for s in q["sub_items"])
    assert set(legibilities) <= {"deferred", "none"} and "deferred" in legibilities


def test_choice_only_truly_prevents_explanation_ocr(tmp_path, monkeypatch):
    key = make_key()
    policies = {q.id: "choice_only" for q in key.questions}
    result, rt, out = _run(tmp_path, monkeypatch, key=key, policies=policies,
                           responses=_responses(ocr=[]))
    assert rt.total_calls == 0
    assert all(r["skip_reasons"] and r["skip_reasons"][0] == "choice_only"
               for r in _records(out))


def test_rescue_policy_still_processes_the_explanation(tmp_path, monkeypatch):
    """explanation_can_rescue_wrong_choice: wrong MC must NOT early-exit —
    lazy OCR runs, then the grader."""
    key = _wrong_key()
    policies = {q.id: "explanation_can_rescue_wrong_choice" for q in key.questions}
    ocr = [ExplanationTranscription(sub_item_id="1", transcription="הסבר של הסטודנט",
                                    legibility="full")]
    result, rt, out = _run(tmp_path, monkeypatch, key=key, policies=policies,
                           responses=_responses(ocr=ocr))
    assert rt.calls.get("ocr_primary", 0) >= 1               # OCR executed
    assert rt.calls.get("grade_primary", 0) >= 1             # grading executed
    recs = _records(out)
    assert any(s["stage"] == "ocr_explanation" and s["status"] == "executed"
               and s["task"] == "ocr_primary" for r in recs for s in r["stages"])


def test_independent_policy_processes_the_explanation_component(tmp_path, monkeypatch):
    key = make_key()
    policies = {q.id: "choice_and_explanation_independent" for q in key.questions}
    ocr = [ExplanationTranscription(sub_item_id="1", transcription="הסבר קצר",
                                    legibility="full")]
    result, rt, out = _run(tmp_path, monkeypatch, key=key, policies=policies,
                           responses=_responses(ocr=ocr))
    assert rt.calls.get("ocr_primary", 0) >= 1
    assert rt.calls.get("grade_primary", 0) >= 1
    # lazily produced transcriptions are frozen back into extraction.json
    ext = json.loads((out / "extraction.json").read_text(encoding="utf-8"))
    filled = [s for q in ext["questions"] for s in q["sub_items"]
              if s["explanation_transcription"]]
    assert filled and all(s["explanation_legibility"] == "full" for s in filled)


def test_correct_mc_under_wrong_choice_zero_still_grades_the_explanation(tmp_path, monkeypatch):
    """The early exit fires only on WRONG selections: a correct one continues
    into lazy OCR + grading."""
    key = make_key()                                          # fixture answers correctly
    policies = {q.id: "wrong_choice_zero" for q in key.questions}
    ocr = [ExplanationTranscription(sub_item_id="1", transcription="נימוק נכון",
                                    legibility="full")]
    result, rt, out = _run(tmp_path, monkeypatch, key=key, policies=policies,
                           responses=_responses(ocr=ocr))
    assert rt.calls.get("ocr_primary", 0) >= 1
    assert rt.calls.get("grade_primary", 0) >= 1


def test_without_an_ocr_primary_route_extraction_stays_eager(tmp_path, monkeypatch):
    """No ocr_primary route configured -> the validated eager extraction runs
    unchanged and the reliability route consumes its transcriptions."""
    key = make_key()
    result, rt, out = _run(tmp_path, monkeypatch, key=key, policies={},
                           responses=_grade_responses())     # NO ocr_primary route
    assert rt.calls.get("grade_primary", 0) >= 1
    ext = json.loads((out / "extraction.json").read_text(encoding="utf-8"))
    assert all(s["explanation_legibility"] != "deferred"
               for q in ext["questions"] for s in q["sub_items"])
