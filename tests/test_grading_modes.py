"""Grading pipeline modes at the REAL orchestration seam (§1, §2, §8, §9).

Everything runs through ``cli.run_grade_pipeline`` with a mock vision backend
and a mock task gateway: no provider, no network, no inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from autograder import grade as grade_mod
from autograder import extract as extract_mod
from autograder import orchestrator as orch
from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.cli import run_grade_pipeline
from autograder.escalation import GradeResult, OCRVerifyResult, RubricItemGrade
from autograder.gateway import ModelGateway
from autograder.key_parser import save_answer_key
from autograder.requestcache import RequestCache
from autograder.reliability import GradingModeError
from autograder.usage import BudgetLimits, BudgetManager, UsageLedger
from tests.conftest import make_pdf
from tests.test_grade import make_key
from tests.test_offline_pipeline import (EXPECTED_TOTAL, _fixture_extraction, _fixture_judgement,
                                         _fixture_survey)


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class FakeRuntime:
    """A real Runtime shape with a mock-backed gateway and per-task counters."""

    def __init__(self, tmp_path: Path, responses: dict[str, list], routes=None):
        self.calls: dict[str, int] = {}
        self.blocks: dict[str, list] = {}
        queues = {t: list(v) for t, v in responses.items()}
        tasks = routes or list(responses)

        def factory(cfg):
            task = cfg.model                      # the model name doubles as the task tag

            def responder(model, system, blocks):
                self.calls[task] = self.calls.get(task, 0) + 1
                self.blocks.setdefault(task, []).append(blocks)
                q = queues.get(task) or []
                if not q:
                    raise AssertionError(f"unexpected extra call to {task}")
                return q.pop(0) if len(q) > 1 else q[0]
            return MockBackend(config=cfg, responder=responder)

        self.gateway = ModelGateway.from_dict(
            {"models": {t: {"backend": "mock", "model": t} for t in tasks}},
            backend_factory=factory,
            cache=RequestCache(tmp_path / "cache"),
            ledger=UsageLedger(tmp_path / "usage.jsonl"))
        self.gateway.budget = BudgetManager(BudgetLimits(), ledger=None, warn=lambda m: None)
        self.cache, self.ledger, self.budget = None, None, self.gateway.budget
        self.root = tmp_path
        self.warnings: list[str] = []

    @property
    def total_calls(self) -> int:
        return sum(self.calls.values())


def _backend():
    fixtures = {"ExamSurvey": _fixture_survey(), "QuestionExtraction": _fixture_extraction(),
                "ExplanationJudgement": _fixture_judgement()}
    return MockBackend(config=BackendConfig(backend="mock", model="recording"),
                       responder=lambda model, system, blocks:
                           fixtures[model.__name__].model_copy(deep=True))


def _run(tmp_path, monkeypatch, *, mode, runtime=None, policies=None, rag_policy="RAG_DISABLED",
         out_name="out"):
    exam = make_pdf(tmp_path / "01_50.pdf")
    key_path = tmp_path / "answer_key.json"
    save_answer_key(make_key(), key_path)
    if policies is not None:
        (tmp_path / "policies.json").write_text(json.dumps(policies), encoding="utf-8")
    if runtime is not None:
        monkeypatch.setattr(orch, "setup_from_config", lambda *a, **kw: runtime)
    ns = argparse.Namespace(
        key=str(key_path), rubric=None, resume=False, version="auto", exam=str(exam),
        grading_mode=mode, rag_policy=rag_policy,
        models_config=(str(tmp_path / "models.toml") if runtime is not None else None),
        grading_policies=(str(tmp_path / "policies.json") if policies is not None else None))
    backend = _backend()
    out = tmp_path / out_name
    result = run_grade_pipeline(ns, backend, out, 800, exam_path=exam, exam_label="exam-001")
    return result, backend, out


def _judge_calls(backend) -> int:
    return sum(1 for c in backend.calls if c.output_model == "ExplanationJudgement")


def _grade_responses(score=4.0, verify_text="DC filter"):
    # ocr_verify is the INDEPENDENT contract: the scripted verifier returns
    # its own full-legibility reading; agreement is computed locally, so the
    # default echoes the suspicious fixture text (TEXT_SUSPICIOUS).
    from autograder.escalation import OCRVerifyTranscription
    return {"grade_primary": [GradeResult(score=score)],
            "grade_escalate": [GradeResult(score=score)],
            "ocr_verify": [OCRVerifyTranscription(transcription=verify_text,
                                                  legibility="full")]}


# --------------------------------------------------------------------------
# §1 modes
# --------------------------------------------------------------------------


def test_legacy_is_the_default_and_is_unchanged(tmp_path, monkeypatch):
    result, backend, out = _run(tmp_path, monkeypatch, mode="legacy")
    assert result.total_awarded == EXPECTED_TOTAL
    assert _judge_calls(backend) == 1                     # the validated judge ran
    assert not (out / "decisions.jsonl").exists()         # no new artefacts in legacy
    assert not (out / "shadow_comparison.json").exists()


def test_reliability_mode_replaces_the_legacy_judge(tmp_path, monkeypatch):
    rt = FakeRuntime(tmp_path, _grade_responses())
    result, backend, out = _run(tmp_path, monkeypatch, mode="reliability", runtime=rt)
    assert _judge_calls(backend) == 0                     # the legacy judge did NOT run
    assert rt.calls.get("grade_primary") == 8             # one call per written answer
    assert result.total_awarded == EXPECTED_TOTAL         # agreeing grader -> same score
    assert (out / "decisions.jsonl").exists()


def test_reliability_mode_requires_the_gateway(tmp_path, monkeypatch):
    with pytest.raises(GradingModeError):
        _run(tmp_path, monkeypatch, mode="reliability")   # no --models-config


def test_shadow_runs_both_and_legacy_stays_authoritative(tmp_path, monkeypatch):
    legacy, _, _ = _run(tmp_path, monkeypatch, mode="legacy", out_name="legacy_out")
    rt = FakeRuntime(tmp_path, _grade_responses(score=0.0))   # a grader that DISAGREES
    result, backend, out = _run(tmp_path, monkeypatch, mode="shadow", runtime=rt,
                                out_name="shadow_out")
    assert _judge_calls(backend) == 1 and rt.calls.get("grade_primary") == 8   # both ran
    # the authoritative result is byte-identical to the legacy run (bar timestamps)
    assert result.total_awarded == legacy.total_awarded == EXPECTED_TOTAL
    assert [r.reason for r in result.needs_human_review] == \
           [r.reason for r in legacy.needs_human_review]
    stored = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert stored["total_awarded"] == EXPECTED_TOTAL
    cmp = json.loads((out / "shadow_comparison.json").read_text(encoding="utf-8"))
    assert cmp["authoritative"] == "legacy"
    assert cmp["totals"]["legacy_total"] == EXPECTED_TOTAL
    assert cmp["totals"]["reliability_total"] < EXPECTED_TOTAL      # recorded, not applied


def test_shadow_failure_cannot_break_the_authoritative_grade(tmp_path, monkeypatch):
    rt = FakeRuntime(tmp_path, {"grade_primary": []})     # every call raises
    result, backend, out = _run(tmp_path, monkeypatch, mode="shadow", runtime=rt)
    assert result.total_awarded == EXPECTED_TOTAL
    assert _judge_calls(backend) == 1


def test_unknown_mode_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(GradingModeError):
        _run(tmp_path, monkeypatch, mode="vibes",
             runtime=FakeRuntime(tmp_path, _grade_responses()))


# --------------------------------------------------------------------------
# §8 early exits stay in front of the new route
# --------------------------------------------------------------------------


def test_wrong_choice_zero_short_circuits_before_any_model_work(tmp_path, monkeypatch):
    """A resolved WRONG selection under wrong_choice_zero must score 0 and
    stop: no OCR, no RAG, no grader, no escalation, no REVIEW."""
    key = make_key()
    policies = {q.id: "wrong_choice_zero" for q in key.questions}
    rt = FakeRuntime(tmp_path, _grade_responses())
    # the fixture answers Q1 correctly, so make every accepted answer differ
    for q in key.questions:
        for s in q.sub_items:
            s.correct_by_version = {v: ["Z"] for v in key.versions}
    key_path = tmp_path / "answer_key.json"
    save_answer_key(key, key_path)
    (tmp_path / "policies.json").write_text(json.dumps(policies), encoding="utf-8")
    monkeypatch.setattr(orch, "setup_from_config", lambda *a, **kw: rt)
    exam = make_pdf(tmp_path / "01_50.pdf")
    ns = argparse.Namespace(key=str(key_path), rubric=None, resume=False, version="A1",
                            exam=str(exam), grading_mode="reliability",
                            rag_policy="RAG_DISABLED",
                            models_config=str(tmp_path / "models.toml"),
                            grading_policies=str(tmp_path / "policies.json"))
    result = run_grade_pipeline(ns, _backend(), tmp_path / "out", 800, exam_path=exam,
                                exam_label="exam-001")
    assert rt.total_calls == 0                       # no model work of ANY kind
    assert result.total_awarded == 0.0
    records = [json.loads(l) for l in
               (tmp_path / "out" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records and all(r["final_state"] == "AUTO" for r in records)
    for r in records:
        skipped = {s["stage"]: s["skip_reason"] for s in r["stages"] if s["status"] == "skipped"}
        assert skipped == {"ocr_explanation": "wrong_choice_zero",
                           "grading_rag": "wrong_choice_zero",
                           "grade_primary": "wrong_choice_zero",
                           "grade_escalate": "wrong_choice_zero"}
        assert not [s for s in r["stages"] if s["status"] == "executed"]
    assert not [r for r in result.needs_human_review if r.question_id == "1"]


def test_choice_only_skips_the_explanation_even_when_one_was_written(tmp_path, monkeypatch):
    rt = FakeRuntime(tmp_path, _grade_responses())
    key = make_key()
    policies = {q.id: "choice_only" for q in key.questions}
    result, backend, out = _run(tmp_path, monkeypatch, mode="reliability", runtime=rt,
                                policies=policies)
    assert rt.total_calls == 0
    records = [json.loads(l) for l in
               (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["skip_reasons"] and r["skip_reasons"][0] == "choice_only" for r in records)


# --------------------------------------------------------------------------
# §9 OCR vs grading uncertainty, at the wired seam
# --------------------------------------------------------------------------


def test_unreadable_handwriting_never_reaches_the_grader(tmp_path, monkeypatch):
    """OCR trouble is answered with OCR/review logic — a stronger grader is
    never given an untrusted transcription."""
    from autograder.reliability import ReliabilityConfig, run_reliability_judging
    from autograder.schema import ExamExtraction, QuestionExtraction, SubItemExtraction

    key = make_key()
    ext = ExamExtraction(questions=[QuestionExtraction(
        question_id="1", source_pages=[1], authoritative_source="sheet",
        sub_items=[SubItemExtraction(sub_item_id=s.id, status="answered", final_answer="F",
                                     explanation_transcription=None,
                                     explanation_legibility="illegible",
                                     interpretation_rationale="", confidence=1.0)
                   for s in key.questions[0].sub_items])])
    rt = FakeRuntime(tmp_path, _grade_responses())
    run = run_reliability_judging(key=key, extraction=ext, version="A1",
                                  config=ReliabilityConfig(mode="reliability"),
                                  gateway=rt.gateway, packs={}, exam_id="exam-001")
    assert rt.total_calls == 0                                   # no grader, no verifier
    assert {d.reason_code for d in run.decisions} == {"OCR_UNRESOLVED"}
    assert all(d.final_state == "REVIEW" for d in run.decisions)
    assert all(r.ocr_status == "OCR_UNRESOLVED" and r.grade_status is None for r in run.records)


def test_grading_uncertainty_never_reruns_ocr(tmp_path, monkeypatch):
    """Grading trouble is answered with grading work only: the accepted
    transcription is not re-read."""
    from autograder.gradingpack import build_all_packs
    from autograder.reliability import ReliabilityConfig, run_reliability_judging
    from autograder.schema import ExamExtraction, QuestionExtraction, SubItemExtraction

    key = make_key()
    ext = ExamExtraction(questions=[QuestionExtraction(
        question_id="1", source_pages=[1], authoritative_source="sheet",
        sub_items=[SubItemExtraction(sub_item_id="1", status="answered", final_answer="F",
                                     explanation_transcription="התדרים הגבוהים נשמרים בתמונה",
                                     explanation_legibility="full",
                                     interpretation_rationale="", confidence=1.0)])])
    rt = FakeRuntime(tmp_path, {"grade_primary": [GradeResult(score=2.0, uncertain=True)],
                                "grade_escalate": [GradeResult(score=2.0)],
                                "ocr_verify": [OCRVerifyResult(verdict="supported")]})
    run = run_reliability_judging(key=key, extraction=ext, version="A1",
                                  config=ReliabilityConfig(mode="reliability"),
                                  gateway=rt.gateway, packs=build_all_packs(key, {}),
                                  exam_id="exam-001")
    assert rt.calls.get("grade_primary") == 1 and rt.calls.get("grade_escalate") == 1
    assert rt.calls.get("ocr_verify") is None                    # OCR was accepted: not re-run
    rec = run.records[0]
    assert rec.ocr_status == "OCR_OK" and rec.grade_status == "GRADE_OK"
    assert run.decisions[0].final_state == "AUTO"


def test_evidence_that_is_not_in_the_answer_escalates_at_the_seam(tmp_path):
    from autograder.gradingpack import build_all_packs
    from autograder.reliability import ReliabilityConfig, run_reliability_judging
    from autograder.schema import ExamExtraction, QuestionExtraction, SubItemExtraction

    key = make_key()
    key.questions[0].grading_notes = "identifies that high frequencies survive"
    ext = ExamExtraction(questions=[QuestionExtraction(
        question_id="1", source_pages=[1], authoritative_source="sheet",
        sub_items=[SubItemExtraction(sub_item_id="1", status="answered", final_answer="F",
                                     explanation_transcription="התדרים הגבוהים נשמרים בתמונה",
                                     explanation_legibility="full",
                                     interpretation_rationale="", confidence=1.0)])])
    fabricated = GradeResult(score=4.0, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="משפט שלא נכתב מעולם")])
    rt = FakeRuntime(tmp_path, {"grade_primary": [fabricated], "grade_escalate": [fabricated],
                                "ocr_verify": [OCRVerifyResult(verdict="supported")]})
    run = run_reliability_judging(key=key, extraction=ext, version="A1",
                                  config=ReliabilityConfig(mode="reliability"),
                                  gateway=rt.gateway, packs=build_all_packs(key, {}),
                                  exam_id="exam-001")
    assert rt.calls.get("grade_escalate") == 1               # invalid evidence -> escalation
    d = run.decisions[0]
    assert d.final_state == "REVIEW" and d.reason_code in ("EVIDENCE_INVALID", "GRADE_DISAGREEMENT")
    assert run.records[0].evidence["fabricated"] >= 1
