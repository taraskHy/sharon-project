"""Structural fixes from the 2026-08 OCR-verifier audit (docs/ocr-verifier-audit.md).

Pins the FIXED behaviors: budget pause instead of silent per-item REVIEW,
fail-closed confidence default, digit/formula suspicion, the OCR model's own
partial-legibility self-report as a routing signal, crop-aware empty
transcriptions, failed-call trace honesty, and ledger meta attribution.
All offline; mock gateway only.
"""

from __future__ import annotations

import pytest

from autograder.escalation import OCRVerifyResult, escalate_ocr, ocr_suspicion
from autograder.usage import BudgetExceeded, BudgetLimits, BudgetManager
from autograder.trace import DecisionTraceStore
from tests.test_grading_modes import FakeRuntime, _grade_responses
from tests.test_reliability_guarantees import (TEXT_HE, TEXT_SUSPICIOUS, _extraction,
                                               _png, _run)


# ------------------------------------------------- budget pause (F1) --------


def test_budget_exhaustion_at_the_verifier_pauses_the_run(tmp_path):
    """BudgetExceeded during ocr_verify is a job-level PAUSE, mirroring the
    grade path — never a silent per-item REVIEW labeled 'verifier failed'."""
    from autograder.gradingpack import build_all_packs
    from autograder.reliability import ReliabilityConfig, run_reliability_judging
    from tests.test_grade import make_key

    key = make_key()
    rt = FakeRuntime(tmp_path, _grade_responses())
    # cloud_only=False so the mock backend counts; limit 0 -> first check raises
    rt.gateway.budget = BudgetManager(BudgetLimits(max_calls_per_job=0), cloud_only=False)
    run = run_reliability_judging(
        key=key, extraction=_extraction(text=TEXT_SUSPICIOUS, n=1), version="A1",
        config=ReliabilityConfig(mode="reliability"), gateway=rt.gateway,
        packs=build_all_packs(key, {}), exam_id="exam-001",
        crops={("1", "1"): _png()})
    assert run.paused is True
    assert run.decisions[0].final_state == "PAUSED"
    assert run.decisions[0].reason_code == "BUDGET_PAUSED"


def test_escalate_ocr_reraises_budget_exceeded():
    class _BudgetGateway:
        def route(self, task):
            return object()

        def call(self, **kw):
            raise BudgetExceeded("hard budget reached")

    with pytest.raises(BudgetExceeded):
        escalate_ocr(transcription=TEXT_SUSPICIOUS, crop_png_b64="AAAA",
                     gateway=_BudgetGateway())


# ------------------------------------- fail-closed confidence default (F2) --


def test_omitted_self_assessment_defaults_fail_closed_and_blocks_auto(tmp_path):
    # the research-only verdict schema keeps its fail-closed confidence...
    assert OCRVerifyResult(verdict="supported").confidence == "low"
    # ...and the production INDEPENDENT schema fails closed on legibility: an
    # omitted self-assessment reads as "illegible" and can never support AUTO,
    # even when the returned text happens to match.
    from autograder.escalation import OCRVerifyTranscription
    assert OCRVerifyTranscription(transcription=TEXT_SUSPICIOUS).legibility == "illegible"
    responses = _grade_responses()
    responses["ocr_verify"] = [OCRVerifyTranscription(transcription=TEXT_SUSPICIOUS)]
    run, rt = _run(tmp_path, _extraction(text=TEXT_SUSPICIOUS, n=1),
                   responses=responses, crops={("1", "1"): _png()})
    d = run.decisions[0]
    assert d.final_state == "REVIEW"
    assert d.reason_code == "OCR_UNRESOLVED"
    # evidence-backed disagreement: the grader was withheld entirely
    assert rt.calls.get("grade_primary") is None


def test_full_legibility_independent_agreement_still_supports_auto(tmp_path):
    run, _rt = _run(tmp_path, _extraction(text=TEXT_SUSPICIOUS, n=1),
                    crops={("1", "1"): _png()})  # scripted echo, legibility full
    assert run.decisions[0].final_state == "AUTO"


# ------------------------------------------- digit/formula suspicion (F4) --


def test_short_formulas_trip_the_technical_token_signal():
    for text in ("x=3", "5+3=8", "y = 12"):
        s = ocr_suspicion(text)
        assert s.suspicious and "short_technical_token" in s.signals, text
    assert not ocr_suspicion(TEXT_HE).suspicious  # long Hebrew stays clean


# ------------------------- self-declared partial legibility routes (F3) ----


def test_partial_legibility_with_text_flags_but_never_costs_points(tmp_path):
    clean, _ = _run(tmp_path, _extraction(text=TEXT_HE, n=1))
    partial, _ = _run(tmp_path, _extraction(text=TEXT_HE, n=1, legibility="partial"))
    d = partial.decisions[0]
    assert clean.decisions[0].final_state == "AUTO"
    assert d.final_state == "REVIEW"                   # flagged for a human...
    assert d.reason_code == "OCR_UNRESOLVED"
    assert d.proposed_score == clean.decisions[0].proposed_score   # ...score untouched
    assert (partial.evaluations["1"]["1"].verdict
            == clean.evaluations["1"]["1"].verdict)
    assert "self_declared_partial" in (d.record.signals["ocr"]["suspicion_signals"])
    assert d.record.signals["ocr"]["primary_legibility"] == "partial"


# ------------------------------------ crop-aware empty transcription (F5) --


def test_empty_transcription_over_inked_crop_is_reviewed_not_auto(tmp_path):
    run, rt = _run(tmp_path, _extraction(text="", n=1, legibility="none"),
                   crops={("1", "1"): _png()})       # ink present
    d = run.decisions[0]
    assert d.final_state == "REVIEW" and d.reason_code == "OCR_UNRESOLVED"
    assert rt.calls.get("grade_primary") is None


def test_empty_transcription_over_blank_crop_stays_auto_missing(tmp_path):
    run, _rt = _run(tmp_path, _extraction(text="", n=1, legibility="none"),
                    crops={("1", "1"): _png(blank=True)})
    assert run.decisions[0].final_state == "AUTO"


def test_empty_transcription_without_crop_stays_auto_missing(tmp_path):
    run, _rt = _run(tmp_path, _extraction(text="", n=1, legibility="none"))
    assert run.decisions[0].final_state == "AUTO"    # unchanged production shape


def test_contradictory_full_legibility_with_empty_text_reviews(tmp_path):
    run, _rt = _run(tmp_path, _extraction(text="", n=1, legibility="full"))
    d = run.decisions[0]
    assert d.final_state == "REVIEW" and d.reason_code == "OCR_UNRESOLVED"


# ------------------------------------------------ trace honesty (F7/F8) ----


def test_failed_verifier_call_is_traced_failed_not_skipped(tmp_path):
    store = DecisionTraceStore(tmp_path / "decisions.jsonl")
    responses = _grade_responses()
    responses["ocr_verify"] = []                     # queue empty -> call raises
    run, _rt = _run(tmp_path, _extraction(text=TEXT_SUSPICIOUS, n=1),
                    responses=responses, crops={("1", "1"): _png()}, store=store)
    stages = [s for r in store.read() for s in r["stages"] if s["stage"] == "ocr_verify"]
    assert stages and stages[0]["status"] == "failed"
    assert not stages[0].get("avoided")              # no phantom savings credit
    assert run.decisions[0].final_state == "REVIEW"  # provider failure stays closed


def test_executed_verifier_trace_carries_real_call_metadata(tmp_path):
    store = DecisionTraceStore(tmp_path / "decisions.jsonl")
    _run(tmp_path, _extraction(text=TEXT_SUSPICIOUS, n=1),
         crops={("1", "1"): _png()}, store=store)
    stages = [s for r in store.read() for s in r["stages"] if s["stage"] == "ocr_verify"]
    assert stages and stages[0]["status"] == "executed"
    assert stages[0]["cache_hit"] is False           # real call, not a guess
    assert stages[0]["cloud"] is False               # mock route correctly non-cloud
    assert stages[0]["model"] == "ocr_verify"        # from the CallResult route


# ------------------------------------------------- meta attribution (F6) ---


def test_gateway_meta_carries_exam_id_for_ledger_attribution(tmp_path):
    rt = FakeRuntime(tmp_path, _grade_responses())
    captured: list[dict] = []
    original = rt.gateway.call

    def spy(**kw):
        captured.append(dict(kw.get("meta") or {}))
        return original(**kw)

    rt.gateway.call = spy
    from autograder.gradingpack import build_all_packs
    from autograder.reliability import ReliabilityConfig, run_reliability_judging
    from tests.test_grade import make_key

    key = make_key()
    run_reliability_judging(
        key=key, extraction=_extraction(text=TEXT_HE, n=1), version="A1",
        config=ReliabilityConfig(mode="reliability"), gateway=rt.gateway,
        packs=build_all_packs(key, {}), exam_id="exam-001", job_id="job-7")
    assert captured
    for meta in captured:
        assert meta.get("exam_id") == "exam-001"
        assert meta.get("job_id") == "job-7"
