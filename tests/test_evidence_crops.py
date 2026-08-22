"""Production OCR-verifier wiring around the (explicitly unavailable) crop
producer: interface, fail-closed behaviour, trace/review reason."""
from __future__ import annotations

import base64

from autograder.escalation import escalate_ocr
from autograder.evidencecrops import (CROP_AVAILABLE, CROP_UNAVAILABLE, PRODUCTION_UNAVAILABLE_REASON,
                                      StaticCropProvider, UnavailableCropProvider, collect_crops,
                                      production_crop_provider)
from autograder.schema import (AnswerKey, ExamExtraction, KeyQuestion, KeySubItem, QuestionExtraction,
                               SubItemExtraction)


def _key() -> AnswerKey:
    q = KeyQuestion(id="1", title="q1", type="selection_with_explanation", max_points=4.0,
                    explanation_required=True,
                    sub_items=[KeySubItem(id="1", prompt="p", correct_by_version={"A": ["a"]}, points=4.0,
                                          reference_explanation="because")])
    return AnswerKey(exam_title="t", versions=["A"], questions=[q], total_points=4.0)


def test_production_provider_is_explicitly_unavailable_and_never_a_full_page():
    p = production_crop_provider()
    assert isinstance(p, UnavailableCropProvider)
    r = p.crop("1", "1")
    assert r.status == CROP_UNAVAILABLE and r.png_b64 is None and not r.available
    assert r.reason == PRODUCTION_UNAVAILABLE_REASON
    d = p.describe()
    assert d["status"] == CROP_UNAVAILABLE and "fail-closed" in d["fallback"]
    assert "full page is never sent" in d["reason"]


def test_collect_crops_reports_availability_without_inventing_crops():
    crops, report = collect_crops(production_crop_provider(), _key())
    assert crops == {} and report["items_with_crop"] == 0 and report["items_without_crop"] == 1
    assert report["status"] == CROP_UNAVAILABLE
    png = base64.b64encode(b"\x89PNG fake").decode()
    crops, report = collect_crops(StaticCropProvider({("1", "1"): png}, name="fixture"), _key())
    assert crops == {("1", "1"): png} and report["status"] == CROP_AVAILABLE and report["provider"] == "fixture"


def test_suspicious_reading_without_crop_fails_closed_to_review_without_a_call():
    calls = []

    class _GW:
        def route(self, task):
            return object()

        def call(self, **kw):
            calls.append(kw)
            raise AssertionError("verifier must not be called without an evidence crop")

    d = escalate_ocr(transcription="ab", crop_png_b64=None, gateway=_GW(), extra_suspicion=["self_declared_partial"])
    assert d.outcome == "review" and d.status == "OCR_UNRESOLVED" and calls == []
    assert "no evidence crop available" in d.reason and d.verify is None and not d.attempted


def test_reliability_route_records_the_missing_crop_as_review_reason():
    """End to end through run_reliability_judging: a partial reading with
    no crop -> REVIEW with the OCR_UNRESOLVED typed reason, verifier skipped
    (no call), crop availability recorded on the run."""
    from autograder.backends.mock import MockBackend
    from autograder.gateway import ModelGateway
    from autograder.reliability import ReliabilityConfig, run_reliability_judging

    calls = []

    def _responder(model, system, blocks):
        calls.append(model.__name__)
        raise AssertionError("no model call expected for an OCR-unresolved item")

    gw = ModelGateway.from_dict({"models": {"grade_primary": {"backend": "mock", "model": "m"},
                                            "ocr_verify": {"backend": "mock", "model": "m"}}},
                                backend_factory=lambda c: MockBackend(config=c, responder=_responder))
    key = _key()
    extraction = ExamExtraction(questions=[QuestionExtraction(
        question_id="1", source_pages=[1], authoritative_source="answer_table",
        sub_items=[SubItemExtraction(sub_item_id="1", status="answered", final_answer="a",
                                     explanation_transcription="x", explanation_legibility="partial",
                                     interpretation_rationale="r", confidence=0.9)])])
    crops, report = collect_crops(production_crop_provider(), key)
    run = run_reliability_judging(key=key, extraction=extraction, version="A",
                                  config=ReliabilityConfig(mode="reliability"), gateway=gw, crops=crops)
    run.evidence_crops = report
    assert run.evidence_crops["status"] == CROP_UNAVAILABLE
    assert calls == []
    d = run.decisions[0]
    # heuristic-only suspicion with no crop: the reading is UNRESOLVED (never
    # AUTO), the verifier stage is SKIPPED with the crop reason, the item is
    # flagged for a human; no model is called (no grading pack here).
    assert d.final_state == "REVIEW"
    assert d.record.ocr_status == "OCR_UNRESOLVED"
    stages = {s.stage: s for s in d.record.stages}
    assert stages["ocr_verify"].status == "skipped"
    import json as _json
    assert "no evidence crop available" in _json.dumps(d.record.as_dict(), ensure_ascii=False)
    assert run.review_items and run.review_items[0].reason.startswith("[")
