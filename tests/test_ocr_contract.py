"""The minimal cloud-OCR contract and the immutable transcription.

What may leave the machine for OCR is the smallest legitimate request: the
crop/page image, the answer-free question structure needed to locate the
writing, and a registered verbatim-transcription instruction. Never a rubric,
solution, RAG chunk, score, or another student's work — and the verify pass is
an INDEPENDENT reading compared locally. No provider is contacted here.
"""

from __future__ import annotations

import json

import pytest

from autograder.backends.mock import MockBackend
from autograder.escalation import (OCR_VERIFY_INDEPENDENT_SYSTEM, OCRVerifyTranscription,
                                   compare_transcriptions, escalate_ocr)
from autograder.extract import OCRPageSelectionError, lazy_explanation_ocr
from autograder.gateway import ModelGateway
from autograder.gradingpack import CONTEXT_HEADERS
from autograder.ingest import PageImage
from autograder.prompts import EXPLANATION_OCR_SYSTEM
from autograder.schema import ExamSurvey, ExplanationTranscription, PageInfo, SubItemExtraction
from tests.test_grade import make_key


def _survey(q_on_page: int | None = 1) -> ExamSurvey:
    pages = []
    for n in (1, 2):
        pages.append(PageInfo(page_number=n, content_summary=f"page {n}",
                              question_ids=(["1"] if q_on_page == n else [])))
    return ExamSurvey(pages=pages, student_ink_description="blue pen",
                      grader_annotations_description="")


def _pages() -> list[PageImage]:
    return [PageImage(page_number=n, png_bytes=b"\x89PNG-fake", width=10, height=10, text="")
            for n in (1, 2)]


def _capturing_gateway(response):
    captured = {}

    def factory(cfg):
        def responder(model, system, blocks):
            captured["system"] = system
            captured["blocks"] = blocks
            return response
        return MockBackend(config=cfg, responder=responder)

    gw = ModelGateway.from_dict(
        {"models": {"ocr_primary": {"backend": "mock", "model": "ocr"},
                    "ocr_verify": {"backend": "mock", "model": "verify"}}},
        backend_factory=factory)
    return gw, captured


def _q():
    return make_key().questions[0]


def _se():
    return SubItemExtraction(sub_item_id="1", status="answered",
                             explanation_legibility="deferred",
                             interpretation_rationale="test fixture", confidence=1.0)


# --------------------------------------------------------------------------
# §13 1-5: the production OCR payload is OCR-only
# --------------------------------------------------------------------------


def test_lazy_ocr_payload_carries_images_and_structure_but_no_grading_material():
    gw, cap = _capturing_gateway(ExplanationTranscription(
        sub_item_id="1", transcription="ההסבר של הסטודנט", legibility="full"))
    lazy_explanation_ocr(gw, _q(), _se(), _survey(1), _pages())
    assert cap["system"] == EXPLANATION_OCR_SYSTEM
    text = "\n".join(b.get("text", "") for b in cap["blocks"] if b.get("type") == "text")
    key = make_key()
    # no rubric / solution / RAG headers, no reference explanations, no scores
    for header in CONTEXT_HEADERS:
        assert header not in text, header
    for sub in key.questions[0].sub_items:
        if getattr(sub, "reference_explanation", None):
            assert sub.reference_explanation not in text, "official solution leaked into OCR"
    for banned in ("correct_by_version", "instructor", "score", "Rubric", "solution"):
        assert banned not in text, banned
    # the images sent are the pages the survey placed the question on — not
    # the whole exam
    imgs = [b for b in cap["blocks"] if b.get("type") == "image"]
    assert len(imgs) == 1


def test_lazy_ocr_passes_the_production_cloud_boundary():
    """The REAL payload builder output clears the boundary's payload contract
    — proving the contract permits the honest path while banning grading
    content (test_cloud_boundary pins the banning half)."""
    from autograder.cloudboundary import check_cloud_call
    gw, cap = _capturing_gateway(ExplanationTranscription(
        sub_item_id="1", transcription="טקסט", legibility="full"))
    lazy_explanation_ocr(gw, _q(), _se(), _survey(1), _pages())
    check_cloud_call(task="ocr_primary", backend="openrouter", base_url=None,
                     execution_mode="production", system=cap["system"],
                     content_blocks=cap["blocks"])


def test_no_survey_placement_refuses_instead_of_sending_the_whole_exam():
    gw, cap = _capturing_gateway(ExplanationTranscription(
        sub_item_id="1", transcription="x", legibility="full"))
    with pytest.raises(OCRPageSelectionError):
        lazy_explanation_ocr(gw, _q(), _se(), _survey(q_on_page=None), _pages())
    assert "blocks" not in cap, "nothing was sent"


# --------------------------------------------------------------------------
# the independent verify pass
# --------------------------------------------------------------------------


def test_verify_sends_the_crop_only_never_the_primary_reading_or_a_rubric():
    gw, cap = _capturing_gateway(OCRVerifyTranscription(transcription="מסנן DC נשאר",
                                                        legibility="full"))
    d = escalate_ocr(transcription="מסנן DC נשאר", crop_png_b64="AAA=", gateway=gw)
    assert d.outcome == "auto"
    assert cap["system"] == OCR_VERIFY_INDEPENDENT_SYSTEM
    assert [b["type"] for b in cap["blocks"]] == ["image"], \
        "the verifier receives the crop and NOTHING else"
    blob = json.dumps(cap["blocks"], ensure_ascii=False)
    assert "מסנן" not in blob, "the primary transcription must not anchor the verifier"


def test_verify_agreement_is_computed_locally_and_preserves_the_primary_text():
    gw, _ = _capturing_gateway(OCRVerifyTranscription(transcription="טקסט אחר לגמרי",
                                                      legibility="full"))
    d = escalate_ocr(transcription="מסנן DC נשאר", crop_png_b64="AAA=", gateway=gw)
    assert d.outcome == "review"
    assert d.transcription == "מסנן DC נשאר", \
        "disagreement flags the item; it never rewrites the primary reading"


def test_verifier_self_reported_partial_reading_cannot_auto():
    gw, _ = _capturing_gateway(OCRVerifyTranscription(transcription="מסנן DC נשאר",
                                                      legibility="partial"))
    d = escalate_ocr(transcription="מסנן DC נשאר", crop_png_b64="AAA=", gateway=gw)
    assert d.outcome == "review"


def test_comparison_is_deterministic_and_normalization_aware():
    same = compare_transcriptions("שלום  עולם", "שלום עולם")   # whitespace only
    assert same["similarity"] == 1.0
    diff = compare_transcriptions("התדר גבוה", "התדר נמוך")
    assert diff["similarity"] < 0.95 and diff["substitutions"] == 1


# --------------------------------------------------------------------------
# §13-6: mistakes are preserved, not repaired
# --------------------------------------------------------------------------


def test_both_ocr_contracts_pin_verbatim_transcription():
    for prompt in (EXPLANATION_OCR_SYSTEM, OCR_VERIFY_INDEPENDENT_SYSTEM):
        low = prompt.lower()
        assert "never correct" in low, "the contract must forbid corrections"
        assert "exactly" in low or "verbatim" in low
    # and neither asks for grading, explanation, or confidence essays
    for banned in ("rubric", "grade", "score", "solution", "reasoning"):
        assert banned not in EXPLANATION_OCR_SYSTEM.lower()
        assert banned not in OCR_VERIFY_INDEPENDENT_SYSTEM.lower()


def test_the_ocr_response_schema_is_minimal():
    assert set(ExplanationTranscription.model_fields) == {"sub_item_id", "transcription",
                                                          "legibility"}
    assert set(OCRVerifyTranscription.model_fields) == {"transcription", "legibility"}


# --------------------------------------------------------------------------
# §13-15: nothing on the RAG/grading side can mutate the transcription
# --------------------------------------------------------------------------


def test_rag_attachment_cannot_touch_the_transcription():
    """The grading pack has no transcription field at all, the retrieval
    query is built from question+rubric+solution, and escalate_grade passes
    the frozen text through unchanged."""
    from autograder.escalation import GradeResult, escalate_grade
    from autograder.gradingpack import QuestionGradingPack, RagEvidence, rag_query

    pack = QuestionGradingPack(
        question_id="1", question_text="שאלה", question_type="multiple_choice", max_score=4.0,
        correct_by_version={}, rubric=["רובריקה"], scoring_rules=[],
        grading_policy="choice_plus_explanation", official_solution={"1": "פתרון"},
        evidence_policy="disabled", rag_policy="RAG_ON_UNCERTAIN")
    assert "transcription" not in {f.name for f in pack.__dataclass_fields__.values()}
    q = rag_query(pack)
    assert "רובריקה" in q and "שאלה" in q

    seen = {}

    def factory(cfg):
        def responder(model, system, blocks):
            seen.setdefault("texts", []).append("\n".join(b.get("text", "") for b in blocks))
            return GradeResult(score=99.0, uncertain=True)     # unclean -> triggers RAG retry
        return MockBackend(config=cfg, responder=responder)

    gw = ModelGateway.from_dict(
        {"models": {"grade_primary": {"backend": "mock", "model": "g"}}}, backend_factory=factory)

    frozen = "הטקסט הקפוא של הסטודנט עם שגיאת כתיב"

    def rag_attach(p):
        import dataclasses
        return dataclasses.replace(
            p, rag_evidence=[RagEvidence(chunk_id="c", source="s", text="course chunk",
                                         page=1, similarity=0.9)])

    escalate_grade(pack=pack, selected="A", transcription=frozen, version="default",
                   selection_correct=True, gateway=gw, rag_attach=rag_attach)
    assert len(seen["texts"]) == 2, "primary + RAG retry"
    for t in seen["texts"]:
        assert frozen in t, "the frozen transcription reaches the grader verbatim, every time"
