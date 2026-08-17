"""Low-review escalation engine — offline, mocked gateway."""

from __future__ import annotations

from autograder.backends.mock import MockBackend
from autograder.escalation import (GradeResult, OCRVerifyResult, ReviewMetrics,
                                   escalate_grade, escalate_ocr, ocr_suspicion,
                                   validate_grade)
from autograder.gateway import ModelGateway
from autograder.gradingpack import build_pack
from tests.test_grade import make_key


def _gw(responses_by_task: dict):
    calls = {t: 0 for t in responses_by_task}
    queues = {t: list(v) for t, v in responses_by_task.items()}

    def factory(cfg):
        task = cfg.model  # model name doubles as the task tag in tests

        def responder(model, system, blocks):
            calls[task] += 1
            return queues[task].pop(0)
        return MockBackend(config=cfg, responder=responder)

    gw = ModelGateway.from_dict({"models": {t: {"backend": "mock", "model": t} for t in responses_by_task}},
                                backend_factory=factory)
    return gw, calls


# ------------------------------------------------------------------ OCR ------


def test_ocr_suspicion_is_conservative():
    assert not ocr_suspicion("ניתן לראות שהתדרים הגבוהים נשמרים בתמונה לאחר הסינון").suspicious
    assert "protocol_artifact" in ocr_suspicion('{"transcription": "ניתן לראות').signals
    assert "repetition" in ocr_suspicion("האם יש תנאי קיפול? " * 6).signals
    assert "empty_or_tiny" in ocr_suspicion("").signals
    assert "short_technical_token" in ocr_suspicion("מסנן DC נשאר").signals   # dangerous class
    assert "no_hebrew" in ocr_suspicion("date 7.10.15 the defendant filed").signals


def test_ocr_not_suspicious_means_no_verifier_call():
    gw, calls = _gw({"ocr_verify": [OCRVerifyResult(verdict="review")]})
    d = escalate_ocr(transcription="ניתן לראות שהתדרים הגבוהים נשמרים בתמונה לאחר הסינון",
                     crop_png_b64="AAA=", gateway=gw)
    assert d.outcome == "auto" and calls["ocr_verify"] == 0


def test_ocr_suspicious_verifier_supports_then_auto_else_review():
    gw, calls = _gw({"ocr_verify": [OCRVerifyResult(verdict="supported", confidence="high"),
                                    OCRVerifyResult(verdict="review", substitutions=["DC->pc"])]})
    ok = escalate_ocr(transcription="מסנן DC נשאר", crop_png_b64="AAA=", gateway=gw)
    assert ok.outcome == "auto" and calls["ocr_verify"] == 1
    bad = escalate_ocr(transcription="מסנן DC נשאר", crop_png_b64="AAA=", gateway=gw)
    assert bad.outcome == "review" and bad.verify["substitutions"] == ["DC->pc"]


def test_ocr_suspicious_without_verifier_is_review():
    d = escalate_ocr(transcription="", crop_png_b64=None, gateway=None)
    assert d.outcome == "review"


# --------------------------------------------------------------- grading -----


def _pack(policy="choice_and_explanation_independent"):
    key = make_key()
    return build_pack(key, key.questions[0], grading_policy=policy)


def test_validate_grade_rules():
    pack = _pack()
    ok = validate_grade(GradeResult(score=4, rubric_items_met=[]), pack, selection_correct=True, selected="F")
    assert ok.ok
    bad = validate_grade(GradeResult(score=99, rubric_items_met=["R9"], uncertain=True), pack,
                         selection_correct=True, selected="F")
    assert not bad.ok and any("outside" in p for p in bad.problems) and any("uncertainty" in p for p in bad.problems)
    wcz = _pack("wrong_choice_zero")
    v = validate_grade(GradeResult(score=2), wcz, selection_correct=False, selected="Z")
    assert not v.ok and any("wrong_choice_zero" in p for p in v.problems)


def test_grade_primary_clean_is_auto_no_escalation_call():
    gw, calls = _gw({"grade_primary": [GradeResult(score=3, uncertain=False)],
                     "grade_escalate": [GradeResult(score=3)]})
    d = escalate_grade(pack=_pack(), selected="F", transcription="הסבר", version="A1", selection_correct=True,
                       gateway=gw)
    assert d.outcome == "auto" and d.stage == "primary" and calls["grade_escalate"] == 0


def test_grade_uncertain_primary_resolved_by_escalation():
    gw, calls = _gw({"grade_primary": [GradeResult(score=2, uncertain=True)],
                     "grade_escalate": [GradeResult(score=3, uncertain=False)]})
    d = escalate_grade(pack=_pack(), selected="F", transcription="הסבר", version="A1", selection_correct=True,
                       gateway=gw)
    assert d.outcome == "auto" and d.stage == "escalated" and d.result.score == 3
    assert calls["grade_escalate"] == 1


def test_grade_unresolved_disagreement_is_review():
    gw, calls = _gw({"grade_primary": [GradeResult(score=99)],           # invalid range
                     "grade_escalate": [GradeResult(score=1, uncertain=True)]})  # still uncertain
    d = escalate_grade(pack=_pack(), selected="F", transcription="הסבר", version="A1", selection_correct=True,
                       gateway=gw)
    assert d.outcome == "review" and d.stage == "escalated" and d.problems


def test_grade_no_escalation_model_configured_is_review():
    gw, calls = _gw({"grade_primary": [GradeResult(score=1, uncertain=True)]})
    d = escalate_grade(pack=_pack(), selected="F", transcription="הסבר", version="A1", selection_correct=True,
                       gateway=gw)
    assert d.outcome == "review" and "no escalation model" in d.reason


def test_grade_prompt_is_small_and_pack_hash_in_meta():
    from autograder.escalation import grade_prompt
    pack = _pack()
    blocks = grade_prompt(pack, selected="F", transcription="x", version="A1")
    assert len(blocks) == 1 and len(blocks[0]["text"]) < 4000   # tiny by design (no full pages)


def test_review_metrics_pairs_rates():
    m = ReviewMetrics()
    for _ in range(8):
        m.bump("items")
    m.bump("auto", 6); m.bump("review", 2); m.bump("escalated", 3); m.bump("mc_early_exit", 4)
    d = m.as_dict()
    assert d["auto_pct"] == 75.0 and d["review_pct"] == 25.0 and d["escalation_pct"] == 37.5
    assert d["mc_early_exit_pct"] == 50.0
