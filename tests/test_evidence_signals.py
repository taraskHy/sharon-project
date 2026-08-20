"""Evidence-grounded grading (§1) + OCR/grading uncertainty separation (§2)
+ raw decision signals (§3). Offline, mocked gateway, no provider calls."""

from __future__ import annotations

import copy

import pytest

from autograder.escalation import (GradeResult, OCRVerifyResult, RubricItemGrade,
                                   escalate_grade, escalate_ocr, validate_grade)
from autograder.evidence import (CreditedItem, MAX_EVIDENCE_CHARS, evidence_supported,
                                 normalize_for_evidence, validate_evidence)
from autograder.gradingpack import RubricItemSpec, build_pack
from autograder.signals import (DecisionSignals, GradingSignals, MCSignals, OCRSignals,
                                grade_status_from, ocr_status_from, route_item)
from tests.test_escalation import _gw
from tests.test_grade import make_key

TRANSCRIPTION = (
    "התדרים הגבוהים נשמרים בתמונה לאחר הסינון, ולכן הפירמידה היא לפלסיאן. "
    "the DC term stays unchanged"
)
OTHER_ANSWER = "התדרים הנמוכים בלבד נשמרים כאן, ולכן מדובר בפירמידה גאוסיאנית"


def _pack(policy="choice_and_explanation_independent", **kw):
    key = make_key()
    p = build_pack(key, key.questions[0], grading_policy=policy)
    p.rubric_items = kw.pop("rubric_items", None) or [
        RubricItemSpec(id="R1", text="identifies that high frequencies survive"),
        RubricItemSpec(id="R2", text="names the pyramid type"),
    ]
    for k, v in kw.items():
        setattr(p, k, v)
    p.compute_hash()
    return p


# ------------------------------------------------------------- normalisation --


def test_normalisation_covers_only_harmless_protocol_differences():
    # bidi controls, niqqud, whitespace runs, quote variants, latin case
    assert normalize_for_evidence("‏התדרים   הגבוהים‬") == "התדרים הגבוהים"
    assert normalize_for_evidence("שָׁלוֹם") == normalize_for_evidence("שלום")
    assert normalize_for_evidence('the "DC" term') == normalize_for_evidence("the “DC” term")
    # it never rewrites letters, so a different word stays different
    assert normalize_for_evidence("הגבוהים") != normalize_for_evidence("הנמוכים")


# ---------------------------------------------------------------- §1 evidence --


def test_valid_exact_evidence_is_supported():
    assert evidence_supported("התדרים הגבוהים נשמרים", TRANSCRIPTION)
    # quoting punctuation around the span is a protocol difference, not a defect
    assert evidence_supported('"התדרים הגבוהים נשמרים…"', TRANSCRIPTION)
    assert evidence_supported("the dc term stays", TRANSCRIPTION)


def test_fabricated_and_foreign_evidence_are_rejected():
    assert not evidence_supported("הפירמידה היא גאוסיאנית", TRANSCRIPTION)   # invented
    assert not evidence_supported(OTHER_ANSWER[:30], TRANSCRIPTION)          # another student's answer
    assert not evidence_supported("", TRANSCRIPTION)                         # empty is never support


def test_missing_evidence_is_a_problem_only_where_required():
    specs = {"R1": RubricItemSpec(id="R1", text="semantic"),
             "R2": RubricItemSpec(id="R2", text="left blank", requires_evidence=False)}
    v = validate_evidence(credited=[CreditedItem("R1", None), CreditedItem("R2", None)],
                          transcription=TRANSCRIPTION, specs=specs)
    assert not v.ok and v.missing == ["R1"] and "R2" not in v.missing


def test_empty_evidence_where_allowed_is_ok():
    specs = {"R2": RubricItemSpec(id="R2", text="blank item", requires_evidence=False)}
    v = validate_evidence(credited=[CreditedItem("R2", None)], transcription=TRANSCRIPTION, specs=specs)
    assert v.ok and not v.problems


def test_unverifiable_evidence_without_a_transcription_is_a_problem():
    v = validate_evidence(credited=[CreditedItem("R1", "התדרים הגבוהים")], transcription=None)
    assert not v.ok and "no transcription" in v.problems[0]


def test_evidence_length_is_bounded():
    long = "א" * (MAX_EVIDENCE_CHARS + 50)
    v = validate_evidence(credited=[CreditedItem("R1", long)], transcription=long)
    assert any("characters" in p for p in v.problems)


def test_transcription_is_never_mutated_byte_for_byte():
    original = TRANSCRIPTION
    frozen = copy.deepcopy(original)
    raw = frozen.encode("utf-8")
    validate_evidence(credited=[CreditedItem("R1", "לא קיים"), CreditedItem("R2", "התדרים הגבוהים")],
                      transcription=frozen)
    assert frozen == original and frozen.encode("utf-8") == raw


# ------------------------------------------------- §1 integration: escalation --


def test_grade_with_verified_evidence_validates_clean():
    pack = _pack()
    g = GradeResult(score=3, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="התדרים הגבוהים נשמרים"),
        RubricItemGrade(id="R2", met=False)])
    v = validate_grade(g, pack, selection_correct=True, selected="F", transcription=TRANSCRIPTION)
    assert v.ok and v.evidence["verified"] == ["R1"] and v.evidence["fabricated"] == []


def test_grade_with_fabricated_evidence_is_invalid_and_escalates():
    pack = _pack()
    fabricated = GradeResult(score=4, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="הפירמידה היא גאוסיאנית")])
    clean = GradeResult(score=2, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="התדרים הגבוהים נשמרים")])
    v = validate_grade(fabricated, pack, selection_correct=True, selected="F", transcription=TRANSCRIPTION)
    assert not v.ok and v.evidence["fabricated"] == ["R1"]

    gw, calls = _gw({"grade_primary": [fabricated], "grade_escalate": [clean]})
    d = escalate_grade(pack=pack, selected="F", transcription=TRANSCRIPTION, version="A1",
                       selection_correct=True, gateway=gw)
    assert calls["grade_escalate"] == 1                       # invalid evidence -> escalation
    assert d.outcome == "review" and d.status == "GRADE_DISAGREEMENT"
    assert any("evidence absent" in p for p in d.problems)


def test_evidence_quoted_from_a_different_answer_escalates():
    pack = _pack()
    g = GradeResult(score=4, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence=OTHER_ANSWER[:25])])
    gw, calls = _gw({"grade_primary": [g]})
    d = escalate_grade(pack=pack, selected="F", transcription=TRANSCRIPTION, version="A1",
                       selection_correct=True, gateway=gw)
    assert d.outcome == "review" and d.signals.evidence_fabricated == 1


def test_grader_cannot_earn_credit_by_asserting_without_a_span():
    """The legacy id-only field cannot buy rubric credit under an
    evidence-required pack."""
    pack = _pack()
    g = GradeResult(score=4, rubric_items_met=["R1", "R2"])
    v = validate_grade(g, pack, selection_correct=True, selected="F", transcription=TRANSCRIPTION)
    assert not v.ok and sorted(v.evidence["missing"]) == ["R1", "R2"]


def test_choice_only_packs_disable_evidence_checking():
    pack = build_pack(make_key(), make_key().questions[1], grading_policy="choice_only")
    assert pack.evidence_policy == "disabled"
    v = validate_grade(GradeResult(score=0), pack, selection_correct=False, selected="Z",
                       transcription=None)
    assert v.ok


# ------------------------------------------------ §2 OCR vs grading routing ---


def test_ocr_unclear_routes_to_ocr_work_even_when_grading_is_hard():
    d = route_item(ocr_status="OCR_UNRESOLVED", grade_status="GRADE_UNCERTAIN",
                   ocr_escalation_available=True, grade_escalation_available=True)
    assert d.route == "OCR_ESCALATION" and d.reason_code == "OCR_UNRESOLVED"


def test_hard_grading_never_invokes_stronger_ocr():
    d = route_item(ocr_status="OCR_OK", grade_status="GRADE_UNCERTAIN",
                   ocr_escalation_available=True, grade_escalation_available=True)
    assert d.route == "GRADE_ESCALATION"
    # ...and with no grading escalation configured it is REVIEW, not OCR work
    d2 = route_item(ocr_status="OCR_OK", grade_status="GRADE_INVALID",
                    ocr_escalation_available=True, grade_escalation_available=False)
    assert d2.route == "REVIEW" and d2.reason_code == "GRADE_INVALID"


def test_unreadable_crop_never_reaches_a_stronger_grader():
    d = route_item(ocr_status="OCR_UNRESOLVED", grade_status=None,
                   grade_escalation_available=True, ocr_escalation_available=False)
    assert d.route == "REVIEW" and d.reason_code == "OCR_UNRESOLVED"


def test_both_clear_is_auto_and_disagreement_is_review():
    assert route_item(ocr_status="OCR_OK", grade_status="GRADE_OK").route == "AUTO"
    assert route_item(ocr_status="OCR_OK", grade_status=None).route == "AUTO"
    d = route_item(ocr_status="OCR_OK", grade_status="GRADE_DISAGREEMENT",
                   grade_escalation_available=True)
    assert d.route == "REVIEW" and d.reason_code == "GRADE_DISAGREEMENT"
    assert route_item(ocr_status="OCR_OK", grade_status="GRADE_OK", paused=True).route == "PAUSED"


def test_status_mapping_helpers():
    assert ocr_status_from(suspicious=False) == "OCR_OK"
    assert ocr_status_from(suspicious=True) == "OCR_UNRESOLVED"
    assert ocr_status_from(suspicious=True, verifier_verdict="supported") == "OCR_OK"
    # an image-quality verdict outranks a supportive verifier
    assert ocr_status_from(suspicious=False, verifier_verdict="supported",
                           quality_status="BLANK") == "OCR_UNRESOLVED"
    assert grade_status_from(validation_ok=True) == "GRADE_OK"
    assert grade_status_from(validation_ok=True, uncertain=True) == "GRADE_UNCERTAIN"
    assert grade_status_from(validation_ok=False) == "GRADE_INVALID"
    assert grade_status_from(validation_ok=True, disagreement=True) == "GRADE_DISAGREEMENT"


def test_bad_image_quality_short_circuits_the_ocr_verifier():
    gw, calls = _gw({"ocr_verify": [OCRVerifyResult(verdict="supported", confidence="high")]})
    d = escalate_ocr(transcription="מסנן DC נשאר", crop_png_b64="AAA=", gateway=gw,
                     quality_status="BLANK")
    assert d.outcome == "review" and d.status == "OCR_UNRESOLVED" and calls["ocr_verify"] == 0


# --------------------------------------------------------- §3 signal capture --


def test_decision_signals_round_trip_and_persist_raw_inputs():
    s = DecisionSignals(item_id="item-7", question_id="1",
                        mc=MCSignals(cv_score=0.81, cv_margin=0.4, candidate_cells=2,
                                     resolver_source="deterministic", model_reported_confidence="high"),
                        ocr=OCRSignals(output_chars=42, suspicion_signals=["short_technical_token"]),
                        grading=GradingSignals(primary_score=3.0, explicit_uncertainty=False),
                        raw={"crop_sha": "abc"})
    d = s.as_dict()
    assert d["mc"]["cv_margin"] == 0.4 and d["raw"]["crop_sha"] == "abc"
    back = DecisionSignals.from_dict(d)
    assert back.mc.resolver_source == "deterministic" and back.ocr.output_chars == 42
    assert back.as_dict() == d


def test_escalation_paths_emit_signals():
    pack = _pack()
    good = GradeResult(score=2, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="התדרים הגבוהים נשמרים")])
    gw, _ = _gw({"grade_primary": [good]})
    d = escalate_grade(pack=pack, selected="F", transcription=TRANSCRIPTION, version="A1",
                       selection_correct=True, gateway=gw)
    assert d.outcome == "auto" and d.signals.evidence_verified == 1
    assert d.signals.invariants_ok and d.signals.explicit_uncertainty is False

    ocr = escalate_ocr(transcription="מסנן DC נשאר", crop_png_b64=None, gateway=None)
    assert ocr.signals.suspicion_signals and ocr.signals.output_chars == len("מסנן DC נשאר")


def test_model_reported_confidence_alone_is_not_enough_for_auto():
    """A 'high confidence' grader whose evidence does not exist still escalates."""
    pack = _pack()
    confident_but_wrong = GradeResult(score=4, uncertain=False, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="טענה שלא נכתבה")])
    gw, calls = _gw({"grade_primary": [confident_but_wrong]})
    d = escalate_grade(pack=pack, selected="F", transcription=TRANSCRIPTION, version="A1",
                       selection_correct=True, gateway=gw)
    assert d.outcome == "review"
