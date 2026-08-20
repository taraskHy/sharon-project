"""Deterministic grade invariants (§6). No model is ever trusted with
arithmetic or structural score validity."""

from __future__ import annotations

from autograder.escalation import GradeResult, RubricItemGrade
from autograder.gradingpack import RubricItemSpec, build_pack
from autograder.invariants import (check_exam_invariants, check_question_invariants,
                                   recompute_exam_totals, repair_arithmetic)
from autograder.schema import ExamResult, QuestionResult, SubItemResult
from tests.test_grade import make_key


def _pack(policy="choice_and_explanation_independent", items=None, **kw):
    key = make_key()
    p = build_pack(key, key.questions[0], grading_policy=policy)
    p.max_score = 8.0
    p.rubric_items = items if items is not None else [
        RubricItemSpec(id="R1", text="a", points=2),
        RubricItemSpec(id="R2", text="b", points=3),
        RubricItemSpec(id="R3", text="c", points=3),
    ]
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _g(score, met=(), dupes=()):
    items = [RubricItemGrade(id=i, met=True, student_evidence="x") for i in met]
    items += [RubricItemGrade(id=i, met=True, student_evidence="x") for i in dupes]
    return GradeResult(score=score, rubric_items=items)


# --------------------------------------------------------------- question ----


def test_legal_result_passes_every_check():
    r = check_question_invariants(_g(5, met=("R1", "R3")), _pack())
    assert r.ok and not r.problems and "subscore_sum" in r.checked


def test_score_overflow_and_negative_scores_are_invalid():
    over = check_question_invariants(_g(99, met=("R1",)), _pack())
    assert not over.ok and any("exceeds question maximum" in p for p in over.problems)
    neg = check_question_invariants(_g(-1), _pack())
    assert not neg.ok and any("negative" in p for p in neg.problems)


def test_unknown_rubric_id_is_invalid():
    r = check_question_invariants(_g(2, met=("R9",)), _pack())
    assert not r.ok and any("not in the rubric" in p for p in r.problems)


def test_duplicate_rubric_credit_is_invalid():
    r = check_question_invariants(_g(4, met=("R1",), dupes=("R1",)), _pack())
    assert not r.ok and any("more than once" in p for p in r.problems)


def test_subscore_sum_must_equal_the_question_score():
    r = check_question_invariants(_g(7, met=("R1", "R2")), _pack())   # 2+3 = 5, not 7
    assert not r.ok and any("sum of credited rubric points" in p for p in r.problems)


def test_wrong_choice_zero_violation_is_invalid():
    p = _pack("wrong_choice_zero", items=[RubricItemSpec(id="R1", text="a")])
    r = check_question_invariants(_g(3, met=("R1",)), p, selection_correct=False)
    assert not r.ok and any("wrong_choice_zero" in x for x in r.problems)
    ok = check_question_invariants(_g(0), p, selection_correct=False)
    assert ok.ok


def test_choice_only_forbids_rubric_credit():
    p = _pack("choice_only", items=[RubricItemSpec(id="R1", text="a")])
    r = check_question_invariants(_g(2, met=("R1",)), p)
    assert not r.ok and any("choice_only" in x for x in r.problems)


def test_mutually_exclusive_items_cannot_both_be_credited():
    items = [RubricItemSpec(id="R1", text="a", excludes=["R2"]), RubricItemSpec(id="R2", text="b")]
    r = check_question_invariants(_g(4, met=("R1", "R2")), _pack(items=items))
    assert not r.ok and any("mutually exclusive" in p for p in r.problems)
    assert check_question_invariants(_g(4, met=("R1",)), _pack(items=items)).ok


def test_prerequisite_items_are_enforced():
    items = [RubricItemSpec(id="R1", text="a"), RubricItemSpec(id="R2", text="b", requires=["R1"])]
    r = check_question_invariants(_g(4, met=("R2",)), _pack(items=items))
    assert not r.ok and any("requires ['R1']" in p for p in r.problems)
    assert check_question_invariants(_g(4, met=("R1", "R2")), _pack(items=items)).ok


def test_score_granularity_policy_is_enforced_when_configured():
    p = _pack(items=[RubricItemSpec(id="R1", text="a")], score_granularity=0.5)
    assert check_question_invariants(_g(2.5, met=("R1",)), p).ok
    bad = check_question_invariants(_g(2.3, met=("R1",)), p)
    assert not bad.ok and any("granularity" in x for x in bad.problems)


def test_arithmetic_repair_is_explicit_and_only_for_unambiguous_components():
    p = _pack()
    score, repaired = repair_arithmetic(_g(7, met=("R1", "R2")), p)
    assert (score, repaired) == (5.0, True)
    # no declared points -> nothing is unambiguous -> no silent correction
    p2 = _pack(items=[RubricItemSpec(id="R1", text="a")])
    score2, repaired2 = repair_arithmetic(_g(7, met=("R1",)), p2)
    assert (score2, repaired2) == (7.0, False)


# ------------------------------------------------------------------- exam ----


def _sub(qid, sid, sel, exp, mx):
    return SubItemResult(question_id=qid, sub_item_id=sid, question_type="multiple_choice",
                         status="answered", student_answer="A", accepted_answers=["A"],
                         selection_correct=True, explanation_transcription=None,
                         explanation_evaluation=None, points_selection=sel, points_explanation=exp,
                         points_total=sel + exp, points_max=mx, reason="")


def _exam(total=None, sub_total_override=None, awarded=None):
    subs = [_sub("1", "1", 2.0, 1.0, 4.0), _sub("1", "2", 4.0, 0.0, 4.0)]
    if sub_total_override is not None:
        subs[0].points_total = sub_total_override
    q = QuestionResult(question_id="1", question_type="multiple_choice",
                       points_awarded=7.0 if awarded is None else awarded,
                       points_max=8.0, sub_results=subs, summary="")
    return ExamResult(exam_file="e", graded_at="", model="", detected_version="A1",
                      version_detection="", total_awarded=7.0 if total is None else total,
                      total_max=8.0, questions=[q])


def test_exam_level_legal_result_passes():
    assert check_exam_invariants(_exam()).ok


def test_final_total_must_be_the_deterministic_sum():
    r = check_exam_invariants(_exam(total=99.0))
    assert not r.ok and any("total_awarded" in p for p in r.problems)
    assert recompute_exam_totals(_exam(total=99.0)) == 7.0


def test_component_sum_and_cap_violations_are_detected():
    r = check_exam_invariants(_exam(sub_total_override=9.0))
    assert not r.ok and any("components" in p for p in r.problems)
    cap = check_exam_invariants(_exam(awarded=8.5))
    assert not cap.ok and any("awarded" in p for p in cap.problems)


def test_exam_maxima_are_checked_against_the_key():
    key = make_key()
    r = check_exam_invariants(_exam(), key)
    assert any("question_max_matches_key" == c for c in r.checked)


# --------------------------------------------- integration: the live path ----


def test_grade_exam_runs_the_deterministic_self_check(monkeypatch):
    """The scoring arithmetic is plain Python, so a violation is a real defect:
    it is reported and routed to a human, never silently corrected."""
    from autograder import grade as grade_mod
    from autograder.config import GraderConfig
    from autograder.grade import VersionDecision, grade_exam
    from autograder.schema import ExamExtraction, QuestionExtraction, SubItemExtraction

    key = make_key()
    q = key.questions[0]
    ext = ExamExtraction(questions=[QuestionExtraction(
        question_id=qq.id, source_pages=[1], authoritative_source="sheet",
        sub_items=[SubItemExtraction(sub_item_id=s.id, status="unanswered",
                                     interpretation_rationale="", confidence=1.0)
                   for s in qq.sub_items]) for qq in key.questions])
    vd = VersionDecision("A1", "", False)
    clean = grade_exam(key, ext, {}, vd, GraderConfig())
    assert not any(r.question_id == "*" and "consistency check" in r.reason
                   for r in clean.needs_human_review)

    # a corrupted total must be caught by the check, not accepted
    import autograder.invariants as inv_mod
    real = inv_mod.check_exam_invariants

    def broken(result, key=None):
        result.total_awarded = 999.0
        return real(result, key)

    monkeypatch.setattr(inv_mod, "check_exam_invariants", broken)
    flagged = grade_exam(key, ext, {}, vd, GraderConfig())
    assert flagged.needs_human_review[0].question_id == "*"
    assert "consistency check" in flagged.needs_human_review[0].reason
    assert any("invariants violated" in m for m in flagged.mark_interpretations)


def test_undecodable_crop_spends_no_model_calls():
    """imagequality in the live MC chain: an unreadable crop cannot be read by
    any model, so no call is made to discover that."""
    from autograder.mcresolve import resolve_row

    calls = {"n": 0}

    class _GW:
        def route(self, task):
            return True

        def call(self, **kw):
            calls["n"] += 1
            raise AssertionError("no model call should happen on an undecodable crop")

    res, trace = resolve_row(band_png=b"not-an-image", letters=["A", "B"], candidates=["A", "B"],
                             gateway=_GW())
    assert calls["n"] == 0 and res.source == "review"
    assert any(s["stage"] == "image_quality" for s in trace.stages)
