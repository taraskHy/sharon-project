"""Grading-policy early exits + MC resolution chain — offline, all mocked."""

from __future__ import annotations

import pytest

from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.gateway import ModelGateway
from autograder.mcresolve import MCRead, MCResolverStats, resolve_row
from autograder.policies import (MCResolution, decide_before_ocr,
                                 infer_policy_from_key)


def mc(sel, state="single_mark", conf=0.95, cands=None):
    return MCResolution(sel, state, conf, "deterministic", cands or ([sel] if sel else []))


# ------------------------------------------------------------ policies ------


def test_wrong_choice_zero_wrong_answer_zero_calls_and_zero_score():
    d = decide_before_ocr(policy="wrong_choice_zero", mc=mc("B"), accepted=["D"],
                          points_selection=4, points_max=10)
    assert d.action == "score_locally" and d.score == 0.0 and d.skip_explanation
    assert d.persist_flag == "deterministic_zero_wrong_choice" and d.selection_correct is False


def test_wrong_choice_zero_correct_answer_continues_to_ocr():
    d = decide_before_ocr(policy="wrong_choice_zero", mc=mc("D"), accepted=["D"],
                          points_selection=4, points_max=10)
    assert d.action == "ocr_explanation" and not d.skip_explanation and d.selection_correct is True


def test_explanation_required_if_correct():
    wrong = decide_before_ocr(policy="explanation_required_if_correct", mc=mc("A"), accepted=["D"],
                              points_selection=4, points_max=10)
    assert wrong.action == "score_locally" and wrong.skip_explanation and wrong.score == 0.0
    right = decide_before_ocr(policy="explanation_required_if_correct", mc=mc("D"), accepted=["D"],
                              points_selection=4, points_max=10)
    assert right.action == "ocr_explanation"
    proc = decide_before_ocr(policy="explanation_required_if_correct", mc=mc("A"), accepted=["D"],
                             points_selection=4, points_max=10, wrong_answer_rule="process")
    assert proc.action == "ocr_explanation"


def test_rescue_policy_never_early_exits_on_wrong():
    d = decide_before_ocr(policy="explanation_can_rescue_wrong_choice", mc=mc("A"), accepted=["D"],
                          points_selection=4, points_max=10)
    assert d.action == "ocr_explanation" and not d.skip_explanation


def test_choice_only_never_ocrs():
    d = decide_before_ocr(policy="choice_only", mc=mc("D"), accepted=["D"], points_selection=4, points_max=4)
    assert d.action == "score_locally" and d.score == 4 and d.skip_explanation
    d2 = decide_before_ocr(policy="choice_only", mc=mc("A"), accepted=["D"], points_selection=4, points_max=4)
    assert d2.score == 0.0 and d2.skip_explanation


def test_ambiguous_mc_never_yields_invalid_early_exit():
    amb = mc(None, state="multiple_marks", conf=0.0, cands=["B", "D"])
    for policy in ("wrong_choice_zero", "explanation_required_if_correct",
                   "explanation_can_rescue_wrong_choice", "choice_and_explanation_independent"):
        d = decide_before_ocr(policy=policy, mc=amb, accepted=["D"], points_selection=4, points_max=10)
        assert d.action == "ocr_explanation" and d.score is None and not d.skip_explanation, policy
    d = decide_before_ocr(policy="choice_only", mc=amb, accepted=["D"], points_selection=4, points_max=4)
    assert d.action == "review"
    # low-confidence single mark is likewise not resolved
    low = mc("A", conf=0.5)
    d = decide_before_ocr(policy="wrong_choice_zero", mc=low, accepted=["D"], points_selection=4, points_max=10)
    assert d.action == "ocr_explanation" and not d.skip_explanation


def test_policy_inference_from_key():
    assert infer_policy_from_key(False, 0.0, None)[0] == "choice_only"
    assert infer_policy_from_key(True, 0.5, "Wrong answer zero for the whole question")[0] == "wrong_choice_zero"
    assert infer_policy_from_key(True, 0.5, "credit for explanation even if the choice is wrong")[0] == "explanation_can_rescue_wrong_choice"
    assert infer_policy_from_key(True, 0.5, "graded separately")[0] == "choice_and_explanation_independent"
    assert infer_policy_from_key(True, 1.0, None)[0] == "explanation_can_rescue_wrong_choice"
    assert infer_policy_from_key(True, 0.5, None)[0] is None  # ambiguous -> next inference stage


# ------------------------------------------------- policy hook in grade.py --


def test_grade_hook_skips_judge_calls_for_early_exit_rows():
    from autograder import grade
    from autograder.schema import QuestionExtraction, SubItemExtraction
    from tests.test_grade import make_key

    key = make_key()
    q1 = key.questions[0]           # matching_with_explanation, explanation_required
    subs = []
    for s in q1.sub_items[:3]:
        correct = s.correct_by_version["A1"][0]
        wrong = "Z"
        subs.append(SubItemExtraction(
            sub_item_id=s.id, status="answered", answer_origin="answer_sheet",
            final_answer=(correct if s.id == "1" else wrong),
            explanation_transcription="some explanation text", explanation_legibility="full",
            interpretation_rationale="t", confidence=0.95))
    ext_q = QuestionExtraction(question_id="1", source_pages=[1], authoritative_source="t",
                               answer_sheet_status="present", sub_items=subs)
    calls = []

    def responder(model, system, blocks):
        calls.append(blocks)
        from autograder.schema import ExplanationEvaluation, ExplanationJudgement
        return ExplanationJudgement(evaluations=[
            ExplanationEvaluation(sub_item_id="1", verdict="valid", reasoning="ok")])

    llm = MockBackend(config=BackendConfig(backend="mock", model="m"), responder=responder)
    grade.set_grading_policies({"1": "wrong_choice_zero"})
    try:
        evs = grade.judge_question(llm, q1, ext_q, "A1")
    finally:
        grade.set_grading_policies(None)
    # wrong rows 2 and 3 early-exited: never in the judge payload
    assert len(calls) == 1
    payload_text = "".join(b["text"] for b in calls[0] if b["type"] == "text")
    assert '"sub_item_id": "1"' in payload_text
    assert '"sub_item_id": "2"' not in payload_text and '"sub_item_id": "3"' not in payload_text
    assert "deterministic_zero_wrong_choice" in evs["2"].reasoning
    log = grade.early_exit_log()
    assert not log  # cleared by set_grading_policies(None)


# --------------------------------------------------------- MC resolve chain --


def _gw(local_reads=None, cloud_reads=None, tasks=("mc_resolve", "mc_resolve_cloud")):
    calls = {"mc_resolve": 0, "mc_resolve_cloud": 0}
    reads = {"mc_resolve": list(local_reads or []), "mc_resolve_cloud": list(cloud_reads or [])}

    def factory(cfg):
        task = "mc_resolve" if cfg.model == "local-q" else "mc_resolve_cloud"

        def responder(model, system, blocks):
            calls[task] += 1
            return reads[task].pop(0)
        return MockBackend(config=cfg, responder=responder)

    models = {}
    if "mc_resolve" in tasks:
        models["mc_resolve"] = {"backend": "mock", "model": "local-q"}
    if "mc_resolve_cloud" in tasks:
        models["mc_resolve_cloud"] = {"backend": "mock", "model": "cloud-m"}
    return ModelGateway.from_dict({"models": models}, backend_factory=factory), calls


def _band_png() -> bytes:
    """A real (tiny) band image: the resolver now triages the crop before
    spending model calls, so a stub byte string is rejected as an undecodable
    image — correctly."""
    import numpy as np

    from autograder.tablecrop import _encode_png_gray

    a = np.full((40, 200), 255, dtype=np.uint8)
    a[10:30, 40:60] = 20            # one dark mark
    return _encode_png_gray(a)


PNG = _band_png()


def test_deterministic_confident_never_calls_local():
    """Only ambiguous rows enter the chain: a confident deterministic row is
    settled in extract.py before resolve_row is ever consulted."""
    from autograder import extract
    assert extract.get_mc_resolver() is None   # default: chain not installed


def test_uncertain_row_calls_local_and_agreement_resolves():
    gw, calls = _gw(local_reads=[MCRead(selected="B", candidates=["B", "D"], state="single_mark", confidence="high")])
    res, trace = resolve_row(band_png=PNG, letters=["A", "B", "C", "D"], candidates=["B", "D"], gateway=gw)
    assert calls["mc_resolve"] == 1 and calls["mc_resolve_cloud"] == 0
    assert res.selected == "B" and res.resolved() and res.source == "local_model"


def test_local_unclear_then_cloud_resolves():
    gw, calls = _gw(local_reads=[MCRead(selected=None, candidates=["B", "D"], state="unclear", confidence="low")],
                    cloud_reads=[MCRead(selected="D", candidates=["B", "D"], state="single_mark", confidence="high")])
    res, trace = resolve_row(band_png=PNG, letters=["A", "B", "C", "D"], candidates=["B", "D"], gateway=gw)
    assert calls["mc_resolve"] == 1 and calls["mc_resolve_cloud"] == 1
    assert res.selected == "D" and res.source == "cloud_model"


def test_local_cloud_disagreement_escalates_to_review():
    gw, calls = _gw(local_reads=[MCRead(selected="B", candidates=["B", "D"], state="single_mark", confidence="low")],
                    cloud_reads=[MCRead(selected="D", candidates=["B", "D"], state="single_mark", confidence="high")])
    res, trace = resolve_row(band_png=PNG, letters=["A", "B", "C", "D"], candidates=["B", "D"], gateway=gw)
    assert res.source == "review" and res.selected is None
    assert any(s["stage"] == "conflict" for s in trace.stages)


def test_unresolved_everywhere_is_review_and_stats():
    gw, calls = _gw(local_reads=[MCRead(selected=None, candidates=["B", "D"], state="multiple_marks", confidence="high")],
                    cloud_reads=[MCRead(selected=None, candidates=["B", "D"], state="unclear", confidence="low")])
    res, trace = resolve_row(band_png=PNG, letters=["A", "B", "C", "D"], candidates=["B", "D"], gateway=gw)
    assert res.source == "review"
    st = MCResolverStats()
    st.observe(res, trace, deterministic_only=False)
    st.observe(res, trace, deterministic_only=True)
    d = st.as_dict()
    assert d["rows"] == 2 and d["review_rows"] == 1 and d["local_fallback_rate"] == 50.0


def test_no_gateway_means_review_and_letter_outside_candidates_rejected():
    res, _ = resolve_row(band_png=PNG, letters=["A", "B"], candidates=["A", "B"], gateway=None)
    assert res.source == "review"
    gw, calls = _gw(local_reads=[MCRead(selected="C", candidates=["C"], state="single_mark", confidence="high")],
                    cloud_reads=[MCRead(selected="C", candidates=["C"], state="single_mark", confidence="high")])
    res, _ = resolve_row(band_png=PNG, letters=["A", "B", "C"], candidates=["A", "B"], gateway=gw)
    assert res.source == "review"   # a letter outside deterministic candidates never wins


def test_cloud_disabled_stops_after_local():
    gw, calls = _gw(local_reads=[MCRead(selected=None, candidates=["A", "B"], state="unclear", confidence="low")],
                    cloud_reads=[MCRead(selected="A", candidates=["A"], state="single_mark", confidence="high")])
    res, _ = resolve_row(band_png=PNG, letters=["A", "B"], candidates=["A", "B"], gateway=gw, allow_cloud=False)
    assert calls["mc_resolve_cloud"] == 0 and res.source == "review"
