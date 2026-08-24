"""grade-v3: `score` means EXPLANATION QUALITY, not the student's final grade.

grade-v2 asked the model for "the score (within the stated maximum)" and showed
it the pack's final-score composition rules ("explanation weight 0", "no credit
for an answer without an explanation"). The model answered the question it was
asked — the student's FINAL sub-item grade — and with no selection in the
prompt that grade is correctly 0.

Both live smoke runs proved it. Every candidate returned raw 0.0 on every case,
and their own evidence said why:

    "does not identify the correct histogram (G) ... no letter/answer given"
    "does not identify the correct image ... no specific image identified"

Production never wanted that number: it maps `score` through
``reliability._verdict_from_score`` onto an explanation verdict and computes the
final grade itself from the deterministically-resolved selection. v3 makes the
REQUESTED quantity match the CONSUMED quantity.

No provider is contacted anywhere in this file.
"""

from __future__ import annotations

import re

import pytest

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.roles import GradeAdapter
from autograder.escalation import GRADE_SYSTEM, explanation_scale, grade_prompt
from autograder.gradingpack import QuestionGradingPack

MAX = 4.0


def _pack(**kw) -> QuestionGradingPack:
    from autograder.benchmark.roles import pack_from_inputs

    base = {
        "question_id": "1", "question_text": "q", "question_type": "matching_with_explanation",
        "max_score": MAX, "rubric": ["explain the directional blur"], "rubric_items": [],
        "scoring_rules": ["explanation weight 0",
                          "no credit will be given for an answer without an explanation",
                          "answer every question"],
        "grading_policy": "choice_and_explanation_independent",
        "official_solution": {"1": "image F shows 45-degree directional blur"},
        "evidence_policy": "required", "correct_by_version": {},
    }
    base.update(kw)
    return pack_from_inputs(base)


def _prompt(selected=None, transcription="the waves are stretched in one direction",
            version=None, **packkw) -> str:
    blocks = grade_prompt(_pack(**packkw), selected=selected, transcription=transcription,
                          version=version)
    return blocks[0]["text"]


# ---------- 1. the prompt states what `score` means -------------------------


def test_the_system_prompt_says_score_is_explanation_quality_not_the_final_grade():
    s = GRADE_SYSTEM.lower()
    assert "explanation-quality value only" in s
    assert "not the student's final score" in s
    assert "not computing the student's final score" in s


def test_the_user_prompt_repeats_it_next_to_the_scale():
    body = _prompt().lower()
    assert "explanation-quality value" in body
    assert "not the student's final score for this question" in body


def test_the_system_prompt_hands_the_final_score_to_a_downstream_step():
    s = GRADE_SYSTEM.lower()
    assert "deterministic step" in s and "not yours" in s


# ---------- 2. no selection language leaks in -------------------------------

#: phrasings that would make a grader treat a missing selection as a fact about
#: the student rather than a fact about this task
_FORBIDDEN = [
    r"\bno answer\b", r"\bblank answer\b", r"\bwrong choice\b", r"\bwrong answer\b",
    r"\bmust identify\b", r"\bdid not answer\b", r"\bunanswered\b",
    r"student selected option", r"\(none\)",
]


@pytest.mark.parametrize("pattern", _FORBIDDEN)
def test_no_selection_never_produces_answer_shaming_language(pattern):
    body = _prompt(selected=None)
    assert not re.search(pattern, body, re.I), f"{pattern!r} appears in the v3 prompt"


def test_the_selection_is_never_rendered_even_when_supplied():
    """v3 judges the explanation only, so showing the selection can only bias a
    judgement that must not depend on it."""
    with_sel = _prompt(selected="F")
    without = _prompt(selected=None)
    assert with_sel == without
    assert "F\n" not in with_sel.replace("image F", "")


def test_the_final_score_composition_rules_are_not_shown_to_the_grader():
    """`explanation weight 0` told a v2 grader that all points ride on the
    selection. It is a downstream composition rule, not an explanation rubric."""
    body = _prompt()
    assert "explanation weight" not in body
    assert "Scoring rules:" not in body
    assert "without an explanation" not in body


def test_the_grader_is_told_not_to_require_a_letter():
    s = GRADE_SYSTEM.lower()
    assert "do not require the explanation to name" in s
    assert "unless a rubric item explicitly demands it" in s


def test_absence_of_a_selection_is_explicitly_declared_uninformative():
    s = GRADE_SYSTEM.lower()
    assert "never lower your judgement because no selection appears" in s


# ---------- 3/4/5. the three encodings --------------------------------------


def test_a_valid_explanation_may_score_max_with_no_selection():
    body = _prompt(selected=None)
    assert f"  {MAX:g}  = valid" in body
    # nothing in the prompt conditions the max on a selection being present
    scale = body.split("using exactly one of:")[1]
    assert "select" not in scale.lower() and "option" not in scale.lower()


def test_a_partial_explanation_maps_to_half_max():
    assert f"  {MAX / 2:g}  = partially valid" in _prompt()


def test_an_invalid_explanation_maps_to_zero():
    assert "  0  = invalid" in _prompt()


def test_the_scale_offers_exactly_three_values():
    lines = [l for l in explanation_scale(MAX).splitlines() if l.strip()]
    assert len(lines) == 3
    assert [l.strip().split()[0] for l in lines] == ["0", "2", "4"]


def test_the_scale_follows_max_score():
    lines = [l.strip().split()[0] for l in explanation_scale(10.0).splitlines() if l.strip()]
    assert lines == ["0", "5", "10"]


def test_the_three_values_are_exactly_the_ones_production_can_distinguish():
    """Any other number is silently collapsed by _verdict_from_score, so the
    scale must name the values that survive."""
    from autograder.benchmark.verdicts import verdict_from_model_score

    assert verdict_from_model_score(0.0, MAX) == "invalid"
    assert verdict_from_model_score(MAX / 2, MAX) == "partially_valid"
    assert verdict_from_model_score(MAX, MAX) == "valid"


# ---------- 6. production composition unchanged ------------------------------


def test_production_score_to_final_grade_is_unchanged():
    from autograder.benchmark.verdicts import final_score_for, verdict_from_model_score

    for raw, sel, expected in [
        (4.0, True, 4.0), (2.0, True, 2.0), (0.0, True, 0.0),
        (4.0, False, 0.0), (2.0, False, 0.0), (0.0, False, 0.0),
    ]:
        v = verdict_from_model_score(raw, MAX)
        assert final_score_for(selection_correct=sel, verdict=v, max_points=MAX) == expected


def test_the_verdict_factor_table_is_unchanged():
    from autograder.config import GraderConfig
    from autograder.grade import _verdict_factor

    cfg = GraderConfig()
    assert (_verdict_factor("valid", cfg), _verdict_factor("partially_valid", cfg),
            _verdict_factor("invalid", cfg)) == (1.0, 0.5, 0.0)


# ---------- 7. one conversion, shared -----------------------------------------


def test_benchmark_and_production_use_the_same_conversion():
    from autograder.benchmark.verdicts import verdict_from_model_score
    from autograder.reliability import _verdict_from_score

    for i in range(0, 401):
        s = i / 100
        assert verdict_from_model_score(s, MAX) == _verdict_from_score(s, MAX)


# ---------- 8. no label leakage on the real frozen cases ----------------------


@pytest.fixture(scope="module")
def manifest():
    return load_manifest("grade_primary")


def _user_text(adapter, case) -> str:
    """Only the per-case content the model is shown. The SYSTEM prompt is fixed
    instruction text and legitimately talks ABOUT selections and scores; a leak
    test that scans it would flag its own wording."""
    req = adapter.build_request(dict(case.inputs), None)
    return "\n".join(b["text"] for b in req.content_blocks if b.get("type") == "text")


def test_no_instructor_or_target_label_leaks_into_any_frozen_request(manifest):
    """Every case in the dataset, not just the smoke ones."""
    from autograder.benchmark.runner import leakage_check

    adapter = GradeAdapter()
    #: label FIELD NAMES that would betray the evaluation side if rendered
    banned = ("explanation_verdict", "explanation_verdict_derivable",
              "selection_correct", "marked_option", "instructor",
              "partially_valid", "label_score", "final_labels")
    for case in manifest.cases:
        req = adapter.build_request(dict(case.inputs), None)
        leakage_check(case, req, adapter.model_visible_fields)   # raises on leak
        body = _user_text(adapter, case)
        low = body.lower()
        for key in banned:
            assert key not in low, f"{case.case_id}: {key!r} in the per-case prompt"
        # and the case's own instructor score must not appear as a labelled value
        assert "score:" not in low and "grade:" not in low


def test_the_marked_option_from_the_selection_audit_never_reaches_a_prompt(manifest):
    """The audit recorded letters like A/D/H/I. None may enter a request."""
    import json
    from pathlib import Path

    p = Path("evaluation/model_selection/datasets/grade_primary/selection_audit.json")
    if not p.exists():
        pytest.skip("no selection audit in this checkout")
    audit = json.loads(p.read_text(encoding="utf-8"))["decisions"]
    adapter = GradeAdapter()
    by_id = {c.case_id: c for c in manifest.cases}
    for cid, entry in audit.items():
        case = by_id.get(cid)
        if case is None:
            continue
        body = _user_text(adapter, case).lower()
        assert "student selected option" not in body, cid
        letter = (entry.get("selected_option") or "").strip().lower()
        # the audited letter must not be rendered as this case's selection
        assert f"selected option: {letter}" not in body, cid


# ---------- 9. production / benchmark request parity --------------------------


def test_the_benchmark_request_is_the_production_request(manifest):
    """The adapter must not build its own variant: same system prompt object,
    same block builder, same output model."""
    from autograder.escalation import GradeResult

    case = next(c for c in manifest.cases if c.split == "DEV")
    adapter = GradeAdapter()
    req = adapter.build_request(dict(case.inputs), None)

    from autograder.benchmark.roles import pack_from_inputs

    pack = pack_from_inputs(case.inputs["pack"])
    expected = grade_prompt(pack, selected=case.inputs.get("selected"),
                            transcription=case.inputs["transcription"],
                            version=case.inputs.get("version"))
    assert req.system is GRADE_SYSTEM
    assert req.content_blocks == expected
    assert req.output_model is GradeResult
    assert req.prompt_version == "grade-v3"


def test_the_prompt_version_is_declared_consistently():
    import tomllib
    from pathlib import Path

    assert GradeAdapter.prompt_version == "grade-v3"
    for name in ("models.toml", "models.example.toml"):
        p = Path(name)
        if not p.exists():
            continue
        cfg = tomllib.loads(p.read_text(encoding="utf-8"))
        for task in ("grade_primary", "grade_escalate"):
            assert cfg["models"][task]["prompt_version"] == "grade-v3", f"{name}:{task}"


def test_the_adapter_version_moved_with_the_scoring_semantics():
    assert GradeAdapter.adapter_version == "grade-bench-v2"
