"""No selection -> no "Student selected option" line.

The prompt used to render a missing selection as "(none)". A grader reads that
as "the student left the answer blank", which is a WRONG choice under several
grading policies — so an unresolved or not-applicable selection silently became
a wrong one, depressing the score for a reason that has nothing to do with the
answer being graded.

The frozen 67-case GRADE_PRIMARY benchmark carries selected=None on every case
by construction (manifest policy: "model-visible: pack (...), selected=None,
frozen audited transcription, version=None"), so this affected all 67.

A genuinely blank multiple-choice response is a DIFFERENT fact and must get its
own explicit state; these tests pin that None is not it.

Offline: no model, network or OCR calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autograder.benchmark.roles import GradeAdapter, pack_from_inputs
from autograder.escalation import GradeResult, grade_prompt, validate_grade
from tests.test_escalation import _pack

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
MARKER = "Student selected option"


def _text(pack, selected, *, transcription="הסבר כלשהו", version=None) -> str:
    return grade_prompt(pack, selected=selected, transcription=transcription,
                        version=version)[0]["text"]


# --------------------------------------------------------- 1. absent -> no line


@pytest.mark.parametrize("selected", [None, ""])
def test_no_selection_means_the_line_is_absent_entirely(selected):
    text = _text(_pack(), selected)
    assert MARKER not in text, "the selection line must be omitted, not rendered empty"
    assert "(none)" not in text.split("Allowed rubric item ids")[0], (
        "'(none)' must not appear where a selection would have been")


def test_the_omission_leaves_the_rest_of_the_prompt_intact():
    """Only that one line goes; the grader still gets everything it needs."""
    pack = _pack()
    with_sel = _text(pack, "C")
    without = _text(pack, None)
    assert without == with_sel.replace(f"{MARKER}: C\n", "")
    for required in ("Student explanation", "Allowed rubric item ids",
                     f"Score range: 0..{pack.max_score:g}", pack.question_text):
        assert required in without


# ------------------------------------------------------- 2. present -> rendered


@pytest.mark.parametrize("selected", ["C", "A", "I", "B"])
def test_a_real_selection_is_still_shown(selected):
    text = _text(_pack(), selected)
    assert f"{MARKER}: {selected}" in text


def test_a_real_selection_survives_alongside_the_correct_option_line():
    """version set -> the correct-option line appears; the selection must too."""
    pack = _pack()
    text = _text(pack, "C", version="A1")
    assert f"{MARKER}: C" in text
    assert "Correct option(s) for this exam version" in text


# ------------------------------- 3. None is not blank / wrong-choice semantics


def test_none_does_not_trigger_wrong_choice_zero():
    """`selection_correct is False` is the wrong-choice trigger. None is not False.

    If None were treated as a blank (= wrong) answer, a full-credit explanation
    would be invalidated under wrong_choice_zero.
    """
    wcz = _pack("wrong_choice_zero")
    unresolved = validate_grade(GradeResult(score=4, rubric_items_met=[]), wcz,
                                selection_correct=None, selected=None, transcription="הסבר")
    assert unresolved.ok, f"an unresolved selection was scored as wrong: {unresolved.problems}"

    actually_wrong = validate_grade(GradeResult(score=4, rubric_items_met=[]), wcz,
                                    selection_correct=False, selected="Z", transcription="הסבר")
    assert not actually_wrong.ok, "a genuinely wrong choice must still be caught"
    assert any("wrong_choice_zero" in p for p in actually_wrong.problems)


def test_no_blank_or_empty_wording_reaches_the_grader():
    """Nothing in the prompt may suggest the student answered nothing."""
    text = _text(_pack(), None).lower()
    head = text.split("student explanation")[0]
    for word in ("(none)", "blank", "no answer", "did not select", "unanswered", "left empty"):
        assert word not in head, f"the prompt implies a blank response via {word!r}"


# ------------------------------------------- the 67 frozen cases, all of them


def _dataset_cases() -> list[dict]:
    return [json.loads(l) for l in
            (DATASET / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def test_none_of_the_67_frozen_cases_mentions_a_selection():
    cases = _dataset_cases()
    assert len(cases) == 67
    adapter = GradeAdapter()
    for c in cases:
        assert c.get("selected") is None, f"{c['case_id']}: dataset changed; it now carries a selection"
        text = adapter.build_request(dict(c), DATASET).content_blocks[0]["text"]
        assert MARKER not in text, f"{c['case_id']}: selection line leaked into the prompt"


def test_the_67_cases_still_carry_everything_the_grader_needs():
    """Removing a line must not remove the question, rubric or answer."""
    adapter = GradeAdapter()
    for c in _dataset_cases():
        pack = pack_from_inputs(c["pack"])
        text = adapter.build_request(dict(c), DATASET).content_blocks[0]["text"]
        assert c["transcription"] in text, f"{c['case_id']}: the student answer is missing"
        assert pack.question_text.strip()[:40] in text, f"{c['case_id']}: the question is missing"
        assert f"Score range: 0..{pack.max_score:g}" in text
        for rid in pack.rubric_item_ids():
            assert rid in text, f"{c['case_id']}: rubric id {rid} is missing"


def test_the_leakage_check_still_passes_on_every_frozen_case():
    """The prompt change must not have opened a path for label information."""
    from autograder.benchmark.manifests import load_manifest
    from autograder.benchmark.runner import leakage_check

    adapter = GradeAdapter()
    manifest = load_manifest("grade_primary")
    assert len(manifest.cases) == 67
    for case in manifest.cases:
        request = adapter.build_request(dict(case.inputs), DATASET)
        leakage_check(case, request, adapter.model_visible_fields)   # raises on a leak


def test_the_prompt_version_was_bumped_with_the_prompt():
    """A changed prompt under an unchanged version would make two different
    runs look comparable in the artifacts."""
    assert GradeAdapter.prompt_version == "grade-v2"
    import tomllib
    for name in ("models.example.toml", "models.toml"):
        p = REPO / name
        if not p.exists():
            continue
        cfg = tomllib.loads(p.read_text(encoding="utf-8"))
        for task in ("grade_primary", "grade_escalate"):
            assert cfg["models"][task]["prompt_version"] == "grade-v2", f"{name}:{task}"
