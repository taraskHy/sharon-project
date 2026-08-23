"""Guards that must hold BEFORE the first paid grading call.

Both were found by a zero-cost preflight audit of the model-selection path:

* the leakage backstop compared only STRING label values, so the grading
  target — a number — was the one field it never checked, and `score` was not
  among the banned label names either;
* scripts/grading_rag_ab.py calls a live model directly, outside the bench
  harness, and its cell list was pre-registered BEFORE the writer splits were
  frozen — two of its five cells landed in CALIBRATION and HELD_OUT, where none
  of the harness guards apply.

No model, network or OCR calls.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.roles import GradeAdapter
from autograder.benchmark.runner import LeakageError, leakage_check

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
real_dataset = pytest.mark.skipif(not (DATASET / "manifest.json").exists(),
                                  reason="grade_primary dataset not built")


# ------------------------------------------------- the numeric label backstop --

@real_dataset
def test_every_real_request_passes_the_leakage_check():
    """The tightened check must not flag the legitimate 67 — a guard that cries
    wolf gets switched off."""
    m = load_manifest("grade_primary")
    ad = GradeAdapter("grade_primary")
    for c in m.cases:
        leakage_check(c, ad.build_request(c.inputs, c.label), ad.model_visible_fields)


@real_dataset
@pytest.mark.parametrize("injected", [3.5, 2.5, 0.5, 1.5, 12.5])
def test_the_grading_target_reaching_the_prompt_is_caught(injected):
    """The number the model is measured against must never be in its prompt."""
    m = load_manifest("grade_primary")
    ad = GradeAdapter("grade_primary")
    c = m.cases[0]
    req = ad.build_request(c.inputs, c.label)
    req.content_blocks[0]["text"] += f"\nHint: the awarded score was {injected}"
    case = dataclasses.replace(c, label={**c.label, "score": injected})
    with pytest.raises(LeakageError, match="'score'"):
        leakage_check(case, req, ad.model_visible_fields)


@real_dataset
def test_a_bookkeeping_count_is_not_mistaken_for_the_target():
    """`lines_no_text_artifact: 0` shares a digit with "Score range: 0..4";
    only the target fields are compared, so this must not fire."""
    m = load_manifest("grade_primary")
    ad = GradeAdapter("grade_primary")
    c = m.cases[0]
    case = dataclasses.replace(c, label={**c.label, "lines_no_text_artifact": 0, "line_count": 4})
    leakage_check(case, ad.build_request(c.inputs, c.label), ad.model_visible_fields)


@real_dataset
def test_the_target_field_names_are_banned_from_the_prompt():
    from autograder.benchmark.runner import _LABEL_NAMES_NEVER_IN_PROMPT
    for name in ("score", "label_score", "ground_truth_score", "owner_note"):
        assert name in _LABEL_NAMES_NEVER_IN_PROMPT
    m = load_manifest("grade_primary")
    ad = GradeAdapter("grade_primary")
    c = m.cases[0]
    req = ad.build_request(c.inputs, c.label)
    req.content_blocks[0]["text"] += "\nthe ground_truth_score is withheld"
    case = dataclasses.replace(c, label={**c.label, "ground_truth_score": 1.0})
    with pytest.raises(LeakageError, match="label field name"):
        leakage_check(case, req, ad.model_visible_fields)


# ------------------------------------------- the out-of-harness HELD_OUT gate --

@real_dataset
def test_the_rag_ab_script_refuses_a_non_dev_cell():
    """It calls a live model outside the bench harness, so nothing else stops it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rag_ab", REPO / "scripts" / "grading_rag_ab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    splits = {r["case_id"]: r["split"] for r in
              (json.loads(l) for l in (DATASET / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip())}
    held_out = next(c for c, s in splits.items() if s == "HELD_OUT")
    calibration = next(c for c, s in splits.items() if s == "CALIBRATION")
    dev = [c for c, s in splits.items() if s == "DEV"][:2]

    for bad in (held_out, calibration):
        with pytest.raises(SystemExit, match="REFUSED"):
            mod.assert_cells_are_development_only([bad])
    mod.assert_cells_are_development_only(dev)            # DEV cells are fine


@real_dataset
def test_the_preregistered_cell_list_still_contains_a_non_dev_cell():
    """Documents WHY the gate exists: the 2026-08-17 pre-registration predates
    the writer splits, so the historical list is not DEV-only. The list is left
    intact as the record; execution is what is gated."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rag_ab", REPO / "scripts" / "grading_rag_ab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    splits = {r["case_id"]: r["split"] for r in
              (json.loads(l) for l in (DATASET / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip())}
    non_dev = {c: splits.get(c) for c in mod.PREREGISTERED_CELLS if splits.get(c) not in (None, "DEV")}
    assert non_dev, "if this is empty the pre-registration was re-done; the gate can then be relaxed"
    with pytest.raises(SystemExit, match="REFUSED"):
        mod.assert_cells_are_development_only(list(mod.PREREGISTERED_CELLS))
