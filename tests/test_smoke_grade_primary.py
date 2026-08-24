"""The GRADE_PRIMARY smoke subset — the first cases any grading model is paid to see.

The point of freezing is that nobody can pick the cases after seeing a result.

As of 2026-08-24 the slots are the three canonical EXPLANATION VERDICT classes
(the model's actual responsibility), not the old final-score buckets. The v2
score-bucket subset was frozen against a benchmark target since classified
INVALID FOR MODEL SELECTION; it is preserved beside the smoke directory under
its SUPERSEDED name and is deliberately not loaded here.

The selection-correctness audit completed 8/8 on 2026-08-25 and produced NO
`invalid` label: every DEV zero-score case had a WRONG selection, so its
explanation was never the reason for the zero. That makes the missing class
STRUCTURAL rather than pending — on this dataset the verdict benchmark is a
two-class problem. These tests assert that state rather than pretending
otherwise, and will fail loudly if a dataset ever does contain an invalid case.

Offline: no model, network or OCR calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.smoke import (DEFAULT_SMOKE_ROOT, SMOKE_DIVERSITY, SMOKE_ORDER,
                                        SMOKE_RULES, SmokeError, freeze_smoke, load_smoke,
                                        propose_smoke, smoke_case_ids)

ROLE = "grade_primary"
SLOTS = ["verdict_invalid", "verdict_partially_valid", "verdict_valid"]


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(ROLE)


@pytest.fixture(scope="module")
def proposal(manifest):
    return propose_smoke(ROLE, manifest)


@pytest.fixture(scope="module")
def frozen(manifest):
    try:
        return load_smoke(ROLE, manifest)
    except SmokeError as e:
        pytest.skip(f"no frozen verdict-target smoke subset yet: {e}")


def _by_id(manifest):
    return {c.case_id: c for c in manifest.cases}


def _verdict(case):
    return (case.label.get("explanation_verdict")
            if case.label.get("explanation_verdict_derivable") else None)


# ------------------------------------------------------------------ rules ----


def test_the_slots_are_the_three_canonical_verdict_classes():
    assert [slot for slot, _, _ in SMOKE_RULES[ROLE]] == SLOTS


def test_every_slot_matches_its_production_factor():
    """One slot per distinct verdict factor: 0, 0.5, 1."""
    from autograder.benchmark.verdicts import factor_for

    assert factor_for("invalid") == 0.0
    assert factor_for("partially_valid") == 0.5
    assert factor_for("valid") == 1.0


def test_a_slot_never_accepts_a_case_whose_verdict_is_not_derivable(manifest):
    """A zero-score case with an unknown selection has no explanation ground
    truth. Letting it fill `verdict_invalid` would put a fabricated label in
    front of the first paid call."""
    dev = manifest.by_split("DEV", None)
    undecidable = [c for c in dev if not c.label.get("explanation_verdict_derivable")]
    assert undecidable, "expected DEV cases with no derivable verdict"
    for _slot, _why, pred in SMOKE_RULES[ROLE]:
        for c in undecidable:
            assert not pred(c), f"{c.case_id} must not qualify for any slot"


def test_the_proposal_is_dev_only(proposal):
    assert proposal["split"] == "DEV"
    assert all(c["split"] == "DEV" for c in proposal["cases"])


def test_the_proposal_reports_slots_it_could_not_fill(proposal):
    filled = {c["slot"] for c in proposal["cases"]}
    unfilled = set(proposal["unfilled_slots"])
    assert filled | unfilled == set(SLOTS)
    assert not (filled & unfilled)


def test_each_proposed_case_really_carries_its_slots_verdict(manifest, proposal):
    by_id = _by_id(manifest)
    want = {"verdict_invalid": "invalid", "verdict_partially_valid": "partially_valid",
            "verdict_valid": "valid"}
    for c in proposal["cases"]:
        assert _verdict(by_id[c["case_id"]]) == want[c["slot"]], c


def test_it_spans_distinct_students(manifest, proposal):
    by_id = _by_id(manifest)
    writers = [SMOKE_DIVERSITY[ROLE](by_id[c["case_id"]]) for c in proposal["cases"]]
    assert len(set(writers)) == len(writers), f"a writer repeats: {writers}"


def test_it_takes_the_hardest_case_in_each_class_not_the_easiest(manifest, proposal):
    """SMOKE_ORDER puts the LONGEST answer first: the shortest answer in a
    class is usually a fragment that tests nothing."""
    by_id = _by_id(manifest)
    order = SMOKE_ORDER[ROLE]
    chosen = {c["slot"]: by_id[c["case_id"]] for c in proposal["cases"]}
    dev = manifest.by_split("DEV", None)
    used_writers: set = set()
    for slot, _why, pred in SMOKE_RULES[ROLE]:
        if slot not in chosen:
            continue
        picked = chosen[slot]
        eligible = [c for c in dev if pred(c)
                    and SMOKE_DIVERSITY[ROLE](c) not in used_writers]
        assert picked is min(eligible, key=order), slot
        used_writers.add(SMOKE_DIVERSITY[ROLE](picked))


def test_the_proposal_is_reproducible_from_the_rules_alone(manifest, proposal):
    assert propose_smoke(ROLE, manifest) == proposal


# --------------------------------------------------- the freeze gate today ----


def test_the_invalid_class_has_no_support_in_this_dataset(manifest):
    """The selection audit completed 8/8 on 2026-08-25 and produced NO invalid
    label. That is structural, not pending:

    an `invalid` verdict requires instructor score 0 AND a correct selection.
    Every DEV zero-score case turned out to have a WRONG selection, so the
    explanation was never the reason for the zero and the case carries no
    explanation ground truth at all.

    Consequence: on this dataset the verdict benchmark is a TWO-class problem.
    Whether a grader can withhold credit from a bad explanation attached to a
    CORRECT choice is not measurable here, and no further auditing changes
    that — it needs exam data containing such a case.
    """
    dev = manifest.by_split("DEV", None)
    assert not [c for c in dev if _verdict(c) == "invalid"]
    assert {_verdict(c) for c in dev} == {"valid", "partially_valid", None}


def test_the_invalid_slot_is_therefore_unfilled(proposal):
    """Asserted rather than described, so a future dataset that DOES contain
    an invalid case makes this test fail loudly instead of passing silently."""
    if "verdict_invalid" not in proposal["unfilled_slots"]:
        pytest.skip("an invalid case now exists; the slot is filled")
    assert "verdict_invalid" in proposal["unfilled_slots"]


def test_freezing_refuses_while_a_slot_is_unfilled(tmp_path, manifest, proposal):
    """Freezing an incomplete subset would hide the gap behind a hash."""
    if not proposal["unfilled_slots"]:
        pytest.skip("every slot is filled; nothing to refuse")
    with pytest.raises(SmokeError, match="could not be filled"):
        freeze_smoke(ROLE, manifest, tmp_path)


def test_an_incomplete_subset_can_still_be_frozen_deliberately(tmp_path, manifest, proposal):
    """...but only on purpose, and the gap is recorded in the frozen file."""
    if not proposal["unfilled_slots"]:
        pytest.skip("every slot is filled")
    d = freeze_smoke(ROLE, manifest, tmp_path, allow_unfilled=True)
    assert d["unfilled_slots"] == proposal["unfilled_slots"]
    assert json.loads((tmp_path / f"{ROLE}_smoke.json").read_text(encoding="utf-8"))


def test_the_superseded_score_bucket_subset_is_preserved_not_deleted():
    p = DEFAULT_SMOKE_ROOT / f"{ROLE}_smoke.SUPERSEDED_final_score_target_2026-08-23.json"
    if not p.exists():
        pytest.skip("superseded subset not present in this checkout")
    old = json.loads(p.read_text(encoding="utf-8"))
    assert old["selection_sha256"] == \
        "ad104dbe3e9171c485dd9fe4e6b109880d574883ec4a0e59e33bd35712725cbd"
    assert [c["slot"] for c in old["cases"]] == ["no_credit", "partial_credit", "full_credit"]


# ------------------------------------------------- once a subset is frozen ----


def test_the_frozen_subset_is_exactly_what_the_rules_produce(manifest, frozen):
    fresh = propose_smoke(ROLE, manifest)
    assert [c["case_id"] for c in frozen["cases"]] == [c["case_id"] for c in fresh["cases"]]
    assert frozen["selection_sha256"] == fresh["selection_sha256"]


def test_the_frozen_subset_is_dev_only_and_covers_every_class(manifest, frozen):
    assert all(c["split"] == "DEV" for c in frozen["cases"])
    assert not frozen["unfilled_slots"], frozen["unfilled_slots"]
    assert sorted(c["slot"] for c in frozen["cases"]) == sorted(SLOTS)


def test_a_tampered_frozen_file_is_refused(tmp_path, manifest, frozen):
    d = json.loads(json.dumps(frozen))
    d["cases"][0]["case_id"] = "e999_q9_r9"
    root = tmp_path / "smoke"
    root.mkdir()
    (root / f"{ROLE}_smoke.json").write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SmokeError, match="hash mismatch"):
        load_smoke(ROLE, manifest, root)


def test_the_runner_resolves_the_same_ids(manifest, frozen):
    assert sorted(smoke_case_ids(ROLE, manifest)) == \
        sorted(c["case_id"] for c in frozen["cases"])


def test_the_diversity_hook_does_not_disturb_the_roles_that_do_not_use_it(manifest):
    assert "ocr_primary" not in SMOKE_DIVERSITY
    assert "ocr_verify" not in SMOKE_DIVERSITY
