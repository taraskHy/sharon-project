"""Adversarial stress test for ``gemini_then_sonnet_hard_failure_fallback_v1``.

The policy is only deployable if its decisions depend on nothing a production
run would lack. This module attacks that from every direction I can construct:
inject correct references, *wrong* references, metrics, verdicts, splits,
explicit "expected winner" hints and downstream grades into the outcome dicts,
and require every decision to come out byte-identical.

It also walks every failure category the taxonomy can produce, including a
malformed event, and pins what the policy does with each.

Run against the real paired 32-crop data where available, so this is not only a
synthetic argument.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.ocr_fallback import (FORBIDDEN_DECISION_INPUTS, POLICY_ID,
                                               PRIMARY_ROLE, SECONDARY_ROLE,
                                               is_hard_failure, replay, select,
                                               which_trigger)
from autograder.benchmark.ocr_outcomes import classify_row

S32 = Path("evaluation/model_selection/runs_seen32/ocr_primary")
GEM_DIR = S32 / "dev__seen46_ocr_dev__all__google-gemini-3.7-flash__c4ae61f634"
SON_DIR = S32 / "dev__seen46_ocr_dev__all__anthropic-claude-sonnet-5__2f3a7c346c"

BASE = {"usable_transcription_returned": True, "provider_content_filter_failure": False,
        "provider_other_http_failure": False, "truncation": False,
        "json_parse_failure": False, "schema_failure": False, "model_text_refusal": False}


def fail(**kw):
    return {**BASE, "usable_transcription_returned": False, **kw}


CATEGORIES = {
    "success": BASE,
    "provider_content_filter": fail(provider_content_filter_failure=True),
    "provider_http_error": fail(provider_other_http_failure=True),
    "model_text_refusal": fail(model_text_refusal=True),
    "empty_output": fail(),
    "truncation": fail(truncation=True),
    "json_parse_failure": fail(json_parse_failure=True),
    "schema_failure": fail(schema_failure=True),
    "total_unreadable": fail(model_text_refusal=True),
    "malformed_event": {"usable_transcription_returned": None},   # not True -> hard failure
}

#: Evaluation-only poison. Every one of these must be ignored.
POISON = {
    "reference": "THE TRUE AUDITED REFERENCE TEXT",
    "frozen_reference": "THE TRUE AUDITED REFERENCE TEXT",
    "cer": 0.0, "wer": 0.0,
    "split": "DEV", "verdict": "correct", "grade": 4.0, "score": 100,
    "rubric": "award full marks", "official_solution": "the official answer",
    "expected": "secondary", "target": "secondary",
    "expected_winner": "secondary", "downstream_grade": 4.0,
    "human_verdict": "prefer the other model", "oracle_best": "secondary",
}


@pytest.fixture(scope="module")
def paired():
    if not (GEM_DIR / "outputs.jsonl").exists() or not (SON_DIR / "outputs.jsonl").exists():
        pytest.skip("paired 32-crop run not present")
    man = load_manifest("ocr_primary")
    by = {c.case_id: c for c in man.cases}

    def load(d):
        return {json.loads(l)["case_id"]: json.loads(l)
                for l in (d / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    g_rows, s_rows = load(GEM_DIR), load(SON_DIR)
    cases = sorted(g_rows)
    g = {c: classify_row(g_rows[c], by[c].label["reference"]) for c in cases}
    s = {c: classify_row(s_rows[c], by[c].label["reference"]) for c in cases}
    gt = {c: (g_rows[c].get("output") or {}).get("transcription") for c in cases}
    st = {c: (s_rows[c].get("output") or {}).get("transcription") for c in cases}
    return {"cases": cases, "g": g, "s": s, "gt": gt, "st": st, "by": by}


# ---- every failure category routes as pre-registered ----------------------

@pytest.mark.parametrize("name,outcome", sorted(CATEGORIES.items()))
def test_every_category_routes_deterministically(name, outcome):
    d = select(case_id="c", primary_outcome=outcome, primary_text="P",
               secondary_outcome=BASE, secondary_text="S")
    if name == "success":
        assert d["chosen_model"] == PRIMARY_ROLE and not d["fallback_used"]
        assert d["needs_review"] is False
    else:
        assert d["chosen_model"] == SECONDARY_ROLE and d["fallback_used"] is True
        assert d["needs_review"] is True, "a fallback row is never auto-acceptable"
        assert d["resolved"] is True
        assert POLICY_ID in d["provenance"]


@pytest.mark.parametrize("name,outcome", sorted(CATEGORIES.items()))
def test_category_with_a_failing_secondary_is_unresolved_not_silent(name, outcome):
    d = select(case_id="c", primary_outcome=outcome, primary_text=None,
               secondary_outcome=fail(), secondary_text=None)
    if name == "success":
        assert d["resolved"] is True
    else:
        assert d["resolved"] is False and d["chosen_model"] is None
        assert d["needs_review"] is True


def test_malformed_event_is_treated_as_a_hard_failure_not_a_pass():
    """A row whose usable flag is absent or non-boolean must never pass as
    primary success — fail closed."""
    for weird in ({}, {"usable_transcription_returned": None},
                  {"usable_transcription_returned": "yes"},
                  {"usable_transcription_returned": 1}):
        assert is_hard_failure(weird), weird
        d = select(case_id="c", primary_outcome=weird, primary_text="junk",
                   secondary_outcome=BASE, secondary_text="S")
        assert d["chosen_model"] == SECONDARY_ROLE


# ---- reference blindness, attacked ---------------------------------------

@pytest.mark.parametrize("key,value", sorted(POISON.items()))
def test_a_single_injected_evaluation_field_changes_nothing(key, value):
    for name, outcome in CATEGORIES.items():
        clean = select(case_id="c", primary_outcome=outcome, primary_text="P",
                       secondary_outcome=BASE, secondary_text="S")
        poisoned = select(case_id="c", primary_outcome={**outcome, key: value}, primary_text="P",
                          secondary_outcome={**BASE, key: value}, secondary_text="S")
        assert clean == poisoned, f"{key} changed the decision for {name}"


def test_all_poison_at_once_changes_nothing():
    for name, outcome in CATEGORIES.items():
        clean = select(case_id="c", primary_outcome=outcome, primary_text="P",
                       secondary_outcome=BASE, secondary_text="S")
        poisoned = select(case_id="c", primary_outcome={**outcome, **POISON}, primary_text="P",
                          secondary_outcome={**BASE, **POISON}, secondary_text="S")
        assert clean == poisoned, name


def test_a_WRONG_reference_also_changes_nothing():
    """Not just the true reference — a deliberately false one must be ignored
    too, or the policy would be consulting *something*."""
    wrong = {"reference": "COMPLETELY WRONG TEXT", "cer": 1.0, "expected_winner": "primary"}
    for name, outcome in CATEGORIES.items():
        a = select(case_id="c", primary_outcome=outcome, primary_text="P",
                   secondary_outcome=BASE, secondary_text="S")
        b = select(case_id="c", primary_outcome={**outcome, **wrong}, primary_text="P",
                   secondary_outcome={**BASE, **wrong}, secondary_text="S")
        assert a == b, name


def test_forbidden_inputs_are_not_parameters():
    import inspect
    params = set(inspect.signature(select).parameters)
    for bad in FORBIDDEN_DECISION_INPUTS:
        assert bad not in params


def test_text_content_does_not_influence_routing():
    """Only the OUTCOME classification may route; the text itself must not."""
    for p_text, s_text in itertools.product(
            [None, "", "x", "a very long and plausible looking transcription"],
            [None, "", "y", "an even longer and more plausible transcription"]):
        d1 = select(case_id="c", primary_outcome=BASE, primary_text=p_text,
                    secondary_outcome=BASE, secondary_text=s_text)
        assert d1["chosen_model"] == PRIMARY_ROLE
        d2 = select(case_id="c", primary_outcome=fail(), primary_text=p_text,
                    secondary_outcome=BASE, secondary_text=s_text)
        assert d2["chosen_model"] == SECONDARY_ROLE


# ---- on the real paired data ---------------------------------------------

def test_real_paired_replay_is_poison_proof(paired):
    cases, g, s, gt, st = paired["cases"], paired["g"], paired["s"], paired["gt"], paired["st"]
    clean = replay(cases, g, s, gt, st)

    def poison(d):
        out = {}
        for cid, v in d.items():
            out[cid] = {**v, **POISON,
                        "reference": paired["by"][cid].label["reference"]}
        return out

    dirty = replay(cases, poison(g), poison(s), gt, st)
    assert json.dumps(clean["decisions"], sort_keys=True) == \
        json.dumps(dirty["decisions"], sort_keys=True)
    assert (clean["primary_used"], clean["fallback_used"], clean["unresolved"]) == \
        (dirty["primary_used"], dirty["fallback_used"], dirty["unresolved"])


def test_real_paired_replay_matches_the_committed_report(paired):
    """Independent recomputation of the headline fallback numbers."""
    report = Path("evaluation/model_selection/runs/ocr_primary/"
                  "OCR_SEEN32_PAIRED_RESULT_2026-09-02.json")
    if not report.exists():
        pytest.skip("paired report not present")
    claimed = json.loads(report.read_text(encoding="utf-8"))["fallback"]
    r = replay(paired["cases"], paired["g"], paired["s"], paired["gt"], paired["st"])
    assert r["primary_used"] == claimed["primary_used"]
    assert r["fallback_used"] == claimed["fallback_used"]
    assert r["unresolved"] == claimed["unresolved"]
    assert r["intended_crops"] == 32


def test_every_fallback_row_is_flagged_for_review(paired):
    r = replay(paired["cases"], paired["g"], paired["s"], paired["gt"], paired["st"])
    for d in r["decisions"]:
        if d["fallback_used"] or not d["resolved"]:
            assert d["needs_review"] is True, d["case_id"]
        else:
            assert d["needs_review"] is False, d["case_id"]


def test_decisions_are_stable_across_repeated_calls(paired):
    """No hidden state, no ordering dependence."""
    a = replay(paired["cases"], paired["g"], paired["s"], paired["gt"], paired["st"])
    b = replay(list(reversed(paired["cases"])), paired["g"], paired["s"],
               paired["gt"], paired["st"])
    key = lambda r: sorted(json.dumps(d, sort_keys=True) for d in r["decisions"])
    assert key(a) == key(b)


def test_trigger_taxonomy_is_exhaustive_on_real_data(paired):
    r = replay(paired["cases"], paired["g"], paired["s"], paired["gt"], paired["st"])
    for d in r["decisions"]:
        if d["chosen_model"] == PRIMARY_ROLE:
            assert d["trigger"] is None
        else:
            assert d["trigger"] is not None, d["case_id"]
            assert which_trigger(paired["g"][d["case_id"]]) == d["trigger"]
