"""The explicit OCR outcome taxonomy, and the prospective fallback policy.

Two reporting failures motivate this module:

* Stage-1c's ``metrics.json`` called three provider content-filter outcomes
  "schema failures".
* The Stage-1c summary table reported Gemini ``Refusals = 0`` on that same arm.
  Only true of model-TEXT refusals; a reader sees "zero operational refusals".

And one design requirement: the fallback must be deployable, which means it can
never consult a reference. That is enforced by the signature and proved below.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.ocr_fallback import (FORBIDDEN_DECISION_INPUTS, POLICY_ID,
                                               PRIMARY_ROLE, SECONDARY_ROLE, is_hard_failure,
                                               replay, select, which_trigger)
from autograder.benchmark.ocr_outcomes import (OUTCOME_FIELDS, classify_row,
                                               is_bare_marker, reference_is_readable,
                                               summarize)
from autograder.benchmark.ocr_views import is_handwritten

R = Path("evaluation/model_selection/runs/ocr_primary")
S1C = Path("evaluation/model_selection/runs_stage1c/ocr_primary/"
           "dev__smoke__all__google-gemini-3.7-flash__45297cdd83")
LUNA = R / "dev__smoke__all__openai-gpt-5.6-luna-pro__c6f10f3603"
SONNET = R / "dev__smoke__all__anthropic-claude-sonnet-5__0481873207"
GEM_S1 = R / "dev__smoke__all__google-gemini-3.7-flash__feceaa6084"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest("ocr_primary")


def _rows(d):
    if not (d / "outputs.jsonl").exists():
        pytest.skip(f"{d} not present")
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in (d / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


def _tax(d, manifest):
    by = {c.case_id: c for c in manifest.cases}
    return {cid: classify_row(r, by[cid].label["reference"]) for cid, r in _rows(d).items()}


# ---- taxonomy ------------------------------------------------------------

def test_content_filter_is_not_a_schema_failure(manifest):
    t = _tax(S1C, manifest)
    cf = [c for c, v in t.items() if v["provider_content_filter_failure"]]
    assert len(cf) == 3
    for c in cf:
        assert t[c]["schema_failure"] is False
        assert t[c]["json_parse_failure"] is False
        assert t[c]["truncation"] is False
        assert t[c]["model_text_refusal"] is False, "the model never spoke"
        assert t[c]["total_line_loss"] is True


def test_a_content_filtered_row_received_a_body_but_did_not_complete(manifest):
    """The distinction the Stage-1c wording blurred."""
    t = _tax(S1C, manifest)
    cf = [v for v in t.values() if v["provider_content_filter_failure"]]
    assert cf
    for v in cf:
        assert v["provider_http_response_received"] is True
        assert v["provider_request_completed"] is False


def test_gemini_stage1c_has_zero_model_text_refusals_but_three_provider_failures(manifest):
    """Both halves of the sentence that must never be shortened to 'Refusals = 0'."""
    s = summarize(_tax(S1C, manifest))
    assert s["model_text_refusal"] == 0
    assert s["provider_content_filter_failure"] == 3
    assert s["usable_transcription_returned"] == 5


def test_luna_bare_unreadable_is_a_model_text_refusal_not_coverage(manifest):
    t = _tax(LUNA, manifest)
    hw = [c for c in t if is_handwritten(c)]
    s = summarize(t, hw)
    assert s["model_text_refusal"] == 4, "Luna declined 4 of 5 handwritten crops in text"
    assert s["usable_transcription_returned"] == 1
    assert s["provider_request_completed"] == 5, "every request completed — that is the point"
    assert s["usable_coverage"] == "1/5"


def test_stage1_gemini_http400s_are_other_http_failures(manifest):
    s = summarize(_tax(GEM_S1, manifest))
    assert s["provider_other_http_failure"] == 8
    assert s["provider_http_response_received"] == 0
    assert s["provider_content_filter_failure"] == 0
    assert s["schema_failure"] == 0


def test_a_reference_that_is_itself_unreadable_makes_a_marker_not_a_refusal():
    """hc_e002_q2_r6's audited reference ends in [לא קריא]; a model flagging the
    same word agrees with the auditor rather than refusing."""
    assert reference_is_readable("יש טשטוש בכל התדרים") is True
    assert reference_is_readable("... להכפלה ב-2 [לא קריא]") is False
    row = {"ok": True, "output": {"transcription": "[?]"}}
    assert classify_row(row, "... [לא קריא]")["model_text_refusal"] is False
    assert classify_row(row, "יש טשטוש בכל התדרים")["model_text_refusal"] is True


def test_bare_marker_detection():
    assert is_bare_marker("[unreadable]") and is_bare_marker("  [?] ")
    assert not is_bare_marker("יש טשטוש [?]"), "a partial marker is still a transcription"
    assert not is_bare_marker(None) and not is_bare_marker("")


def test_fabrication_is_never_inferred():
    """It is semantic; nothing in the classifier may guess it."""
    row = {"ok": True, "output": {"transcription": "completely unrelated fluent text"}}
    assert classify_row(row, "the real reference")["fabrication_detected"] is None


def test_every_axis_is_present_on_every_row(manifest):
    for d in (S1C, LUNA, SONNET, GEM_S1):
        for v in _tax(d, manifest).values():
            for f in OUTCOME_FIELDS:
                assert f in v, f


def test_usable_and_total_line_loss_are_complementary(manifest):
    for d in (S1C, LUNA, SONNET, GEM_S1):
        for v in _tax(d, manifest).values():
            assert v["usable_transcription_returned"] != v["total_line_loss"]


# ---- fallback ------------------------------------------------------------

USABLE = {"usable_transcription_returned": True, "provider_content_filter_failure": False,
          "provider_other_http_failure": False, "truncation": False,
          "json_parse_failure": False, "schema_failure": False, "model_text_refusal": False}
FILTERED = {**USABLE, "usable_transcription_returned": False,
            "provider_content_filter_failure": True}
REFUSED = {**USABLE, "usable_transcription_returned": False, "model_text_refusal": True}
TRUNCATED = {**USABLE, "usable_transcription_returned": False, "truncation": True}


def test_primary_wins_whenever_it_returns_usable_text():
    d = select(case_id="c", primary_outcome=USABLE, primary_text="gem",
               secondary_outcome=USABLE, secondary_text="son")
    assert d["chosen_model"] == PRIMARY_ROLE and d["chosen_text"] == "gem"
    assert d["fallback_used"] is False and d["needs_review"] is False


def test_primary_wins_even_when_the_secondary_looks_better():
    """The rule must not consult quality — that is the whole design."""
    d = select(case_id="c", primary_outcome=USABLE, primary_text="a",
               secondary_outcome=USABLE, secondary_text="a much longer nicer looking answer")
    assert d["chosen_model"] == PRIMARY_ROLE


@pytest.mark.parametrize("outcome,trigger", [
    (FILTERED, "provider_content_filter_failure"),
    (REFUSED, "model_text_refusal"),
    (TRUNCATED, "truncation"),
    ({**USABLE, "usable_transcription_returned": False, "provider_other_http_failure": True},
     "provider_other_http_failure"),
    ({**USABLE, "usable_transcription_returned": False, "schema_failure": True}, "schema_failure"),
    ({**USABLE, "usable_transcription_returned": False, "json_parse_failure": True},
     "json_parse_failure"),
])
def test_each_hard_failure_triggers_the_fallback(outcome, trigger):
    assert is_hard_failure(outcome)
    assert which_trigger(outcome) == trigger
    d = select(case_id="c", primary_outcome=outcome, primary_text=None,
               secondary_outcome=USABLE, secondary_text="son")
    assert d["chosen_model"] == SECONDARY_ROLE and d["fallback_used"] is True
    assert d["trigger"] == trigger
    assert d["needs_review"] is True, "a fallback result is never auto-acceptable"
    assert POLICY_ID in d["provenance"]


def test_both_failing_is_unresolved_not_silently_empty():
    d = select(case_id="c", primary_outcome=FILTERED, primary_text=None,
               secondary_outcome=FILTERED, secondary_text=None)
    assert d["resolved"] is False and d["chosen_model"] is None
    assert d["needs_review"] is True


def test_an_unknown_future_failure_mode_still_routes_to_fallback():
    """usable_transcription_returned is authoritative, so a new provider failure
    nobody has enumerated cannot silently pass as a primary success."""
    weird = {"usable_transcription_returned": False}
    assert is_hard_failure(weird)
    assert which_trigger(weird) == "empty_or_missing_output"


# ---- the reference-blindness proof ---------------------------------------

def test_select_has_no_parameter_that_could_carry_a_reference():
    import inspect
    params = set(inspect.signature(select).parameters)
    assert params == {"case_id", "primary_outcome", "primary_text",
                      "secondary_outcome", "secondary_text",
                      "primary_model", "secondary_model"}
    for bad in FORBIDDEN_DECISION_INPUTS:
        assert bad not in params


def test_stripping_every_evaluation_field_changes_no_decision(manifest):
    """The requirement, proved on real data: run the policy over the Stage-1c
    outcomes, then again with every reference/metric/split/grade field deleted
    from the inputs, and require byte-identical decisions."""
    by = {c.case_id: c for c in manifest.cases}
    gem = _tax(S1C, manifest)
    son = _tax(SONNET, manifest)
    gem_rows, son_rows = _rows(S1C), _rows(SONNET)
    cases = sorted(gem)
    gem_text = {c: (gem_rows[c].get("output") or {}).get("transcription") for c in cases}
    son_text = {c: (son_rows[c].get("output") or {}).get("transcription") for c in cases}

    full = replay(cases, gem, son, gem_text, son_text)

    def strip(d):
        out = {}
        for cid, v in d.items():
            clean = {k: val for k, val in v.items()
                     if k not in FORBIDDEN_DECISION_INPUTS}
            # also inject decoys that MUST be ignored
            clean["cer"] = 0.0
            clean["reference"] = by[cid].label["reference"]
            clean["split"] = "DEV"
            out[cid] = clean
        return out

    decoyed = replay(cases, strip(gem), strip(son), gem_text, son_text)
    assert json.dumps(full["decisions"], sort_keys=True) == \
        json.dumps(decoyed["decisions"], sort_keys=True), \
        "a reference or metric changed a fallback decision"
    assert full["primary_used"] == decoyed["primary_used"]
    assert full["fallback_used"] == decoyed["fallback_used"]


def test_replay_reports_a_full_denominator(manifest):
    gem, son = _tax(S1C, manifest), _tax(SONNET, manifest)
    cases = sorted(gem)
    gem_rows, son_rows = _rows(S1C), _rows(SONNET)
    r = replay(cases,
               gem, son,
               {c: (gem_rows[c].get("output") or {}).get("transcription") for c in cases},
               {c: (son_rows[c].get("output") or {}).get("transcription") for c in cases})
    assert r["intended_crops"] == len(cases)
    assert r["primary_used"] + r["fallback_used"] + r["unresolved"] == len(cases)
    assert r["policy_id"] == POLICY_ID


def test_a_cache_replay_is_not_a_provider_request(manifest):
    """Found by independent verification: classify_row hardcoded
    provider_request_attempted=True, contradicting its own docstring and
    inflating the provider-request count on cached rows. The paired 32-crop
    Gemini arm has exactly 2 such rows."""
    t = _tax(Path("evaluation/model_selection/runs_seen32/ocr_primary/"
                  "dev__seen46_ocr_dev__all__google-gemini-3.7-flash__c4ae61f634"), manifest)
    cached = [c for c, v in t.items() if v["served_from_cache"]]
    assert len(cached) == 2, cached
    for c in cached:
        assert t[c]["provider_request_attempted"] is False
        assert t[c]["usable_transcription_returned"] is True, "both cache hits were successes"
    s = summarize(t)
    assert s["provider_request_attempted"] == 30, "62 real requests across both arms, 30 here"
    assert s["served_from_cache"] == 2
    assert s["usable_transcription_returned"] == 14, "coverage is unchanged by the fix"
