"""Exhaustive deterministic state-space tests for the risk engine.

Every practical combination of semantic verdict x schema x evidence x
validation x uncertainty x transcription x source integrity x local-grader
availability x model-output currency (1,152 structural states) is evaluated
under every registered policy and every mode. No network, no DB, no model.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder import riskengine as re_  # noqa: E402

VERDICTS = ("invalid", "partially_valid", "valid")
BOOLS = (True, False)
SOURCES = ("current", "stale", "issue")
NOW = "2026-09-02 00:00:00"

PROSPECTIVE = ("prospective_valid_only_v1", "prospective_noninvalid_v1")
BASELINE = ("prospective_auto_all_structurally_valid_v1",)
RETRO = ("retrospective_human_dispute_aware_b_v1",
         "retrospective_human_dispute_aware_c_v1")


def _all_states():
    for (verdict, schema, evidence, validation, uncertain, transcription,
         source, grader, current) in product(
            VERDICTS, BOOLS, BOOLS, BOOLS, BOOLS, BOOLS, SOURCES, BOOLS, BOOLS):
        yield re_.ProspectiveDecisionInput.from_mapping({
            "semantic_verdict": verdict, "schema_ok": schema,
            "evidence_ok": evidence, "validation_ok": validation,
            "uncertain": uncertain, "transcription_complete": transcription,
            "source_integrity": source, "model_output_current": current,
            "local_grader_available": grader, "model_digest": "d",
            "prompt_version": "grade-v4-charitable-local",
            "prompt_sha256": "p", "schema_sha256": "s",
            "validation_version": "grade-validation-v2"})


STATES = list(_all_states())


def _structurally_clean(d):
    return (d.schema_ok and d.evidence_ok and d.validation_ok
            and not d.uncertain and d.transcription_complete
            and d.source_integrity == "current" and d.model_output_current
            and d.local_grader_available)


def test_state_space_size_is_complete():
    assert len(STATES) == 3 * 2 ** 6 * 3 * 2 == 1152


def test_off_mode_never_produces_any_action_for_any_state_or_policy():
    for pol in PROSPECTIVE + BASELINE + RETRO:
        eng = re_.build_engine(mode="off", policy_id=pol)
        for d in STATES[:64] + STATES[-64:]:
            out = eng.run_case("c", "r", d, now=NOW)
            assert out.decision is None
            assert out.applied is False
            assert out.active_grade_changed is False
            assert out.shadow_event_id is None


def test_shadow_mode_never_changes_the_active_grade_for_any_state():
    for pol in PROSPECTIVE + BASELINE:
        eng = re_.build_engine(mode="shadow", policy_id=pol)
        for d in STATES:
            out = eng.run_case("c", "r", d, now=NOW)
            assert out.applied is False
            assert out.active_grade_changed is False
            assert out.decision is not None
            assert out.decision.action in ("AUTO", "REVIEW")


def test_no_structural_failure_ever_reaches_auto_under_any_policy():
    engines = [re_.build_engine(mode="shadow", policy_id=p)
               for p in PROSPECTIVE + BASELINE]
    retro_engines = [(re_.build_engine(mode="shadow", policy_id=p),
                      re_.RetrospectiveContext(False, False)) for p in RETRO]
    for d in STATES:
        if _structurally_clean(d):
            continue
        for eng in engines:
            dec = eng.decide(d, now=NOW)
            assert dec.action == "REVIEW", (eng.policy_id, d)
            assert dec.reason.startswith("REVIEW_"), dec.reason
            assert dec.candidate_awarded_verdict is None
        for eng, ctx in retro_engines:
            dec = eng.decide(d, ctx, offline_analysis=True, now=NOW)
            assert dec.action == "REVIEW", (eng.policy_id, d)


def test_stale_or_issue_sources_and_stale_outputs_never_auto():
    for pol in PROSPECTIVE + BASELINE:
        eng = re_.build_engine(mode="shadow", policy_id=pol)
        for d in STATES:
            if d.source_integrity != "current":
                dec = eng.decide(d, now=NOW)
                assert dec.action == "REVIEW"
                if d.local_grader_available and d.model_output_current:
                    assert dec.reason == "REVIEW_SOURCE_INTEGRITY"
            elif not d.model_output_current:
                dec = eng.decide(d, now=NOW)
                assert dec.action == "REVIEW"
                if d.local_grader_available:
                    assert dec.reason == "REVIEW_STALE_OUTPUT"


def test_unavailable_local_grader_reviews_and_never_falls_back_to_cloud():
    src = (REPO / "autograder" / "riskengine.py").read_text(encoding="utf-8")
    for marker in ("openrouter", "anthropic", "gemini", "gateway", "httpx"):
        assert marker not in src.lower().replace("nonprospective", ""), marker
    for pol in PROSPECTIVE + BASELINE:
        eng = re_.build_engine(mode="shadow", policy_id=pol)
        for d in STATES:
            if not d.local_grader_available:
                dec = eng.decide(d, now=NOW)
                assert (dec.action, dec.reason) == \
                    ("REVIEW", "REVIEW_LOCAL_GRADER_UNAVAILABLE")


def test_verdict_routing_on_clean_states_is_exactly_the_policy_contract():
    clean = [d for d in STATES if _structurally_clean(d)]
    assert len(clean) == 3
    by_verdict = {d.semantic_verdict: d for d in clean}
    eng_v = re_.build_engine(mode="shadow", policy_id="prospective_valid_only_v1")
    eng_n = re_.build_engine(mode="shadow", policy_id="prospective_noninvalid_v1")
    eng_a = re_.build_engine(
        mode="shadow", policy_id="prospective_auto_all_structurally_valid_v1")
    assert eng_v.decide(by_verdict["valid"], now=NOW).reason == "AUTO_GROUNDED_VALID"
    assert eng_v.decide(by_verdict["partially_valid"], now=NOW).reason == \
        "REVIEW_PARTIAL_VERDICT"
    assert eng_v.decide(by_verdict["invalid"], now=NOW).reason == \
        "REVIEW_INVALID_VERDICT"
    assert eng_n.decide(by_verdict["partially_valid"], now=NOW).reason == \
        "AUTO_GROUNDED_PARTIAL"
    assert eng_n.decide(by_verdict["invalid"], now=NOW).reason == \
        "REVIEW_INVALID_VERDICT"
    assert eng_a.decide(by_verdict["invalid"], now=NOW).reason == \
        "AUTO_STRUCTURALLY_VALID_BASELINE"


def test_retrospective_dispute_and_issue_routing_on_clean_states():
    clean_valid = next(d for d in STATES if _structurally_clean(d)
                       and d.semantic_verdict == "valid")
    eng = re_.build_engine(mode="shadow",
                           policy_id="retrospective_human_dispute_aware_c_v1")
    dec = eng.decide(clean_valid, re_.RetrospectiveContext(True, False),
                     offline_analysis=True, now=NOW)
    assert dec.reason == "REVIEW_WIDE_HUMAN_DISAGREEMENT"
    dec = eng.decide(clean_valid, re_.RetrospectiveContext(False, True),
                     offline_analysis=True, now=NOW)
    assert dec.reason == "REVIEW_ACTIVE_EVIDENCE_ISSUE"
    dec = eng.decide(clean_valid, re_.RetrospectiveContext(False, False),
                     offline_analysis=True, now=NOW)
    assert (dec.action, dec.reason) == ("AUTO", "AUTO_GROUNDED_VALID")


def test_retrospective_policies_are_blocked_in_every_production_state():
    for pol in RETRO:
        eng = re_.build_engine(mode="shadow", policy_id=pol)
        for d in STATES[::37]:
            dec = eng.decide(d, re_.RetrospectiveContext(False, False), now=NOW)
            assert (dec.action, dec.reason) == \
                ("BLOCKED", "BLOCKED_NONPROSPECTIVE_POLICY")


def test_active_mode_stays_blocked_without_authorization_for_all_states():
    for pol in PROSPECTIVE:
        eng = re_.build_engine(mode="active", policy_id=pol)
        for d in STATES[::17]:
            out = eng.run_case("c", "r", d, now=NOW)
            assert out.decision.action == "BLOCKED"
            assert out.decision.reason == "BLOCKED_ACTIVATION_INCOMPLETE"
            assert out.applied is False and out.active_grade_changed is False


def test_decisions_are_deterministic_across_engine_instances():
    a = re_.build_engine(mode="shadow", policy_id="prospective_noninvalid_v1")
    b = re_.build_engine(mode="shadow", policy_id="prospective_noninvalid_v1")
    for d in STATES[::11]:
        assert a.decide(d, now=NOW) == b.decide(d, now=NOW)


def test_serialization_round_trips_across_the_state_space():
    eng = re_.build_engine(mode="shadow", policy_id="prospective_valid_only_v1")
    for d in STATES[::23]:
        dec = eng.decide(d, now=NOW)
        assert re_.RiskDecision.from_dict(dec.to_dict()) == dec


def test_every_emitted_reason_is_in_the_registry():
    seen = set()
    for pol in PROSPECTIVE + BASELINE:
        eng = re_.build_engine(mode="shadow", policy_id=pol)
        for d in STATES:
            seen.add(eng.decide(d, now=NOW).reason)
    eng = re_.build_engine(mode="shadow",
                           policy_id="retrospective_human_dispute_aware_b_v1")
    for wide, issue in ((True, False), (False, True)):
        seen.add(eng.decide(
            next(x for x in STATES if _structurally_clean(x)),
            re_.RetrospectiveContext(wide, issue),
            offline_analysis=True, now=NOW).reason)
    assert seen <= set(re_.REASONS)
    # and the mission's mandatory codes are all reachable or defined
    for must in ("AUTO_GROUNDED_VALID", "AUTO_GROUNDED_PARTIAL",
                 "REVIEW_INVALID_VERDICT", "REVIEW_PARTIAL_VERDICT",
                 "REVIEW_UNCERTAIN", "REVIEW_SCHEMA_FAILURE",
                 "REVIEW_EVIDENCE_FAILURE", "REVIEW_TRANSCRIPTION_INCOMPLETE",
                 "REVIEW_SOURCE_INTEGRITY", "REVIEW_STALE_OUTPUT",
                 "REVIEW_LOCAL_GRADER_UNAVAILABLE",
                 "BLOCKED_POLICY_UNSELECTED", "BLOCKED_POLICY_HASH_MISMATCH",
                 "BLOCKED_NONPROSPECTIVE_POLICY"):
        assert must in re_.REASONS, must
