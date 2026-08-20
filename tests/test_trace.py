"""Decision records (§14) + local early-exit accounting (§15)."""

from __future__ import annotations

import pytest

from autograder.signals import DecisionSignals, MCSignals
from autograder.trace import (DecisionTrace, DecisionTraceStore, EarlyExitLedger,
                              record_from_early_exit)


def deterministic_mc_record(exam="exam001", qid="2", sid="7"):
    t = DecisionTrace(exam, qid, sid)
    t.package(variant="variant_1", variant_source="reused", alignment_source="deterministic",
              grading_policy="wrong_choice_zero", pack_hash="abc123")
    t.deterministic("single clean mark in column C, ink margin 0.42")
    t.signals(DecisionSignals(item_id="item-1", mc=MCSignals(cv_score=0.9, cv_margin=0.42,
                                                            resolver_source="deterministic")))
    t.skipped("mc_resolve_local", "deterministic_mc", detail="confident CV resolution",
              avoided={"mc": 1})
    t.skipped("ocr_explanation", "wrong_choice_zero", detail="wrong selection scores 0",
              avoided={"ocr": 1, "cloud": 1})
    t.skipped("grade_primary", "wrong_choice_zero", detail="no rubric judgement needed",
              avoided={"grading": 1, "cloud": 1})
    return t.finish("AUTO", "AUTO", "policy settled the item locally",
                    points_awarded=0.0, points_max=2.0)


def escalated_record(exam="exam002"):
    t = DecisionTrace(exam, "1", "3", grading_policy="choice_and_explanation_independent")
    t.executed("ocr_primary", task="ocr_primary", model="local-vlm", cloud=False, latency_s=1.2)
    t.executed("grade_primary", task="grade_primary", model="cloud-a", cloud=True,
               usage={"total_tokens": 800, "reported_cost": 0.002}, request_id="gen-1")
    t.executed("grade_escalate", task="grade_escalate", model="cloud-b", cloud=True,
               usage={"total_tokens": 1200}, request_id="gen-2")
    return t.finish("AUTO", "AUTO", "escalation resolved consistently",
                    points_awarded=3.0, points_max=4.0)


# ------------------------------------------------------------------ §14 ------


def test_record_answers_why_this_was_auto_graded_without_review():
    r = deterministic_mc_record()
    text = r.explain()
    assert "AUTO" in text and "wrong_choice_zero" in text
    assert "single clean mark" in text and "variant_1" in text
    assert "skipped grade_primary" in text
    assert len(text.splitlines()) <= 12          # compact by design


def test_executed_and_skipped_stages_and_their_reasons():
    r = deterministic_mc_record()
    assert [s.stage for s in r.skipped()] == ["mc_resolve_local", "ocr_explanation", "grade_primary"]
    assert r.executed() == []
    assert r.why_skipped("ocr_explanation") == "wrong_choice_zero"
    assert r.why_skipped("never_ran") is None


def test_model_calls_carry_usage_ids_for_cost_reconciliation():
    r = escalated_record()
    calls = r.model_calls()
    assert [c["task"] for c in calls] == ["ocr_primary", "grade_primary", "grade_escalate"]
    assert calls[1]["request_id"] == "gen-1" and calls[1]["tokens"] == 800
    assert calls[1]["cost"] == 0.002


def test_signals_and_package_provenance_are_recorded():
    r = deterministic_mc_record()
    assert r.signals["mc"]["cv_margin"] == 0.42
    assert r.variant_source == "reused" and r.alignment_source == "deterministic"
    assert r.pack_hash == "abc123"


def test_unknown_states_and_skip_reasons_are_rejected():
    t = DecisionTrace("e", "1")
    with pytest.raises(ValueError):
        t.skipped("ocr", "because_i_said_so")
    with pytest.raises(ValueError):
        t.finish("MAYBE", "AUTO")


def test_records_persist_and_are_retrievable(tmp_path):
    store = DecisionTraceStore(tmp_path / "traces" / "decisions.jsonl")
    store.append(deterministic_mc_record())
    store.append(escalated_record())
    rows = store.read()
    assert len(rows) == 2 and rows[0]["skip_reasons"][0] == "deterministic_mc"
    found = store.find("exam002", "1", "3")
    assert found and found["final_state"] == "AUTO" and len(found["model_calls"]) == 3


# ------------------------------------------------------------------ §15 ------


def test_early_exit_accounting_aggregates_avoided_work():
    ledger = EarlyExitLedger()
    for i in range(9):
        ledger.add(deterministic_mc_record(exam=f"exam{i:03d}"))
    ledger.add(escalated_record())
    d = ledger.as_dict()
    assert d["questions"] == 10
    assert d["ocr_calls_avoided"] == 9 and d["grading_calls_avoided"] == 9
    assert d["mc_calls_avoided"] == 9 and d["cloud_calls_avoided"] == 18
    assert d["explanations_skipped"] == 9
    assert d["by_skip_reason"] == {"deterministic_mc": 9, "wrong_choice_zero": 18}
    assert d["fully_local_questions"] == 9 and d["pct_graded_fully_locally"] == 90.0
    assert d["by_final_state"]["AUTO"] == 10


def test_escalation_that_prevented_a_review_is_counted():
    ledger = EarlyExitLedger()
    ledger.add(escalated_record())
    assert ledger.as_dict()["review_cases_avoided"] == 1


def test_cache_hits_are_free_and_keep_an_item_fully_local():
    t = DecisionTrace("exam003", "1", "1")
    t.executed("grade_primary", task="grade_primary", model="cloud-a", cloud=True, cache_hit=True)
    r = t.finish("AUTO", "AUTO", "cache hit")
    assert r.fully_local
    ledger = EarlyExitLedger()
    ledger.add(r)
    d = ledger.as_dict()
    assert d["cache_hits"] == 1 and d["pct_graded_fully_locally"] == 100.0


def test_bridge_from_the_existing_policy_early_exit_log():
    entry = {"question_id": "2", "sub_item_id": "5", "policy": "choice_only",
             "flag": "deterministic_choice_only", "reason": "choice_only: local MC score"}
    r = record_from_early_exit(entry, exam_id="exam007")
    assert r.final_state == "AUTO" and r.grading_policy == "choice_only"
    assert r.why_skipped("ocr_explanation") == "choice_only"
    assert r.avoided()["cloud"] == 2 and r.fully_local


def test_trace_records_a_failed_stage_without_pretending_it_ran():
    t = DecisionTrace("exam004", "1", "2")
    t.executed("ocr_primary", task="ocr_primary", model="m", cloud=True)
    t.failed("grade_primary", "provider timed out", task="grade_primary")
    r = t.finish("REVIEW", "PROVIDER_FAILED", "grading provider unavailable")
    assert r.final_state == "REVIEW" and "FAILED grade_primary" in r.explain()
    assert [c["task"] for c in r.model_calls()] == ["ocr_primary"]
