"""The deterministic risk engine: taxonomy, fail-closed inputs, matrix
integrity, activation lock, typed refusals, serialization, rare-event math.

No model / provider / network call anywhere in this file. Everything runs on
temp files; no live database, no benchmark content, no HELD_OUT anything.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder import rare_events, riskengine as re_  # noqa: E402

POLICY_PATH = REPO / "evaluation" / "model_selection" / "policies" / \
    "asymmetric_grading_risk_v1.json"

CLEAN = {"semantic_verdict": "valid", "schema_ok": True, "evidence_ok": True,
         "validation_ok": True, "uncertain": False,
         "transcription_complete": True, "source_integrity": "current",
         "model_output_current": True, "local_grader_available": True,
         "model_digest": "digest", "prompt_version": "grade-v4-charitable-local",
         "prompt_sha256": "p" * 8, "schema_sha256": "s" * 8,
         "validation_version": "grade-validation-v2"}


def _input(**patch):
    return re_.ProspectiveDecisionInput.from_mapping({**CLEAN, **patch})


def _engine(policy="prospective_valid_only_v1", mode="shadow", **kw):
    return re_.build_engine(mode=mode, policy_id=policy, **kw)


def _full_activation(policy_id="prospective_valid_only_v1"):
    spec = re_.POLICY_REGISTRY[policy_id]
    matrix = re_.load_risk_matrix()
    return re_.ActivationRecord(
        owner_ack=re_.ACTIVATION_ACK, policy_id=policy_id,
        policy_sha256=spec.sha256(), matrix_name=matrix.name,
        matrix_sha256=matrix.matrix_sha256, model_digest="d",
        prompt_version="grade-v4-charitable-local", prompt_sha256="p",
        schema_sha256="s", validation_version="grade-validation-v2",
        ocr_policy_version="ocr-validation-NOT-YET-RUN",
        final_validation_record="none-yet", stale_artifacts_check_passed=True,
        configured_at="2026-09-02")


# --------------------------------------------------------------- taxonomy ---


def test_policy_taxonomy_is_exactly_the_registered_five():
    scopes = {p: s.scope for p, s in re_.POLICY_REGISTRY.items()}
    assert scopes == {
        "prospective_valid_only_v1": "PROSPECTIVE_DEPLOYABLE",
        "prospective_noninvalid_v1": "PROSPECTIVE_DEPLOYABLE",
        "prospective_auto_all_structurally_valid_v1": "ANALYSIS_BASELINE_ONLY",
        "retrospective_human_dispute_aware_b_v1": "RETROSPECTIVE_HUMAN_ASSISTED",
        "retrospective_human_dispute_aware_c_v1": "RETROSPECTIVE_HUMAN_ASSISTED",
    }
    for row in re_.policy_table():
        assert row["online_deployable"] == (row["scope"] == "PROSPECTIVE_DEPLOYABLE")
        assert row["uses_human_or_reference_data"] == \
            (row["scope"] == "RETROSPECTIVE_HUMAN_ASSISTED")
        assert row["policy_sha256"]


def test_observability_inventory_is_versioned_and_unknown_fails_closed():
    inv = re_.observability_inventory()
    assert inv["inventory_version"] == "risk-observability-v1"
    assert re_.classify_field("semantic_verdict") == "ONLINE_OBSERVABLE"
    assert re_.classify_field("reviewer_disagreement") == "POST_HOC_ONLY"
    assert re_.classify_field("reviewer_note") == "ADMIN_ONLY"
    assert re_.classify_field("some_future_field") == "UNKNOWN"
    # every field a prospective input carries is ONLINE_OBSERVABLE
    for f in re_.ProspectiveDecisionInput.__dataclass_fields__:
        assert re_.classify_field(f) == "ONLINE_OBSERVABLE", f


# ------------------------------------------------------ fail-closed inputs --


def test_unknown_field_fails_closed():
    with pytest.raises(re_.RiskInputError, match="unknown field"):
        re_.ProspectiveDecisionInput.from_mapping({**CLEAN, "surprise": 1})


@pytest.mark.parametrize("bad", ["reference_verdict", "final_verdict",
                                 "reviewer_disagreement", "adjudicated_verdict",
                                 "instructor_score", "label_verdict",
                                 "expected_verdict", "human_reference",
                                 "wide_human_disagreement", "benchmark_correct",
                                 "strict_loss", "target_verdict"])
def test_post_hoc_target_fields_are_refused_by_name(bad):
    with pytest.raises(re_.RiskInputError, match="POST_HOC|refused"):
        re_.ProspectiveDecisionInput.from_mapping({**CLEAN, bad: "valid"})


def test_missing_field_and_bad_types_fail_closed():
    with pytest.raises(re_.RiskInputError, match="missing required"):
        re_.ProspectiveDecisionInput.from_mapping(
            {k: v for k, v in CLEAN.items() if k != "uncertain"})
    with pytest.raises(re_.RiskInputError, match="unknown semantic verdict"):
        _input(semantic_verdict="mostly_valid")
    with pytest.raises(re_.RiskInputError, match="source_integrity"):
        _input(source_integrity="fine")
    with pytest.raises(re_.RiskInputError, match="must be a bool"):
        _input(uncertain=1)
    with pytest.raises(re_.RiskInputError, match="non-empty string"):
        _input(model_digest="")
    with pytest.raises(re_.RiskInputError, match="mapping"):
        re_.ProspectiveDecisionInput.from_mapping(None)


def test_retrospective_context_type_is_enforced():
    with pytest.raises(re_.RiskInputError):
        re_.RetrospectiveContext(wide_human_disagreement=1,
                                 active_review_issue=False)


# --------------------------------------------------------- matrix integrity -


def test_real_matrix_loads_and_matches_the_frozen_policy():
    m = re_.load_risk_matrix()
    assert m.name == "asymmetric_grading_risk_v1"
    assert m.policy_file_sha256.startswith("11e65e79e0f36cf6")
    assert m.matrix["invalid"]["valid"] == 12
    assert m.matrix["partially_valid"]["valid"] == 5


def _write_policy(tmp_path, mutate=None, raw=None) -> Path:
    doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if mutate:
        mutate(doc)
    p = tmp_path / "policy.json"
    p.write_text(raw if raw is not None
                 else json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.mark.parametrize("case,mutate,raw,msg", [
    ("corrupted json", None, "{not json", "not valid JSON"),
    ("truncated json", None,
     POLICY_PATH.read_text(encoding="utf-8")[:200], "not valid JSON"),
    ("missing cell", lambda d: d["cost_matrix"]["invalid"].pop("valid"),
     None, "self-hash|exactly the three"),
    ("unknown verdict row",
     lambda d: d["cost_matrix"].update({"sort_of_valid": {}}), None,
     "self-hash|exactly the three"),
    ("wrong numeric type float",
     lambda d: d["cost_matrix"]["invalid"].update({"valid": 12.5}), None,
     "self-hash|integer"),
    ("wrong numeric type bool",
     lambda d: d["cost_matrix"]["valid"].update({"invalid": True}), None,
     "self-hash|integer"),
    ("negative loss",
     lambda d: d["cost_matrix"]["valid"].update({"invalid": -3}), None,
     "self-hash|negative"),
    ("nonzero diagonal",
     lambda d: d["cost_matrix"]["valid"].update({"valid": 1}), None,
     "self-hash|diagonal"),
    ("ordering violation",
     lambda d: d["cost_matrix"]["invalid"].update({"valid": 2}), None,
     "self-hash|largest"),
    ("hash mismatch / tampered value",
     lambda d: d["cost_matrix"]["partially_valid"].update({"valid": 4}), None,
     "self-hash"),
    ("extra top-level field", lambda d: d.update({"bonus": 1}), None,
     "self-hash"),
    ("missing required key", lambda d: d.pop("schema_version"), None,
     "missing|self-hash"),
])
def test_malformed_matrix_artifacts_are_typed_refusals(tmp_path, case, mutate,
                                                       raw, msg):
    p = _write_policy(tmp_path, mutate=mutate, raw=raw)
    with pytest.raises(re_.RiskMatrixError, match=msg):
        re_.load_risk_matrix(p)


def test_structural_matrix_defects_survive_a_recomputed_self_hash(tmp_path):
    """Even an attacker who fixes the self-hash cannot pass a broken matrix."""
    import hashlib
    for mutate, msg in (
            (lambda d: d["cost_matrix"]["valid"].update({"valid": 1}), "diagonal"),
            (lambda d: d["cost_matrix"]["valid"].update({"invalid": -3}), "negative"),
            (lambda d: d["cost_matrix"]["invalid"].update({"valid": 2}), "largest"),
            (lambda d: d["cost_matrix"]["invalid"].pop("valid"), "exactly the three")):
        doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        mutate(doc)
        payload = json.dumps({k: v for k, v in doc.items()
                              if k != "policy_sha256"},
                             ensure_ascii=False, sort_keys=True)
        doc["policy_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        p = tmp_path / "evil.json"
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(re_.RiskMatrixError, match=msg):
            re_.load_risk_matrix(p)


# ------------------------------------------------------------- BLOCKED ------


def test_unknown_or_unselected_policy_blocks():
    m = re_.load_risk_matrix()
    for pid in (None, "no_such_policy_v9"):
        eng = re_.RiskEngine(mode="shadow", policy_id=pid,
                             expected_policy_sha256=None, matrix=m,
                             expected_matrix_sha256=m.matrix_sha256)
        dec = eng.decide(_input(), now="t")
        assert (dec.action, dec.reason) == ("BLOCKED", "BLOCKED_POLICY_UNSELECTED")


def test_policy_and_matrix_hash_mismatches_block():
    m = re_.load_risk_matrix()
    eng = re_.RiskEngine(mode="shadow", policy_id="prospective_valid_only_v1",
                         expected_policy_sha256="deadbeef", matrix=m,
                         expected_matrix_sha256=m.matrix_sha256)
    assert eng.decide(_input(), now="t").reason == "BLOCKED_POLICY_HASH_MISMATCH"
    spec = re_.POLICY_REGISTRY["prospective_valid_only_v1"]
    eng = re_.RiskEngine(mode="shadow", policy_id="prospective_valid_only_v1",
                         expected_policy_sha256=spec.sha256(), matrix=m,
                         expected_matrix_sha256="deadbeef")
    assert eng.decide(_input(), now="t").reason == "BLOCKED_MATRIX_HASH_MISMATCH"


def test_retrospective_policies_are_blocked_outside_offline_analysis():
    for pid in ("retrospective_human_dispute_aware_b_v1",
                "retrospective_human_dispute_aware_c_v1"):
        dec = _engine(pid).decide(
            _input(), re_.RetrospectiveContext(False, False), now="t")
        assert (dec.action, dec.reason) == ("BLOCKED",
                                            "BLOCKED_NONPROSPECTIVE_POLICY")


def test_prospective_policy_refuses_retrospective_context_entirely():
    with pytest.raises(re_.RiskInputError, match="must not receive"):
        _engine("prospective_valid_only_v1").decide(
            _input(), re_.RetrospectiveContext(True, True), now="t")


def test_retrospective_policy_without_context_refuses_to_guess():
    with pytest.raises(re_.RiskInputError, match="refusing to guess"):
        _engine("retrospective_human_dispute_aware_c_v1").decide(
            _input(), None, offline_analysis=True, now="t")


# -------------------------------------------------------- ACTIVE stays shut -


def test_active_mode_without_activation_blocks():
    dec = _engine(mode="active").decide(_input(), now="t")
    assert (dec.action, dec.reason) == ("BLOCKED", "BLOCKED_ACTIVATION_INCOMPLETE")


@pytest.mark.parametrize("break_field", [
    "owner_ack", "policy_sha256", "matrix_sha256", "model_digest",
    "prompt_version", "prompt_sha256", "schema_sha256", "validation_version",
    "ocr_policy_version", "final_validation_record", "configured_at"])
def test_every_missing_activation_requirement_blocks(break_field):
    import dataclasses
    a = _full_activation()
    a = dataclasses.replace(a, **{break_field: ""})
    dec = _engine(mode="active", activation=a).decide(_input(), now="t")
    assert dec.reason == "BLOCKED_ACTIVATION_INCOMPLETE"


def test_stale_artifacts_flag_and_wrong_ack_block():
    import dataclasses
    a = dataclasses.replace(_full_activation(), stale_artifacts_check_passed=False)
    assert _engine(mode="active", activation=a).decide(_input(), now="t") \
        .reason == "BLOCKED_ACTIVATION_INCOMPLETE"
    a = dataclasses.replace(_full_activation(), owner_ack="yes please")
    assert _engine(mode="active", activation=a).decide(_input(), now="t") \
        .reason == "BLOCKED_ACTIVATION_INCOMPLETE"


def test_nonprospective_policies_never_activate_even_fully_authorized():
    a = _full_activation("prospective_auto_all_structurally_valid_v1")
    eng = _engine("prospective_auto_all_structurally_valid_v1", mode="active",
                  activation=a)
    assert eng.decide(_input(), now="t").reason == "BLOCKED_NONPROSPECTIVE_POLICY"


def test_fully_authorized_active_path_exists_but_has_no_production_caller():
    eng = _engine(mode="active", activation=_full_activation())
    out = eng.run_case("case", "run", _input(), now="t")
    assert out.decision.action == "AUTO" and out.applied is True
    # …and NO module outside riskengine/tests CONSTRUCTS an activation
    # record or configures active mode: the lock is structural, not
    # configurational (prose mentions in reports are fine)
    for pkg in ("autograder", "review46_app", "labeling_app", "scripts"):
        for py in (REPO / pkg).rglob("*.py"):
            if py.name == "riskengine.py":
                continue
            src = py.read_text(encoding="utf-8", errors="replace")
            assert "ActivationRecord(" not in src, py
            assert 'mode="active"' not in src and "mode='active'" not in src, py


# --------------------------------------------------- decisions + rounding ---


def test_decisions_are_deterministic_and_never_mutate_the_verdict():
    eng = _engine("prospective_noninvalid_v1")
    for verdict in ("invalid", "partially_valid", "valid"):
        d = _input(semantic_verdict=verdict)
        a = eng.decide(d, now="2026-09-02 00:00:00")
        b = eng.decide(d, now="2026-09-02 00:00:00")
        assert a == b
        assert a.semantic_verdict == verdict
        assert a.candidate_awarded_verdict == (verdict if a.action == "AUTO"
                                               else None)


def test_serialization_round_trip_preserves_meaning():
    dec = _engine().decide(_input(), now="t")
    again = re_.RiskDecision.from_dict(json.loads(
        json.dumps(dec.to_dict(), ensure_ascii=False)))
    assert again == dec


def test_unknown_action_reason_or_field_fail_closed_on_deserialization():
    base = _engine().decide(_input(), now="t").to_dict()
    with pytest.raises(re_.RiskInputError, match="unknown reason"):
        re_.RiskDecision.from_dict({**base, "reason": "סיבה_לא_ידועה"})
    with pytest.raises(re_.RiskInputError, match="unknown action"):
        re_.RiskDecision.from_dict({**base, "action": "MAYBE"})
    with pytest.raises(re_.RiskInputError, match="unknown decision fields"):
        re_.RiskDecision.from_dict({**base, "extra": 1})
    with pytest.raises(re_.RiskInputError, match="missing decision fields"):
        re_.RiskDecision.from_dict({k: v for k, v in base.items()
                                    if k != "reason"})


def test_no_benchmark_case_id_is_hardcoded_in_the_engine():
    src = (REPO / "autograder" / "riskengine.py").read_text(encoding="utf-8")
    assert not re.search(r"e0\d\d_q\d_r\d", src)
    for banned in ("gateway", "httpx", "requests", "urllib.request", "socket",
                   "openrouter", "ollama", "11434"):
        assert banned not in src, banned


# ----------------------------------------------------------- rare events ----


def test_zero_events_over_five_cases_bounds_at_45_percent():
    r = rare_events.severe_event_report(0, 5)
    assert r["observed"] == 0 and r["denominator"] == 5
    assert abs(r["one_sided_upper_95"] - (1 - 0.05 ** 0.2)) < 1e-6
    assert 0.45 < r["one_sided_upper_95"] < 0.4512
    assert r["two_sided_95"][0] == 0.0


def test_minimum_sample_table_is_exact():
    assert rare_events.min_n_for_zero_event_bound(0.10) == 29
    assert rare_events.min_n_for_zero_event_bound(0.05) == 59
    assert rare_events.min_n_for_zero_event_bound(0.02) == 149
    assert rare_events.min_n_for_zero_event_bound(0.01) == 299
    # each n actually achieves the bound and n-1 does not
    for bound, n in ((0.10, 29), (0.05, 59), (0.02, 149), (0.01, 299)):
        assert rare_events.exact_upper_bound(0, n) <= bound
        assert rare_events.exact_upper_bound(0, n - 1) > bound


def test_nonzero_counts_get_coherent_exact_intervals():
    lo, hi = rare_events.two_sided_interval(4, 13)
    assert 0 < lo < 4 / 13 < hi < 1
    # exactness: at the bounds the tail probabilities equal alpha/2
    assert abs(rare_events.binomial_cdf(4, 13, hi) - 0.025) < 1e-6
    assert abs((1 - rare_events.binomial_cdf(3, 13, lo)) - 0.025) < 1e-6


def test_rare_event_math_refuses_invalid_inputs():
    for k, n in ((-1, 5), (6, 5), (0, 0)):
        with pytest.raises(ValueError):
            rare_events.exact_upper_bound(k, n)
    with pytest.raises(ValueError):
        rare_events.min_n_for_zero_event_bound(0.0)
