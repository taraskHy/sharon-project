"""OCR_PROMPT_V2_NEUTRAL_FRAMING result integrity. ZERO provider calls.

Pins the properties that make the neutral arm a valid one-variable experiment
and that keep its conclusion honest: population identity with the control, the
explicit outcome taxonomy, paired-transition accounting, the annotation guard,
immutability of the pre-registered drop rule, cost reconciliation, and HELD_OUT
exclusion.
"""
import hashlib
import json
from pathlib import Path

import pytest

R = Path("evaluation/model_selection/runs/ocr_primary")
E = Path("evaluation/model_selection/experiments")
RESULT = R / "OCR_PROMPT_V2_PAIRED_RESULT_2026-09-03.json"
PREREG = E / "OCR_PROMPT_V2_NEUTRAL_FRAMING_2026-09-02.json"
FREEZE = E / "OCR_NEUTRAL_V2_PROMPT_FREEZE_2026-09-02.json"

pytestmark = pytest.mark.skipif(not RESULT.exists(), reason="neutral arm has not been run")


@pytest.fixture(scope="module")
def res():
    return json.loads(RESULT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prereg():
    return json.loads(PREREG.read_text(encoding="utf-8"))


def _self_hash(path, field):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    body = json.dumps({k: v for k, v in d.items() if k != field},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest(), d[field]


# ---- freeze integrity -------------------------------------------------------

def test_preregistration_self_hash_verifies():
    got, stored = _self_hash(PREREG, "experiment_sha256")
    assert got == stored
    assert stored.startswith("05185839e204ca61")


def test_prompt_freeze_self_hash_verifies():
    got, stored = _self_hash(FREEZE, "freeze_sha256")
    assert got == stored


def test_result_self_hash_verifies():
    got, stored = _self_hash(RESULT, "content_sha256")
    assert got == stored


# ---- one variable / population identity -------------------------------------

def test_population_is_identical_to_the_control(res):
    p = res["population_identity"]
    assert p["n"] == 32
    assert p["crop_hashes_identical"] is True
    assert p["reference_hashes_identical"] is True
    assert p["order_identical"] is True


def test_held_out_and_calibration_are_excluded(res):
    assert res["population_identity"]["HELD_OUT"] == 0
    assert res["population_identity"]["CALIBRATION"] == 0
    for r in res["case_matrix"]:
        assert not r["case_id"].startswith(("e004", "e005", "e006"))


def test_both_arms_are_the_same_model(res):
    assert {r["model"] for r in res["case_matrix"]} == {"google/gemini-3.7-flash"}


def test_the_neutral_prompt_differs_from_the_control_in_exactly_one_line():
    from autograder.benchmark.roles import load_ocr_prompts
    v1, v2 = load_ocr_prompts("m2-strict-v1"), load_ocr_prompts("ocr-neutral-v2")
    for cat in ("handwritten_line", "handwritten_cell"):
        a, b = v1[cat].splitlines(), v2[cat].splitlines()
        assert len(a) == len(b)
        assert [i for i in range(len(a)) if a[i] != b[i]] == [0]
        assert a[1:] == b[1:], "the rules block must be byte-identical"


def test_cache_keys_are_separated(res):
    from autograder.benchmark.roles import load_ocr_prompts
    v1, v2 = load_ocr_prompts("m2-strict-v1"), load_ocr_prompts("ocr-neutral-v2")
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    assert len({h(v1["handwritten_line"]), h(v1["handwritten_cell"]),
                h(v2["handwritten_line"]), h(v2["handwritten_cell"])}) == 4
    # the live arm resolved no cache replays, so nothing was reused
    assert res["arms"]["neutral_ocr-neutral-v2"]["cache_hits"] == 0


# ---- explicit outcome taxonomy ----------------------------------------------

TAXONOMY = ("provider_request_attempted", "provider_http_response_received",
            "provider_request_completed", "provider_content_filter_failure",
            "provider_other_http_failure", "model_text_refusal",
            "usable_transcription_returned", "fabrication_detected", "truncation",
            "json_parse_failure", "schema_failure", "total_line_loss",
            "annotation_inclusion_error")


def test_every_row_carries_the_full_taxonomy(res):
    for r in res["case_matrix"]:
        for f in TAXONOMY:
            assert f in r, f"{r['case_id']} is missing {f}"


def test_taxonomy_is_not_collapsed_into_one_success_field(res):
    for r in res["case_matrix"]:
        assert "success" not in r and "refusal" not in r


def test_a_provider_response_is_not_automatically_usable(res):
    neutral = [r for r in res["case_matrix"] if r["arm"] == "neutral_ocr-neutral-v2"]
    responded = [r for r in neutral if r["provider_http_response_received"]]
    usable = [r for r in neutral if r["usable_transcription_returned"]]
    assert len(responded) > len(usable)


def test_arm_counts_match_the_rows(res):
    for arm, key in (("control_m2-strict-v1", "control"), ("neutral_ocr-neutral-v2", "neutral")):
        rows = [r for r in res["case_matrix"] if r["arm"] == arm]
        s = res["arms"][arm]
        assert len(rows) == 32 == s["intended"]
        assert sum(1 for r in rows if r["usable_transcription_returned"]) == s["usable"]
        assert sum(1 for r in rows if r["provider_content_filter_failure"]) == s["provider_content_filter"]
        assert s["usable"] + s["hard_failures"] == 32


# ---- paired transitions -----------------------------------------------------

def test_every_crop_has_exactly_one_transition(res):
    assert len(res["paired_transitions"]) == 32
    assert sum(res["transition_counts"].values()) == 32
    assert len({p["case_id"] for p in res["paired_transitions"]}) == 32


def test_rescued_and_broken_lists_agree_with_the_transitions(res):
    P = res["paired_transitions"]
    assert sorted(res["rescued_crops"]) == sorted(
        p["case_id"] for p in P if not p["control_usable"] and p["neutral_usable"])
    assert sorted(res["newly_broken_crops"]) == sorted(
        p["case_id"] for p in P if p["control_usable"] and not p["neutral_usable"])


def test_mcnemar_counts_match_the_discordant_pairs(res):
    P, t = res["paired_transitions"], res["paired_test"]
    assert t["b_control_usable_treatment_not"] == sum(
        1 for p in P if p["control_usable"] and not p["neutral_usable"])
    assert t["c_treatment_usable_control_not"] == sum(
        1 for p in P if p["neutral_usable"] and not p["control_usable"])
    assert t["discordant"] == t["b_control_usable_treatment_not"] + t["c_treatment_usable_control_not"]
    assert 0.0 <= t["p_value"] <= 1.0


def test_matched_pairs_only_uses_crops_usable_in_both_arms(res):
    m = res["matched_pairs_quality"]
    C = {r["case_id"]: r for r in res["case_matrix"] if r["arm"] == "control_m2-strict-v1"}
    T = {r["case_id"]: r for r in res["case_matrix"] if r["arm"] == "neutral_ocr-neutral-v2"}
    for cid in m["case_ids"]:
        assert C[cid]["usable_transcription_returned"] and T[cid]["usable_transcription_returned"]
    assert m["n_usable_in_both_arms"] == len(m["case_ids"])
    assert m["improved"] + m["regressed"] + m["unchanged"] == m["n_usable_in_both_arms"]


# ---- annotation contamination ------------------------------------------------

def test_annotation_contamination_is_measured_and_no_crop_was_excluded(res):
    for arm in ("control_m2-strict-v1", "neutral_ocr-neutral-v2"):
        rows = [r for r in res["case_matrix"] if r["arm"] == arm]
        assert len(rows) == 32, "no crop may be dropped after results arrive"
        assert res["arms"][arm]["annotation_inclusion_errors"] == sum(
            1 for r in rows if r["annotation_inclusion_error"])


# ---- drop-rule immutability ---------------------------------------------------

def test_drop_rule_text_is_carried_verbatim_from_the_preregistration(res, prereg):
    assert (res["pre_registered_drop_rule"]["rule_text_as_committed"]
            == prereg["advancement_and_drop_rules_stated_in_advance"])


def test_drop_rule_threshold_is_ten_and_was_applied_as_written(res):
    d = res["pre_registered_drop_rule"]
    assert ">= 10/32" in d["rule_text_as_committed"]["DROP_gemini_as_primary"]
    hf = d["neutral_hard_failures"]
    expected = ("DROP_gemini_as_primary" if hf >= 10
                else "ADOPT_and_rerun_paired" if hf <= 4 and d["neutral_successful_only_cer"] <= 0.20
                else "REPORT_ONLY")
    assert d["outcome"] == expected
    assert d["threshold_not_changed_after_seeing_results"] is True


def test_the_result_does_not_claim_production_readiness(res):
    blob = json.dumps(res, ensure_ascii=False).lower()
    assert "production-ready" not in blob and "production ready" not in blob


# ---- accounting ---------------------------------------------------------------

def test_ledger_is_authoritative_and_reconciles(res):
    a = res["accounting"]
    assert a["ledger_rows_after"] - a["ledger_rows_before"] == a["new_rows"] == 32
    assert a["billable_rows"] + a["nonbillable_rows"] == 32
    assert round(a["ending_ledger_usd"] - a["starting_ledger_usd"], 6) == a["run_attributed_cost_usd"]
    assert round(a["case_row_attributed_usd"] + a["unattributed_billed_failure_usd"], 6) == \
        a["run_attributed_cost_usd"]


def test_account_usage_matches_the_ledger(res):
    a = res["accounting"]
    assert round(a["ending_account_usage_usd"] - a["starting_account_usage_usd"], 6) == \
        a["run_attributed_cost_usd"]
    assert abs(a["rounding_difference_usd"]) < 1e-6
    assert a["account_matches_ledger"] is True


def test_spend_stayed_within_the_authorized_actual_ceiling(res):
    a = res["accounting"]
    assert a["run_attributed_cost_usd"] <= a["authorized_actual_ceiling_usd"]
    assert a["within_authorized_ceiling"] is True


def test_finish_reasons_account_for_every_call(res):
    assert sum(res["accounting"]["finish_reasons"].values()) == 32


# ---- append-only ---------------------------------------------------------------

def test_the_control_run_artifacts_were_not_modified():
    ctrl = Path("evaluation/model_selection/runs_seen32/ocr_primary/"
                "dev__seen46_ocr_dev__all__google-gemini-3.7-flash__c4ae61f634")
    rows = [json.loads(l) for l in (ctrl / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 32
    assert sum(1 for r in rows if r["ok"]) == 14, "the control arm must still read 14/32"


def test_control_numbers_in_the_result_match_the_frozen_control(res):
    c = res["arms"]["control_m2-strict-v1"]
    assert c["usable"] == 14 and c["provider_content_filter"] == 10
    assert c["successful_only_mean_cer"] == 0.1155
    assert c["failure_aware_cer"] == 0.613
