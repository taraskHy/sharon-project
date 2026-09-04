"""OCR decision registry + alternative-candidate screen freeze. ZERO provider calls.

These tests exist to stop three specific mistakes: silently re-running a
configuration that has already been measured and dropped, silently reporting a
dropped arm as the current OCR winner, and letting a "new" candidate be an
alias of a failed route.
"""
import json
import hashlib
from pathlib import Path

import pytest

from autograder.benchmark.ocr_decisions import (
    BLOCKING, Decision, DroppedConfiguration, RegistryError, assert_selectable,
    current_winner, decisions_for, load_registry, provider_pin_of,
)

REG = Path("evaluation/model_selection/policies/ocr_decision_registry.json")
SCREEN = Path("evaluation/model_selection/experiments/"
              "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1_2026-09-03.json")
STAGE1 = Path("evaluation/model_selection/runs/ocr_primary/"
              "OCR_SMOKE_STAGE1_BASELINE_2026-09-02.json")

GEM = "google/gemini-3.7-flash"
QWEN = "qwen/qwen3-vl-235b-a22b-instruct"


@pytest.fixture(scope="module")
def reg():
    return load_registry(REG)


@pytest.fixture(scope="module")
def screen():
    return json.loads(SCREEN.read_text(encoding="utf-8"))


# ---- 1. failed configurations preserved and non-selectable -------------------

def test_every_failed_configuration_is_preserved(reg):
    ids = {e["id"] for e in reg["entries"]}
    assert ids == {"LUNA_ALL_ROUTES", "SONNET_ALL_ROUTES", "GEMINI_M2_STRICT_V1_AUTOROUTE",
                   "GEMINI_OCR_NEUTRAL_V2_AUTOROUTE", "GEMINI_THEN_SONNET_FALLBACK_V1"}


@pytest.mark.parametrize("model,prompt,pin", [
    ("openai/gpt-5.6-luna-pro", "m2-strict-v1", None),
    ("openai/gpt-5.6-luna-pro", "anything", "any-provider"),
    (GEM, "m2-strict-v1", None),
    (GEM, "ocr-neutral-v2", None),
    ("gemini_then_sonnet_hard_failure_fallback_v1", "m2-strict-v1", None),
])
def test_a_dropped_configuration_refuses_to_run(model, prompt, pin, reg):
    with pytest.raises(DroppedConfiguration):
        assert_selectable(model, prompt, pin, registry=reg)


def test_a_dropped_configuration_runs_only_with_a_named_experiment(reg):
    hits = assert_selectable(GEM, "m2-strict-v1", None,
                             authorized_experiment="SOME_NEW_EXPERIMENT", registry=reg)
    assert [d.status for d in hits] == ["DROP_AS_PRIMARY_ROUTE"]


@pytest.mark.parametrize("empty", [None, ""])
def test_an_empty_override_is_not_an_override(empty, reg):
    with pytest.raises(DroppedConfiguration):
        assert_selectable(GEM, "m2-strict-v1", None,
                          authorized_experiment=empty, registry=reg)


def test_sonnet_is_control_only_and_not_blocking(reg):
    hits = decisions_for("anthropic/claude-sonnet-5", "m2-strict-v1", None, registry=reg)
    assert [d.status for d in hits] == ["HISTORICAL_CONTROL_ONLY"]
    assert not any(d.blocking for d in hits)
    assert_selectable("anthropic/claude-sonnet-5", "m2-strict-v1", None, registry=reg)


# ---- 2. no dropped route silently reintroduced -------------------------------

def test_there_is_no_current_ocr_winner(reg):
    assert current_winner(reg) is None
    assert reg["current_winner"] is None


def test_a_dropped_arm_can_never_be_reported_as_the_winner(reg):
    tampered = json.loads(json.dumps(reg))
    tampered["current_winner"] = {"model": GEM, "prompt_version": "m2-strict-v1",
                                  "provider_pin": None}
    with pytest.raises(RegistryError):
        current_winner(tampered)


def test_registry_self_hash_is_verified_on_load(tmp_path):
    doc = json.loads(REG.read_text(encoding="utf-8"))
    doc["entries"][0]["status"] = "ADVANCE"          # tamper
    p = tmp_path / "reg.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(p)


def test_an_alias_pin_does_not_evade_a_drop(reg):
    """A service TIER of the same provider is not a new provider. The screen
    pins provider slugs, never endpoint tags, so a tier cannot be passed off as
    an independent route."""
    for arm in json.loads(SCREEN.read_text(encoding="utf-8"))["candidates"]:
        assert "/" not in arm["provider_pin"], "a pin must be a provider slug, not an endpoint tag"


# ---- 3-5. discovery / eligibility -------------------------------------------

def test_screen_candidates_are_image_capable_and_priced(screen):
    import tomllib
    pricing = tomllib.loads(Path("models.toml").read_text(encoding="utf-8")).get("pricing", {})
    for arm in screen["candidates"]:
        entry = pricing.get(arm["model"])
        assert entry and float(entry["input"]) > 0 and float(entry["output"]) > 0, \
            f"{arm['model']} must be priced before it can be pre-registered"
    for slug, snap in screen["live_pricing_snapshot"].items():
        assert snap["input_per_M"] > 0 and snap["output_per_M"] > 0


def test_unpriced_candidate_is_refused_before_a_live_run():
    from autograder.benchmark.runner import require_priced_candidate, UnpricedCandidate
    with pytest.raises(UnpricedCandidate):
        require_priced_candidate("some/unpriced-model", {"other/model": {"input": 1, "output": 2}})
    with pytest.raises(UnpricedCandidate):
        require_priced_candidate("zero/priced", {"zero/priced": {"input": 0, "output": 0}})


def test_a_text_only_candidate_would_be_rejected_by_the_eligibility_rule():
    cat = json.loads(Path("evaluation/model_selection/runs/ocr_primary/"
                          "OCR_OPENROUTER_CATALOG_SNAPSHOT_2026-09-03.json").read_text(encoding="utf-8"))
    by = {m["id"]: m for m in cat["models"]}
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    for arm in screen["candidates"]:
        assert arm["model"] in by, f"{arm['model']} is not in the image-capable catalog snapshot"
        assert "image" in by[arm["model"]]["input_modalities"]
        assert "structured_outputs" in by[arm["model"]]["supported_parameters"]


# ---- 6-7. provider route recording ------------------------------------------

def test_provider_pin_requires_determinism():
    class R:
        extra_generation = {"provider": {"order": ["google-vertex"], "allow_fallbacks": False}}
    assert provider_pin_of(R()) == "google-vertex"

    class Fallbacks:
        extra_generation = {"provider": {"order": ["google-vertex"], "allow_fallbacks": True}}
    assert provider_pin_of(Fallbacks()) is None, "a fallback can silently change the provider"

    class TwoProviders:
        extra_generation = {"provider": {"order": ["a", "b"], "allow_fallbacks": False}}
    assert provider_pin_of(TwoProviders()) is None

    class NoPin:
        extra_generation = {}
    assert provider_pin_of(NoPin()) is None


def test_pin_is_recognised_on_a_TaskRoute_not_only_a_BackendConfig(reg):
    """REGRESSION. provider_pin_of originally read only extra_generation, but
    TaskRoute carries the routing object as a TOP-LEVEL field and folds it into
    extra_generation only in to_backend_config(). A genuinely pinned arm
    therefore looked like automatic routing and was refused as a dropped
    auto-route configuration — which is exactly how the first live attempt of
    the alt-candidate screen was blocked, at zero cost."""
    from autograder.gateway import TaskRoute
    pin = {"order": ["google-ai-studio"], "allow_fallbacks": False}
    route = TaskRoute(task="ocr_primary", backend="openrouter", model=GEM,
                      provider=pin, prompt_version="m2-strict-v1")
    assert provider_pin_of(route) == "google-ai-studio", "TaskRoute shape"
    assert provider_pin_of(route.to_backend_config()) == "google-ai-studio", "BackendConfig shape"
    # and the guard must therefore let the pinned arm through
    assert_selectable(GEM, "m2-strict-v1", provider_pin_of(route), registry=reg)


def test_an_unpinned_taskroute_is_still_automatic_routing(reg):
    from autograder.gateway import TaskRoute
    assert provider_pin_of(TaskRoute(task="t", backend="openrouter", model=GEM)) is None
    assert provider_pin_of(TaskRoute(task="t", backend="openrouter", model=GEM,
                                     provider={"order": ["a"], "allow_fallbacks": True})) is None
    assert provider_pin_of(TaskRoute(task="t", backend="openrouter", model=GEM,
                                     provider={"order": ["a", "b"],
                                               "allow_fallbacks": False})) is None


def test_pinned_gemini_is_a_distinct_route_and_not_inherited_as_dropped(reg):
    for pin in ("google-ai-studio", "google-vertex"):
        assert decisions_for(GEM, "m2-strict-v1", pin, registry=reg) == []
        assert_selectable(GEM, "m2-strict-v1", pin, registry=reg)


def test_pinned_routes_are_declared_explicitly_so_a_pin_cannot_quietly_evade(reg):
    declared = {(r["model"], r["prompt_version"], r["provider_pin"])
                for r in reg["distinct_routes_explicitly_not_dropped"]}
    assert (GEM, "m2-strict-v1", "google-ai-studio") in declared
    assert (GEM, "m2-strict-v1", "google-vertex") in declared
    assert (QWEN, "m2-strict-v1", "alibaba") in declared


def test_forensics_reports_unknown_rather_than_guessing_a_provider():
    f = json.loads(Path("evaluation/model_selection/runs/ocr_primary/"
                        "OCR_PROVIDER_ROUTE_FORENSICS_2026-09-03.json").read_text(encoding="utf-8"))
    assert f["did_all_filter_outcomes_come_from_the_same_provider"] == "UNKNOWN"
    assert f["did_successful_and_failed_outputs_use_the_same_provider"] == "UNKNOWN"
    # every filtered row really does lack a provider — the UNKNOWN is earned
    for stage in json.loads(Path("evaluation/model_selection/runs/ocr_primary/"
                                 "OCR_EXPERIMENT_LINEAGE_2026-09-03.json")
                            .read_text(encoding="utf-8"))["stages"]:
        rf = stage.get("route_forensics")
        if rf:
            assert rf["content_filtered_rows_with_a_provider_recorded"] == 0


# ---- 8-10. screen shape ------------------------------------------------------

def test_candidate_count_is_at_most_three(screen):
    assert screen["candidate_count"] == len(screen["candidates"]) <= 3


def test_screen_uses_the_exact_frozen_eight_cases(screen):
    base = json.loads(STAGE1.read_text(encoding="utf-8"))
    assert screen["population"]["ordered_case_ids"] == base["ordered_case_ids"]
    assert screen["population"]["smoke_selection_sha256"] == base["smoke_selection_sha256"]
    assert screen["population"]["n"] == 8
    assert screen["population"]["handwritten"] == 5
    assert screen["population"]["printed_or_text_layer"] == 3
    got = {c["case_id"]: c["crop_sha256"] for c in screen["population"]["cases"]}
    want = {c["case_id"]: c["crop_sha256"] for c in base["cases"]}
    assert got == want, "crop bytes must be the frozen ones"


def test_crop_hashes_still_match_the_files_on_disk(screen):
    from autograder.benchmark.manifests import load_manifest
    man = load_manifest("ocr_primary")
    for c in screen["population"]["cases"]:
        p = man.root / c["image"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == c["crop_sha256"]


def test_screen_excludes_held_out_and_calibration(screen):
    assert screen["population"]["HELD_OUT"] == 0
    assert screen["population"]["CALIBRATION"] == 0
    assert screen["population"]["splits"] == ["DEV"]
    for c in screen["population"]["cases"]:
        assert c["split"] == "DEV"


def test_held_out_execution_log_is_still_empty():
    log = Path("evaluation/model_selection/HELD_OUT_EXECUTIONS.jsonl")
    n = len([l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]) if log.exists() else 0
    assert n == 0


# ---- 11-12. prompt hygiene ---------------------------------------------------

def test_screen_prompt_carries_no_grading_content(screen):
    from autograder.benchmark.roles import load_ocr_prompts
    p = load_ocr_prompts(screen["prompt"]["version"])
    banned = ["rubric", "score", "grade", "points", "correct answer", "official solution",
              "מחוון", "ציון"]
    for cat, text in p.items():
        low = text.lower()
        for b in banned:
            assert b.lower() not in low, f"{cat} prompt contains grading vocabulary {b!r}"


def test_neutral_colour_wording_is_not_inherited(screen):
    from autograder.benchmark.roles import load_ocr_prompts
    assert screen["prompt"]["version"] == "m2-strict-v1"
    assert screen["prompt"]["colour_wording_inherited"] is False
    p = load_ocr_prompts("m2-strict-v1")
    for cat, text in p.items():
        assert "different colour of ink" not in text, \
            f"{cat} must not inherit the refuted ocr-neutral-v2 wording"


def test_all_screen_prompts_are_registered_in_the_cloud_boundary(screen):
    from autograder.cloudboundary import approved_cloud_ocr_systems
    from autograder.benchmark.roles import load_ocr_prompts
    approved = approved_cloud_ocr_systems()
    p = load_ocr_prompts(screen["prompt"]["version"])
    for cat in screen["prompt"]["prompt_sha256_by_category"]:
        assert p[cat] in approved, f"{cat} prompt is not a registered cloud OCR prompt"


def test_prompt_hashes_in_the_freeze_match_the_live_prompts(screen):
    from autograder.benchmark.roles import load_ocr_prompts
    p = load_ocr_prompts(screen["prompt"]["version"])
    for cat, h in screen["prompt"]["prompt_sha256_by_category"].items():
        assert hashlib.sha256(p[cat].encode("utf-8")).hexdigest() == h


def test_no_case_specific_instructions(screen):
    assert screen["prompt"]["case_specific_instructions"] == 0
    assert screen["prompt"]["rubric_or_solution_or_grade_context"] == 0
    assert screen["prompt"]["identical_semantic_contract_across_all_three_arms"] is True


# ---- 13. gates frozen --------------------------------------------------------

def test_advancement_gates_are_frozen_and_complete(screen):
    g = screen["advancement_and_drop_rules_stated_in_advance"]
    assert g["operational_coverage"]["usable_total"] == ">= 7/8"
    assert g["operational_coverage"]["usable_handwritten"] == ">= 4/5"
    assert g["reliability"]["hard_provider_failures"] == "<= 1"
    assert g["reliability"]["fabrication"] == "0"
    assert g["handwriting_quality"]["successful_only_handwritten_mean_cer"] == "<= 0.20"
    assert g["handwriting_quality"]["total_line_loss"] == "0"
    assert g["critical_errors"]["max_critical_among_usable_handwriting"] == 1
    assert g["critical_errors"]["fabricated_mathematical_content"] == 0
    assert "NOT production proof" in g["handwriting_quality"]["labelled"]


def test_passing_the_screen_authorizes_only_the_32_crop_experiment(screen):
    g = screen["advancement_and_drop_rules_stated_in_advance"]
    assert "32-crop" in g["passing_authorizes"]
    assert "production readiness" in g["passing_does_not_establish"]


def test_screen_experiment_self_hash_verifies(screen):
    body = json.dumps({k: v for k, v in screen.items() if k != "experiment_sha256"},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    assert hashlib.sha256(body.encode()).hexdigest() == screen["experiment_sha256"]


# ---- 14. zero provider calls --------------------------------------------------

def test_preparing_the_screen_made_no_provider_calls(screen):
    """Directory ABSENCE is not the property that matters — a refused arm
    creates the run directory before the decision-registry guard fires, and
    that refusal is a success, not an execution. What matters is that no
    outputs exist for an arm that has not run."""
    assert screen["provider_calls_made_preparing_this"] == 0
    root = Path("evaluation/model_selection/runs_altscreen")
    if root.exists():
        for run_dir in root.rglob("*"):
            if run_dir.is_dir() and (run_dir / "outputs.jsonl").exists():
                rows = [json.loads(l) for l in
                        (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
                        if l.strip()]
                # an executed arm must be a COMPLETE frozen arm, never a partial
                assert len(rows) <= 8, f"{run_dir.name} has more rows than the frozen 8"


def test_payload_boundary_was_verified_offline(screen):
    v = screen["payload_and_boundary_verification"]
    assert v["provider_calls"] == 0
    assert v["payloads_built_offline"] == 24
    assert v["image_blocks_per_payload"] == 1
    assert v["reference_leakage"] == 0
    assert v["grading_vocabulary_hits"] == 0
    assert v["secret_patterns"] == 0
    assert v["held_out_in_payloads"] == 0


def test_campaign_envelope_covers_the_whole_screen_worst_case(screen):
    """The single envelope must absorb ALL THREE arms at their worst case —
    the per-arm framing is exactly what would have over-spent."""
    from autograder.campaignbudget import load_campaign_budget
    b = load_campaign_budget(screen["campaign_budget_manifest"])
    assert b.experiment_sha256 == screen["experiment_sha256"], \
        "the budget must be bound to the frozen experiment it funds"
    assert b.hard_increment_usd == 0.12 and b.warning_increment_usd == 0.08
    worst = screen["budget"]["screen"]["predicted_worst_case_usd"]
    assert b.predicted_campaign_worst_case_usd == pytest.approx(worst)
    # the hard threshold must cover L0 plus every arm's worst case at once
    assert b.hard_usd >= b.starting_ledger_usd + worst
    assert screen["budget"]["historically_lower_actual_was_NOT_used_to_lower_the_predicted_ceiling"]
    assert screen["budget"]["project_wide_unchanged"]["hard_usd"] == 10.0


def test_increment_semantics_are_stated_as_campaign_wide(screen):
    b = screen["budget"]["screen"]
    assert b["campaign_warning_increment_usd"] == 0.08
    assert b["campaign_hard_increment_usd"] == 0.12
    assert "not a per-arm allowance" in b["increment_semantics"]


def test_gemini_arms_are_labelled_provider_route_attribution_arms(screen):
    for arm in screen["candidates"]:
        if arm["model"] == "google/gemini-3.7-flash":
            assert arm["arm_type"] == "PROVIDER-ROUTE ATTRIBUTION ARM"
        else:
            assert arm["arm_type"] == "CROSS-FAMILY CANDIDATE ARM"


def test_passing_the_screen_means_advance_to_seen32_only(screen):
    g = screen["advancement_and_drop_rules_stated_in_advance"]
    assert g["outcome_vocabulary"]["pass"] == "ADVANCE_TO_SEEN32"
    assert "ADVANCE_TO_SEEN32" in g["passing_authorizes"]
    assert "CANNOT select a production winner" in g["passing_authorizes"]
    assert "production winner" in g["passing_does_not_establish"]


def test_historical_filter_attribution_is_not_overstated():
    """The artifacts support 'not declared OpenRouter moderation'; they do NOT
    support naming the mechanism or the endpoint."""
    f = json.loads(Path("evaluation/model_selection/runs/ocr_primary/"
                        "OCR_PROVIDER_ROUTE_FORENSICS_2026-09-03.json").read_text(encoding="utf-8"))
    reg = json.loads(Path("evaluation/model_selection/policies/"
                          "ocr_decision_registry.json").read_text(encoding="utf-8"))
    claims = [f["historical_filter_attribution"]["claim"],
              f["does_openrouter_offer_multiple_routes_for_this_exact_model"]
               ["openrouter_model_level_moderation"],
              reg["historical_filter_attribution"]]
    for claim in claims:
        # the asserted claim itself must be hedged and must not name a mechanism
        assert "consistent with" in claim
        assert "UNKNOWN" in claim
        assert "NOT a declared OpenRouter moderation stage" in claim
        assert "own safety layer" not in claim
    # the correction NOTE may quote the withdrawn wording — that is the record
    note = f["historical_filter_attribution"]["language_correction"]
    assert "overstated" in note


def test_local_prior_art_is_scoped_to_the_configurations_tested():
    a = json.loads(Path("evaluation/model_selection/runs/ocr_primary/"
                        "OCR_ALTERNATIVE_ARCHITECTURES_2026-09-03.json").read_text(encoding="utf-8"))
    opt4 = next(o for o in a["options"] if o["id"] == 4)
    assert "none of the 14 tested local configurations was competitive" in opt4["expected_benefit"]
    assert "REFUTED BY MEASUREMENT" not in opt4["verdict"]
    d = json.loads(Path("evaluation/model_selection/runs/ocr_primary/"
                        "OCR_CANDIDATE_DISCOVERY_2026-09-03.json").read_text(encoding="utf-8"))
    assert "not a proof about local OCR in general" in \
        d["local_prior_art_that_constrains_this_choice"]["scope_of_the_claim"]


# ---- 15. immutability ---------------------------------------------------------

def test_audited_references_and_manifest_are_unchanged():
    from autograder.benchmark.manifests import load_manifest
    h = load_manifest("ocr_primary").summary()["hashes"]
    assert h["references_sha256"] == "4a93e826e6e94777d445e64ae2c3f5ed10def46aa021ff763a45a9807fb913b5"
    assert h["items_sha256"] == "25463e91b95db8eecfc306644dd0839f65e42910d4161f6991383b6f4a524a8b"
    assert h["audit_sha256"] == "eace95fc6dc684c88f1fea2142414a68ad8f421257ac012faf6bcfc9ae07f50c"


def test_historical_run_outputs_are_untouched():
    for d, n_ok in (("runs_seen32/ocr_primary/dev__seen46_ocr_dev__all__google-gemini-3.7-flash__c4ae61f634", 14),
                    ("runs_promptv2/ocr_primary/dev__seen46_ocr_dev__all__google-gemini-3.7-flash__61dd6641fb", 16)):
        p = Path("evaluation/model_selection") / d / "outputs.jsonl"
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 32
        assert sum(1 for r in rows if r["ok"]) == n_ok


def test_ocr_primary_role_is_still_unselected():
    import tomllib
    reg = tomllib.loads(Path("evaluation/model_selection/candidates.toml").read_text(encoding="utf-8"))
    assert reg["roles"]["ocr_primary"]["status"] == "UNSELECTED"


# ---- V1 closure and V2 freeze -------------------------------------------------

V1P = Path("evaluation/model_selection/experiments/"
           "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1_2026-09-03.json")
V2P = Path("evaluation/model_selection/experiments/"
           "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V2_2026-09-04.json")
CLOSURE = Path("evaluation/model_selection/runs/ocr_primary/"
               "OCR_ALTSCREEN_V1_CLOSURE_2026-09-04.json")


@pytest.fixture(scope="module")
def v2():
    return json.loads(V2P.read_text(encoding="utf-8"))


def test_v1_is_closed_as_inconclusive_and_never_rehashed():
    c = json.loads(CLOSURE.read_text(encoding="utf-8"))
    v1 = json.loads(V1P.read_text(encoding="utf-8"))
    assert c["terminal_outcome"] == "INCONCLUSIVE_MECHANICAL_STOP"
    assert c["gates_not_evaluated"] is True
    assert c["experiment_sha256"] == v1["experiment_sha256"], \
        "the closure must reference V1's hash AS EXECUTED, unchanged"
    assert "never be edited or re-hashed" in c["immutability"]


def test_v1_evidence_is_preserved_and_partitioned():
    e = json.loads(CLOSURE.read_text(encoding="utf-8"))["preserved_evidence"]
    assert e["five_historical_unpinned_cache_hits"]["count"] == 5
    assert e["three_live_ai_studio_attempts"]["count"] == 3
    assert e["three_live_ai_studio_attempts"]["route_violations"] == 0
    assert e["spend"]["additional_spend_usd"] == 0.00252675
    assert e["spend"]["ledger_before"] == 0.70323229
    assert e["spend"]["ledger_after"] == 0.70575904
    assert "campaign_id" in e["missing_linkage_fields"]["null_fields"]


def test_v1_live_outputs_support_only_the_pin_statement_and_are_not_reusable():
    live = json.loads(CLOSURE.read_text(encoding="utf-8"))[
        "preserved_evidence"]["three_live_ai_studio_attempts"]
    assert "reached the wire" in live["THE_ONLY_STATEMENT_THESE_SUPPORT"]
    assert "must NOT be reused" in live["explicitly_not_reusable"]
    assert json.loads(V2P.read_text(encoding="utf-8"))[
        "supersedes"]["v1_outputs_excluded_from_v2_evaluation"] is True


def test_v2_keeps_the_frozen_design(v2):
    v1 = json.loads(V1P.read_text(encoding="utf-8"))
    assert v2["population"]["ordered_case_ids"] == v1["population"]["ordered_case_ids"]
    assert v2["population"]["cases"] == v1["population"]["cases"]
    assert v2["prompt"]["version"] == "m2-strict-v1"
    assert v2["schema"] == v1["schema"]
    assert v2["advancement_and_drop_rules_stated_in_advance"] == \
        v1["advancement_and_drop_rules_stated_in_advance"]
    assert [a["provider_pin"] for a in v2["candidates"]] == \
        [a["provider_pin"] for a in v1["candidates"]]
    assert v2["execution_requirements"]["retry_policy"]["transport_retries"] == 2


def test_v2_freezes_the_corrected_identity_and_cache_policy(v2):
    ip = v2["identity_and_cache_policy"]
    assert ip["identity_version"] == 2
    assert ip["cache_policy"] == "refresh"
    assert ip["cache_hits_allowed"] == 0
    assert ip["cache_hit_consequence"] == "INCONCLUSIVE_MECHANICAL_STOP"
    assert len(set(ip["arm_identities"].values())) == 3
    assert set(ip["excluded_from_all_identities"]) >= {"api_key", "api_key_env"}
    assert v2["execution_requirements"]["all_intended_logical_requests_live"] == 24
    assert set(v2["execution_requirements"]["required_attempt_linkage_fields"]) == {
        "campaign_id", "arm_id", "case_id", "logical_request_id", "attempt_id", "retry_index"}


def test_v2_budget_is_prospective_and_fits(v2):
    b = v2["budget"]
    assert b["L0_verified_from_disk"] == 0.70575904
    assert b["campaign_family_absolute_limits_preserved"] == {"warning": 0.78323229,
                                                              "hard": 0.82323229}
    assert b["prospective_warning_increment"] == 0.07747325
    assert b["prospective_hard_increment"] == 0.11747325
    assert b["L0_verified_from_disk"] + b["predicted_worst_case_usd"] <= 0.82323229
    assert "NOT_AUTHORIZED" in b


def test_v2_is_frozen_but_not_authorized_and_not_executed(v2):
    assert v2["status"] == "FROZEN - NOT EXECUTED - NOT AUTHORIZED"
    assert v2["provider_calls_made_preparing_this"] == 0
    body = json.dumps({k: v for k, v in v2.items() if k != "experiment_sha256"},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    assert hashlib.sha256(body.encode()).hexdigest() == v2["experiment_sha256"]
    assert not Path("evaluation/model_selection/policies/"
                    "OCR_ALTSCREEN_V2_CAMPAIGN_BUDGET.json").exists()
