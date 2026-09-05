"""V4 closure and V5 freeze: the artifacts must say what the campaign relies on.

V5 exists for exactly one reason: the `alibaba` slug had no evidenced display
name, V4 preregistered no procedure for resolving one, and amending a frozen
artifact is not available. So V4 was closed unexecuted and V5 was frozen on top
of a catalogue captured under a protocol that was itself frozen first.

These tests hold that chain together. ZERO network.
"""
import hashlib
import json
from pathlib import Path

import pytest

E = Path("evaluation/model_selection/experiments")
R = Path("evaluation/model_selection/runs/ocr_primary")
V4_P = E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V4_2026-09-05.json"
V5_P = E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V5_2026-09-06.json"
PROTO_P = E / "OCR_PROVIDER_METADATA_CAPTURE_PROTOCOL_V1_2026-09-06.json"
SNAP_P = R / "OCR_METADATA_SNAPSHOT_2026-09-06.json"
CLOSURE_P = R / "OCR_ALTSCREEN_V4_CLOSURE_2026-09-06.json"

HARD_ABS = 0.82323229
WARN_ABS = 0.78323229


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _selfhash(doc, field):
    body = {k: v for k, v in doc.items() if k != field}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, indent=1,
                                     sort_keys=True, default=str).encode()).hexdigest()


@pytest.fixture(scope="module")
def v5():
    return _load(V5_P)


@pytest.fixture(scope="module")
def snap():
    return _load(SNAP_P)


# =============================================================================
# the chain of custody
# =============================================================================

@pytest.mark.parametrize("path,field", [
    (V5_P, "experiment_sha256"),
    (PROTO_P, "content_sha256"),
    (SNAP_P, "content_sha256"),
    (CLOSURE_P, "content_sha256"),
])
def test_every_artifact_in_the_chain_verifies_its_own_hash(path, field):
    doc = _load(path)
    assert doc[field] == _selfhash(doc, field), f"{path.name} does not match its recorded {field}"


def test_v4_is_closed_unexecuted_and_v5_records_the_closure(v5):
    closure = _load(CLOSURE_P)
    assert closure["outcome"] == "SUPERSEDED_BEFORE_EXECUTION"
    assert closure["paid_model_calls"] == 0
    assert closure["metadata_calls"] == 0
    assert closure["additional_spend_usd"] == 0.0
    assert closure["all_v4_outputs_nonexistent"] is True
    assert closure["v4_budget_manifest_created"] is False
    sup = v5["supersedes"]
    assert sup["experiment"] == "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V4"
    assert sup["experiment_sha256"] == _load(V4_P)["experiment_sha256"]
    assert sup["closure_sha256"] == closure["content_sha256"]
    assert sup["outcome"] == "SUPERSEDED_BEFORE_EXECUTION"
    assert sup["v1_v2_v3_v4_outputs_excluded_from_v5_evaluation"] is True


def test_the_protocol_was_frozen_before_the_capture_it_governs(v5, snap):
    """The whole point: acceptance criteria cannot be chosen after seeing data."""
    proto = _load(PROTO_P)
    assert snap["protocol"] == proto["protocol"]
    assert snap["protocol_sha256"] == proto["content_sha256"]
    assert v5["metadata_prerequisite"]["protocol_sha256"] == proto["content_sha256"]
    assert v5["metadata_prerequisite"]["snapshot_sha256"] == snap["content_sha256"]


def test_v5_changed_nothing_but_the_slug_resolution(v5):
    """A superseding freeze that quietly moved the experiment would invalidate
    the comparison it exists to make."""
    v4 = _load(V4_P)
    for key in ("question", "population", "prompt", "schema", "adapter_version",
                "advancement_and_drop_rules_stated_in_advance", "prohibitions",
                "execution_requirements", "live_pricing_snapshot"):
        assert v5[key] == v4[key], f"V5 silently changed {key!r}"
    for a5, a4 in zip(v5["candidates"], v4["candidates"]):
        assert a5["arm_id"] == a4["arm_id"]
        assert a5["experiment_identity"] == a4["experiment_identity"]
        assert a5["cli_argv"] == a4["cli_argv"]
        assert a5["provider_pin"] == a4["provider_pin"]


# =============================================================================
# the three arms
# =============================================================================

def test_all_three_arms_are_retained_including_qwen(v5):
    ids = [a["arm_id"] for a in v5["candidates"]]
    assert ids == ["gemini_pinned_ai_studio", "gemini_pinned_vertex",
                   "qwen3_vl_235b_pinned_alibaba"]
    assert v5["candidate_count"] == 3


def test_every_arm_now_has_a_VERIFIED_provider_mapping(v5, snap):
    for arm in v5["candidates"]:
        pm = arm["provider_mapping"]
        assert pm["slug_mapping_status"] == "VERIFIED", arm["arm_id"]
        assert pm["requested_slug"] == arm["provider_pin"]
        assert pm["resolved_display_name"] == \
            snap["acceptance"]["resolved_mapping"][arm["provider_pin"]]


def test_the_arms_are_three_distinct_experiments(v5):
    assert len({a["experiment_identity"] for a in v5["candidates"]}) == 3
    for a in v5["candidates"]:
        assert len(a["experiment_identity"]) == 64
        assert set(a["experiment_identity"]) <= set("0123456789abcdef")


def test_every_arm_has_a_semantic_identity_for_all_eight_cases(v5):
    order = v5["population"]["ordered_case_ids"]
    assert len(order) == 8
    seen = set()
    for a in v5["candidates"]:
        sem = a["semantic_request_identity_by_case"]
        assert list(sem) == order, a["arm_id"]
        assert all(len(h) == 64 for h in sem.values())
        # a per-case identity that repeated would mean the image is not in it
        assert len(set(sem.values())) == 8, a["arm_id"]
        seen |= set(sem.values())
    assert len(seen) == 24, "no semantic identity may collide across arms"


def test_the_qwen_arm_clears_reasoning_rather_than_inheriting_it(v5):
    """models.toml's production default would send `reasoning`, which this model
    does not support — that is the Stage-1 HTTP 400, waiting to happen again."""
    qwen = next(a for a in v5["candidates"] if a["arm_id"] == "qwen3_vl_235b_pinned_alibaba")
    eg = qwen["effective_config"]["extra_generation"]
    assert "reasoning" not in eg, eg
    assert qwen["effective_config"]["max_tokens"] == 1000


def test_local_qwen_failures_are_recorded_as_risk_not_as_a_verdict(v5):
    risk = v5["qwen_arm_prior_risk"]
    assert "RETAINED" in risk["status"]
    assert "prior risk" in risk["what_that_evidence_IS"]
    assert "not" in risk["what_that_evidence_IS_NOT"].lower()
    assert "dominated" in risk["what_that_evidence_IS_NOT"]


# =============================================================================
# the metadata prerequisite
# =============================================================================

def test_the_capture_accepted_with_zero_failures(v5, snap):
    mp = v5["metadata_prerequisite"]
    assert mp["result"] == "ACCEPTED"
    assert mp["failure_count"] == 0
    assert snap["acceptance"]["result"] == "ACCEPTED"


def test_acceptance_is_recomputable_from_the_archived_raw_bodies(snap):
    """The recorded verdict must be a function of the recorded evidence, not a
    claim about it.

    Note the snapshot keeps its derived sections under `parsed`, which is not
    the shape `evaluate_acceptance` reads — so this re-derives from the raw
    bodies instead. That is the stronger check: primary evidence, re-parsed,
    through the same pure evaluator that was frozen before the capture.
    """
    from autograder.metadatacapture import reevaluate_from_archive

    again = reevaluate_from_archive(snap, R / "OCR_METADATA_RAW_2026-09-06")
    rec = snap["acceptance"]
    assert again["result"] == rec["result"] == "ACCEPTED"
    assert again["failure_count"] == rec["failure_count"] == 0
    assert again["resolved_mapping"] == rec["resolved_mapping"]
    assert again["effective_prices_per_m"] == rec["effective_prices_per_m"]
    assert again["conservative_arm_costs_usd"] == rec["conservative_arm_costs_usd"]


def test_re_derivation_refuses_a_raw_body_that_changed(snap, tmp_path):
    """If the archived evidence were altered, re-evaluation must fail loudly
    rather than quietly re-accept."""
    import shutil

    from autograder.metadatacapture import reevaluate_from_archive

    shutil.copytree(R / "OCR_METADATA_RAW_2026-09-06", tmp_path / "raw")
    p = tmp_path / "raw" / "providers.json"
    p.write_text(p.read_text(encoding="utf-8").replace("Alibaba", "Alibaba Cloud", 1),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="has changed"):
        reevaluate_from_archive(snap, tmp_path / "raw")


def test_raw_bodies_are_archived_and_hash_to_what_the_snapshot_claims(snap):
    raw_dir = R / "OCR_METADATA_RAW_2026-09-06"
    assert raw_dir.is_dir()
    for key, rec in snap["raw_bodies"].items():
        f = raw_dir / Path(rec["file"]).name
        assert f.exists(), f"missing archived body for {key}"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == rec["raw_body_sha256"], key


def test_no_credential_was_sent_and_none_is_archived(snap):
    assert snap["method"] == "GET"
    assert snap["authorization_header_sent"] is False
    assert snap["credentials_sent"] is False
    assert snap["request_body_sent"] is None
    assert snap["redirects_followed"] == 0
    assert snap["paid_calls"] == 0
    assert snap["model_calls"] == 0
    assert snap["inference_calls"] == 0
    assert snap["additional_spend_usd"] == 0.0
    for key, req in snap["requests"].items():
        assert req["http_status"] == 200, key
        assert req["headers"]["content-type"].startswith("application/json"), key
        assert req["archived"] is True and req["parse_error"] is None, key
        assert not ({h.lower() for h in req["headers"]}
                    & {"authorization", "x-api-key", "api-key", "cookie", "set-cookie"}), key
    # Scan for credential VALUES, not field names: `authorization_header_sent`
    # is a declaration that none was sent and must be allowed to appear.
    import re

    secret = re.compile(r"sk-or-v1-[A-Za-z0-9]|sk-[A-Za-z0-9]{16}|bearer\s+[A-Za-z0-9._-]{8}",
                        re.I)
    blob = json.dumps(snap, ensure_ascii=False)
    assert not secret.search(blob), "a secret-shaped value appears in the snapshot"
    for raw in (R / "OCR_METADATA_RAW_2026-09-06").glob("*.json"):
        assert not secret.search(raw.read_text(encoding="utf-8")), raw.name


def test_a_cheaper_live_price_never_lowers_the_ceiling(snap):
    """max(frozen, live). A price drop is not a reason to plan a larger spend."""
    from autograder.metadatacapture import FROZEN_PRICES

    for model, eff in snap["acceptance"]["effective_prices_per_m"].items():
        frozen = FROZEN_PRICES[model]
        assert eff["input_per_m"] >= frozen["input_per_m"]
        assert eff["output_per_m"] >= frozen["output_per_m"]


def test_the_snapshot_may_not_be_refreshed_by_editing_the_freeze(v5):
    rule = v5["metadata_prerequisite"]["staleness_rule"]
    assert "re-accept with 0 failures" in rule
    assert "14 days" in rule
    assert "never 'refreshed' by editing" in rule


# =============================================================================
# money — V5 authorizes nothing
# =============================================================================

def test_v5_creates_no_budget_manifest_and_authorizes_no_spend(v5):
    b = v5["budget"]
    assert b["no_v5_budget_manifest_created"] is True
    assert b["v4_authorization_unused_and_non_transferable"] is True
    assert "authorizes nothing" in b["NOT_AUTHORIZED"]
    assert "new explicit authorization" in b["NOT_AUTHORIZED"]
    assert v5["status"] == "FROZEN - NOT EXECUTED - NOT AUTHORIZED"
    assert not list(R.glob("OCR_ALTSCREEN_V5_CAMPAIGN_BUDGET*.json"))


def test_L0_matches_the_ledger_on_disk(v5):
    rows = [json.loads(l) for l in (Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl")
                                    ).read_text(encoding="utf-8").splitlines() if l.strip()]
    spent = round(sum(float(r.get("reported_cost") or 0) for r in rows
                      if r.get("cloud") and not r.get("cache_hit")), 8)
    assert v5["budget"]["L0_verified_from_disk"] == spent
    assert v5["budget"]["ledger_rows_at_freeze"] == len(rows)


def test_the_absolute_family_limits_were_not_raised(v5):
    lim = v5["budget"]["campaign_family_absolute_limits_preserved"]
    assert lim["warning"] == WARN_ABS and lim["hard"] == HARD_ABS


def test_the_complete_campaign_fits_under_the_hard_limit(v5):
    cm = v5["cost_model"]
    L0 = v5["budget"]["L0_verified_from_disk"]
    assert cm["complete_campaign_fits_under_the_hard_limit"] is True
    assert round(L0 + cm["conservative_campaign_maximum_usd"], 8) <= HARD_ABS
    assert cm["headroom_after_the_completion_bound_usd"] > 0


def test_the_conservative_maximum_is_the_sum_of_the_arms(v5):
    cm = v5["cost_model"]
    assert round(sum(cm["conservative_arm_costs_usd"].values()), 8) == \
        round(cm["conservative_campaign_maximum_usd"], 8)
    assert cm["matches_frozen_maximum"] is True


def test_nothing_was_spent_or_called_to_produce_this_freeze(v5):
    assert v5["provider_calls_made_preparing_this"] == 0
    assert v5["metadata_calls_made_preparing_this"] == 0


def test_held_out_remains_sealed(v5):
    assert v5["population"]["HELD_OUT"] == 0
    log = Path("evaluation/model_selection/HELD_OUT_EXECUTIONS.jsonl")
    rows = [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()] if log.exists() else []
    assert rows == []
