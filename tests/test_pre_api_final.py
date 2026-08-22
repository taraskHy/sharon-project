"""Final pre-API phase: reference provenance (129), smoke subsets, held-out
no-dry-run / no-preview, dataset label isolation (grading / MC / variant),
owner-label persistence, spend preflight (mocked), models.toml UNSELECTED,
readiness headline with no key. Offline only."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from autograder.benchmark.manifests import (VALID_REFERENCE_CLASSES, BenchmarkIntegrityError, load_manifest,
                                            reference_breakdown, validate_reference_provenance)
from autograder.benchmark.ownerlabels import OwnerLabelError, OwnerLabelStore, merge_owner_labels
from autograder.benchmark.roles import adapter_for
from autograder.benchmark.runner import HeldOutRefused, RunSpec, leakage_check, run_benchmark
from autograder.benchmark.smoke import (SmokeError, freeze_smoke, load_smoke, propose_smoke, smoke_case_ids,
                                        smoke_status)
from autograder.benchmark.status import all_role_statuses, role_dataset_status

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "evaluation" / "model_selection" / "datasets"
SMOKE = REPO / "evaluation" / "model_selection" / "smoke"


# ------------------------------------------------------------ references --

def test_all_129_references_have_an_explicit_valid_provenance():
    m = load_manifest("ocr_primary")
    b = reference_breakdown(m)
    assert b["total"] == 129
    h, o = b["handwritten_manual_audit"], b["other_categories_text_layer"]
    assert (h["count"], h["confirmed"], h["corrected"], h["ambiguous"]) == (102, 69, 33, 0)
    assert o["count"] == 27 and o["invalid"] == 0
    assert b["by_provenance_class"] == {"audited_confirmed": 69, "audited_corrected": 33, "text_layer_mechanical": 27}
    assert b["all_valid_for_strict_scoring"] and b["invalid_items"] == []
    validate_reference_provenance(m)                 # does not raise
    assert set(c.label["provenance_class"] for c in m.cases) <= set(VALID_REFERENCE_CLASSES)


def test_strict_scoring_refuses_an_item_without_admissible_provenance():
    m = load_manifest("ocr_primary")
    bad = m.cases[0]
    bad.label["provenance_class"] = "INVALID:audit_status_unchecked"
    bad.label["provenance_valid"] = False
    with pytest.raises(BenchmarkIntegrityError, match="lack an admissible reference provenance"):
        validate_reference_provenance(m, [bad.case_id])
    ad = adapter_for("ocr_primary")
    row = ad.score(bad, {"transcription": bad.label["reference"]}, None)
    assert not row["scored"] and row["skip_reason"].startswith("invalid_reference_source")
    agg = ad.aggregate([row], [])
    assert agg["refused_invalid_reference"] == 1 and agg["overall"]["cases"] == 0


# ----------------------------------------------------------------- smoke --

def test_frozen_smoke_subsets_are_dev_only_and_hash_bound():
    for role, n in (("ocr_primary", 8), ("ocr_verify", 12)):
        m = load_manifest(role)
        d = load_smoke(role, m)
        assert len(d["cases"]) == n and d["split"] == "DEV"
        ids = smoke_case_ids(role, m)
        assert all(c.split == "DEV" for c in m.cases if c.case_id in ids)
        # the frozen selection equals the deterministic rule output
        assert propose_smoke(role, m)["selection_sha256"] == d["selection_sha256"]
        assert smoke_status(role, m)["valid"]
    # verifier smoke covers REAL and SYNTHETIC, supported and error slots
    v = load_smoke("ocr_verify")
    slots = {c["slot"] for c in v["cases"]}
    assert {"supported_1", "real_omission", "real_substitution", "real_unsupported_addition",
            "real_number_sign_formula", "synthetic_digit_substitution"} <= slots


def test_smoke_subset_is_immutable(tmp_path):
    m = load_manifest("ocr_primary")
    froze = freeze_smoke("ocr_primary", m, tmp_path)
    with pytest.raises(SmokeError, match="never re-selected"):
        freeze_smoke("ocr_primary", m, tmp_path)
    p = tmp_path / "ocr_primary_smoke.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["cases"][0]["case_id"] = "hl_e003_q1_r3__l1"              # tamper
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SmokeError, match="hash mismatch"):
        load_smoke("ocr_primary", m, tmp_path)
    # a DEV-only check: a non-DEV id is refused even with a matching hash
    d = json.loads(json.dumps(froze))
    held = next(c.case_id for c in m.cases if c.split == "HELD_OUT")
    d["cases"][0]["case_id"] = held
    from autograder.benchmark.smoke import _selection_hash
    d["selection_sha256"] = _selection_hash(d["cases"])
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SmokeError, match="DEV only"):
        load_smoke("ocr_primary", m, tmp_path)


def test_smoke_subset_runs_only_on_dev(tmp_path):
    with pytest.raises(ValueError, match="DEV-only"):
        run_benchmark(RunSpec(role="ocr_verify", split="calibration", candidate="openai/gpt-5.6-luna",
                              subset="smoke", runs_root=tmp_path, state_root=tmp_path / "s",
                              held_out_log=tmp_path / "HO.jsonl", dry_run=True))
    res = run_benchmark(RunSpec(role="ocr_verify", split="dev", candidate="openai/gpt-5.6-luna", subset="smoke",
                                runs_root=tmp_path, state_root=tmp_path / "s", held_out_log=tmp_path / "HO.jsonl",
                                dry_run=True))
    assert res.cases_selected == 12 and res.cases_done == 12 and "smoke" in res.run_id


# -------------------------------------------------------------- held-out --

def test_held_out_cannot_be_dry_run_even_when_confirmed(tmp_path):
    for kw in ({}, {"confirm_held_out": True}, {"confirm_held_out": True, "final_evaluation": True}):
        with pytest.raises(HeldOutRefused, match="cannot be previewed/dry-run"):
            run_benchmark(RunSpec(role="ocr_verify", split="held_out", candidate="openai/gpt-5.6-luna",
                                  runs_root=tmp_path, state_root=tmp_path / "s",
                                  held_out_log=tmp_path / "HO.jsonl", dry_run=True, **kw))
    assert not (tmp_path / "HO.jsonl").exists()


def test_held_out_is_never_previewed_by_the_cli(capsys):
    from autograder.cli import main
    rc = main(["bench", "inspect", "--role", "ocr_verify", "--split", "held_out", "--preview"])
    assert rc == 3
    err = capsys.readouterr().err
    assert "cannot be previewed/dry-run" in err
    rc = main(["bench", "dry-run", "--role", "ocr_verify", "--split", "held_out", "--candidate", "openai/gpt-5.6-luna"])
    assert rc == 3
    rc = main(["bench", "run", "--role", "ocr_verify", "--split", "held_out", "--candidate", "openai/gpt-5.6-luna",
               "--i-understand-this-spends-money"])
    assert rc == 3
    assert not (REPO / "evaluation" / "model_selection" / "HELD_OUT_EXECUTIONS.jsonl").exists()


# ---------------------------------------------------------------- datasets --

@pytest.mark.skipif(not (DATASETS / "grade_primary" / "manifest.json").exists(), reason="grading dataset not built")
def test_grading_dataset_is_no_rag_and_labels_stay_separate():
    m = load_manifest("grade_primary")
    assert m.cases and m.status == "FROZEN"
    ad = adapter_for("grade_primary")
    for c in m.cases[:10]:
        assert set(c.inputs) == {"case_id", "pack", "selected", "transcription", "version"}
        assert not c.inputs["pack"].get("rag_evidence")
        req = ad.build_request(dict(c.inputs), m.root)
        leakage_check(c, req, ad.model_visible_fields)
        text = req.text_for_inspection()
        assert c.inputs["transcription"] in text
        for k in ("evidence_images", "transcription_items", "label_status"):
            assert k not in text
    st = role_dataset_status("grade_primary", m)
    assert st["status"] in ("NEEDS_OWNER_LABELS", "PARTIALLY_READY", "READY")
    if st["status"] != "READY":
        assert st["owner_action"]


@pytest.mark.skipif(not (DATASETS / "mc_resolve_cloud" / "manifest.json").exists(), reason="mc dataset not built")
def test_mc_dataset_never_shows_the_audited_answer():
    m = load_manifest("mc_resolve_cloud")
    ad = adapter_for("mc_resolve_cloud")
    assert len(m.cases) == 10
    for c in m.cases:
        assert set(c.inputs) == {"case_id", "band_png", "letters", "candidates"}
        assert "answer" in c.label and "label_provenance" in c.label
        req = ad.build_request(dict(c.inputs), m.root)
        leakage_check(c, req, ad.model_visible_fields)
        text = req.text_for_inspection()
        assert "audit" not in text.lower() and "answer:" not in text.lower()
        assert c.label["answer"] in c.inputs["candidates"]
    assert role_dataset_status("mc_resolve_cloud", m)["status"] == "READY"


@pytest.mark.skipif(not (DATASETS / "variant_resolve" / "manifest.json").exists(), reason="variant dataset not built")
def test_variant_dataset_isolates_the_variant_label():
    m = load_manifest("variant_resolve")
    ad = adapter_for("variant_resolve")
    assert len(m.cases) == 16
    for c in m.cases:
        assert set(c.inputs) == {"case_id", "versions", "cover_png"}
        req = ad.build_request(dict(c.inputs), m.root)
        leakage_check(c, req, ad.model_visible_fields)
        text = req.text_for_inspection()
        # the label variant id appears only as one of the generic version ids, never singled out
        assert c.label["variant"] in c.inputs["versions"]
        assert "suit" not in text.lower().replace("card-suit", "") or "Version ids" in text
        assert "label" not in text.lower()
    # scoring: a confident catalogue with the right variant is correct; wrong+confident is unsafe
    c = m.cases[0]
    ok = ad.score(c, {"n_variants": 1, "markers": [{"id": "m", "variant": c.label["variant"], "description": "x"}],
                      "confident": True}, None)
    bad = ad.score(c, {"n_variants": 1, "markers": [{"id": "m", "variant": "nope", "description": "x"}],
                       "confident": True}, None)
    assert ok["correct"] and not ok["unsafe_automatic"]
    assert bad["unsafe_automatic"]


def test_align_dataset_is_honestly_not_available():
    st = role_dataset_status("align_resolve")
    assert st["status"] == "NOT_AVAILABLE"
    assert all_role_statuses()["grade_escalate"]["status"] in ("PENDING_OTHER_EXPERIMENT", "NOT_AVAILABLE", "READY",
                                                              "PARTIALLY_READY", "NEEDS_OWNER_LABELS")


# ------------------------------------------------------------ owner labels --

def test_owner_label_store_persists_incrementally_and_merges(tmp_path):
    store = OwnerLabelStore(tmp_path)
    assert store.entries == {}
    store.record("e003_q1_r1", score=3.5, max_score=4.0, rubric_met=["r1"], note="good", now="2026-08-22 10:00:00")
    with pytest.raises(OwnerLabelError):
        store.record("e003_q1_r2", score=5.0, max_score=4.0)
    with pytest.raises(OwnerLabelError):
        store.record("e003_q1_r2", score=None, status="confirmed")
    store.record("e003_q1_r2", score=None, status="skipped", note="unreadable")
    again = OwnerLabelStore(tmp_path)                     # reload from disk
    assert again.get("e003_q1_r1")["score"] == 3.5 and again.get("e003_q1_r2")["status"] == "skipped"
    assert not list(tmp_path.glob("*.tmp"))
    labels = {"e003_q1_r1": {"score": None}, "e003_q1_r2": {"score": None}, "e003_q1_r3": {"score": None}}
    n = merge_owner_labels(labels, again)
    assert n == 1 and labels["e003_q1_r1"]["score"] == 3.5 and labels["e003_q1_r1"]["rubric_met"] == ["r1"]
    assert labels["e003_q1_r2"]["score"] is None and labels["e003_q1_r2"]["owner_status"] == "skipped"
    s = again.summary(list(labels))
    assert (s["confirmed"], s["skipped"], s["remaining"]) == (1, 1, 1)
    again.reset("e003_q1_r1")
    assert OwnerLabelStore(tmp_path).get("e003_q1_r1") is None


@pytest.mark.skipif(not (DATASETS / "grade_primary" / "manifest.json").exists(), reason="grading dataset not built")
def test_owner_labels_flow_into_the_manifest_without_touching_frozen_files(tmp_path):
    src = DATASETS / "grade_primary"
    dst = tmp_path / "grade_primary"
    shutil.copytree(src, dst)
    before = {p.name: p.read_bytes() for p in dst.iterdir() if p.is_file()}
    m0 = load_manifest("grade_primary", datasets_root=tmp_path)
    cid = m0.cases[0].case_id
    OwnerLabelStore(dst).record(cid, score=2.0, max_score=4.0)
    m1 = load_manifest("grade_primary", datasets_root=tmp_path)
    assert next(c for c in m1.cases if c.case_id == cid).label["score"] == 2.0
    assert "owner_labels_sha256" in m1.hashes and m1.extra["owner_labels_merged"] == 1
    assert role_dataset_status("grade_primary", m1)["status"] == "PARTIALLY_READY"
    for name, data in before.items():
        assert (dst / name).read_bytes() == data            # frozen files untouched


# ------------------------------------------------------------------- spend --

def test_campaign_preflight_sequence_with_mocked_key_metadata(tmp_path):
    from autograder.spend import campaign_preflight, key_usage_checkpoints
    from autograder.usage import UsageLedger
    led = UsageLedger(tmp_path / "usage.jsonl")
    led.record({"task": "ocr_primary", "backend": "openrouter", "model": "m", "cloud": True, "cache_hit": False,
                "reported_cost": 1.0})
    # no credential -> refused at step 1
    d = campaign_preflight(credential_present=False, fetch_key_metadata=None, ledger=led, state_root=tmp_path)
    assert not d["allowed"] and "credential" in d["reason"]
    # credential + key metadata (mock) + ledger agree -> allowed; checkpoint recorded
    meta = {"ok": True, "usage": 12.0, "limit": 20.0, "limit_remaining": 8.0}
    d = campaign_preflight(credential_present=True, fetch_key_metadata=lambda: meta, ledger=led,
                           state_root=tmp_path, predicted_cost=0.01, now="2026-08-22 10:00:00")
    assert d["allowed"] and [s["step"] for s in d["steps"]] == [
        "credential_present", "key_metadata", "record_starting_key_usage", "compare_ledger_with_key", "budget_safe"]
    assert len(key_usage_checkpoints(tmp_path)) == 1
    # later: account usage grew by 2.5 while the ledger says 1.0 -> both shown, disagreement refuses
    meta2 = {"ok": True, "usage": 14.5, "limit": 20.0, "limit_remaining": 5.5}
    d2 = campaign_preflight(credential_present=True, fetch_key_metadata=lambda: meta2, ledger=led,
                            state_root=tmp_path, predicted_cost=0.01)
    cmp = d2["steps"][3]["comparison"]
    assert cmp["local_ledger_usd"] == 1.0 and cmp["key_usage_attributable_usd"] == 2.5 and cmp["disagree"]
    assert not d2["allowed"] and "DISAGREE" in d2["reason"].upper()
    # hard stop: predicted spend would cross $10 -> refused BEFORE any request
    led.record({"task": "x", "backend": "openrouter", "model": "m", "cloud": True, "cache_hit": False,
                "reported_cost": 8.95})
    d3 = campaign_preflight(credential_present=True, fetch_key_metadata=None, ledger=led, state_root=tmp_path,
                            predicted_cost=0.10)
    assert not d3["allowed"] and "refused BEFORE" in d3["reason"]
    d4 = campaign_preflight(credential_present=True, fetch_key_metadata=None, ledger=led, state_root=tmp_path,
                            predicted_cost=0.01)
    assert d4["allowed"] and "WARNING" in d4["reason"]


# ------------------------------------------------------------- model config --

def test_models_toml_keeps_every_cloud_role_unselected_without_env_slugs():
    from autograder.readiness import CLOUD_TASKS, role_status
    cfg = REPO / "models.toml"
    if not cfg.exists():
        pytest.skip("local models.toml not created on this machine")
    rs = role_status(cfg)
    for t, v in rs["tasks"].items():
        if t in CLOUD_TASKS and v["status"] != "ABSENT":
            assert v["status"] == "UNSELECTED", t
    assert rs["env_slug_cloud_tasks"] == []
    assert rs["configured_cloud"] == []
    b = rs["budget_section"]
    assert float(b["max_cost_total"]) == 10.0 and float(b["soft_fraction"]) == 0.8
    assert "OPENROUTER_API_KEY" not in cfg.read_text(encoding="utf-8").replace("OPENROUTER_API_KEY environment", "")


def test_readiness_headline_without_key(tmp_path, monkeypatch, no_network):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GRADER_JOBS_DIR", str(tmp_path / "jobs"))
    from autograder.readiness import format_readiness, readiness_report
    cfg = REPO / "models.toml"
    if not cfg.exists():
        pytest.skip("local models.toml not created on this machine")
    rep = readiness_report(models_config=cfg, state_root=tmp_path / "state")
    cats = {c["category"]: c for c in rep["categories"]}
    assert cats["API KEY"]["status"] == "NOT INSTALLED"
    assert cats["MODEL CONFIG"]["ok"] and cats["BUDGET"]["ok"] and cats["HELD_OUT PROTECTION"]["ok"]
    assert rep["pre_api_setup_complete"] is True and rep["ready_for_api_key"] is True
    assert rep["network_calls"] == 0
    text = format_readiness(rep)
    assert "PRE-API SETUP COMPLETE: YES" in text and "READY FOR API KEY: YES" in text
    assert "OWNER ACTION REQUIRED" in text
