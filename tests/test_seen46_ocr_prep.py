"""The two OCR-campaign preparation gaps, closed and pinned.

1. seen46 OCR subsets (frozen, split-scoped, campaign-derived);
2. report-time per-writer CER/WER/omission/digit-sign metrics;
plus the zero-cost dry-run validation artifacts and the payload contract.

No model / provider / network call anywhere in this file.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder.benchmark import ocr_writer_metrics as owm  # noqa: E402
from autograder.benchmark import subsets  # noqa: E402
from autograder.benchmark.manifests import load_manifest  # noqa: E402

CAMPAIGN = REPO / "evaluation" / "model_selection" / "experiments" / \
    "OCR_VALIDATION_CAMPAIGN_2026-09-02.json"
RUNS_OCR = REPO / "evaluation" / "model_selection" / "runs" / "ocr_primary"
HELD_PAT = re.compile(r"(?<![0-9a-fA-F])(e005|e006)(?![0-9a-fA-F])")

needs_campaign = pytest.mark.skipif(not CAMPAIGN.exists(),
                                    reason="campaign freeze absent")


@pytest.fixture(scope="module")
def manifest():
    try:
        return load_manifest("ocr_primary")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ocr_primary manifest not loadable here: {e}")


@pytest.fixture(scope="module")
def campaign():
    return json.loads(CAMPAIGN.read_text(encoding="utf-8"))


# ------------------------------------------------ gap 1: frozen subsets -----


@needs_campaign
def test_seen46_subsets_are_frozen_split_scoped_and_complete(manifest, campaign):
    dev = subsets.load_subset("ocr_primary", "seen46_ocr_dev", manifest)
    cal = subsets.load_subset("ocr_primary", "seen46_ocr_calibration", manifest)
    assert dev["case_count"] == 32 and dev["split"] == "DEV"
    assert cal["case_count"] == 21 and cal["split"] == "CALIBRATION"
    assert sorted({r["writer"] for r in dev["cases"]}) == ["e002", "e003", "e007"]
    assert sorted({r["writer"] for r in cal["cases"]}) == ["e004"]
    for row in dev["cases"] + cal["cases"]:
        assert row["provenance_valid"] is True
        assert not HELD_PAT.search(row["case_id"])
        # identity + provenance only — never reference text or grade fields
        assert "reference" not in row and "explanation_verdict" not in row
        assert row["verdict"] is None
    # the union covers exactly the campaign crops that exist as bench items:
    # 53 of 54; the single missing crop is the human-repaired line that lives
    # in the grade dataset's repair store, not in hebrew_bench_v2
    campaign_crops = {rel.split("/")[-1] for c in campaign["cases"]
                      for rel in c["evidence_crops"]}
    subset_crops = {r["image"].split("/")[-1] for r in dev["cases"] + cal["cases"]}
    assert subset_crops <= campaign_crops
    assert campaign_crops - subset_crops == {"e004_q2_r3__l2.png"}


@needs_campaign
def test_subset_selection_is_deterministic_and_tamper_refused(manifest, tmp_path):
    prop = subsets.propose_subset("ocr_primary", "seen46_ocr_dev", manifest)
    again = subsets.propose_subset("ocr_primary", "seen46_ocr_dev", manifest)
    assert prop["selection_sha256"] == again["selection_sha256"]
    frozen = subsets.load_subset("ocr_primary", "seen46_ocr_dev", manifest)
    assert frozen["selection_sha256"] == prop["selection_sha256"]
    # tampered copy is refused
    doc = json.loads(subsets.subset_path("ocr_primary", "seen46_ocr_dev")
                     .read_text(encoding="utf-8"))
    doc["cases"] = doc["cases"][:-1]
    root = tmp_path / "subsets"
    root.mkdir()
    (root / "ocr_primary__seen46_ocr_dev.json").write_text(
        json.dumps(doc), encoding="utf-8")
    with pytest.raises(subsets.SubsetError, match="hash mismatch"):
        subsets.load_subset("ocr_primary", "seen46_ocr_dev", manifest, root)
    # a frozen subset is never re-selected
    with pytest.raises(subsets.SubsetError, match="never re-selected"):
        subsets.freeze_subset("ocr_primary", "seen46_ocr_dev", manifest)


def test_cli_exposes_the_new_subsets():
    src = (REPO / "autograder" / "benchmark" / "cli.py").read_text(encoding="utf-8")
    assert "seen46_ocr_dev" in src and "seen46_ocr_calibration" in src


@needs_campaign
def test_campaign_tamper_fails_the_subset_predicate(tmp_path, monkeypatch):
    doc = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    doc["cases"] = doc["cases"][:10]
    bad = tmp_path / "campaign.json"
    bad.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(subsets, "_SEEN46_CAMPAIGN", bad)
    monkeypatch.setattr(subsets, "_SEEN46_CROPS", None)
    with pytest.raises(subsets.SubsetError, match="self-hash"):
        subsets._seen46_campaign_crops()


# ------------------------------------------ gap 2: per-writer WER metrics ---


def test_pair_metrics_known_values():
    ident = owm.pair_metrics("יש טשטוש בתמונה", "יש טשטוש בתמונה")
    assert ident["cer"] == 0.0 and ident["wer"] == 0.0
    assert ident["digit_sign_error"] is False
    # the minus sign: invisible to normalized CER, caught by the signature
    sign = owm.pair_metrics("x = -3 לכן", "x = 3 לכן")
    assert sign["digit_sign_error"] is True
    lost = owm.pair_metrics("ההסבר נכון", None)
    assert lost["line_lost"] is True and lost["scored"] is False
    with pytest.raises(owm.OcrMetricsError, match="reference"):
        owm.pair_metrics("", "hyp")


def test_writer_metrics_grouping_and_refusals():
    pairs = [
        {"case_id": "hl_e002_q1_r1__l1", "reference": "אבג דהו",
         "hypothesis": "אבג דהו", "provenance_valid": True},
        {"case_id": "hl_e002_q1_r2__l1", "reference": "אבג דהו",
         "hypothesis": "אבג", "provenance_valid": True},       # omission
        {"case_id": "hc_e003_q1_r1", "reference": "שלום",
         "hypothesis": None, "provenance_valid": True},        # line lost
        {"case_id": "hc_e004_q1_r1", "reference": "x",
         "hypothesis": "y", "provenance_valid": False},        # refused
        {"case_id": "pr_docA_p1_b1", "reference": "5 + 5 = 10",
         "hypothesis": "5 + 5 = 10", "provenance_valid": True},
    ]
    m = owm.writer_metrics(pairs)
    assert m["metrics_version"] == "ocr-writer-metrics-v1"
    assert set(m["per_writer"]) == {"e002", "e003", "no_writer"}
    e2 = m["per_writer"]["e002"]
    assert e2["cases"] == 2 and e2["scored"] == 2
    assert e2["mean_omission_rate"] > 0
    assert m["per_writer"]["e003"]["line_loss"] == 1
    assert m["refused_invalid_reference"] == ["hc_e004_q1_r1"]
    assert m["worst_writer_by_mean_cer"] == "e002"
    assert m["overall"]["cases"] == 4                # refusal not counted
    # deterministic
    assert owm.writer_metrics(list(pairs)) == m


def test_pairs_from_run_fails_closed(tmp_path, manifest):
    with pytest.raises(owm.OcrMetricsError, match="no outputs"):
        owm.pairs_from_run(tmp_path, manifest)
    p = tmp_path / "outputs.jsonl"
    p.write_text('{"case_id": "not_a_case", "output": {}}\n', encoding="utf-8")
    with pytest.raises(owm.OcrMetricsError, match="not in\\s+the frozen manifest|not in"):
        owm.pairs_from_run(tmp_path, manifest)
    p.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(owm.OcrMetricsError, match="malformed"):
        owm.pairs_from_run(tmp_path, manifest)
    real = manifest.cases[0].case_id
    p.write_text(json.dumps({"case_id": real, "output":
                             {"transcription": "טקסט"}}) + "\n",
                 encoding="utf-8")
    pairs = owm.pairs_from_run(tmp_path, manifest)
    assert len(pairs) == 1 and pairs[0]["case_id"] == real
    assert pairs[0]["hypothesis"] == "טקסט"


# -------------------------------- dry-run validation + payload contract -----

DRY_DIRS = {
    "smoke": "dev__smoke__all__google-gemini-3.7-flash__feceaa6084",
    "seen46_dev": "dev__seen46_ocr_dev__all__google-gemini-3.7-flash__cc0cad4c52",
    "seen46_cal":
        "calibration__seen46_ocr_calibration__all__google-gemini-3.7-flash__35819d025d",
}


@pytest.mark.parametrize("name,expected_cases", [("smoke", 8),
                                                 ("seen46_dev", 32),
                                                 ("seen46_cal", 21)])
def test_dry_run_plans_are_zero_cost_priced_and_held_out_free(name,
                                                              expected_cases):
    d = RUNS_OCR / DRY_DIRS[name]
    if not (d / "plan.json").exists():
        pytest.skip("dry-run plan not present on this machine")
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    assert plan["mode"] == "dry_run"
    assert plan["cases"] == expected_cases
    assert plan["leakage_check"] == "passed"
    assert plan["pricing_table_available"] is True
    assert plan["predicted_cost_total"] and plan["predicted_cost_total"] > 0
    for row in plan["rows"]:
        assert not HELD_PAT.search(row["case_id"])
        assert row["images"] == 1 and row["predicted_cost"] > 0
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    assert all(h["mode"] == "dry_run" for h in run["history"])
    assert not (d / "outputs.jsonl").exists()        # zero provider calls


def test_ocr_request_is_crop_plus_transcription_instruction_only(manifest):
    from autograder.benchmark.manifests import DEFAULT_BENCH_ROOT
    from autograder.benchmark.roles import OcrPrimaryAdapter
    from autograder.benchmark.runner import files_root_for
    from autograder.cloudboundary import (check_cloud_call,
                                          forbidden_cloud_markers)
    adapter = OcrPrimaryAdapter()
    root = files_root_for(manifest, DEFAULT_BENCH_ROOT)
    for cid in ("hc_e002_q1_r1", "hl_e003_q1_r5__l1"):
        case = next(c for c in manifest.cases if c.case_id == cid)
        req = adapter.build_request(case.inputs, root)
        # exactly one image block; no extra text rides along
        assert [b.get("type") for b in req.content_blocks] == ["image"]
        low = req.system.lower()
        assert "transcri" in low                     # exact-transcription task
        for marker in forbidden_cloud_markers():
            assert marker not in req.system
        for banned in ("rubric", "official solution", "scoring_rules",
                       "explanation_verdict", "retrieval", "course material"):
            assert banned not in low
        # the frozen prompt now clears the production boundary for OpenRouter
        check_cloud_call(task="ocr_primary", backend="openrouter",
                         base_url=None, execution_mode="production",
                         system=req.system, content_blocks=req.content_blocks)


def test_stage1_smoke_candidates_are_registered_for_ocr_primary():
    """The owner-named stage-1 candidates must be registered ocr_primary
    candidates (the runner refuses unlisted slugs), and — when the
    machine-local pricing table exists — priced (an unpriced candidate
    cannot run live)."""
    import tomllib
    reg = tomllib.loads((REPO / "evaluation" / "model_selection" /
                         "candidates.toml").read_text(encoding="utf-8"))
    listed = reg["roles"]["ocr_primary"]["candidates"]
    for slug in ("google/gemini-3.7-flash", "openai/gpt-5.6-luna-pro",
                 "anthropic/claude-sonnet-5"):
        assert slug in listed, slug
    assert reg["roles"]["ocr_primary"]["status"] == "UNSELECTED"
    models_toml = REPO / "models.toml"
    if models_toml.exists():
        pricing = tomllib.loads(models_toml.read_text(encoding="utf-8")).get(
            "pricing") or {}
        for slug in ("google/gemini-3.7-flash", "openai/gpt-5.6-luna-pro",
                     "anthropic/claude-sonnet-5"):
            assert pricing.get(slug), f"{slug} unpriced: cannot run live"


def test_grade_impact_metric_definition_is_owner_gated(campaign):
    m = campaign["metrics"]
    assert "verdict-flip" in m["downstream_grade_impact"]
    assert "owner-gated" in m["downstream_grade_impact"]
    assert "digit/operator signature" in " ".join(m["secondary"])
