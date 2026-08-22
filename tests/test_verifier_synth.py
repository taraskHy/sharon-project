"""SYNTHETIC_NEAR_MISS layer (scripts/verifier_synth.py) — offline tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "verifier_synth", REPO_ROOT / "scripts" / "verifier_synth.py")
vsyn = importlib.util.module_from_spec(_spec)
sys.modules["verifier_synth"] = vsyn
_spec.loader.exec_module(vsyn)
refaudit, vsel = vsyn.refaudit, vsyn.vsel

from tests.test_verifier_select import bench as _real_bench  # noqa: E402,F401 (fixture)
from tests.test_verifier_select import ITEMS  # noqa: E402


@pytest.fixture()
def frozen(_real_bench):
    """A bench with the REAL selected benchmark frozen (split A)."""
    store = refaudit.AuditStore(_real_bench)
    sel = vsel.build_selection(store, vsel.load_raw_pool(store), "A")
    vsel.freeze_selected(store, sel, "A", rationale="test")
    return _real_bench


# ------------------------------------------------------------------ rules ----


def test_rules_are_deterministic_and_divergent():
    ref = "x = 3.5 ולכן התשובה נכונה"
    for rule in vsyn.ALL_RULES:
        a, b = vsyn.apply_rule(rule, ref, "item1"), vsyn.apply_rule(rule, ref, "item1")
        assert a == b, rule                          # deterministic
        if a is not None:
            assert a != ref, rule                    # a real perturbation
    assert vsyn.apply_rule("digit_substitution", ref, "i") in ("x = 4.5 ולכן התשובה נכונה",
                                                             "x = 3.6 ולכן התשובה נכונה")
    assert vsyn.apply_rule("decimal_point_corruption", ref, "i") == "x = 35 ולכן התשובה נכונה"
    assert vsyn.apply_rule("operator_substitution", "y = 2", "i") == "y 2"   # '=' removed
    assert vsyn.apply_rule("superscript_subscript_loss", "x² + y^2", "i") == "x2 + y2"
    assert vsyn.apply_rule("digit_substitution", "אין ספרות כאן", "i") is None
    # Hebrew prefix hyphen is not an operator
    assert vsyn.apply_rule("operator_substitution", "רק ב-High pass", "i") is None
    # char deletion removes exactly one interior letter of a long word
    out = vsyn.apply_rule("char_deletion", "ההיסטוגרמה מתארת", "i")
    assert out is not None and len(out) == len("ההיסטוגרמה מתארת") - 1
    assert vsyn.apply_rule("short_token_omission", "כן", "i") is None      # too short


# --------------------------------------------------------------- building ----


def test_build_respects_split_inheritance_and_per_image_cap(frozen):
    store = refaudit.AuditStore(frozen)
    synth = vsyn.build_synthetic(store, vsyn.load_selected_with_candidates(store))
    r = synth["report"]
    assert r["synthetic_cases_total"] > 0
    assert r["zero_image_overlap_between_splits"] is True
    per_image = {}
    for l in synth["labels"]:
        per_image[l["item_id"]] = per_image.get(l["item_id"], 0) + 1
    assert max(per_image.values()) <= vsyn.MAX_PER_IMAGE
    # splits come from the frozen REAL selected manifest (writer-level A)
    wsplit = {w: s for s, ws in vsel.SPLIT_PROPOSALS["A"].items() for w in ws}
    for l in synth["labels"]:
        assert l["split"] == wsplit[l["writer"]]
        assert l["source"] == "synthetic_near_miss"
        assert l["expected_verdict"] == "review"
    # the numeric-bearing reference gets a numeric corruption first
    w3 = [l for l in synth["labels"] if l["item_id"] == "w3_a"]
    assert w3 and w3[0]["corruption_group"] == "numeric" or any(
        l["corruption_group"] == "numeric" for l in w3)
    # every synthetic candidate differs from its reference and from real negatives
    for row, lab in zip(synth["inputs"], synth["labels"]):
        assert row["candidate_transcription"] != lab["audited_reference"]
        assert set(row) == {"case_id", "crop", "candidate_transcription"}
        raw = json.dumps(row, ensure_ascii=False).lower()
        assert "synthetic" not in raw and "reference" not in raw


def test_dedup_against_real_negatives_and_no_effect_removal(frozen):
    store = refaudit.AuditStore(frozen)
    selected = vsyn.load_selected_with_candidates(store)
    # Force a collision: make the digit-substitution output of w3_a equal a
    # real negative by injecting it into the selected labels' candidates.
    ref = store.entry("w3_a")["audited_reference"]
    forced = vsyn.apply_rule("digit_substitution", ref, "w3_a")
    for lab in selected["labels"]:
        if lab["item_id"] == "w3_a" and lab["polarity"] == "negative":
            lab["_candidate"] = forced
            break
    synth = vsyn.build_synthetic(store, selected)
    assert synth["report"]["duplicate_removals"].get("duplicate_of_real_negative", 0) >= 1
    assert all(not (l["item_id"] == "w3_a" and l["corruption_type"] == "digit_substitution")
               for l in synth["labels"])


def test_propose_writes_only_proposal_and_freeze_writes_component(frozen):
    store = refaudit.AuditStore(frozen)
    synth = vsyn.build_synthetic(store, vsyn.load_selected_with_candidates(store))
    out = vsyn.write_proposal(store, synth)
    assert out.name == "synthetic_near_miss_proposal.json"
    assert not (frozen / "verifier_bench" / "synthetic").exists()
    # the REAL selected component is untouched by the synthetic build
    sel_dir = frozen / "verifier_bench" / "selected"
    before = {p.name: p.read_bytes() for p in sel_dir.iterdir()}
    manifest = vsyn.freeze_synthetic(store, synth)
    assert {p.name: p.read_bytes() for p in sel_dir.iterdir()} == before
    syn_dir = frozen / "verifier_bench" / "synthetic"
    assert (syn_dir / "cases_inputs.jsonl").exists() and (syn_dir / "CHECKSUMS.sha256").exists()
    assert "SEPARATELY" in manifest["_policy"]
    assert manifest["report"]["metrics_contract"]["COMBINED"].startswith("secondary")


def test_refuses_without_frozen_real_selected(_real_bench):
    store = refaudit.AuditStore(_real_bench)          # audit frozen, selected NOT frozen
    with pytest.raises(vsyn.SynthError):
        vsyn.load_selected_with_candidates(store)


def test_no_network_no_model_imports(frozen, no_network):
    store = refaudit.AuditStore(frozen)
    vsyn.build_synthetic(store, vsyn.load_selected_with_candidates(store))
    src = (REPO_ROOT / "scripts" / "verifier_synth.py").read_text(encoding="utf-8")
    for banned in ("backends", "gateway", "httpx", "urllib", "requests", "openrouter",
                   "anthropic", "openai", "ollama"):
        assert banned not in src, banned
