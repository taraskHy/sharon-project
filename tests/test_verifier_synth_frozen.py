"""Frozen SYNTHETIC_NEAR_MISS manifest validation (scripts/verifier_synth.py
freeze_synthetic / verify_frozen_synthetic) — offline regression tests."""

from __future__ import annotations

import json

import pytest

from tests.test_verifier_synth import frozen as _frozen  # noqa: F401 (fixture)
from tests.test_verifier_synth import _real_bench  # noqa: F401 (fixture dependency)
from tests.test_verifier_synth import vsyn, refaudit, vsel  # noqa: E402

SYN = "verifier_bench/synthetic"


def _freeze(bench):
    store = refaudit.AuditStore(bench)
    synth = vsyn.build_synthetic(store, vsyn.load_selected_with_candidates(store))
    vsyn.write_proposal(store, synth)
    manifest = vsyn.freeze_synthetic(store, synth)
    return store, synth, manifest


def test_freeze_refuses_without_or_against_a_different_proposal(_frozen):
    store = refaudit.AuditStore(_frozen)
    synth = vsyn.build_synthetic(store, vsyn.load_selected_with_candidates(store))
    with pytest.raises(vsyn.SynthError):            # no saved proposal
        vsyn.freeze_synthetic(store, synth)
    vsyn.write_proposal(store, synth)
    # an explicit expectation that disagrees with the built composition refuses
    with pytest.raises(vsyn.SynthError):
        vsyn.freeze_synthetic(store, synth, expect={"total": synth["report"]["synthetic_cases_total"] + 1})
    # a saved proposal with a different composition refuses
    prop = _frozen / "verifier_bench" / "synthetic_near_miss_proposal.json"
    doc = json.loads(prop.read_text(encoding="utf-8"))
    doc["synthetic_cases_total"] += 1
    prop.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(vsyn.SynthError):
        vsyn.freeze_synthetic(store, synth)
    assert not (_frozen / SYN).exists()


def test_frozen_manifest_persists_required_fields_and_verifies(_frozen):
    store, synth, manifest = _freeze(_frozen)
    r = synth["report"]
    for field in ("rules_version", "selection_policy_version", "source_audit_sha256",
                  "real_benchmark", "split_assignment", "image_ids_per_split",
                  "case_ids_per_split", "composition", "zero_image_overlap_between_splits",
                  "inputs_sha256", "labels_sha256"):
        assert field in manifest, field
    assert manifest["rules_version"] == vsyn.RULES_VERSION
    assert manifest["selection_policy_version"] == vsyn.SELECTION_POLICY_VERSION
    assert manifest["split_assignment"] == vsel.SPLIT_PROPOSALS["A"]
    assert manifest["composition"]["total"] == r["synthetic_cases_total"]
    assert manifest["composition"]["text"] + manifest["composition"]["numeric_math"] == r["synthetic_cases_total"]
    assert sum(len(v) for v in manifest["case_ids_per_split"].values()) == r["synthetic_cases_total"]
    assert len(manifest["real_benchmark"]["manifest_sha256"]) == 64
    # labels carry the evaluation-only metadata incl. the synthetic group
    labels = [json.loads(l) for l in (_frozen / SYN / "cases_labels.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert {l["synthetic_group"] for l in labels} <= {"text", "numeric_math"}
    for l in labels:
        for k in ("expected_verdict", "corruption_type", "item_id", "writer", "split",
                  "audited_reference", "cer_vs_audited"):
            assert k in l, k
    # post-freeze verification passes and reports the checksums
    result = vsyn.verify_frozen_synthetic(store)
    assert result["ok"] and result["cases"] == r["synthetic_cases_total"]
    checks = dict(line.split(None, 1)[::-1] for line in
                  (_frozen / SYN / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines())
    assert set(checks) == {"cases_inputs.jsonl", "cases_labels.jsonl", "manifest.json"}
    assert result["manifest_sha256"] == checks["manifest.json"]


@pytest.mark.parametrize("tamper", ["inputs", "label_split", "drop_case", "real_file", "raw_file"])
def test_verify_detects_tampering(_frozen, tamper):
    store, _synth, _manifest = _freeze(_frozen)
    syn_dir = _frozen / SYN
    if tamper == "inputs":
        p = syn_dir / "cases_inputs.jsonl"
        p.write_text(p.read_text(encoding="utf-8").replace("\n", " \n", 1), encoding="utf-8")
    elif tamper == "label_split":
        p = syn_dir / "cases_labels.jsonl"
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        rows[0]["split"] = "HELD_OUT" if rows[0]["split"] != "HELD_OUT" else "DEV"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                     encoding="utf-8")
    elif tamper == "drop_case":
        p = syn_dir / "cases_labels.jsonl"
        lines = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    elif tamper == "real_file":
        p = _frozen / "verifier_bench" / "selected" / "cases_labels.jsonl"
        p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif tamper == "raw_file":
        p = _frozen / "verifier_bench" / "cases_inputs.jsonl"
        p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(vsyn.SynthError):
        vsyn.verify_frozen_synthetic(store)


def test_model_visible_file_has_no_label_vocabulary(_frozen):
    store, _synth, _manifest = _freeze(_frozen)
    text = (_frozen / SYN / "cases_inputs.jsonl").read_text(encoding="utf-8").lower()
    for tok in vsyn.ALL_RULES + ("supported", "review", "synthetic", "reference", "numeric_math"):
        assert tok not in text, tok
    assert vsyn.verify_frozen_synthetic(store)["ok"]
