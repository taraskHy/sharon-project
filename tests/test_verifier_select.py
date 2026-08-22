"""Verifier benchmark selection layer (scripts/verifier_select.py) — offline.

Pins: one positive per audited item, opaque/indistinguishable model-visible
rows, per-image dedup + the <=2-negatives rule (with the documented
number/sign exception), writer-level splits with zero image overlap, raw
pool untouched by propose/freeze, freeze artifacts + hashes, no network.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "verifier_select", REPO_ROOT / "scripts" / "verifier_select.py")
vsel = importlib.util.module_from_spec(_spec)
sys.modules["verifier_select"] = vsel
_spec.loader.exec_module(vsel)
refaudit = vsel.refaudit

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAA"
    "AABJRU5ErkJggg==")

# item -> (writer, reference)
ITEMS = {
    "w2_a": ("e002", "התדרים הגבוהים נשמרים בתמונה"),
    "w3_a": ("e003", "x = 3 ולכן התשובה נכונה"),
    "w4_a": ("e004", "ההיסטוגרמה מתארת עוצמות אפור"),
    "w5_a": ("e005", "מסנן מעביר גבוהים מסיר DC"),
    "w6_a": ("e006", "שורה פשוטה לבדיקה"),
    "w7_a": ("e007", "עוד שורה אחת"),
}
# historical outputs: config -> item -> candidate
OUTPUTS = {
    "cfgA": {"w2_a": "התדרים הגבוהים נשמרים בתמונה",        # correct (== ref)
             "w3_a": "x = 8 ולכן התשובה נכונה",             # number corruption
             "w4_a": "ההיסטוגרמה מתארת",                    # omission
             "w5_a": "מסנן מעביר גבוהים",                    # omission
             "w6_a": "שורה פשוטה לבדיקה נוספת",             # addition
             "w7_a": "עוד שורה"},
    "cfgB": {"w2_a": "התדרים הגבוהים נשמרו בתמונה",         # subtle substitution
             "w3_a": "x = 8 ולכן התשובה נכונה",             # duplicate of cfgA (dedup)
             "w4_a": "ההיסטוגרמה מתארת עוצמות אפור בתמונה", # addition
             "w5_a": "מסנן מעביר גבוהים מסיר DC ורעש",       # addition
             "w6_a": "שורה",                                 # severe omission
             "w7_a": "עוד שורה אחת"},                         # correct
    "cfgC": {"w2_a": "התדרים",                               # severe
             "w3_a": "x = -3 ולכן התשובה נכונה",             # minus corruption
             "w4_a": "ההיסטוגרמה מתארת עוצמות",              # omission (subtle)
             "w5_a": "מסנן מעביר נמוכים מסיר DC"},           # substitution
}


@pytest.fixture()
def bench(tmp_path: Path) -> Path:
    d = tmp_path / "bench"
    (d / "crops").mkdir(parents=True)
    items, refs = [], {"_policy": "t"}
    for item_id, (writer, text) in ITEMS.items():
        (d / "crops" / f"{item_id}.png").write_bytes(_PNG)
        items.append({"id": item_id, "category": "handwritten_line", "tier": "owner",
                      "hard": False, "image": f"crops/{item_id}.png",
                      "writer": writer, "task": "transcribe"})
        refs[item_id] = {"text": text, "provenance": "owner annotation (htr_pilot)"}
    (d / "items.json").write_text(json.dumps({"version": 2, "items": items},
                                             ensure_ascii=False), encoding="utf-8")
    (d / "references.json").write_text(json.dumps(refs, ensure_ascii=False),
                                       encoding="utf-8")
    for config, outs in OUTPUTS.items():
        run = d / "outputs" / config / "run1"
        run.mkdir(parents=True)
        for item_id, text in outs.items():
            (run / f"{item_id}.json").write_text(json.dumps(
                {"transcription": text, "item": item_id, "run": 1},
                ensure_ascii=False), encoding="utf-8")
    store = refaudit.AuditStore(d)
    for item_id in ITEMS:
        store.record(item_id, "confirmed")
    refaudit.freeze_manifest(store)
    refaudit.verifier_prep(store, emit=True)      # the RAW pool
    return d


def _raw_bytes(bench: Path) -> dict:
    raw = bench / "verifier_bench"
    return {n: (raw / n).read_bytes()
            for n in ("cases_inputs.jsonl", "cases_labels.jsonl", "manifest.json")}


def test_one_positive_per_audited_item_and_opaque_rows(bench):
    store = refaudit.AuditStore(bench)
    sel = vsel.build_selection(store, vsel.load_raw_pool(store), "A")
    labels = sel["labels"]
    positives = [l for l in labels if l["polarity"] == "positive"]
    assert len(positives) == len(ITEMS)
    assert {l["item_id"] for l in positives} == set(ITEMS)
    assert all(l["expected_verdict"] == "supported" for l in positives)
    by_id = {l["case_id"]: l for l in labels}
    for row in sel["inputs"]:
        assert set(row) == {"case_id", "crop", "candidate_transcription"}
        raw = json.dumps(row, ensure_ascii=False).lower()
        for banned in ("positive", "negative", "reference", "verdict", "audited", "cfg"):
            assert banned not in raw, banned
        lab = by_id[row["case_id"]]
        if lab["polarity"] == "positive":   # candidate IS the audited reference
            assert row["candidate_transcription"] == ITEMS[lab["item_id"]][1]
    # ordering is by opaque id: not grouped by polarity
    polarities = [by_id[r["case_id"]]["polarity"] for r in sel["inputs"]]
    assert polarities != sorted(polarities) and polarities != sorted(polarities, reverse=True)


def test_dedup_and_at_most_two_negatives_per_image(bench):
    store = refaudit.AuditStore(bench)
    sel = vsel.build_selection(store, vsel.load_raw_pool(store), "A")
    r = sel["report"]
    neg_per_item = {}
    for l in sel["labels"]:
        if l["polarity"] == "negative":
            neg_per_item[l["item_id"]] = neg_per_item.get(l["item_id"], 0) + 1
    # w3_a: cfgA and cfgB produced the identical "x = 8" candidate -> one group;
    # plus the minus-corruption -> 2 unique negatives, both selected
    assert neg_per_item["w3_a"] == 2
    w3 = [l for l in sel["labels"] if l["item_id"] == "w3_a" and l["polarity"] == "negative"]
    x8 = next(l for l in w3 if "cfgA" in l["source_configs"])
    assert set(x8["source_configs"]) == {"cfgA", "cfgB"}  # dedup kept both configs
    assert len(x8["raw_case_ids"]) == 2
    assert all(n <= vsel.MAX_NEGATIVES_PER_IMAGE + 1 for n in neg_per_item.values())
    assert all(n <= vsel.MAX_NEGATIVES_PER_IMAGE for item, n in neg_per_item.items()
               if item not in {e["item_id"] for e in r["documented_exceptions"]})
    # w2_a has 3 unique negatives (subtle, severe, correct-duplicate excluded):
    # the subtle substitution must be picked first, then the severe one
    w2 = sorted((l for l in sel["labels"] if l["item_id"] == "w2_a" and l["polarity"] == "negative"),
                key=lambda l: l["cer_vs_audited"])
    assert len(w2) == 2 and w2[0]["severity"] == "subtle" and w2[1]["severity"] == "severe"
    assert r["real_correct_raw_candidates_not_added"] == 2   # cfgA w2_a, cfgB w7_a
    assert r["unique_negatives_after_dedup"] < len(vsel.load_raw_pool(store)["labels"])


def test_multi_label_error_kinds_preserved_and_number_sign_covered(bench):
    store = refaudit.AuditStore(bench)
    sel = vsel.build_selection(store, vsel.load_raw_pool(store), "A")
    kinds = sel["report"]["error_kind_counts_overlapping"]
    assert kinds.get("number_sign_formula", 0) >= 2        # x=8 and x=-3
    assert kinds.get("omission", 0) >= 1 and kinds.get("unsupported_addition", 0) >= 1
    multi = [l for l in sel["labels"] if len(l["error_kinds"]) > 1]
    assert multi, "multi-label negatives must keep all their kinds"
    assert sel["report"]["number_sign_formula_coverage"]["images_with_number_sign_case"] >= 1


def test_writer_level_split_has_zero_image_overlap(bench):
    store = refaudit.AuditStore(bench)
    for split_name in ("A", "B"):
        sel = vsel.build_selection(store, vsel.load_raw_pool(store), split_name)
        r = sel["report"]
        assert r["zero_image_overlap_between_splits"] is True
        assert r["images_in_multiple_splits"] == []
        # every case of an item sits in the split of its writer
        wsplit = {w: s for s, ws in vsel.SPLIT_PROPOSALS[split_name].items() for w in ws}
        for l in sel["labels"]:
            assert l["split"] == wsplit[l["writer"]]
        assert sum(v["cases"] for v in r["by_split"].values()) == r["total_selected_cases"]


def test_propose_and_freeze_keep_raw_pool_byte_identical(bench):
    before = _raw_bytes(bench)
    store = refaudit.AuditStore(bench)
    raw = vsel.load_raw_pool(store)
    sel = vsel.build_selection(store, raw, "A")
    proposal = vsel.write_proposal(store, sel)
    assert proposal.name == "selection_proposal.json"
    assert _raw_bytes(bench) == before
    manifest = vsel.freeze_selected(store, sel, "A")
    assert _raw_bytes(bench) == before
    out = bench / "verifier_bench" / "selected"
    assert (out / "cases_inputs.jsonl").exists() and (out / "cases_labels.jsonl").exists()
    assert len(manifest["inputs_sha256"]) == 64
    assert manifest["report"]["positive_cases"] == len(ITEMS)
    assert "FALSE ACCEPT RATE" in manifest["_policy"]
    inputs = [json.loads(l) for l in (out / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(set(r) == {"case_id", "crop", "candidate_transcription"} for r in inputs)


def test_refuses_without_frozen_audit_or_tampered_raw_pool(bench):
    store = refaudit.AuditStore(bench)
    # tamper the raw pool -> manifest hash mismatch -> refuse
    raw_inputs = bench / "verifier_bench" / "cases_inputs.jsonl"
    raw_inputs.write_text(raw_inputs.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(vsel.SelectionError):
        vsel.load_raw_pool(store)


def test_no_network_and_no_model_imports(bench, no_network):
    store = refaudit.AuditStore(bench)
    vsel.build_selection(store, vsel.load_raw_pool(store), "A")
    src = (REPO_ROOT / "scripts" / "verifier_select.py").read_text(encoding="utf-8")
    for banned in ("backends", "gateway", "httpx", "urllib", "requests", "openrouter",
                   "anthropic", "openai", "ollama"):
        assert banned not in src, banned
