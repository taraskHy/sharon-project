"""Audit SCOPE regression tests: only human-transcribed (tier "owner")
references are auditable; printed/text-layer/mechanical references are out
of scope and keep their original benchmark references. The rule is
provenance-based, never category-based (scripts/refaudit.py)."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_BENCH = REPO_ROOT / "evaluation" / "hebrew_bench_v2"

spec = importlib.util.spec_from_file_location(
    "refaudit_scope", REPO_ROOT / "scripts" / "refaudit.py")
refaudit = importlib.util.module_from_spec(spec)
sys.modules["refaudit_scope"] = refaudit
spec.loader.exec_module(refaudit)

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAA"
    "AABJRU5ErkJggg==")

# (id, category, tier, provenance, expected_eligible)
CASES = [
    ("hl_1", "handwritten_line", "owner",
     "owner annotation 2026-07-16 19:19:13 (htr_pilot)", True),
    ("hc_1", "handwritten_cell", "owner",
     "owner-verified 16-cell benchmark (July campaign)", True),
    # handwritten mixed He/En, manually transcribed -> IN scope
    ("mx_hand", "mixed_he_en", "owner",
     "owner annotation 2026-07-20 (htr_pilot)", True),
    # printed mixed He/En from the text layer -> OUT of scope
    ("mx_print", "mixed_he_en", "text-layer",
     "embedded text layer of docA (born-digital)", False),
    # handwritten formula, manually transcribed -> IN scope
    ("fm_hand", "formula_handwritten", "owner",
     "owner annotation 2026-07-21 (htr_pilot)", True),
    # printed formula from the text layer -> OUT
    ("fm_print", "formula_printed", "text-layer",
     "embedded text layer of docB (born-digital)", False),
    ("pr_1", "printed_rtl", "text-layer",
     "embedded text layer of docA (born-digital)", False),
    ("assoc_1", "option_row_association", "text-layer",
     "embedded text layer of docB (born-digital); pairs from word geometry", False),
]


@pytest.fixture()
def bench(tmp_path: Path) -> Path:
    d = tmp_path / "bench"
    (d / "crops").mkdir(parents=True)
    items, refs = [], {"_policy": "test"}
    for item_id, category, tier, provenance, _exp in CASES:
        (d / "crops" / f"{item_id}.png").write_bytes(_PNG)
        items.append({"id": item_id, "category": category, "tier": tier,
                      "hard": False, "image": f"crops/{item_id}.png",
                      "writer": "e003" if tier == "owner" else None,
                      "task": "transcribe"})
        refs[item_id] = {"text": f"טקסט {item_id}", "provenance": provenance}
    (d / "items.json").write_text(json.dumps({"version": 2, "items": items},
                                             ensure_ascii=False), encoding="utf-8")
    (d / "references.json").write_text(json.dumps(refs, ensure_ascii=False),
                                       encoding="utf-8")
    run = d / "outputs" / "cfgA" / "run1"
    run.mkdir(parents=True)
    for item_id, *_ in CASES:
        (run / f"{item_id}.json").write_text(json.dumps(
            {"transcription": f"טקסט {item_id} שגוי", "item": item_id, "run": 1},
            ensure_ascii=False), encoding="utf-8")
    return d


def test_eligibility_follows_provenance_not_category(bench):
    store = refaudit.AuditStore(bench)
    for item_id, _cat, _tier, _prov, expected in CASES:
        assert store.is_eligible(item_id) is expected, (item_id, store.eligibility(item_id))
    assert store.eligible_ids == ["hl_1", "hc_1", "mx_hand", "fm_hand"]
    assert store.excluded_ids == ["mx_print", "fm_print", "pr_1", "assoc_1"]


def test_summary_denominator_is_eligible_count(bench):
    store = refaudit.AuditStore(bench)
    s = store.summary()
    assert s["total"] == 4 and s["benchmark_total"] == 8
    assert s["excluded_not_in_scope"] == 4 and s["remaining"] == 4


def test_recording_an_out_of_scope_item_is_refused(bench):
    store = refaudit.AuditStore(bench)
    with pytest.raises(refaudit.RefAuditError):
        store.record("pr_1", "confirmed")
    with pytest.raises(refaudit.RefAuditError):
        store.record("fm_print", "corrected", audited_text="x")
    assert store.summary()["checked"] == 0


def test_freeze_does_not_wait_for_excluded_items(bench):
    store = refaudit.AuditStore(bench)
    for item_id in ("hl_1", "hc_1", "mx_hand"):
        store.record(item_id, "confirmed")
    with pytest.raises(refaudit.FreezeError):   # one ELIGIBLE item still open
        refaudit.freeze_manifest(store)
    store.record("fm_hand", "corrected", audited_text="נוסחה מתוקנת")
    manifest = refaudit.freeze_manifest(store)  # excluded items never block
    assert manifest["eligible_items"] == 4 and manifest["excluded_items"] == 4
    assert manifest["benchmark_total"] == 8
    assert set(manifest["excluded_item_ids"]) == {"mx_print", "fm_print", "pr_1", "assoc_1"}
    assert "provenance" in manifest["eligibility_rule"]
    assert refaudit.is_frozen(store)


def test_scoring_uses_original_reference_for_out_of_scope_items(bench):
    store = refaudit.AuditStore(bench)
    store.record("hl_1", "corrected", audited_text="תוקן")
    audited = refaudit.reference_for_scoring(store, "hl_1", mode="final")
    assert audited.reference == "תוקן" and audited.use_for_strict_cer
    for item_id in ("pr_1", "mx_print", "fm_print", "assoc_1"):
        # never audited, never refused: the original is normal ground truth
        res = refaudit.reference_for_scoring(store, item_id, mode="final")
        assert res.status == "not_in_audit_scope"
        assert res.reference == f"טקסט {item_id}"
        assert res.use_for_strict_cer is True
        assert res.source == "original_benchmark_reference"


def test_preview_and_verifier_prep_respect_scope(bench):
    store = refaudit.AuditStore(bench)
    for item_id in ("hl_1", "hc_1", "mx_hand", "fm_hand"):
        store.record(item_id, "confirmed")
    cfg = refaudit.preview_metrics(store)["configs"][0]
    assert cfg["items_with_output_compared"] == 4
    assert cfg["out_of_scope_unchanged"] == 4
    dry = refaudit.verifier_prep(store)
    assert dry["cases"] == 4                        # handwriting subset only
    assert dry["out_of_scope_items_excluded"] == 4
    refaudit.freeze_manifest(store)
    refaudit.verifier_prep(store, emit=True)
    labels = [json.loads(l) for l in (bench / "verifier_bench" / "cases_labels.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert {lab["item_id"] for lab in labels} == {"hl_1", "hc_1", "mx_hand", "fm_hand"}


def test_stray_out_of_scope_entries_are_ignored_not_deleted(bench):
    """An audit file that (from an older scope) contains a decision for an
    excluded item: kept on disk untouched, ignored by every computation."""
    store = refaudit.AuditStore(bench)
    store.record("hl_1", "confirmed")
    doc = json.loads(store.audit_path.read_text(encoding="utf-8"))
    doc["entries"]["pr_1"] = {"item_id": "pr_1", "original_reference": "x",
                              "audited_reference": "x", "status": "confirmed",
                              "note": "", "audited_at": "t"}
    store.audit_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    reloaded = refaudit.AuditStore(bench)
    s = reloaded.summary()
    assert s["checked"] == 1 and s["out_of_scope_entries_ignored"] == 1
    assert "pr_1" not in reloaded.entries_canonical()
    reloaded.record("hc_1", "confirmed")   # a new decision must not drop it
    doc2 = json.loads(reloaded.audit_path.read_text(encoding="utf-8"))
    assert "pr_1" in doc2["entries"]


# ------------------------------------------------------ the real benchmark ----


def test_real_benchmark_scope_is_102_handwriting_references_and_unchanged():
    """Read-only over the frozen benchmark: 102 owner-tier (manual) items are
    eligible, 27 text-layer items are out of scope, and no file is touched."""
    before = {p.name: p.read_bytes() for p in (REAL_BENCH / "items.json",
                                               REAL_BENCH / "references.json")}
    store = refaudit.AuditStore(REAL_BENCH)
    assert len(store.item_ids) == 129
    assert len(store.eligible_ids) == 102
    assert len(store.excluded_ids) == 27
    items = {it["id"]: it for it in store.items}
    assert all(items[i]["tier"] == "owner" for i in store.eligible_ids)
    assert all(items[i]["tier"] == "text-layer" for i in store.excluded_ids)
    assert {items[i]["category"] for i in store.eligible_ids} == {
        "handwritten_line", "handwritten_cell"}
    assert "pr_docA_p1_b0" in store.excluded_ids       # the reported scope bug
    assert store.summary()["total"] == 102
    for p in (REAL_BENCH / "items.json", REAL_BENCH / "references.json"):
        assert p.read_bytes() == before[p.name]
