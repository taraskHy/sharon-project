"""Manual reference-audit toolchain (scripts/refaudit.py) — offline tests.

Covers the required matrix: confirm/correct/ambiguous, resume, atomic
persistence, original-reference immutability, reference_for_scoring rules
(incl. final-mode refusal and ambiguous handling), summary counts, freeze
refusal/success, historical preview, verifier-prep gating and input
isolation, single-item reset, and the no-model/no-network guarantee.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "refaudit", REPO_ROOT / "scripts" / "refaudit.py")
refaudit = importlib.util.module_from_spec(spec)
sys.modules["refaudit"] = refaudit  # dataclasses need the module registered
spec.loader.exec_module(refaudit)

# 1x1 grey PNG (valid image bytes for crops)
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAA"
    "AABJRU5ErkJggg==")

REF = {
    "it1": "התדרים הגבוהים נשמרים",
    "it2": "x = 3 ולכן התשובה נכונה",
    "it3": "ההיסטוגרמה מתארת עוצמות",
}


@pytest.fixture()
def bench(tmp_path: Path) -> Path:
    d = tmp_path / "bench"
    (d / "crops").mkdir(parents=True)
    items = []
    for item_id, _text in REF.items():
        (d / "crops" / f"{item_id}.png").write_bytes(_PNG)
        items.append({"id": item_id, "category": "handwritten_line",
                      "tier": "owner", "hard": False,
                      "image": f"crops/{item_id}.png", "writer": "e003",
                      "task": "transcribe"})
    (d / "items.json").write_text(json.dumps(
        {"version": 2, "items": items}, ensure_ascii=False), encoding="utf-8")
    (d / "references.json").write_text(json.dumps(
        {"_policy": "test", **{k: {"text": v, "provenance": "owner"}
                               for k, v in REF.items()}},
        ensure_ascii=False), encoding="utf-8")
    # persisted historical outputs for two configs
    for config, outs in {
        "cfgA": {"it1": "התדרים הגבוהים נשמרים",       # exact
                 "it2": "x = 8 ולכן התשובה נכונה",      # number corruption+sub
                 "it3": "ההיסטוגרמה"},                  # omission
        "cfgB": {"it1": "התדרים נשמרים"},               # omission
    }.items():
        run = d / "outputs" / config / "run1"
        run.mkdir(parents=True)
        for item_id, text in outs.items():
            (run / f"{item_id}.json").write_text(json.dumps(
                {"transcription": text, "item": item_id, "run": 1},
                ensure_ascii=False), encoding="utf-8")
    return d


def _store(bench: Path) -> "refaudit.AuditStore":
    return refaudit.AuditStore(bench)


# ------------------------------------------------------------- statuses ----


def test_confirm_sets_audited_to_original(bench):
    store = _store(bench)
    entry = store.record("it1", "confirmed", note="clear handwriting")
    assert entry["status"] == "confirmed"
    assert entry["audited_reference"] == REF["it1"] == entry["original_reference"]
    assert entry["note"] == "clear handwriting"
    assert entry["audited_at"]


def test_correction_preserves_both_versions(bench):
    store = _store(bench)
    entry = store.record("it2", "corrected", audited_text="x = 3 ולכן נכונה")
    assert entry["original_reference"] == REF["it2"]
    assert entry["audited_reference"] == "x = 3 ולכן נכונה"
    with pytest.raises(refaudit.RefAuditError):
        store.record("it3", "corrected")  # corrected requires text
    with pytest.raises(refaudit.RefAuditError):
        store.record("it3", "corrected", audited_text="")  # empty poisons CER
    with pytest.raises(refaudit.RefAuditError):
        store.record("it3", "corrected", audited_text="   ")


def test_ambiguous_preserves_entered_text_only_when_entered(bench):
    store = _store(bench)
    with_text = store.record("it1", "ambiguous", audited_text="אולי כך", note="smudged")
    assert with_text["status"] == "ambiguous"
    assert with_text["audited_reference"] == "אולי כך"
    untouched = store.record("it2", "ambiguous", audited_text=REF["it2"])
    assert untouched["audited_reference"] is None  # unchanged text != entered text


def test_invalid_status_rejected(bench):
    with pytest.raises(refaudit.RefAuditError):
        _store(bench).record("it1", "unchecked")


# ------------------------------------------------- resume + persistence ----


def test_resume_from_saved_state(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    store.record("it2", "corrected", audited_text="תוקן")
    reopened = _store(bench)  # fresh instance = closed & reopened UI
    assert reopened.status("it1") == "confirmed"
    assert reopened.entry("it2")["audited_reference"] == "תוקן"
    assert reopened.status("it3") == "unchecked"


def test_stale_store_never_clobbers_other_sessions_decisions(bench):
    """Two store instances (two sessions/processes): a record through a
    STALE instance must merge with, not overwrite, decisions the other
    instance already persisted."""
    session_a = _store(bench)
    session_b = _store(bench)          # loaded while the file was empty
    session_a.record("it1", "confirmed")
    session_b.record("it2", "confirmed")   # stale doc: must not erase it1
    fresh = _store(bench)
    assert fresh.status("it1") == "confirmed"
    assert fresh.status("it2") == "confirmed"


def test_atomic_save_never_corrupts_existing_state(bench, monkeypatch):
    store = _store(bench)
    store.record("it1", "confirmed")
    before = store.audit_path.read_text(encoding="utf-8")

    def boom(*a, **kw):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(refaudit.json, "dump", boom)
    with pytest.raises(RuntimeError):
        store.record("it2", "confirmed")
    monkeypatch.undo()
    # the on-disk file is still the previous complete, valid state
    assert store.audit_path.read_text(encoding="utf-8") == before
    assert _store(bench).status("it1") == "confirmed"
    assert not list(bench.glob("*.tmp"))


def test_original_reference_files_are_never_modified(bench):
    refs_before = (bench / "references.json").read_bytes()
    items_before = (bench / "items.json").read_bytes()
    store = _store(bench)
    store.record("it1", "corrected", audited_text="שונה לגמרי")
    store.record("it2", "ambiguous")
    store.record("it3", "confirmed")
    refaudit.freeze_manifest(store)
    refaudit.write_preview(store)
    refaudit.verifier_prep(store, emit=True)
    assert (bench / "references.json").read_bytes() == refs_before
    assert (bench / "items.json").read_bytes() == items_before


def test_reset_single_item(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    store.record("it2", "confirmed")
    store.reset_item("it1")
    assert store.status("it1") == "unchecked"
    assert store.status("it2") == "confirmed"


def test_global_reset_requires_explicit_confirmation(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    with pytest.raises(refaudit.RefAuditError):
        store.reset_all()
    store.reset_all(confirm="RESET")
    assert store.summary()["checked"] == 0


# --------------------------------------------------- scoring resolution ----


def test_reference_for_scoring_rules(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    store.record("it2", "corrected", audited_text="תוקן")
    store.record("it3", "ambiguous", audited_text="אולי")

    confirmed = refaudit.reference_for_scoring(store, "it1")
    assert confirmed.reference == REF["it1"] and confirmed.use_for_strict_cer

    corrected = refaudit.reference_for_scoring(store, "it2")
    assert corrected.reference == "תוקן" and corrected.use_for_strict_cer

    ambiguous = refaudit.reference_for_scoring(store, "it3")
    assert ambiguous.status == "ambiguous"
    assert ambiguous.use_for_strict_cer is False  # never silent ordinary GT


def test_final_mode_refuses_unchecked(bench):
    store = _store(bench)
    with pytest.raises(refaudit.UncheckedReferenceError):
        refaudit.reference_for_scoring(store, "it1", mode="final")
    preview = refaudit.reference_for_scoring(store, "it1", mode="preview")
    assert preview.source == "original_unaudited"
    assert preview.use_for_strict_cer is False


# ------------------------------------------------- summary + freezing ----


def test_summary_counts(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    store.record("it2", "ambiguous")
    s = store.summary()
    assert {k: s[k] for k in ("total", "checked", "confirmed", "corrected",
                               "ambiguous", "unchecked", "remaining")} == {
        "total": 3, "checked": 2, "confirmed": 1, "corrected": 0,
        "ambiguous": 1, "unchecked": 1, "remaining": 1}
    assert s["benchmark_total"] == 3 and s["excluded_not_in_scope"] == 0


def test_freeze_refused_with_unchecked_items(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    with pytest.raises(refaudit.FreezeError):
        refaudit.freeze_manifest(store)
    assert not store.manifest_path.exists()


def test_freeze_succeeds_when_complete_and_hash_binds_content(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    store.record("it2", "corrected", audited_text="תוקן")
    store.record("it3", "ambiguous")
    manifest = refaudit.freeze_manifest(store)
    assert manifest["summary"]["unchecked"] == 0
    assert len(manifest["audit_sha256"]) == 64
    assert refaudit.is_frozen(store)
    store.record("it1", "corrected", audited_text="שינוי אחרי הקפאה")
    assert not refaudit.is_frozen(store)  # stale manifest detected


def test_frozen_manifest_invalidated_by_benchmark_file_edits(bench):
    store = _store(bench)
    for item_id in store.item_ids:
        store.record(item_id, "confirmed")
    refaudit.freeze_manifest(store)
    assert refaudit.is_frozen(store)
    # editing references.json after the freeze must invalidate it
    refs = json.loads((bench / "references.json").read_text(encoding="utf-8"))
    refs["it1"]["text"] = "טקסט אחר"
    (bench / "references.json").write_text(json.dumps(refs, ensure_ascii=False),
                                           encoding="utf-8")
    assert not refaudit.is_frozen(_store(bench))


# ------------------------------------------------- historical preview ----


def test_preview_reports_old_vs_audited_and_excludes_ambiguous(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    store.record("it2", "corrected", audited_text="x = 8 ולכן התשובה נכונה")
    store.record("it3", "ambiguous")
    report = refaudit.preview_metrics(store)
    cfg_a = next(c for c in report["configs"] if c["config"] == "cfgA")
    assert cfg_a["items_with_output_compared"] == 2  # it3 ambiguous excluded
    assert cfg_a["ambiguous_excluded"] == 1 and cfg_a["ambiguous_item_ids"] == ["it3"]
    # the correction made cfgA's it2 output exactly right -> audited CER drops
    it2 = next(c for c in cfg_a["items"] if c["item_id"] == "it2")
    assert it2["old_cer"] > 0 and it2["audited_cer"] == 0.0
    assert cfg_a["affected_items"] == 1
    assert cfg_a["reference_corrections_in_compared"] == 1
    assert report["historical_results_untouched"] is True
    # historical files untouched, preview written to its own file only
    out = refaudit.write_preview(store, report)
    assert out.name == "audit_metrics_preview.json"
    assert not (bench / "m2_bench_results.csv").exists()


def test_preview_works_mid_audit_reporting_unchecked(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    report = refaudit.preview_metrics(store)
    cfg_a = next(c for c in report["configs"] if c["config"] == "cfgA")
    assert cfg_a["items_with_output_compared"] == 1
    assert cfg_a["unchecked_not_compared"] == 2


# ------------------------------------------------- verifier benchmark ----


def test_verifier_prep_dry_counts_and_emit_gating(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    store.record("it2", "corrected", audited_text="x = 8 ולכן התשובה נכונה")
    dry = refaudit.verifier_prep(store)          # dry counts always available
    assert dry["cases"] == 3 and dry["emitted"] is False
    assert dry["correct_candidate"] == 2         # cfgA it1+it2 now exact
    assert dry["error_candidate"] == 1           # cfgB it1 omission
    assert "omission" in dry["error_kind_counts"]
    with pytest.raises(refaudit.FreezeError):
        refaudit.verifier_prep(store, emit=True)  # audit not frozen yet


def test_verifier_emission_isolates_model_visible_inputs(bench):
    store = _store(bench)
    store.record("it1", "confirmed")
    store.record("it2", "corrected", audited_text="x = 8 ולכן התשובה נכונה")
    store.record("it3", "ambiguous")             # ambiguous never becomes a case
    refaudit.freeze_manifest(store)
    report = refaudit.verifier_prep(store, emit=True)
    assert report["emitted"] is True
    out = bench / "verifier_bench"
    inputs = [json.loads(l) for l in
              (out / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    labels = [json.loads(l) for l in
              (out / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(inputs) == len(labels) == 3
    for case in inputs:  # the model-visible file: opaque id + crop + candidate ONLY
        assert set(case) == {"case_id", "crop", "candidate_transcription"}
        raw = json.dumps(case, ensure_ascii=False)
        assert "reference" not in raw and "verdict" not in raw and "cer" not in raw.lower()
        # no label-predictive shortcut: the producing config must not leak
        assert "cfgA" not in raw and "cfgB" not in raw
    # ambiguous items never become cases (checked via the labels mapping)
    assert all(label["item_id"] != "it3" for label in labels)
    assert {label["source_config"] for label in labels} == {"cfgA", "cfgB"}
    kinds = {k for label in labels for k in label["error_kinds"]}
    assert "omission" in kinds
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cases"] == 3 and len(manifest["inputs_sha256"]) == 64


def test_number_formula_corruption_detected(bench):
    store = _store(bench)
    store.record("it2", "confirmed")  # reference keeps x = 3
    dry = refaudit.verifier_prep(store)
    assert dry["error_kind_counts"].get("number_sign_formula", 0) >= 1


def test_dropped_minus_sign_is_a_review_case_not_supported(bench):
    """normalize() deletes '-', so equality alone would label a dropped/added
    minus as 'supported'. The sign-aware ok-decision must catch it."""
    run = bench / "outputs" / "cfgC" / "run1"
    run.mkdir(parents=True)
    (run / "it2.json").write_text(json.dumps(
        {"transcription": "x = -3 ולכן התשובה נכונה", "item": "it2", "run": 1},
        ensure_ascii=False), encoding="utf-8")
    store = _store(bench)
    store.record("it2", "confirmed")  # audited reference keeps 'x = 3'
    dry = refaudit.verifier_prep(store)
    assert dry["error_kind_counts"].get("number_sign_formula", 0) >= 2  # x=8 and x=-3


def test_hebrew_prefix_hyphen_is_not_a_minus_sign():
    sig = refaudit.digit_op_signature
    assert sig("רק ב-High pass") == sig("רק בHigh pass")  # connector ignored
    assert sig("מ-1 עד 3") == "13"
    assert sig("x = -3") != sig("x = 3")                  # real minus kept


def test_association_items_are_excluded_from_cer_and_verifier_cases(tmp_path):
    d = tmp_path / "bench"
    (d / "crops").mkdir(parents=True)
    (d / "crops" / "as1.png").write_bytes(_PNG)
    (d / "crops" / "hd1.png").write_bytes(_PNG)
    (d / "items.json").write_text(json.dumps({"version": 2, "items": [
        {"id": "as1", "category": "option_row_association", "tier": "owner",
         "hard": False, "image": "crops/as1.png", "writer": "e003", "task": "t"},
        {"id": "hd1", "category": "handwritten_line", "tier": "owner",
         "hard": True, "image": "crops/hd1.png", "writer": "e003", "task": "t"},
    ]}, ensure_ascii=False), encoding="utf-8")
    (d / "references.json").write_text(json.dumps(
        {"as1": {"text": "0.55\n()ד0.51", "provenance": "owner"},
         "hd1": {"text": "כתב יד קשה", "provenance": "owner"}},
        ensure_ascii=False), encoding="utf-8")
    run = d / "outputs" / "cfgX" / "run1"
    run.mkdir(parents=True)
    for item_id, text in (("as1", "ד: 0.55; ג: 0.51"), ("hd1", "כתב יד קשה")):
        (run / f"{item_id}.json").write_text(json.dumps(
            {"transcription": text, "item": item_id, "run": 1},
            ensure_ascii=False), encoding="utf-8")
    store = refaudit.AuditStore(d)
    store.record("as1", "confirmed")
    store.record("hd1", "confirmed")
    # preview mirrors m2_bench_eval: association AND hard excluded from CER
    cfg = refaudit.preview_metrics(store)["configs"][0]
    assert cfg["items_with_output_compared"] == 0
    assert cfg["association_excluded_from_cer"] == 1
    assert cfg["hard_excluded_from_cer"] == 1
    # verifier cases: association excluded (its reference layout would give
    # wrong equality labels); hard kept (audited references are verified)
    dry = refaudit.verifier_prep(store)
    assert dry["association_excluded"] == 1
    assert dry["cases"] == 1 and dry["correct_candidate"] == 1


# ------------------------------------------------------- no model calls ----


def test_refaudit_never_touches_models_or_network(bench, no_network):
    """The whole audit lifecycle under a network-kill fixture, and the module
    source imports no backend/gateway/model machinery."""
    store = _store(bench)
    store.record("it1", "confirmed")
    store.record("it2", "corrected", audited_text="תוקן")
    store.record("it3", "ambiguous")
    refaudit.preview_metrics(store)
    refaudit.freeze_manifest(store)
    refaudit.verifier_prep(store, emit=True)
    source = (REPO_ROOT / "scripts" / "refaudit.py").read_text(encoding="utf-8")
    ui_source = (REPO_ROOT / "scripts" / "reference_audit_ui.py").read_text(encoding="utf-8")
    for banned in ("backends", "gateway", "httpx", "urllib", "requests",
                   "ollama", "openrouter", "anthropic", "openai"):
        assert banned not in source, banned
        assert banned not in ui_source, banned


def test_cli_summary_and_freeze_refusal(bench, capsys):
    assert refaudit.main(["--bench-dir", str(bench), "summary"]) == 0
    assert "unchecked" in capsys.readouterr().out
    assert refaudit.main(["--bench-dir", str(bench), "freeze"]) == 2
    assert "REFUSED" in capsys.readouterr().out
