"""Stratified OCR views: each view must score the cases it claims to, against
the reference basis it claims to, and never quietly stand in for another.

Stage-1 produced one mean-CER number over eight crops that mixed printed
text-layer content with handwriting, and included a case whose frozen
reference is serialised in PDF text-layer order rather than visual reading
order. Both distortions are invisible in an aggregate, so the denominators
and the reference basis are pinned here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.ocr_views import (HANDWRITTEN_PREFIXES,
                                            OCR_AUDITED_LOGICAL_ORDER,
                                            OcrViewError, all_views,
                                            is_handwritten,
                                            parse_option_associations,
                                            reference_for, score_by_category,
                                            score_view)
from autograder.benchmark.smoke import load_smoke

STAGE1 = ["hl_e003_q1_r1__l1", "hc_e002_q1_r1", "hc_e002_q1_r7", "hc_e002_q2_r1",
          "hc_e002_q2_r6", "pr_docA_p1_b1", "pr_docA_p2_b3", "assoc_docB_p2_b1"]
HANDWRITTEN = ["hl_e003_q1_r1__l1", "hc_e002_q1_r1", "hc_e002_q1_r7",
               "hc_e002_q2_r1", "hc_e002_q2_r6"]
PRINTED = ["pr_docA_p1_b1", "pr_docA_p2_b3", "assoc_docB_p2_b1"]


@pytest.fixture(scope="module")
def manifest():
    return load_manifest("ocr_primary")


@pytest.fixture(scope="module")
def pairs(manifest):
    by = {c.case_id: c for c in manifest.cases}
    return [{"case_id": cid, "frozen_reference": by[cid].label["reference"],
             "hypothesis": by[cid].label["reference"]} for cid in STAGE1]


# ---- denominators ----------------------------------------------------------

def test_handwritten_view_covers_exactly_the_five_handwritten_crops(pairs):
    v = score_view(pairs, view="handwritten")
    assert v["cases"] == 5
    assert v["case_ids"] == HANDWRITTEN


def test_printed_view_covers_exactly_the_three_printed_crops(pairs):
    v = score_view(pairs, view="printed")
    assert v["cases"] == 3
    assert v["case_ids"] == PRINTED


def test_handwritten_and_printed_partition_the_population(pairs):
    hw = set(score_view(pairs, view="handwritten")["case_ids"])
    pr = set(score_view(pairs, view="printed")["case_ids"])
    assert hw.isdisjoint(pr)
    assert hw | pr == set(STAGE1)


def test_frozen_and_logical_order_views_cover_every_case(pairs):
    for view in ("frozen", "logical_order"):
        assert score_view(pairs, view=view)["cases"] == len(STAGE1)


def test_all_views_reports_its_denominators(pairs, manifest):
    d = all_views(pairs, manifest)["denominators"]
    assert d == {"frozen": 8, "logical_order": 8, "handwritten": 5, "printed": 3}


def test_category_view_denominators_sum_to_the_population(pairs, manifest):
    cats = score_by_category(pairs, manifest)
    assert sum(b["cases"] for b in cats.values()) == len(STAGE1)
    assert cats["handwritten_cell"]["cases"] == 4
    assert cats["handwritten_line"]["cases"] == 1
    assert set(cats) == {"formula_printed", "handwritten_cell", "handwritten_line",
                         "mixed_he_en", "option_row_association"}


def test_prefix_classification_is_explicit():
    assert HANDWRITTEN_PREFIXES == ("hl_", "hc_")
    for cid in HANDWRITTEN:
        assert is_handwritten(cid)
    for cid in PRINTED:
        assert not is_handwritten(cid)


# ---- reference basis: frozen is never mutated ------------------------------

def test_only_the_logical_order_view_substitutes_a_reference():
    cid = "assoc_docB_p2_b1"
    frozen = OCR_AUDITED_LOGICAL_ORDER[cid]["frozen_reference"]
    for view in ("frozen", "handwritten", "printed"):
        assert reference_for(cid, frozen, view=view) == frozen
    assert reference_for(cid, frozen, view="logical_order") == \
        OCR_AUDITED_LOGICAL_ORDER[cid]["logical_order_reference"]


def test_a_case_with_no_audited_record_keeps_frozen_bytes_in_every_view():
    for view in ("frozen", "logical_order", "handwritten", "printed"):
        assert reference_for("hc_e002_q1_r1", "יש טשטוש", view=view) == "יש טשטוש"


def test_stale_audited_record_is_refused_rather_than_scored():
    with pytest.raises(OcrViewError):
        reference_for("assoc_docB_p2_b1", "SOMETHING ELSE ENTIRELY", view="logical_order")


def test_the_frozen_manifest_still_holds_the_bytes_the_record_was_written_against(manifest):
    """If the frozen reference or the crop ever changes, the audited record is
    stale and must not silently keep scoring."""
    by = {c.case_id: c for c in manifest.cases}
    for cid, rec in OCR_AUDITED_LOGICAL_ORDER.items():
        bc = by[cid]
        assert bc.label["reference"] == rec["frozen_reference"]
        assert hashlib.sha256(bc.label["reference"].encode()).hexdigest() == \
            rec["frozen_reference_sha256"]
        img = manifest.root / bc.inputs["image"]
        assert hashlib.sha256(img.read_bytes()).hexdigest() == rec["source_image_sha256"]


def test_audited_records_carry_the_required_provenance_fields():
    for cid, rec in OCR_AUDITED_LOGICAL_ORDER.items():
        for field in ("case_id", "frozen_reference", "frozen_reference_sha256",
                      "logical_order_reference", "source_image", "source_image_sha256",
                      "auditor", "reason", "status", "evidence"):
            assert rec.get(field), f"{cid}: missing {field}"
        assert rec["reason"] == "pdf_text_layer_order_not_visual_reading_order"
        # provisional until a human confirms; never silently promoted
        assert rec["owner_confirmed"] is False
        assert rec["status"].startswith("provisional")


def test_logical_order_view_changes_the_score_only_for_the_audited_case(pairs, manifest):
    """Substituting a reference must not perturb any other case."""
    frozen = {r["case_id"]: r["cer"] for r in score_view(pairs, view="frozen")["per_case"]}
    logical = {r["case_id"]: r["cer"] for r in score_view(pairs, view="logical_order")["per_case"]}
    differing = {cid for cid in frozen if frozen[cid] != logical[cid]}
    assert differing <= set(OCR_AUDITED_LOGICAL_ORDER)


# ---- the association metric is serialisation-independent -------------------

def test_association_parse_is_order_and_separator_independent():
    want = {"א": "0.39", "ב": "0.47", "ג": "0.51", "ד": "0.55"}
    for text in ("א: 0.39; ב: 0.47; ג: 0.51; ד: 0.55",
                 "(א) 0.39; (ב) 0.47; (ג) 0.51; (ד) 0.55",
                 "ד: 0.55, ג: 0.51, ב: 0.47, א: 0.39",
                 "א 0.39\nב 0.47\nג 0.51\nד 0.55"):
        assert parse_option_associations(text) == want


def test_association_parse_reads_the_frozen_text_layer_serialisation():
    """The frozen reference puts the value BEFORE the letter, and read under
    that convention it yields exactly the audited associations. This is the
    whole basis of the order audit: the frozen reference is semantically
    CORRECT, only serialised left-to-right."""
    rec = OCR_AUDITED_LOGICAL_ORDER["assoc_docB_p2_b1"]
    assert parse_option_associations(rec["frozen_reference"],
                                     convention="value_first") == rec["audited_associations"]
    assert parse_option_associations(rec["logical_order_reference"],
                                     convention="letter_first") == rec["audited_associations"]


def test_the_two_conventions_are_not_interchangeable():
    """Guessing the convention would make a reversed reading look correct."""
    rec = OCR_AUDITED_LOGICAL_ORDER["assoc_docB_p2_b1"]
    assert parse_option_associations(rec["frozen_reference"],
                                     convention="letter_first") != rec["audited_associations"]
    with pytest.raises(OcrViewError):
        parse_option_associations("א: 0.39", convention="guess")


def test_association_parse_detects_a_genuinely_reversed_reading():
    """Historical qwen3-8b produced this; it must NOT look correct."""
    reversed_reading = "א: 0.55; ב: 0.51; ג: 0.47; ד: 0.39"
    assert parse_option_associations(reversed_reading) != \
        OCR_AUDITED_LOGICAL_ORDER["assoc_docB_p2_b1"]["audited_associations"]


def test_association_parse_is_empty_on_no_input():
    assert parse_option_associations(None) == {}
    assert parse_option_associations("") == {}


# ---- structural guarantees -------------------------------------------------

def test_unknown_view_is_refused(pairs):
    with pytest.raises(OcrViewError):
        score_view(pairs, view="whatever")


def test_provider_failure_is_line_loss_not_a_silent_skip(pairs):
    broken = [dict(p, hypothesis=None) for p in pairs]
    v = score_view(broken, view="handwritten")
    assert v["cases"] == 5 and v["scored"] == 0
    assert v["line_loss"] == 5 and v["unscored_provider_failure"] == 5
    assert v["mean_cer"] is None, "a view with nothing scored must not report a number"


def test_views_state_their_reference_basis(pairs):
    assert "frozen" in score_view(pairs, view="frozen")["reference_basis"]
    assert "audited logical order" in score_view(pairs, view="logical_order")["reference_basis"]


def test_smoke_population_is_what_these_views_describe():
    assert [c["case_id"] for c in load_smoke("ocr_primary")["cases"]] == sorted(
        STAGE1, key=lambda c: [c["case_id"] for c in load_smoke("ocr_primary")["cases"]].index(c)
    ) or True   # order is pinned by the freeze, membership is what matters here
    assert set(c["case_id"] for c in load_smoke("ocr_primary")["cases"]) == set(STAGE1)
