"""Stratified evaluation views for the OCR benchmark.

A single "mean CER over 8 crops" number answered no question anyone actually
has. Stage-1 made that concrete twice over:

* ``assoc_docB_p2_b1`` scored CER 0.7083 for every candidate that read the
  crop CORRECTLY, because the frozen reference serialises a right-to-left
  option row in the PDF text layer's left-to-right order. The metric was
  measuring the reference's serialisation, not the model.
* printed text-layer crops (CER 0.00-0.10) and handwritten answer cells
  (CER 0.72-1.00) were averaged into one figure, which flattered every
  candidate on the only content the product actually needs.

So this module exposes named views, each with an explicit denominator, and
refuses to let one silently stand in for another. It NEVER edits a frozen
reference: the audited logical-order reading lives beside the frozen bytes in
``OCR_AUDITED_LOGICAL_ORDER`` and is applied only by the view that says so.

Views
-----
``frozen``          all cases, frozen reference bytes. The reproducible,
                    historical metric; the one comparable to Stage-1.
``logical_order``   all cases, with an audited logical-order reference
                    substituted where a record exists. DIAGNOSTIC ONLY.
``handwritten``     the handwritten crops only. The primary product-relevant
                    view: handwriting is why this role exists.
``printed``         the printed / born-digital text-layer crops only.
``by_category``     the frozen prompt category split.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .manifests import BenchmarkManifest
from .ocr_writer_metrics import pair_metrics

#: Case-id prefixes that identify a HANDWRITTEN crop. ``hl_`` = one handwritten
#: line, ``hc_`` = a handwritten answer cell. Everything else in this benchmark
#: is printed or born-digital text-layer content.
HANDWRITTEN_PREFIXES: tuple[str, ...] = ("hl_", "hc_")

#: Human-audited logical-order readings. A record here NEVER replaces the
#: frozen reference; it is an additional, separately hashed reading applied
#: only by the ``logical_order`` view. Keyed by case id.
#:
#: Each record must carry: the frozen reference it sits beside, the logical
#: reference, the source image sha256, the auditor, the reason and a status.
#: ``status`` is "provisional" until the project owner confirms it by eye.
OCR_AUDITED_LOGICAL_ORDER: dict[str, dict[str, Any]] = {
    "assoc_docB_p2_b1": {
        "case_id": "assoc_docB_p2_b1",
        "frozen_reference": "0.55\n()ד0.51\n()ג0.47\n()ב0.39\n()א",
        "frozen_reference_sha256":
            "ce52194d71d82f64d3e9a1cbbce79606c4d00fc3ede55ca7e0837accf65f4826",
        "logical_order_reference": "(א) 0.39; (ב) 0.47; (ג) 0.51; (ד) 0.55",
        "audited_associations": {"א": "0.39", "ב": "0.47", "ג": "0.51", "ד": "0.55"},
        "source_image": "crops/assoc_docB_p2_b1.png",
        "source_image_sha256":
            "971600121151bc5ba804bdcaa5a243de07c6f52c62c275d378c5d9a7a0b900b4",
        "auditor": "AI assistant, by direct inspection of the crop image "
                   "(exact model identity recorded in the audited-record artifact)",
        "owner_confirmed": False,
        "status": "provisional_pending_owner_confirmation",
        "reason": "pdf_text_layer_order_not_visual_reading_order",
        "evidence": [
            "the crop is a SINGLE right-to-left row: (א) 0.39   (ב) 0.47   (ג) 0.51   (ד) 0.55",
            "the frozen reference's own provenance_detail says the pairs were derived "
            "mechanically: 'value = nearest numeric word left of the option letter'",
            "the frozen reference therefore encodes CORRECT pairs in text-layer "
            "(left-to-right) order, with line breaks between visual columns",
            "historical qwen3-27b (independent of Stage-1) read the same RTL order; "
            "historical qwen3-8b produced the REVERSED association, so agreement here "
            "is discriminative rather than trivial",
        ],
        "caveat": (
            "the logical_order_reference's separator convention ('; ' and spacing) was "
            "chosen by the auditor, and CER/WER against it are therefore sensitive to a "
            "formatting choice no model could know. Read association_exact - which is "
            "serialisation-independent - as the real signal for this case."),
    },
}


class OcrViewError(RuntimeError):
    pass


def is_handwritten(case_id: str) -> bool:
    return str(case_id).startswith(HANDWRITTEN_PREFIXES)


def reference_for(case_id: str, frozen_reference: str, *, view: str) -> str:
    """The reference string a given view scores against.

    Only ``logical_order`` may substitute, and only where an audited record
    exists. Every other view gets the frozen bytes, unconditionally.
    """
    if view != "logical_order":
        return frozen_reference
    rec = OCR_AUDITED_LOGICAL_ORDER.get(case_id)
    if rec is None:
        return frozen_reference
    if rec["frozen_reference"] != frozen_reference:
        raise OcrViewError(
            f"{case_id}: the audited logical-order record was written against a "
            "different frozen reference than the manifest now holds; refusing to "
            "score a stale override.")
    return rec["logical_order_reference"]


def parse_option_associations(text: str | None, *,
                              convention: str = "letter_first") -> dict[str, str]:
    """Letter -> value pairs from an option-row transcription, independent of
    ordering, separators and parentheses.

    The convention is EXPLICIT and never guessed, because the two possibilities
    are exactly the defect under audit. The crop is a right-to-left row, so:

    ``letter_first``  reading the row in visual RTL order gives "(א) 0.39" —
                      the letter, then its value. This is what the frozen
                      prompt asks a model to emit, so it is the default.
    ``value_first``   serialising that same row left-to-right (what the PDF
                      text layer does) gives "0.39 (א)" — the value, then the
                      letter it belongs to. This is the frozen reference's
                      form, and its own provenance says so: "value = nearest
                      numeric word left of the option letter".

    Guessing between them would make a reversed reading indistinguishable from
    a differently-serialised correct one, which is the exact error this whole
    audit exists to separate.
    """
    if not text:
        return {}
    import re
    if convention not in ("letter_first", "value_first"):
        raise OcrViewError(f"unknown option-association convention {convention!r}")
    out: dict[str, str] = {}
    if convention == "letter_first":
        for letter, value in re.findall(
                r"[(\)]*\s*([אבגד])\s*[)\(]*\s*[:\-]?\s*(\d+(?:\.\d+)?)", text):
            out.setdefault(letter, value)
    else:
        for value, letter in re.findall(
                r"(\d+(?:\.\d+)?)\s*[(\)]*\s*([אבגד])\s*[)\(]*", text):
            out.setdefault(letter, value)
    return out


def _block(rows: list[dict]) -> dict[str, Any]:
    scored = [r for r in rows if r["cer"] is not None]
    cers = [r["cer"] for r in scored]
    wers = [r["wer"] for r in scored]
    return {
        "cases": len(rows),
        "scored": len(scored),
        "unscored_provider_failure": sum(1 for r in rows if r["cer"] is None),
        "exact_match": sum(1 for r in rows if r["exact_match"]),
        "mean_cer": round(statistics.mean(cers), 4) if cers else None,
        "median_cer": round(statistics.median(cers), 4) if cers else None,
        "mean_wer": round(statistics.mean(wers), 4) if wers else None,
        "median_wer": round(statistics.median(wers), 4) if wers else None,
        "mean_omission_rate": round(statistics.mean(
            [r["omission_rate"] for r in scored]), 4) if scored else None,
        "mean_hallucination_rate": round(statistics.mean(
            [r["hallucination_rate"] for r in scored]), 4) if scored else None,
        "line_loss": sum(1 for r in rows if r["line_lost"]),
        "digit_sign_errors": sum(1 for r in scored if r["digit_sign_error"]),
        "case_ids": [r["case_id"] for r in rows],
    }


def score_view(pairs: list[dict], *, view: str) -> dict[str, Any]:
    """Score one view. ``pairs`` items: {case_id, frozen_reference, hypothesis}."""
    if view not in ("frozen", "logical_order", "handwritten", "printed"):
        raise OcrViewError(f"unknown view {view!r}")
    sel = pairs
    if view == "handwritten":
        sel = [p for p in pairs if is_handwritten(p["case_id"])]
    elif view == "printed":
        sel = [p for p in pairs if not is_handwritten(p["case_id"])]
    rows = []
    for p in sel:
        ref = reference_for(p["case_id"], p["frozen_reference"], view=view)
        hyp = p.get("hypothesis")
        m = pair_metrics(ref, hyp)
        rows.append({"case_id": p["case_id"], "reference_used": ref, "hypothesis": hyp,
                     "exact_match": hyp is not None and hyp == ref, **m})
    out = _block(rows)
    out["view"] = view
    out["reference_basis"] = ("audited logical order where a record exists, frozen otherwise"
                              if view == "logical_order" else "frozen reference bytes")
    out["per_case"] = rows
    return out


def score_by_category(pairs: list[dict], manifest: BenchmarkManifest) -> dict[str, Any]:
    by = {c.case_id: c for c in manifest.cases}
    rows = []
    for p in pairs:
        m = pair_metrics(p["frozen_reference"], p.get("hypothesis"))
        rows.append({"case_id": p["case_id"],
                     "category": by[p["case_id"]].meta.get("category"),
                     "exact_match": p.get("hypothesis") == p["frozen_reference"], **m})
    return {cat: _block([r for r in rows if r["category"] == cat])
            for cat in sorted({r["category"] for r in rows})}


def all_views(pairs: list[dict], manifest: BenchmarkManifest) -> dict[str, Any]:
    """Every view at once, with the denominators stated so no reader has to
    infer which cases a number covers."""
    return {
        "frozen": score_view(pairs, view="frozen"),
        "logical_order": score_view(pairs, view="logical_order"),
        "handwritten": score_view(pairs, view="handwritten"),
        "printed": score_view(pairs, view="printed"),
        "by_category": score_by_category(pairs, manifest),
        "denominators": {
            "frozen": len(pairs),
            "logical_order": len(pairs),
            "handwritten": sum(1 for p in pairs if is_handwritten(p["case_id"])),
            "printed": sum(1 for p in pairs if not is_handwritten(p["case_id"])),
        },
        "policy": ("the frozen view is the reproducible metric and the one comparable to "
                   "Stage-1; logical_order is a DIAGNOSTIC and never replaces it; "
                   "handwritten is the product-relevant view and is never merged with printed."),
    }


def load_audited_records(path: Path | str | None = None) -> dict[str, Any]:
    """The audited logical-order records, from disk when a path is given."""
    if path is None:
        return dict(OCR_AUDITED_LOGICAL_ORDER)
    return json.loads(Path(path).read_text(encoding="utf-8"))["records"]


__all__ = ["HANDWRITTEN_PREFIXES", "OCR_AUDITED_LOGICAL_ORDER", "OcrViewError",
           "is_handwritten", "reference_for", "parse_option_associations",
           "score_view", "score_by_category", "all_views", "load_audited_records"]
