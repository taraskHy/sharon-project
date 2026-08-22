"""Source provenance in the labeling bundle: carried explicitly from the
upstream records, shown to graders as context only, and PROVEN to refer to
the same benchmark case as the crop (structural + pixel correspondence).
No model/API calls."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from labeling_app.app import create_app
from labeling_app.bundle import (GRADER_PROVENANCE_FIELDS, RESIDUAL_RED_MAX, Bundle, build_bundle, case_provenance,
                                 load_provenance_sources, render_masked_page, strict_red_count)

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
pytestmark = pytest.mark.skipif(not (DATASET / "manifest.json").exists(), reason="grade_primary dataset not built")


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("bundle") / "b"
    build_bundle(DATASET, out, evaluation_root=REPO / "evaluation", page_max_edge=1000, now="2026-08-22 12:00:00")
    return out


def _labels() -> dict[str, dict]:
    return {r["case_id"]: r for r in (json.loads(l) for l in (DATASET / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}


# ------------------------------------------------------------- structural --

def test_provenance_is_explicit_and_matches_the_dataset_case(bundle_dir):
    b = Bundle(bundle_dir)
    labels = _labels()
    sources = load_provenance_sources(REPO)
    assert b.meta["items_with_page"] == 67 and b.meta["pages"] >= 10
    for it in b.items:
        cid = b.id_map[it["item_id"]]
        lab = labels[cid]
        pv = it["provenance"]
        # grader-visible fields only; nothing private leaks
        assert set(pv) <= set(GRADER_PROVENANCE_FIELDS) | {"page_image"}
        assert "source_file" not in pv and "writer" not in pv and "crop_files" not in pv
        # the provenance names the SAME case the crop/transcription came from
        assert pv["case_id"] == cid
        assert pv["exam"] == lab["writer"][1:] and pv["question_id"] == lab["question_id"]
        assert pv["part"] == f"r{lab['sub_item_id']}" and pv["row"] == int(lab["sub_item_id"])
        assert pv["line_count"] == len(lab["transcription_items"])
        # page number comes from an upstream RECORD, never from the id
        priv = b.private_provenance[it["item_id"]]
        assert priv["page_source"] in ("evaluation/hebrew_bench/crops_manifest.json", "evaluation/htr_pilot_sources.json")
        if lab["writer"] == "e002":
            rec = sources["cells"][cid]
            assert pv["page"] == rec["page"] and priv["source_file"] == rec["source"] and rec["row"] == pv["row"]
        else:
            w = sources["writers"][lab["writer"]]
            assert pv["page"] == w["sheets"][lab["question_id"]]["page"] and priv["source_file"] == w["pdf"]
        assert re.fullmatch(r"test/\d{3}_\d+\.pdf", priv["source_file"])       # grade-bearing name stays private
        assert any("line bounding box" in u for u in pv["unavailable"])          # honest: not recorded upstream
        assert pv["page_available"] is True and pv["page_image"] == f"pages/exam{pv['exam']}_p{pv['page']}.png"
        assert (bundle_dir / pv["page_image"]).exists()
        # the crop files recorded privately are exactly the dataset's evidence images for this case
        assert priv["crop_files"] == lab["evidence_images"]


def test_case_provenance_reports_unavailable_instead_of_guessing():
    sources = {"cells": {}, "writers": {}}
    pv = case_provenance("e099_q1_r3", {"transcription_items": ["hl_e099_q1_r3__l1"]}, sources)
    assert pv["page"] is None and pv["source_file"] is None
    assert any("page number" in u for u in pv["unavailable"])
    assert pv["exam"] == "099" and pv["part"] == "r3" and pv["question_id"] == "1"


# -------------------------------------------------------------- the page --

def test_served_page_is_the_recorded_source_page_with_red_ink_removed(bundle_dir):
    """Re-rendering (source_file, page) recorded in the provenance yields the
    exact bytes served — the page corresponds to the recorded source — and no
    strict-red instructor ink survives."""
    b = Bundle(bundle_dir)
    seen = set()
    for it in b.items:
        priv = b.private_provenance[it["item_id"]]
        key = (priv["source_file"], priv["page"])
        if key in seen:
            continue
        seen.add(key)
        served = (bundle_dir / it["provenance"]["page_image"]).read_bytes()
        again, rep = render_masked_page(REPO / priv["source_file"], priv["page"], max_edge=1000)
        assert again == served
        assert rep["strict_red_after"] <= RESIDUAL_RED_MAX and priv["page_report"]["strict_red_before"] > 0
        import fitz
        pix = fitz.Pixmap(served)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        assert strict_red_count(arr) == 0
        if len(seen) >= 4:
            break


def _gray(png: bytes) -> np.ndarray:
    import fitz
    pix = fitz.Pixmap(png)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(np.float32)
    return arr.mean(axis=2)


def _ncc_max(image: np.ndarray, template: np.ndarray) -> float:
    """Max normalized cross-correlation of template over image (FFT)."""
    image = image.astype(np.float64)
    template = template.astype(np.float64)
    h, w = template.shape
    H, W = image.shape
    if h > H or w > W:
        return -1.0
    t = template - template.mean()
    tn = np.sqrt((t ** 2).sum()) + 1e-6
    # sliding sums via integral images
    pad = np.pad(image, ((1, 0), (1, 0)))
    S = pad.cumsum(0).cumsum(1)
    S2 = (pad ** 2).cumsum(0).cumsum(1)
    def win(Sx):
        return Sx[h:, w:] - Sx[:-h, w:] - Sx[h:, :-w] + Sx[:-h, :-w]
    s1, s2 = win(S), win(S2)
    n = h * w
    local_var = s2 - s1 ** 2 / n
    local_std = np.sqrt(np.clip(local_var, 1e-6, None))
    fi = np.fft.rfft2(image)
    ft = np.fft.rfft2(t[::-1, ::-1], s=image.shape)
    corr = np.fft.irfft2(fi * ft, s=image.shape)[h - 1:, w - 1:]
    ncc = corr / (local_std * tn)
    return float(np.nanmax(ncc))


def _box(a: np.ndarray, k: int) -> np.ndarray:
    """Box-downsample by k (low-resolution matching is robust to the thin,
    anti-aliased strokes of handwriting rendered at different zooms)."""
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def _resize(a: np.ndarray, s: float) -> np.ndarray:
    h = max(8, int(round(a.shape[0] * s))); w = max(8, int(round(a.shape[1] * s)))
    ys = (np.arange(h) * a.shape[0] / h).astype(int); xs = (np.arange(w) * a.shape[1] / w).astype(int)
    return a[ys][:, xs]


def _best_scale_ncc(page_gray: np.ndarray, crop_gray: np.ndarray, scales, box: int = 4) -> float:
    page = _box(page_gray, box)
    best = -1.0
    for s in scales:
        t = _box(_resize(crop_gray, s), box)
        if t.shape[0] < 4 or t.shape[1] < 4 or t.shape[0] >= page.shape[0] or t.shape[1] >= page.shape[1]:
            continue
        best = max(best, _ncc_max(page, t))
    return best


def test_crop_pixels_are_found_inside_the_recorded_source_page(bundle_dir):
    """Pixel-level correspondence: the grading crop is located (by normalized
    cross-correlation at the right scale) inside the UNMASKED render of the
    page the provenance names, and clearly better than in another exam's
    page. Checked for a cell case (e002) and a line case (e003+)."""
    import fitz
    b = Bundle(bundle_dir)
    labels = _labels()
    picks = []
    for writer in ("e002", "e003"):
        cid = next(c for c, l in labels.items() if l["writer"] == writer and l["question_id"] == "1")
        picks.append(next(i for i in b.items if b.id_map[i["item_id"]] == cid))
    def render_unmasked(pdf: Path, page_no: int, max_edge: int = 1400) -> np.ndarray:
        doc = fitz.open(str(pdf)); page = doc[page_no - 1]; r = page.rect
        zoom = max_edge / max(r.width, r.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(np.float32)
        doc.close()
        return arr.mean(axis=2)
    for it in picks:
        priv = b.private_provenance[it["item_id"]]
        crop = _gray((bundle_dir / it["images"][0]).read_bytes())
        right = render_unmasked(REPO / priv["source_file"], priv["page"])
        # a different exam's page at the same position (negative control)
        other_src = "test/005_48.pdf" if priv["source_file"] != "test/005_48.pdf" else "test/003_70.pdf"
        wrong = render_unmasked(REPO / other_src, priv["page"])
        # the crops were cut from renders of unknown zoom: search a scale range at
        # low resolution (box 4). Measured: e002 cell 0.95 vs 0.48 on a wrong page;
        # e003 line 0.56 vs 0.28 — a clear, principled margin.
        scales = np.linspace(0.30, 0.70, 17)
        score_right = _best_scale_ncc(right, crop, scales)
        score_wrong = _best_scale_ncc(wrong, crop, scales)
        assert score_right > 0.45, (it["provenance"]["case_id"], score_right)
        assert score_right > score_wrong + 0.20, (it["provenance"]["case_id"], score_right, score_wrong)


# ----------------------------------------------------------------- the API --

def test_grader_api_shows_provenance_and_page_but_never_labels(bundle_dir, tmp_path):
    app = create_app(data_dir=tmp_path / "data", bundle_dir=bundle_dir)
    c = TestClient(app)
    c.post("/api/session", json={"name": "P"})
    it = c.post("/api/next").json()["item"]
    pv = it["provenance"]
    assert pv["exam"] and pv["case_id"] and pv["question_id"] and pv["part"] and pv["page"]
    assert pv["page_url"] == f"/api/pages/{it['item_id']}" and "page_image" not in pv and "source_file" not in pv
    r = c.get(pv["page_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and len(r.content) > 10000
    blob = json.dumps(it, ensure_ascii=False)
    for forbidden in ("expected", "predicted", "confidence", "split", "writer", '"score":', "test/0", ".pdf"):
        assert forbidden not in blob, forbidden
    assert c.get("/api/pages/g0000000000").status_code == 404
    # admin sees the private source file; graders never did
    adm = TestClient(app).get(f"/api/admin/items/{it['item_id']}").json()
    assert adm["provenance_private"]["source_file"].endswith(".pdf")
