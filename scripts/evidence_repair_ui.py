"""Manual GRADE_PRIMARY evidence repair (Streamlit) — human transcription of the
lines the frozen OCR benchmark never audited.

    .venv\\Scripts\\python.exe -m streamlit run scripts\\evidence_repair_ui.py -- --browser.gatherUsageStats false

For each line the dataset reports as missing an audited transcription it shows:
the case and its question/part, "line N of M", the whole answer CELL with every
recorded line marked, the sibling lines that ARE audited (with their text, as
context), the mis-segmented crop as it stands, the proposed repaired crop, and —
on demand — the full source page with the instructor's red ink masked.

You either (a) correct the crop rectangle (1-D: the lines are full-width bands of
the cell, so only the top/bottom edges are adjustable) and type the handwritten
line, or (b) record that the sliver is not a distinct line at all. Both are saved
with provenance to ``manual_evidence_repairs.jsonl`` next to the dataset.

No OCR, no model, no network. The instructor's grade is never displayed.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from autograder.benchmark.evidence_repairs import (  # noqa: E402
    RepairError, RepairStore, case_geometry, expected_repairs, render_band, repair_status, suggested_band,
    verify_repairs)
from autograder.benchmark.manifests import DEFAULT_DATASETS_ROOT  # noqa: E402

st.set_page_config(page_title="Evidence repair — grade_primary", page_icon="✂", layout="wide")
st.title("✂ Manual evidence repair — GRADE_PRIMARY")
st.caption("Human transcription of student lines the OCR benchmark never audited. "
           "The frozen `evaluation/hebrew_bench_v2` is never modified. No OCR, no model, no grades shown.")

EVAL_ROOT = REPO_ROOT / "evaluation"
ds_root = Path(st.sidebar.text_input("Datasets root", str(DEFAULT_DATASETS_ROOT)))
dataset = ds_root / "grade_primary"
if not (dataset / "manifest.json").exists():
    st.error(f"No grade_primary dataset at {dataset}.")
    st.stop()

exp = expected_repairs(dataset)
store = RepairStore(dataset)                      # fresh per rerun: never a stale cached store
recs = store.records()
status = repair_status(dataset)
if not exp:
    st.success("Nothing to repair — every recorded line already has an audited transcription.")
    st.stop()

st.sidebar.markdown(f"**Progress:** {status['repaired']} of {status['expected']} lines repaired")
st.sidebar.progress(status["repaired"] / max(status["expected"], 1))
st.sidebar.caption(f"store: {store.path}")
only_open = st.sidebar.toggle("Show only unrepaired", value=True)
view = [e for e in exp if not (only_open and e["line_id"] in recs)]
if not view:
    st.success("All expected lines are repaired.")
    st.markdown("Fold them into the dataset with:")
    st.code("python -m autograder bench apply-evidence-repairs --role grade_primary", language="bash")
    with st.expander("verification"):
        st.json(verify_repairs(dataset, evaluation_root=EVAL_ROOT))
    st.stop()

# cases_labels.jsonl is deliberately NOT loaded here: it carries the instructor's
# score, and seeing a grade before transcribing would bias the transcription.
inputs = {json.loads(l)["case_id"]: json.loads(l)
          for l in (dataset / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
sources = json.loads((EVAL_ROOT / "htr_pilot_sources.json").read_text(encoding="utf-8"))

ids = [e["line_id"] for e in view]
if st.session_state.get("view_ids") != tuple(ids):
    st.session_state["view_ids"] = tuple(ids)
    st.session_state["pos"] = 0
pos = max(0, min(st.session_state.get("pos", 0), len(ids) - 1))
nav = st.columns([1, 1, 3, 2])
if nav[0].button("◀ Previous", disabled=pos == 0):
    st.session_state["pos"] = pos - 1
    st.rerun()
if nav[1].button("Next ▶", disabled=pos >= len(ids) - 1):
    st.session_state["pos"] = pos + 1
    st.rerun()
nav[3].caption(f"{pos + 1} / {len(ids)}")

item = view[pos]
case_id, line_id = item["case_id"], item["line_id"]
inp = inputs[case_id]
try:
    geo = case_geometry(case_id, evaluation_root=EVAL_ROOT)
except RepairError as e:
    st.error(f"geometry could not be derived for {case_id}: {e}")
    st.stop()
line_rec = next(l for l in geo["lines"] if l["sample_id"] == line_id)
writer, q, r = case_id.split("_")[0], case_id.split("_")[1][1:], case_id.split("_")[2][1:]
src = sources.get(writer, {})
page_no = (src.get("sheets", {}).get(q) or {}).get("page")
existing = recs.get(line_id)

st.subheader(f"`{case_id}` — exam {writer[1:]} · question {q} · part r{r} · "
             f"line {line_rec['line_index']} of {geo['n_lines']}"
             + ("  ✅ repaired" if existing else ""))
st.caption(f"line id `{line_id}` · upstream status **{item['transcription_status']}** · "
           f"cell crop {geo['cell_width']}×{geo['cell_height']} px "
           f"({geo['cell_image']}) · line inventory: {geo['line_inventory_source']}")
st.warning("This line's upstream crop is tagged **bad_segmentation** — the crop geometry is wrong "
           "(a sliver, or two lines in one image). Check the whole cell below and fix the crop BEFORE "
           "transcribing. If the region is not a distinct line of writing, say so instead of inventing text.")

left, right = st.columns([3, 2])
with left:
    st.markdown("**The whole answer cell** (every recorded line marked; this is the authoritative region)")
    st.image(str(Path(geo["cell_image_abs"])), width="stretch")
    rows = [{"line": l["line_index"], "sample_id": l["sample_id"], "y0": l["y0"], "y1": l["y1"],
             "height": l["height"], "status": l["annotation_status"], "audited": l["audited"],
             "this one": l["sample_id"] == line_id} for l in geo["lines"]]
    st.dataframe(rows, width="stretch", hide_index=True)
    if geo["uncovered_bands"]:
        st.caption("cell regions no audited line covers: "
                   + ", ".join(f"y {b['y0']}–{b['y1']} ({b['height']}px)" for b in geo["uncovered_bands"]))

    st.markdown("**Lines of this cell that are already audited** (context — do not retype these)")
    audited_texts = (inp.get("transcription") or "").split("\n")
    ai = 0
    for l in geo["lines"]:
        if l["sample_id"] == line_id:
            st.markdown(f"<div style='opacity:.6'>— line {l['line_index']}: <em>this line, to repair</em></div>",
                        unsafe_allow_html=True)
            continue
        text = audited_texts[ai] if ai < len(audited_texts) else ""
        ai += 1
        st.image(str(EVAL_ROOT / l["image"]), width="stretch")
        st.markdown(f"<div dir='rtl' style='font-size:1.1em;border:1px solid #8884;border-radius:6px;padding:.4em'>"
                    f"{html.escape(text)}</div>", unsafe_allow_html=True)

    with st.expander("Full source page (instructor's red ink masked)"):
        if page_no and src.get("pdf"):
            st.caption(f"{src['pdf']} page {page_no} — rendered locally, red ink whitened")
            try:
                from labeling_app.bundle import render_masked_page
                png, rep = render_masked_page(REPO_ROOT / src["pdf"], int(page_no), max_edge=1600)
                st.image(png, width="stretch")
                st.caption(f"masked {rep['masked_pixels']} red pixels; residual strict-red {rep['strict_red_after']}")
            except Exception as e:  # noqa: BLE001
                st.warning(f"could not render the page: {type(e).__name__}: {e}")
        else:
            st.info("no recorded source page for this writer/question")

with right:
    st.markdown("**Repair the crop** — lines are full-width bands, so only the top/bottom edges move")
    default = suggested_band(geo, line_id)
    prev_geo = (existing or {}).get("crop_geometry") or {}
    ky0, ky1 = f"y0_{line_id}", f"y1_{line_id}"
    st.session_state.setdefault(ky0, int(prev_geo.get("y0", default["y0"])))
    st.session_state.setdefault(ky1, int(prev_geo.get("y1", default["y1"])))
    # the presets run BEFORE the number inputs exist: Streamlit forbids writing a
    # widget's state once that widget has been instantiated in the same run.
    c1, c2, c3 = st.columns(3)
    if c1.button("suggested", key=f"sug_{line_id}",
                 help="the largest region of the cell that no audited line covers"):
        st.session_state[ky0], st.session_state[ky1] = default["y0"], default["y1"]
    if c2.button("original crop", key=f"orig_{line_id}", help="the mis-segmented crop's own band"):
        st.session_state[ky0], st.session_state[ky1] = line_rec["y0"], line_rec["y1"]
    if c3.button("whole cell", key=f"whole_{line_id}"):
        st.session_state[ky0], st.session_state[ky1] = 0, geo["cell_height"]
    y0 = st.number_input("top (y0)", 0, geo["cell_height"] - 1, key=ky0)
    y1 = st.number_input("bottom (y1)", 1, geo["cell_height"], key=ky1)
    if y1 <= y0 + 1:
        st.error("bottom must be below top")
        st.stop()
    crop_png = render_band(geo, y0, y1)
    st.markdown(f"**Repaired crop preview** — y {y0}–{y1} ({y1 - y0} px)")
    st.image(crop_png, width="stretch")
    st.caption("compare with the mis-segmented original:")
    st.image(str(EVAL_ROOT / line_rec["image"]), width="stretch")

    st.divider()
    artifact = st.checkbox("This region is NOT a distinct line of writing (segmentation artifact — no text)",
                           value=(existing or {}).get("disposition") == "no_text_segmentation_artifact",
                           key=f"art_{line_id}")
    text = ""
    if not artifact:
        ktx = f"tx_{line_id}"
        st.session_state.setdefault(ktx, (existing or {}).get("transcription", ""))
        carried = st.session_state.pop("_carry_text", None)     # handed over by the copy button below,
        if carried and carried[0] == line_id:                   # before this widget is instantiated
            st.session_state[ktx] = carried[1]
        text = st.text_area("Transcription of THIS line (exactly as written)", height=110, key=ktx,
                            help="type the handwriting; do not correct spelling, do not translate")
        if text.strip():
            st.markdown(f"<div dir='rtl' style='font-size:1.25em;border:1px solid #8884;border-radius:6px;"
                        f"padding:.5em'>{html.escape(text)}</div>", unsafe_allow_html=True)
    with st.expander("previously recorded text for the mis-segmented crop (unverified — may duplicate another line)"):
        ann = None
        for p in (EVAL_ROOT / "htr_pilot" / "annotations").rglob(f"{line_id}.json"):
            ann = json.loads(p.read_text(encoding="utf-8"))
        if ann and (ann.get("transcription") or "").strip():
            st.caption("recorded on a crop known to be mis-segmented; it is NOT verified and may be a copy of "
                       "another line — read the cell above before trusting any of it")
            st.markdown(f"<div dir='rtl' style='opacity:.75'>{html.escape(ann['transcription'])}</div>",
                        unsafe_allow_html=True)
            if st.button("copy into the box above", key=f"cp_{line_id}", disabled=artifact):
                st.session_state["_carry_text"] = (line_id, ann["transcription"])
                st.rerun()
        else:
            st.caption("no text was ever recorded for this crop")
    note = st.text_input("Note (optional)", value=(existing or {}).get("note", ""), key=f"nt_{line_id}")
    who = st.text_input("Verified by", value=(existing or {}).get("verified_by", "owner"), key=f"by_{line_id}")

    if st.button("💾 Save repair", type="primary", key=f"save_{line_id}"):
        try:
            store.save(case_id=case_id, line_id=line_id,
                       transcription="" if artifact else text,
                       disposition="no_text_segmentation_artifact" if artifact else "transcribed",
                       verified_by=who, crop_png=crop_png,
                       crop_geometry={"cell_image": geo["cell_image"], "cell_sha256": geo["cell_sha256"],
                                      "x0": 0, "y0": int(y0), "x1": geo["cell_width"], "y1": int(y1),
                                      "width": geo["cell_width"], "height": int(y1 - y0),
                                      "derivation": "manual band selection over the cell crop; the cell is proven "
                                                    "by pixel-exact containment of every recorded line crop"},
                       source_pdf=src.get("pdf"), source_page=page_no,
                       original_crop={"image": line_rec["image"], "sha256": line_rec["image_sha256"],
                                      "y0": line_rec["y0"], "y1": line_rec["y1"],
                                      "status": line_rec["annotation_status"]},
                       line_index=line_rec["line_index"], line_count=geo["n_lines"], note=note)
            st.success("saved")
            if pos < len(ids) - 1:
                st.session_state["pos"] = pos + 1
            st.rerun()
        except RepairError as e:
            st.error(str(e))
    if existing and st.button("↺ Remove this repair", key=f"del_{line_id}"):
        store.delete(line_id)
        st.rerun()

st.divider()
with st.expander("Repair status / integrity"):
    st.json(verify_repairs(dataset, evaluation_root=EVAL_ROOT))
