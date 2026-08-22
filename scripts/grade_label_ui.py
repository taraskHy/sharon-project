"""Owner grading-label tool (Streamlit) — creates the HUMAN evaluation labels
for the GRADE_PRIMARY benchmark.

    .venv\\Scripts\\python.exe -m streamlit run scripts\\grade_label_ui.py -- --browser.gatherUsageStats false

What you see per case: the question (canonical text), the official solution
and rubric lines from the frozen key, the student's FROZEN audited
transcription (what the grader will read), and the answer image(s) for your
eyes only. What you decide: the final score (0..max, 0.5 steps), optional
rubric-item decisions, a note, and Confirm / Skip. Decisions are saved
atomically and incrementally to owner_labels.json (separate from the frozen
dataset; originals are never touched); you can revisit any case.

No model is called. Nothing here is a model input.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from autograder.benchmark.manifests import DEFAULT_DATASETS_ROOT, load_manifest  # noqa: E402
from autograder.benchmark.ownerlabels import OwnerLabelError, OwnerLabelStore  # noqa: E402

st.set_page_config(page_title="Grading labels — owner", page_icon="✍", layout="wide")
st.title("✍ Owner grading labels (GRADE_PRIMARY)")

ROLE = "grade_primary"
ds_root = Path(st.sidebar.text_input("Datasets root", str(DEFAULT_DATASETS_ROOT)))
dataset_dir = ds_root / ROLE
if not (dataset_dir / "manifest.json").exists():
    st.error(f"No grade_primary dataset at {dataset_dir}. Build it first: `python -m autograder bench build-grading`.")
    st.stop()

manifest = load_manifest(ROLE, datasets_root=ds_root)
cases = sorted(manifest.cases, key=lambda c: (c.split != "DEV", c.split, c.case_id))
ids = [c.case_id for c in cases]
store = OwnerLabelStore(dataset_dir)            # fresh per rerun: never a stale cached store
summ = store.summary(ids)

st.sidebar.markdown(f"**Progress:** {summ['confirmed']} confirmed · {summ['skipped']} skipped · "
                    f"**{summ['remaining']} remaining** of {summ['total']}")
st.sidebar.progress((summ["confirmed"] + summ["skipped"]) / max(summ["total"], 1))
st.sidebar.caption(f"labels file: {store.path}")
split_filter = st.sidebar.multiselect("Splits", ["DEV", "CALIBRATION", "HELD_OUT"], default=["DEV", "CALIBRATION", "HELD_OUT"])
only_unlabeled = st.sidebar.toggle("Show only unlabeled", value=False)
view_ids = [c.case_id for c in cases if c.split in split_filter and (not only_unlabeled or store.get(c.case_id) is None)]
if not view_ids:
    st.success("Nothing left in this view — every case is labeled.")
    st.stop()

if "pos" not in st.session_state or st.session_state.get("pos_view") != tuple(view_ids):
    st.session_state["pos"] = 0
    st.session_state["pos_view"] = tuple(view_ids)
pos = max(0, min(st.session_state["pos"], len(view_ids) - 1))
nav = st.columns([1, 1, 2, 1])
if nav[0].button("◀ Previous", disabled=pos == 0):
    st.session_state["pos"] = pos - 1
    st.rerun()
if nav[1].button("Next ▶", disabled=pos >= len(view_ids) - 1):
    st.session_state["pos"] = pos + 1
    st.rerun()
if nav[2].button("Jump to next unlabeled"):
    nxt = next((i for i, cid in enumerate(view_ids) if i > pos and store.get(cid) is None), None)
    if nxt is None:
        nxt = next((i for i, cid in enumerate(view_ids) if store.get(cid) is None), pos)
    st.session_state["pos"] = nxt
    st.rerun()
nav[3].caption(f"{pos + 1} / {len(view_ids)}")

case = next(c for c in cases if c.case_id == view_ids[pos])
pack = case.inputs["pack"]
existing = store.get(case.case_id)
max_score = float(pack.get("max_score") or case.label.get("max_score") or 0)

st.subheader(f"Case `{case.case_id}` — {case.split} · max {max_score:g} pts"
             + (f" · ✅ {existing['status']} ({existing.get('score')})" if existing else ""))
left, right = st.columns([3, 2])
with left:
    st.markdown("**Question (canonical)**")
    st.markdown(f"<div dir='rtl' style='font-size:1.05em'>{html.escape(pack.get('question_text', '')).replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True)
    st.markdown("**Official solution**")
    for sid, sol in (pack.get("official_solution") or {}).items():
        st.markdown(f"<div dir='rtl' style='background:rgba(127,127,127,.08);padding:.5em;border-radius:6px'>"
                    f"{html.escape(sol or '(none)')}</div>", unsafe_allow_html=True)
    if pack.get("rubric"):
        st.markdown("**Rubric / rules**")
        for line in pack["rubric"]:
            st.markdown(f"<div dir='rtl'>• {html.escape(str(line))}</div>", unsafe_allow_html=True)
    if pack.get("scoring_rules"):
        with st.expander("scoring rules"):
            for line in pack["scoring_rules"]:
                st.markdown(f"<div dir='rtl'>• {html.escape(str(line))}</div>", unsafe_allow_html=True)
    st.markdown("**Student transcription (FROZEN — exactly what the grader reads)**")
    st.markdown(f"<div dir='rtl' style='font-size:1.25em;line-height:1.7;border:1px solid rgba(127,127,127,.4);"
                f"padding:.7em;border-radius:6px'>{html.escape(case.inputs['transcription']).replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True)
    st.caption(f"transcription source: {case.label.get('transcription_source')} · items {case.label.get('transcription_items')}")
with right:
    st.markdown("**Answer image(s) — for your eyes only (not a model input)**")
    for rel in case.label.get("evidence_images") or []:
        img = REPO_ROOT / "evaluation" / rel
        if img.exists():
            st.image(str(img), width="stretch")
        else:
            st.caption(f"missing image {rel}")

st.divider()
st.markdown("**Your decision**")
d1, d2 = st.columns([1, 2])
score = d1.number_input("Final score", min_value=0.0, max_value=max(max_score, 0.0), step=0.5,
                        value=float(existing["score"]) if existing and existing.get("score") is not None else 0.0,
                        key=f"score_{case.case_id}")
rubric_ids = [ri.get("id") for ri in (pack.get("rubric_items") or []) if ri.get("id")]
met: list[str] = []
if rubric_ids:
    d2.caption("rubric items met (optional)")
    prev = set((existing or {}).get("rubric_met") or [])
    for rid in rubric_ids:
        if d2.checkbox(str(rid), value=rid in prev, key=f"ri_{case.case_id}_{rid}"):
            met.append(rid)
note = st.text_input("Note (optional)", value=(existing or {}).get("note", ""), key=f"note_{case.case_id}")
b = st.columns(4)
if b[0].button("✅ Confirm score", type="primary", key=f"confirm_{case.case_id}"):
    try:
        store.record(case.case_id, score=score, max_score=max_score, rubric_met=met, note=note, status="confirmed")
        st.success("saved")
        nxt = next((i for i, cid in enumerate(view_ids) if i > pos and store.get(cid) is None), None)
        if nxt is not None:
            st.session_state["pos"] = nxt
        st.rerun()
    except OwnerLabelError as e:
        st.error(str(e))
if b[1].button("⏭ Skip (cannot judge)", key=f"skip_{case.case_id}"):
    store.record(case.case_id, score=None, note=note, status="skipped")
    st.rerun()
if existing and b[2].button("↺ Reset this case", key=f"reset_{case.case_id}"):
    store.reset(case.case_id)
    st.rerun()
b[3].caption("Saved atomically to owner_labels.json; the frozen dataset files are never modified.")
