"""Local Streamlit annotation app for the HTR pilot (fully offline).

    .venv\\Scripts\\python.exe -m streamlit run scripts/htr_annotation_app.py

Annotates one LINE CROP at a time (the training unit), with the cleaned
cell and the original cell shown for context. Every button writes the
record to disk immediately (atomic file per sample); closing the app never
loses a saved sample, and relaunching resumes at the first undecided one.

Package root defaults to evaluation/htr_pilot; override with the
HTR_PILOT_ROOT environment variable (used by tests/smoke runs so dummy
annotations never touch the real package).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.htr_annotation_lib import (  # noqa: E402
    UNREADABLE_TOKEN, load_all_annotations, load_annotation, load_samples,
    locked_against_overwrite, make_record, package_root, progress,
    resume_index, save_annotation,
)

st.set_page_config(page_title="HTR pilot annotation", layout="wide")
st.markdown(
    """
    <style>
    textarea { direction: rtl !important; text-align: right !important;
               font-size: 1.25rem !important; }
    div[data-testid="stImage"] img { image-rendering: -webkit-optimize-contrast; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = package_root()
if not (ROOT / "splits").exists():
    st.error(f"package root {ROOT} has no splits/ — run scripts/htr_pilot_build.py first")
    st.stop()


def _load_split(split: str):
    samples = load_samples(ROOT, split)
    annotations = load_all_annotations(ROOT, split)
    return samples, annotations


with st.sidebar:
    split = st.selectbox("Split", ("train", "val", "internal_test"), key="split")
    samples, annotations = _load_split(split)
    prog = progress(samples, annotations)
    st.metric("Verified", f"{prog['verified']} / {prog['total']}")
    st.caption(f"flagged (bad-seg / recrop / skip): {prog['flagged']} · "
               f"untouched: {prog['remaining']}")
    st.progress(prog["verified"] / max(prog["total"], 1))
    annotator = st.text_input("Annotator", value="owner")
    if st.button("Jump to first undecided"):
        st.session_state[f"idx_{split}"] = resume_index(samples, annotations)
        st.rerun()
    st.divider()
    st.caption(
        "Rules: copy EXACTLY what is written, character for character — no "
        "spelling fixes, no completion. Unreadable word → insert "
        f"{UNREADABLE_TOKEN} in its place. Whole line unreadable → the "
        "Unreadable button. Empty line image → Blank. Crop shows parts of "
        "two text lines or cuts one → Bad segmentation. Cell box itself "
        "wrong → Needs recrop. Crossed-out but readable text: transcribe "
        "it; add a note."
    )

if not samples:
    st.warning("split is empty")
    st.stop()

idx_key = f"idx_{split}"
if idx_key not in st.session_state:
    st.session_state[idx_key] = resume_index(samples, annotations)
idx = int(st.session_state[idx_key]) % len(samples)
sample = samples[idx]
sid = sample["sample_id"]
existing = load_annotation(ROOT, split, sid)

left, right = st.columns([3, 2])
with left:
    st.subheader(f"{sid}  ·  {idx + 1}/{len(samples)}")
    st.caption(f"writer {sample['writer']} · Q{sample['question']} row "
               f"{sample['row']} · line {sample['line_index']}/{sample['n_lines']}"
               + (" · **expected BLANK**" if sample.get("expected_blank") else ""))
    st.image(str(ROOT / sample["images"]["line"]), caption="line crop (cleaned)",
             width="stretch")
    with st.expander("cleaned cell (context)", expanded=True):
        st.image(str(ROOT / sample["images"]["cell_clean"]), width="stretch")
    with st.expander("ORIGINAL cell (always check against this)", expanded=True):
        st.image(str(ROOT / sample["images"]["cell_orig"]), width="stretch")

with right:
    if existing:
        st.info(f"saved: **{existing['status']}** at {existing['saved_at']}")
    unlock_key = f"unlock_{split}_{sid}"
    if existing and existing.get("human_verified"):
        st.warning("This record is **owner-verified**. It is locked against "
                   "overwrite; tick below only to deliberately re-annotate.")
        st.checkbox("Unlock this verified record", key=unlock_key)
    locked = locked_against_overwrite(
        existing, st.session_state.get(unlock_key, False))
    text_key, notes_key = f"text_{split}_{sid}", f"notes_{split}_{sid}"
    if text_key not in st.session_state:
        st.session_state[text_key] = (existing or {}).get("transcription", "")
    if notes_key not in st.session_state:
        st.session_state[notes_key] = (existing or {}).get("notes", "")

    if st.button(f"insert {UNREADABLE_TOKEN}"):
        st.session_state[text_key] = (st.session_state[text_key] + " "
                                      + UNREADABLE_TOKEN).strip()
    text = st.text_area("Transcription (RTL)", key=text_key, height=140)
    notes = st.text_input("Notes", key=notes_key)

    def commit(status: str, advance: bool = True) -> None:
        if locked_against_overwrite(
                existing, st.session_state.get(unlock_key, False)):
            st.error("verified record is locked — tick 'Unlock this verified "
                     "record' to deliberately re-annotate it")
            return
        rec = make_record(sample, st.session_state[text_key], status,
                          st.session_state[notes_key], annotator)
        save_annotation(ROOT, rec)
        if advance:
            st.session_state[idx_key] = min(idx + 1, len(samples) - 1)
        st.rerun()

    def autosave_draft() -> None:
        """Persist unsaved edits before navigation (autosave semantics)."""
        stored = (existing or {}).get("transcription", ""), (existing or {}).get("notes", "")
        current = st.session_state[text_key].strip(), st.session_state[notes_key].strip()
        if current != (stored[0], stored[1]) and any(current) and \
                (existing is None or existing["status"] in ("draft", "skipped")):
            rec = make_record(sample, current[0], "draft", current[1], annotator)
            save_annotation(ROOT, rec)

    ok_label = "✔ Save and next"
    if st.button(ok_label, type="primary", use_container_width=True,
                 disabled=locked):
        if not st.session_state[text_key].strip():
            st.error("empty transcription — use Blank / Unreadable instead")
        else:
            commit("ok")
    c1, c2 = st.columns(2)
    if c1.button("Whole line unreadable", use_container_width=True,
                 disabled=locked):
        commit("unreadable_full")
    if c2.button("Blank (no writing)", use_container_width=True,
                 disabled=locked):
        commit("blank")
    c3, c4 = st.columns(2)
    if c3.button("Bad segmentation", use_container_width=True,
                 disabled=locked):
        commit("bad_segmentation")
    if c4.button("Needs recrop", use_container_width=True, disabled=locked):
        commit("needs_recrop")
    if st.button("Skip for now", use_container_width=True, disabled=locked):
        commit("skipped")

    st.divider()
    p1, p2 = st.columns(2)
    if p1.button("◀ Previous", use_container_width=True):
        autosave_draft()
        st.session_state[idx_key] = max(idx - 1, 0)
        st.rerun()
    if p2.button("Next ▶", use_container_width=True):
        autosave_draft()
        st.session_state[idx_key] = min(idx + 1, len(samples) - 1)
        st.rerun()
