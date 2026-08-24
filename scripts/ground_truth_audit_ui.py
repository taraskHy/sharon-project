"""Ground-truth audit UI — the 5 DEV cases where every model scored low.

Launch:

    .venv\\Scripts\\python.exe -m streamlit run scripts\\ground_truth_audit_ui.py -- --browser.gatherUsageStats false

All five carry instructor 4/4, so the derived explanation verdict is `valid`.
All three models independently judged the explanation weaker than that. Two
readings fit that evidence equally well and this tool cannot choose between
them — a human reading the transcription against the rubric can:

    A  derived verdict consistent with rubric/instructor practice
       (the models are simply too harsh)
    B  the instructor appears more lenient than the encoded rubric
       (full credit for a correct choice, generous about the explanation)
    C  transcription/evidence issue
    D  genuinely ambiguous

NOTHING is relabelled here. The decision is recorded in the audit JSON for
later use; changing ground truth remains a separate, deliberate act.

No model, no OCR, no network. Every decision persists atomically on click.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "evaluation" / "model_selection" / "runs" / "grade_primary" / \
    "GROUND_TRUTH_AUDIT_2026-08-25.json"

st.set_page_config(page_title="Ground-truth audit — GRADE_PRIMARY", page_icon="⚖", layout="wide")
st.markdown("""
<style>
.rtl { direction: rtl; text-align: right; font-size: 1.15rem; line-height: 1.9;
       padding: .6rem .8rem; border-radius: .5rem;
       border: 1px solid rgba(128,128,128,.35); white-space: pre-wrap; }
.warn { border-left:3px solid #d08a00; padding:.4rem .7rem; border-radius:.3rem;
        background:rgba(208,138,0,.08); }
</style>
""", unsafe_allow_html=True)


def _save(doc: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(AUDIT.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, AUDIT)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


if not AUDIT.exists():
    st.error(f"audit file not found: {AUDIT}")
    st.stop()

doc = json.loads(AUDIT.read_text(encoding="utf-8"))       # fresh every rerun
cases = doc["cases"]
OPTIONS = doc["options"]

with st.sidebar:
    st.header("Progress")
    done = sum(1 for c in cases if c.get("human_decision"))
    st.metric("decided", f"{done} / {len(cases)}")
    for i, c in enumerate(cases):
        mark = c.get("human_decision") or "·"
        if st.button(f"{mark}  {c['case_id']}", key=f"nav{i}", use_container_width=True):
            st.session_state.pos = i
            st.rerun()
    st.divider()
    st.caption("No model, no OCR, no network. Nothing is relabelled by this tool.")

if "pos" not in st.session_state:
    st.session_state.pos = next((i for i, c in enumerate(cases)
                                 if not c.get("human_decision")), 0)
pos = max(0, min(st.session_state.pos, len(cases) - 1))
c = cases[pos]

st.title(c["case_id"])
a, b, d = st.columns(3)
a.metric("writer", c["writer"])
b.metric("question", f"Q{c['question_id']} · item {c['sub_item_id']}")
d.metric("instructor score", f"{c['instructor_final_score']:g} / {c['max_score']:g}")

st.markdown(f'<div class="warn">{doc["warning"]}</div>', unsafe_allow_html=True)

left, right = st.columns([3, 2])
with left:
    st.subheader("Student explanation (frozen transcription)")
    st.markdown(f'<div class="rtl">{c["frozen_transcription"]}</div>', unsafe_allow_html=True)
    st.subheader("Rubric")
    for r in c["rubric"]:
        st.markdown(f'**{r["id"]}** <div class="rtl">{r["text"]}</div>', unsafe_allow_html=True)
    st.subheader("Official solution")
    for k, v in (c["official_solution"] or {}).items():
        st.markdown(f'**[{k}]** <div class="rtl">{v}</div>', unsafe_allow_html=True)
    with st.expander("question text"):
        st.markdown(f'<div class="rtl">{c["question_text"]}</div>', unsafe_allow_html=True)

with right:
    st.subheader("Derived ground truth")
    st.info(f"**{c['derived_explanation_verdict']}**  ·  {c['derivation_reason']}")
    st.subheader("Model predictions (context only)")
    for model, p in c["model_predictions"].items():
        st.markdown(f"**{model}** — `{p['verdict']}` (raw {p['raw_score']:g})"
                    + ("  ⚠ uncertain" if p["uncertain"] else ""))
        st.caption(p["justification"] or "_(no justification returned)_")
    st.divider()
    st.subheader("Your decision")
    current = c.get("human_decision")
    for key, label in OPTIONS.items():
        if st.button(f"{key} — {label}", key=f"opt{key}_{pos}", use_container_width=True,
                     type="primary" if current == key else "secondary"):
            c["human_decision"] = key
            _save(doc)
            st.rerun()
    note = st.text_area("Note (optional)", value=c.get("human_note", ""), key=f"note{pos}")
    if st.button("save note", use_container_width=True):
        c["human_note"] = note
        _save(doc)
        st.rerun()
    if current:
        st.success(f"recorded: **{current}** — {OPTIONS[current]}")
        if st.button("clear", use_container_width=True):
            c["human_decision"] = None
            _save(doc)
            st.rerun()

st.divider()
n1, n2 = st.columns(2)
if n1.button("← previous", use_container_width=True, disabled=pos == 0):
    st.session_state.pos = pos - 1
    st.rerun()
if n2.button("next →", use_container_width=True, disabled=pos == len(cases) - 1):
    st.session_state.pos = pos + 1
    st.rerun()
