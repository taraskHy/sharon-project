"""Selection-correctness audit UI — the 8 ambiguous GRADE_PRIMARY cases only.

Launch:

    .venv\\Scripts\\python.exe -m streamlit run scripts\\selection_audit_ui.py -- --browser.gatherUsageStats false

(the flag keeps Streamlit's usage telemetry off, matching the repo's
no-network convention)

The auditor answers ONE question per case: which option did the student
actually mark in the answer table?

Deliberately NOT shown, so the audit cannot be reasoned backwards or anchored:

* the instructor's score for the case — the whole point is that a 0 does not
  tell us whether the selection or the explanation caused it;
* the letter that ``qwen3-vl:8b-instruct`` extracted — a model's reading can
  never be the ground truth a model is later benchmarked against.

Shown instead: the exam scan page that the exam's own instructions designate
as the authoritative answer sheet, the row to read, and (only where the exam
version was confirmed by the operator) the option the key accepts.

No model, no OCR, no network. Every decision persists atomically the moment a
button is pressed (scripts/selaudit.py), so closing or rerunning never loses
work — reopening resumes at the first unaudited case.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("selaudit", _HERE / "selaudit.py")
selaudit = importlib.util.module_from_spec(spec)
sys.modules.setdefault("selaudit", selaudit)
spec.loader.exec_module(selaudit)

REPO_ROOT = _HERE.parent

st.set_page_config(page_title="Selection audit — GRADE_PRIMARY",
                   page_icon="🔤", layout="wide")

st.markdown("""
<style>
.big-letter { font-size: 2.2rem; font-weight: 700; letter-spacing:.15em; }
.muted { opacity:.7; font-size:.9rem; }
.warn { border-left:3px solid #d08a00; padding:.4rem .7rem; border-radius:.3rem;
        background:rgba(208,138,0,.08); }
</style>
""", unsafe_allow_html=True)


# A FRESH store every rerun: re-reads the audit file from disk, so the UI can
# never hold (and later write back) a stale in-memory copy.
store = selaudit.SelectionAuditStore()
ids = store.case_ids


@st.cache_data(show_spinner=False)
def render_page(pdf_path: str, page_no: int, dpi: int) -> bytes:
    """One page of the scan as PNG bytes. Pure rendering — no OCR, no model."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        page = doc[page_no - 1]
        return page.get_pixmap(dpi=dpi).tobytes("png")


def _first_unaudited() -> int:
    for i, c in enumerate(ids):
        if store.outcome(c) == "unaudited":
            return i
    return 0


if not ids:
    st.success("No ambiguous cases — nothing to audit.")
    st.stop()

if "pos" not in st.session_state:
    st.session_state.pos = _first_unaudited()

# ------------------------------------------------------------------ sidebar --

with st.sidebar:
    st.header("Progress")
    summary = store.summary()
    counts = summary["counts"]
    st.metric("audited", f"{len(ids) - counts['unaudited']} / {len(ids)}")
    st.write({k: v for k, v in counts.items() if v})
    st.divider()
    st.caption("Cases")
    icon = {"unaudited": "⬜", "correct": "✅", "incorrect": "❌", "unresolved": "❓"}
    for i, c in enumerate(ids):
        if st.button(f"{icon[store.outcome(c)]} {c}", key=f"nav_{c}",
                     use_container_width=True):
            st.session_state.pos = i
            st.rerun()
    st.divider()
    dpi = st.slider("render DPI", 100, 400, 200, 25)
    st.caption("No model, no OCR, no network. "
               "The instructor score and the model's extracted letter are "
               "deliberately hidden.")

# --------------------------------------------------------------------- case --

pos = max(0, min(st.session_state.pos, len(ids) - 1))
case_id = ids[pos]
ctx = store.context(case_id)

st.title(f"{case_id}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("split", ctx["split"])
c2.metric("question", f"Q{ctx['question_id']}")
c3.metric("answer row", ctx["answer_table_row"])
c4.metric("max points", f"{ctx['max_points']:g}")

st.markdown(
    f"**Exam scan:** `{ctx['exam_file']}` &nbsp;·&nbsp; "
    f"**answer sheet page {ctx['answer_table_page']}** &nbsp;·&nbsp; "
    f"read **row {ctx['answer_table_row']}**",
    unsafe_allow_html=True)

if ctx["correct_options"]:
    st.markdown(
        f"Exam version **{ctx['audited_version']}** (operator-confirmed) — the key "
        f"accepts **{', '.join(ctx['correct_options'])}** for this row.")
else:
    st.markdown(
        '<div class="warn">This writer\'s exam version was never audited, so the '
        "accepted option is unknown. Record the letter the student marked; "
        "correctness cannot be settled until the version is confirmed — choose "
        "<b>unresolved</b> if you cannot determine correctness.</div>",
        unsafe_allow_html=True)

pdf = REPO_ROOT / (ctx["exam_file"] or "")
left, right = st.columns([3, 2])

with left:
    if not ctx["exam_file"] or not pdf.exists():
        st.error(f"scan not found: {ctx['exam_file']}")
    else:
        try:
            st.image(render_page(str(pdf), ctx["answer_table_page"], dpi),
                     use_container_width=True,
                     caption=f"{ctx['exam_file']} — page {ctx['answer_table_page']}")
        except Exception as e:  # noqa: BLE001 — a render failure must not lose state
            st.error(f"could not render page: {type(e).__name__}: {e}")
        with st.expander("show neighbouring pages"):
            for delta in (-1, 1):
                p = ctx["answer_table_page"] + delta
                if p < 1:
                    continue
                try:
                    st.image(render_page(str(pdf), p, dpi), use_container_width=True,
                             caption=f"page {p}")
                except Exception:  # noqa: BLE001
                    pass

with right:
    st.subheader("What did the student mark?")
    current = ctx["decision"]
    letter = st.text_input(
        "Option letter in the answer table (leave empty if blank/unclear)",
        value=(current.get("selected_option") or ""),
        max_chars=3, key=f"letter_{case_id}").strip().upper()
    if letter:
        st.markdown(f'<div class="big-letter">{letter}</div>', unsafe_allow_html=True)
    note = st.text_area("Note (optional)", value=current.get("note", ""),
                        key=f"note_{case_id}", height=80)

    st.divider()
    st.caption("Record the SELECTION, not the grade.")

    b1, b2 = st.columns(2)
    if b1.button("✅ selection CORRECT", use_container_width=True, type="primary"):
        store.record(case_id, outcome="correct", selected_option=letter, note=note)
        st.session_state.pos = min(pos + 1, len(ids) - 1)
        st.rerun()
    if b2.button("❌ selection INCORRECT", use_container_width=True):
        store.record(case_id, outcome="incorrect", selected_option=letter, note=note)
        st.session_state.pos = min(pos + 1, len(ids) - 1)
        st.rerun()
    if st.button("❓ genuinely UNRESOLVED (blank, ambiguous, unreadable)",
                 use_container_width=True):
        store.record(case_id, outcome="unresolved", selected_option=letter, note=note)
        st.session_state.pos = min(pos + 1, len(ids) - 1)
        st.rerun()

    if store.outcome(case_id) != "unaudited":
        st.info(f"recorded: **{store.outcome(case_id)}**"
                + (f" · marked `{current.get('selected_option')}`"
                   if current.get("selected_option") else ""))
        if st.button("clear this decision", use_container_width=True):
            store.clear(case_id)
            st.rerun()

    st.divider()
    nav1, nav2 = st.columns(2)
    if nav1.button("← previous", use_container_width=True, disabled=pos == 0):
        st.session_state.pos = pos - 1
        st.rerun()
    if nav2.button("next →", use_container_width=True, disabled=pos == len(ids) - 1):
        st.session_state.pos = pos + 1
        st.rerun()

st.divider()
if store.complete():
    st.success("All ambiguous cases audited. Next: "
               "`python scripts/selaudit.py freeze`")
else:
    st.caption(f"{len(store.remaining())} case(s) left: "
               f"{', '.join(store.remaining())}")
