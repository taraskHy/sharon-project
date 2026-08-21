"""Manual reference-audit UI for evaluation/hebrew_bench_v2 (129 items).

Launch:  .venv\\Scripts\\python.exe -m streamlit run scripts\\reference_audit_ui.py -- --browser.gatherUsageStats false

(the flag keeps Streamlit's usage telemetry off, matching the repo's
no-network convention; `autograder ui` does the same)

Developer/benchmark tooling. The human auditor confirms/corrects/marks each
frozen handwriting reference against its crop image. No OCR, no AI
suggestions, no network. Every decision persists ATOMICALLY the moment a
button is pressed (scripts/refaudit.py), so closing or rerunning the app
never loses work — reopening resumes at the first unchecked item.
"""

from __future__ import annotations

import html
import importlib.util
import json
import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("refaudit", _HERE / "refaudit.py")
refaudit = importlib.util.module_from_spec(spec)
sys.modules.setdefault("refaudit", refaudit)
spec.loader.exec_module(refaudit)

st.set_page_config(page_title="Reference audit — hebrew_bench_v2",
                   page_icon="🖊", layout="wide")

# Hebrew is RTL; theme-aware (inherits Streamlit light/dark), no fixed colors.
st.markdown("""
<style>
.ref-text { direction: rtl; text-align: right; font-size: 1.15rem;
            line-height: 1.9; padding: .6rem .8rem; border-radius: .5rem;
            border: 1px solid rgba(128,128,128,.35);
            white-space: pre-wrap; }  /* multi-line references keep their lines */
textarea { direction: rtl !important; text-align: right !important;
           font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)


# A FRESH store every rerun: it re-reads the audit file from disk, so the UI
# can never hold (and later write back) a stale in-memory copy — external
# edits, restarts, and multiple sessions all see current state. record()
# additionally merges into the latest on-disk state under a lock.
store = refaudit.AuditStore(refaudit.bench_dir_from_env())
ids = store.item_ids
STATUS_ICON = {"unchecked": "⬜", "confirmed": "✅", "corrected": "✏️",
               "ambiguous": "❓"}


def _first_unchecked() -> int:
    for idx, item_id in enumerate(ids):
        if store.status(item_id) == "unchecked":
            return idx
    return 0


if "pos" not in st.session_state:
    st.session_state.pos = _first_unchecked()

# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.title("Reference audit")
    summary = store.summary()
    st.progress(summary["checked"] / summary["total"] if summary["total"] else 0.0,
                text=f"{summary['checked']} / {summary['total']} checked")
    c1, c2 = st.columns(2)
    c1.metric("Confirmed", summary["confirmed"])
    c2.metric("Corrected", summary["corrected"])
    c1.metric("Ambiguous", summary["ambiguous"])
    c2.metric("Remaining", summary["remaining"])

    st.divider()
    flt = st.selectbox("Filter", ["all", "unchecked", "confirmed",
                                  "corrected", "ambiguous"], key="filter")
    view_ids = [i for i in ids if flt == "all" or store.status(i) == flt]
    if not view_ids:
        st.info("No items match this filter.")
        view_ids = ids

    jump = st.selectbox(
        "Jump to item", view_ids, key="jump",
        format_func=lambda i: f"{STATUS_ICON[store.status(i)]} {i}")
    if st.button("Go", key="go", use_container_width=True):
        st.session_state.pos = ids.index(jump)
        st.rerun()
    if st.button("First unchecked", key="first_unchecked", use_container_width=True):
        st.session_state.pos = _first_unchecked()
        st.rerun()

    st.divider()
    if st.button("💾 Save now", key="save_now", use_container_width=True,
                 help="Every decision already saves atomically; this re-persists the state"):
        store.save()
        st.toast("Audit state saved.")
    st.caption(f"Audit file (saved on every decision):\n`{store.audit_path}`")
    st.download_button("Export audit JSON",
                       data=json.dumps({"summary": summary,
                                        "entries": store.entries_canonical()},
                                       ensure_ascii=False, indent=1),
                       file_name="reference_audit_export.json",
                       mime="application/json", key="export",
                       use_container_width=True)

    with st.expander("Danger zone"):
        st.caption("Global reset wipes every decision. Type RESET to enable.")
        confirm_text = st.text_input("Type RESET to confirm", key="reset_confirm")
        if st.button("Reset ALL decisions", key="reset_all",
                     disabled=(confirm_text != "RESET")):
            store.reset_all(confirm="RESET")
            for key in [k for k in st.session_state
                        if k.startswith(("text_", "note_"))]:
                del st.session_state[key]   # editors must show originals again
            st.session_state.pos = 0
            st.rerun()

# --------------------------------------------------------------- main panel --
pos = max(0, min(st.session_state.pos, len(ids) - 1))
item_id = ids[pos]
item = store.item(item_id)
entry = store.entry(item_id)
original = entry["original_reference"]

nav_prev, head, nav_next = st.columns([1, 6, 1])
if nav_prev.button("← Prev", key="prev", disabled=pos == 0, use_container_width=True):
    st.session_state.pos = pos - 1
    st.rerun()
if nav_next.button("Next →", key="next", disabled=pos == len(ids) - 1,
                   use_container_width=True):
    st.session_state.pos = pos + 1
    st.rerun()
head.subheader(f"{STATUS_ICON[entry['status']]} {item_id}")
head.caption(
    f"item {pos + 1} of {len(ids)}  ·  category **{item['category']}**  ·  "
    f"tier {item.get('tier', '?')}  ·  writer {item.get('writer', '?')}"
    + ("  ·  **hard**" if item.get("hard") else ""))

img_col, txt_col = st.columns([1, 1], gap="large")
with img_col:
    crop_path = store.bench_dir / item["image"]
    if crop_path.exists():
        st.image(str(crop_path), use_container_width=True)
    else:
        st.error(f"crop image missing: {crop_path}")
    if entry["status"] != "unchecked":
        st.caption(f"decided {entry['audited_at']}"
                   + (f" — note: {entry['note']}" if entry["note"] else ""))

with txt_col:
    st.markdown("**Original reference** (frozen, never modified)")
    st.markdown(f'<div class="ref-text">{html.escape(original)}</div>',
                unsafe_allow_html=True)
    default_text = (entry["audited_reference"]
                    if entry["audited_reference"] is not None else original)
    audited_text = st.text_area("Audited transcription", value=default_text,
                                key=f"text_{item_id}", height=110)
    note = st.text_input("Note (optional)", value=entry["note"] or "",
                         key=f"note_{item_id}")

    edited = audited_text != original
    b_confirm, b_correct, b_ambiguous = st.columns(3)
    if b_confirm.button("✅ CONFIRM", key="confirm", use_container_width=True,
                        help="The original reference is exactly right"):
        if edited:
            st.warning("The text differs from the original — use CORRECT, or "
                       "restore the text before confirming.")
        else:
            store.record(item_id, "confirmed", note=note)
            st.session_state.pos = min(pos + 1, len(ids) - 1)
            st.rerun()
    if b_correct.button("✏️ CORRECT", key="correct", use_container_width=True,
                        help="Save the edited transcription as the audited reference"):
        if not audited_text.strip():
            st.warning("The audited transcription is empty — every benchmark "
                       "crop contains text. Type the correction, or use "
                       "AMBIGUOUS if it is genuinely unreadable.")
        else:
            store.record(item_id, "corrected", audited_text=audited_text, note=note)
            st.session_state.pos = min(pos + 1, len(ids) - 1)
            st.rerun()
    if b_ambiguous.button("❓ AMBIGUOUS", key="ambiguous", use_container_width=True,
                          help="Genuinely unreadable/undecidable — excluded from strict CER"):
        store.record(item_id, "ambiguous", audited_text=audited_text, note=note)
        st.session_state.pos = min(pos + 1, len(ids) - 1)
        st.rerun()

    if entry["status"] != "unchecked":
        with st.expander("Reset this item"):
            st.caption("Returns THIS item to unchecked. Other decisions are untouched.")
            if st.button("Reset item", key="reset_item"):
                store.reset_item(item_id)
                # Clear the widget state too: the editor must show the frozen
                # original again, or a later CORRECT/AMBIGUOUS click would
                # silently re-record the discarded text.
                st.session_state.pop(f"text_{item_id}", None)
                st.session_state.pop(f"note_{item_id}", None)
                st.rerun()
