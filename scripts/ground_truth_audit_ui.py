"""Ground-truth audit UI — BLINDED. One tool, one set of guards, per-queue files.

Queues (pick with ``--audit-file PATH`` after ``--``, or the ``AUDIT_FILE``
environment variable; default is the DEV queue):

- GROUND_TRUTH_AUDIT_2026-08-25.json — 5 DEV cases where every model scored low
- CALIBRATION_AUDIT_2026-08-26.json — 6 CALIBRATION cases from the v3-vs-v4 A/B
  (four both-model downgrades, two single-model upgrades; which is which is
  hidden until the decision is saved)

Launch (DEV queue):

    .venv\\Scripts\\python.exe -m streamlit run scripts\\ground_truth_audit_ui.py -- --browser.gatherUsageStats false

Launch (CALIBRATION queue):

    .venv\\Scripts\\python.exe -m streamlit run scripts\\ground_truth_audit_ui.py --browser.gatherUsageStats false -- --audit-file evaluation\\model_selection\\runs\\grade_primary\\CALIBRATION_AUDIT_2026-08-26.json

Runs entirely on this machine: local JSON state, local images, PyMuPDF page
rendering with the instructor's red ink masked. No network, no provider, no
labels.db.

All five carry instructor 4/4, so the derived explanation verdict is `valid`.
All three models independently judged the explanation weaker than that. Two
readings fit that evidence equally well:

    A  derived verdict consistent with rubric/instructor practice
       (the models are simply too harsh)
    B  the instructor appears more lenient than the encoded rubric
       (full credit for a correct choice, generous about the explanation)
    C  transcription/evidence issue
    D  genuinely ambiguous

WHY THIS IS BLINDED
-------------------
Three models agreeing is not evidence about the label — it can equally mean
three models are wrong in the same direction. Shown first, it anchors: it is
very hard to read a transcription as "sufficient" once three graders have just
called it thin. So the model outputs are withheld until the human decision for
that case is saved, and only then revealed as post-decision context.

The blinding is enforced by the CODE PATH, not by layout: ``view_payload``
returns the pre-decision payload built from an explicit allow-list, and the UI
renders only what that function hands it. Nothing model-derived is passed to
Streamlit before a decision exists, so it cannot leak through a collapsed
expander, a hidden element, or the page source.

A decision does NOT become editable just because the models are now visible.
Changing it takes an explicit re-adjudication, which is recorded — a judgement
revised AFTER seeing model output is weaker evidence than a blind one, and the
file has to say which it was.

NOTHING is relabelled here. No model, no OCR, no network.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
_RUNS = REPO / "evaluation" / "model_selection" / "runs" / "grade_primary"
AUDIT = _RUNS / "GROUND_TRUTH_AUDIT_2026-08-25.json"


def resolve_audit_path(argv: list[str] | None = None, env: dict | None = None) -> Path:
    """Which audit queue to open.

    One blinded UI serves every queue: the blinding rules are a property of the
    tool, not of a particular file, and duplicating the tool per queue is how
    one copy quietly loses a guard.

    Order: ``--audit-file PATH``, then ``AUDIT_FILE`` in the environment, else
    the default queue.
    """
    import os
    import sys

    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    if "--audit-file" in argv:
        i = argv.index("--audit-file")
        if i + 1 < len(argv):
            return Path(argv[i + 1])
    if env.get("AUDIT_FILE"):
        return Path(env["AUDIT_FILE"])
    return AUDIT

#: The ONLY fields a pre-decision screen may carry. Allow-list, not deny-list:
#: a new model-derived field added to the audit file later is excluded by
#: default instead of silently leaking.
BLINDED_FIELDS: tuple[str, ...] = (
    "case_id", "writer", "question_id", "sub_item_id", "question_text",
    "rubric", "official_solution", "max_score", "frozen_transcription",
    "instructor_final_score", "derived_explanation_verdict", "derivation_reason",
    # legitimate human grading material, not model output: the red-ink-cleaned
    # answer crop and the (masked-on-render) source page reference
    "answer_crop", "source_page",
)

#: Fields that exist only because a model produced them.
MODEL_DERIVED_FIELDS: tuple[str, ...] = ("model_predictions",)


# --------------------------------------------------------------- pure core ---

def is_decided(case: dict) -> bool:
    return bool(case.get("human_decision"))


def pre_decision_payload(case: dict) -> dict[str, Any]:
    """Everything the auditor may see BEFORE deciding — and nothing else."""
    return {k: case.get(k) for k in BLINDED_FIELDS}


def post_decision_payload(case: dict) -> dict[str, Any]:
    """Pre-decision material plus the model outputs, revealed as context."""
    out = pre_decision_payload(case)
    out["model_predictions"] = case.get("model_predictions") or {}
    return out


def view_payload(case: dict) -> dict[str, Any]:
    """What the UI is allowed to render for this case, right now."""
    payload = post_decision_payload(case) if is_decided(case) else pre_decision_payload(case)
    payload["_blinded"] = not is_decided(case)
    return payload


def record_decision(case: dict, option: str, *, note: str = "", auditor: str = "owner",
                    now: str | None = None, manifest_hash: str | None = None) -> dict:
    """Save a decision. Records whether it was made blind.

    A decided case REFUSES a new decision: overwriting takes an explicit
    ``reset_decision`` first, so no saved judgement can be replaced silently.
    Each entry pins the exact pre-decision payload it was made from
    (``payload_sha256``) and, when given, the audit manifest's content hash.
    """
    if is_decided(case):
        raise ValueError(
            f"{case.get('case_id')}: already decided ({case['human_decision']!r}); "
            "reset explicitly before re-adjudicating")
    blind = not case.get("models_revealed_at")
    payload_sha = hashlib.sha256(
        json.dumps(pre_decision_payload(case), ensure_ascii=False, sort_keys=True)
        .encode("utf-8")).hexdigest()
    revision = 1 + sum(1 for e in case.get("decision_history", [])
                       if e.get("action") == "decide")
    entry = {
        "at": now or time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": "decide",
        "decision": option, "note": note, "auditor": auditor,
        "made_blind": blind,
        "payload_sha256": payload_sha,
        "revision": revision,
    }
    if manifest_hash:
        entry["manifest_content_hash"] = manifest_hash
    case["human_decision"] = option
    case["human_note"] = note
    case["decided_at"] = entry["at"]
    case["decided_by"] = auditor
    case["decided_blind"] = blind
    case["decision_revision"] = revision
    case.setdefault("decision_history", []).append(entry)
    if blind:
        # the reveal happens because the decision was saved, never before it
        case["models_revealed_at"] = entry["at"]
    return entry


def reset_decision(case: dict, *, reason: str = "", auditor: str = "owner",
                   now: str | None = None) -> dict:
    """Explicit re-adjudication. The reveal is NOT undone — the auditor has
    already seen the model outputs — so the history records that any new
    decision is a sighted one."""
    entry = {
        "at": now or time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": "reset", "previous_decision": case.get("human_decision"),
        "reason": reason, "auditor": auditor,
        "models_already_revealed": bool(case.get("models_revealed_at")),
    }
    case["human_decision"] = None
    case["decided_at"] = None
    case["decided_by"] = None
    case["decided_blind"] = False
    case.setdefault("decision_history", []).append(entry)
    return entry


def load_audit(path: Path = AUDIT) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_audit(doc: dict, path: Path = AUDIT) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ------------------------------------------------------------------- the UI ---

def main() -> None:  # pragma: no cover - exercised by launching streamlit
    import streamlit as st

    st.set_page_config(page_title="Blinded audit", page_icon="⚖", layout="wide")
    st.markdown("""
    <style>
    .rtl { direction: rtl; text-align: right; font-size: 1.15rem; line-height: 1.9;
           padding: .6rem .8rem; border-radius: .5rem;
           border: 1px solid rgba(128,128,128,.35); white-space: pre-wrap; }
    .warn { border-left:3px solid #d08a00; padding:.4rem .7rem; border-radius:.3rem;
            background:rgba(208,138,0,.08); }
    .blind { border-left:3px solid #4a7; padding:.4rem .7rem; border-radius:.3rem;
             background:rgba(68,170,119,.08); }
    </style>
    """, unsafe_allow_html=True)

    audit_path = resolve_audit_path()
    if not audit_path.exists():
        st.error(f"audit file not found: {audit_path}")
        st.stop()

    doc = load_audit(audit_path)             # fresh read every rerun
    cases = doc["cases"]
    OPTIONS = doc["options"]

    with st.sidebar:
        st.header("Progress")
        done = sum(1 for c in cases if is_decided(c))
        st.metric("decided", f"{done} / {len(cases)}")
        by_opt = {k: sum(1 for c in cases if c.get("human_decision") == k) for k in OPTIONS}
        resets = sum(1 for c in cases
                     for e in c.get("decision_history", []) if e.get("action") == "reset")
        st.caption("  ·  ".join(f"{k}: {n}" for k, n in by_opt.items())
                   + f"  ·  remaining: {len(cases) - done}"
                   + (f"  ·  re-adjudications: {resets}" if resets else ""))
        if by_opt.get("C"):
            st.caption(f"⚠ {by_opt['C']} case(s) need evidence repair (C)")
        if by_opt.get("D"):
            st.caption(f"◌ {by_opt['D']} case(s) marked ambiguous (D)")
        for i, c in enumerate(cases):
            mark = c.get("human_decision") or "·"
            if st.button(f"{mark}  {c['case_id']}", key=f"nav{i}", use_container_width=True):
                st.session_state.pos = i
                st.rerun()
        st.divider()
        st.caption("Blinded: model outputs appear only after you save a decision. "
                   "No model, no OCR, no network. Nothing is relabelled here.")

    if "pos" not in st.session_state:
        st.session_state.pos = next((i for i, c in enumerate(cases) if not is_decided(c)), 0)
    pos = max(0, min(st.session_state.pos, len(cases) - 1))
    case = cases[pos]
    view = view_payload(case)                # <- the only source for rendering

    st.caption(audit_path.name)
    st.title(view["case_id"])
    a, b, d = st.columns(3)
    a.metric("writer", view["writer"])
    b.metric("question", f"Q{view['question_id']} · item {view['sub_item_id']}")
    d.metric("instructor score", f"{view['instructor_final_score']:g} / {view['max_score']:g}")

    if view["_blinded"]:
        st.markdown('<div class="blind">Model outputs are hidden until you save a decision '
                    "for this case. Read the explanation against the rubric and judge it on "
                    "its own.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn">{doc["warning"]}</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Student explanation (frozen transcription)")
        st.markdown(f'<div class="rtl">{view["frozen_transcription"]}</div>',
                    unsafe_allow_html=True)
        if view.get("answer_crop"):
            crop = REPO / view["answer_crop"]
            if crop.exists():
                st.caption("answer crop (red ink already removed)")
                st.image(str(crop), width="stretch")
            else:
                st.warning(f"answer crop missing from this checkout: {view['answer_crop']}. "
                           "Judge from the frozen transcription above.")
        sp = view.get("source_page") or {}
        if sp.get("pdf") and sp.get("page"):
            with st.expander("full source page (instructor's red ink masked)"):
                pdf = REPO / sp["pdf"]
                if not pdf.exists():
                    st.warning(f"source PDF missing from this checkout: {sp['pdf']}")
                else:
                    try:
                        import sys as _sys
                        if str(REPO) not in _sys.path:
                            _sys.path.insert(0, str(REPO))
                        from labeling_app.bundle import render_masked_page
                        png, rep = render_masked_page(pdf, int(sp["page"]), max_edge=1600)
                        if not rep["ok"]:
                            st.error(f"masking left {rep['strict_red_after']} strict-red "
                                     "pixels — page withheld")
                        else:
                            st.image(png, width="stretch")
                            st.caption(f"{sp['pdf']} p.{sp['page']} — rendered locally, "
                                       f"{rep['masked_pixels']} red pixels whitened, "
                                       f"residual {rep['strict_red_after']}")
                    except Exception as e:  # noqa: BLE001
                        st.warning(f"could not render the page: {type(e).__name__}: {e}")
        st.subheader("Rubric")
        for r in view["rubric"]:
            st.markdown(f'**{r["id"]}** <div class="rtl">{r["text"]}</div>',
                        unsafe_allow_html=True)
        st.subheader("Official solution")
        for k, v in (view["official_solution"] or {}).items():
            st.markdown(f'**[{k}]** <div class="rtl">{v}</div>', unsafe_allow_html=True)
        with st.expander("question text"):
            st.markdown(f'<div class="rtl">{view["question_text"]}</div>',
                        unsafe_allow_html=True)

    with right:
        st.subheader("Derived ground truth")
        st.info(f"**{view['derived_explanation_verdict']}**  ·  {view['derivation_reason']}")

        if view["_blinded"]:
            st.subheader("Your decision")
            note = st.text_area("Note (optional)", value=case.get("human_note", ""),
                                key=f"note{pos}")
            for key, label in OPTIONS.items():
                if st.button(f"{key} — {label}", key=f"opt{key}_{pos}",
                             use_container_width=True):
                    record_decision(case, key, note=note,
                                    manifest_hash=doc.get("content_hash"))
                    save_audit(doc, audit_path)
                    st.rerun()
        else:
            st.success(f"recorded: **{case['human_decision']}** — "
                       f"{OPTIONS[case['human_decision']]}"
                       + ("  (decided blind)" if case.get("decided_blind") else
                          "  (decided after reveal)"))
            if case.get("human_note"):
                st.caption(case["human_note"])
            st.subheader("Model predictions — post-decision context")
            for model, p in (view["model_predictions"] or {}).items():
                st.markdown(f"**{model}** — `{p['verdict']}` (raw {p['raw_score']:g})"
                            + ("  ⚠ uncertain" if p["uncertain"] else ""))
                st.caption(p["justification"] or "_(no justification returned)_")
            st.divider()
            with st.expander("re-adjudicate (explicit)"):
                st.caption("Your decision is locked. Reopening it after seeing the model "
                           "outputs is recorded as a sighted judgement.")
                reason = st.text_input("reason for re-adjudication", key=f"rsn{pos}")
                if st.button("reset this decision", key=f"reset{pos}",
                             use_container_width=True, disabled=not reason.strip()):
                    reset_decision(case, reason=reason.strip())
                    save_audit(doc, audit_path)
                    st.rerun()

    st.divider()
    n1, n2 = st.columns(2)
    if n1.button("← previous", use_container_width=True, disabled=pos == 0):
        st.session_state.pos = pos - 1
        st.rerun()
    if n2.button("next →", use_container_width=True, disabled=pos == len(cases) - 1):
        st.session_state.pos = pos + 1
        st.rerun()


if __name__ == "__main__":  # pragma: no cover - streamlit executes this as __main__
    main()
