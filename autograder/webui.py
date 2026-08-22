"""Local web interface for the exam autograder (Streamlit).

Start with:  python -m autograder ui        (documented in the README)

Screens (sidebar navigation):

    Dashboard            what is loaded, how far grading is, what needs you
    Exam setup           course -> exam -> key/rubric -> discovery -> policies -> preflight -> students
    Grading progress     live batch status, pause/stop, calls, cache, cost, estimate
    Review queue         every review with its TYPED reason and the evidence we have
    Results / export     final deterministic grades, per-question scores, history, downloads
    Advanced / diagnostics   role -> model (UNSELECTED marked), budget, ledgers, key status, readiness

Design notes:

- Grading runs in a DETACHED subprocess (``autograder run-job``), so closing
  this app never kills a batch; reopening shows the persisted job state.
- All state lives on disk in the job directory (see autograder/jobs.py);
  the UI only reads/writes those files and spawns/controls the runner.
- Model-visible inputs never contain original filenames or private paths —
  intake anonymizes exam copies; original names appear only in the UI tables.
- The OpenRouter credential is read from the environment by the backend and
  is NEVER displayed, stored, or logged here — only its presence is shown.
- Spend truth comes from the persistent usage ledgers (the batch's own
  ledger under <job>/exams/gateway_ledger and the model-selection campaign
  ledger), never from in-memory counters.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from autograder import jobs  # noqa: E402
from autograder.template import ExamTemplate  # noqa: E402

st.set_page_config(page_title="Exam Autograder", page_icon="📝", layout="wide")


from autograder.reviewui import package_dirs  # noqa: E402  (configurable discovery roots)

SCREEN_DASHBOARD = "🏠 Dashboard"
SCREEN_SETUP = "🧭 Exam setup"
SCREEN_PROGRESS = "⏳ Grading progress"
SCREEN_REVIEW = "🔍 Review queue"
SCREEN_RESULTS = "📄 Results / export"
SCREEN_ADVANCED = "🛠 Advanced / diagnostics"
SCREENS = [SCREEN_DASHBOARD, SCREEN_SETUP, SCREEN_PROGRESS, SCREEN_REVIEW, SCREEN_RESULTS, SCREEN_ADVANCED]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def runner_alive(job_dir: Path) -> bool:
    pid_file = job_dir / "runner.pid"
    if not pid_file.exists():
        return False
    try:
        return _pid_alive(int(pid_file.read_text().strip()))
    except (ValueError, OSError):
        return False


def spawn_runner(job_dir: Path) -> None:
    creationflags = 0
    kwargs: dict = {}
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        [sys.executable, "-m", "autograder", "run-job", "--job-dir", str(job_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        cwd=str(REPO_ROOT),
        **kwargs,
    )
    (job_dir / "runner.pid").write_text(str(proc.pid), encoding="utf-8")


def discover_packages() -> dict[str, dict]:
    """Configured exam packages = a key file with a .template.json sidecar."""
    packages: dict[str, dict] = {}
    for d in package_dirs():
        if not d.is_dir():
            continue
        for tpl_path in sorted(d.glob("*.template.json")):
            stem = tpl_path.name[: -len(".template.json")]
            for key_name in (f"{stem}.json", f"{stem}.pdf"):
                key_path = d / key_name
                if key_path.exists():
                    try:
                        tpl = ExamTemplate.model_validate_json(
                            tpl_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        continue
                    packages[f"{tpl.name or tpl.template_id} ({d.name})"] = {
                        "key": key_path,
                        "template": tpl_path,
                        "template_obj": tpl,
                    }
                    break
    return packages


def list_jobs() -> list[Path]:
    root = jobs.jobs_root()
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if (p / "job.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )


def fmt_elapsed(state: dict) -> str:
    start = state.get("started_at")
    if not start:
        return "—"
    t0 = datetime.fromisoformat(start)
    end = state.get("finished_at")
    t1 = datetime.fromisoformat(end) if end else datetime.now()
    seconds = int((t1 - t0).total_seconds())
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _stage_dir(uploaded, name: str) -> Path | None:
    """Persist an uploaded file into the session staging directory."""
    if uploaded is None:
        return None
    staging = Path(st.session_state.setdefault(
        "staging_dir",
        str(jobs.jobs_root() / f"_staging-{datetime.now():%Y%m%d-%H%M%S}"),
    ))
    staging.mkdir(parents=True, exist_ok=True)
    target = staging / f"{name}{Path(uploaded.name).suffix.lower()}"
    target.write_bytes(uploaded.getvalue())
    return target


def _job_ledger_path(job_dir: Path) -> Path:
    """The batch's OWN persistent usage ledger (written by every grade
    subprocess: orchestrator.setup_from_config(models.toml, <job>/exams))."""
    return job_dir / "exams" / "gateway_ledger" / "usage.jsonl"


def _models_toml() -> Path | None:
    p = REPO_ROOT / "models.toml"
    return p if p.exists() else None


def _job_overview(job_dir: Path) -> dict:
    """Counts for the dashboard/progress cards — tolerant of partial state."""
    job = jobs.load_job(job_dir)
    state = jobs.load_state(job_dir)
    exams = state.get("exams", {}) or {}
    counts = {s: sum(1 for e in exams.values() if e.get("status") == s)
              for s in ("pending", "running", "done", "failed")}
    review = sum(1 for e in exams.values() if (e.get("review_items") or 0) > 0 and e.get("status") == "done")
    auto = sum(1 for e in exams.values() if e.get("status") == "done" and not (e.get("review_items") or 0))
    total = len(exams)
    return {"job": job, "state": state, "exams": exams, "counts": counts, "review": review,
            "auto": auto, "total": total, "progress": (counts["done"] / total) if total else 0.0,
            "alive": runner_alive(job_dir), "current": state.get("current")}


def _package_status(job_dir: Path) -> dict:
    """Preflight over the batch's parsed key (uploads/answer_key.json)."""
    pk = job_dir / "uploads" / "answer_key.json"
    if not pk.exists():
        return {"status": "KEY_NOT_PARSED", "summary": "appears once the answer key has been parsed",
                "report": None}
    try:
        from autograder.key_parser import load_answer_key
        from autograder.preflight import alignment_from_discovery, preflight_package

        k = load_answer_key(pk)
        al_path = job_dir / "uploads" / "answer_key.alignment.json"
        al = json.loads(al_path.read_text(encoding="utf-8")) if al_path.exists() else None
        rep = preflight_package(key=k, variants=list(k.versions),
                                alignment=alignment_from_discovery(al, list(k.versions), k))
        return {"status": rep.status, "summary": rep.summary(), "report": rep, "key": k}
    except Exception as exc:  # noqa: BLE001 — never block the UI on a check
        return {"status": "UNKNOWN", "summary": f"package check unavailable: {type(exc).__name__}", "report": None}


def _results_and_extractions(job_dir: Path) -> tuple[dict, dict]:
    from autograder import reviewui as _rui
    results = _rui.load_job_results(job_dir)
    extractions = {}
    for eid in results:
        ep = job_dir / "exams" / eid / "extraction.json"
        if ep.exists():
            try:
                extractions[eid] = json.loads(ep.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    return results, extractions


def _job_packs(job_dir: Path) -> dict:
    """Grading packs persisted by the batch (rubric items / question context
    for the review screen). Empty when the batch ran in legacy mode."""
    root = job_dir / "exams" / "packs"
    if not root.is_dir():
        return {}
    try:
        from autograder.gradingpack import PackStore
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            packs = PackStore(d).load(d.name)
            if packs:
                return packs
    except Exception:  # noqa: BLE001 — packs are a convenience for display
        pass
    return {}


def _spend(job_dir: Path | None) -> dict:
    """Spend truth: the batch ledger (when any) + the campaign ledger."""
    from autograder.benchmark.runner import DEFAULT_STATE_ROOT
    from autograder.spend import spend_view
    out = {"job": None, "campaign": None}
    if job_dir is not None:
        p = _job_ledger_path(job_dir)
        out["job"] = spend_view(p) if p.exists() else None
    camp = DEFAULT_STATE_ROOT / "gateway_ledger" / "usage.jsonl"
    out["campaign"] = spend_view(camp) if camp.exists() else None
    return out


def _nav(target: str) -> None:
    st.session_state["nav_target"] = target
    st.rerun()


def _reason_title(code: str) -> str:
    from autograder.reviewqueue import REASONS
    spec = REASONS.get(code)
    return spec.title if spec is not None else code


# --------------------------------------------------------------------------
# sidebar: backend configuration + grading route + batch + navigation
# --------------------------------------------------------------------------

st.sidebar.title("📝 Exam Autograder")
st.sidebar.caption("Local open-model grading — cloud roles only when you configure them.")

_toml_defaults = {}
_toml_path = REPO_ROOT / "grader.toml"
if _toml_path.exists():
    import tomllib

    _toml_defaults = tomllib.loads(_toml_path.read_text(encoding="utf-8"))

_b = _toml_defaults.get("backend", {})
_g = _toml_defaults.get("grading", {})

# navigation target requested by a button on the previous run (set BEFORE
# the radio is instantiated, which is the supported way to change it)
if "nav_target" in st.session_state:
    st.session_state["screen"] = st.session_state.pop("nav_target")

with st.sidebar:
    screen = st.radio("Screen", SCREENS, key="screen", label_visibility="collapsed")
    st.divider()
    _all_jobs = list_jobs()
    _names = [p.name for p in _all_jobs]
    selected_job_name = None
    if _names:
        _default = st.session_state.get("selected_job")
        _idx = _names.index(_default) if _default in _names else 0
        selected_job_name = st.selectbox("Current batch", _names, index=_idx, key="job_select")
    else:
        st.caption("No batches yet — create one in Exam setup.")

    with st.expander("Model backend (local)", expanded=False):
        base_url = st.text_input("Server URL", _b.get("base_url", "http://localhost:11434/v1"))
        model = st.text_input("Model", _b.get("model", "qwen3-vl:8b-instruct"))
        structured_mode = st.selectbox(
            "Structured output", ["json_schema", "json_object", "prompt"],
            index=["json_schema", "json_object", "prompt"].index(
                _b.get("structured_mode", "json_schema")
            ),
        )
        timeout_s = st.number_input("Timeout (s)", 30.0, 7200.0,
                                    float(_b.get("timeout_s", 1800.0)), step=30.0)
        max_tokens = st.number_input("Max output tokens", 500, 32000,
                                     int(_g.get("max_tokens", 8000)), step=500)
        max_image_edge = st.number_input("Extraction image edge (px)", 640, 2560,
                                         int(_g.get("max_image_edge", 1400)), step=100)
        survey_image_edge = st.number_input("Survey image edge (px)", 320, 1600,
                                            int(_g.get("survey_image_edge", 640)), step=64)
        if st.button("Check backend"):
            from autograder.backends import BackendConfig, create_backend

            try:
                report = create_backend(BackendConfig(
                    backend="openai", model=model, base_url=base_url,
                    structured_mode=structured_mode, timeout_s=float(timeout_s),
                )).health_check()
                (st.success if report.ok else st.error)(f"{report.detail}")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

backend_args = {
    "--backend": "openai",
    "--base-url": base_url,
    "--model": model,
    "--structured-mode": structured_mode,
    "--timeout": timeout_s,
}
grading_args = {
    "--max-tokens": int(max_tokens),
    "--max-image-edge": int(max_image_edge),
    "--survey-image-edge": int(survey_image_edge),
}

# ---------------------------------------------------------------------------
# sidebar: grading route — the production mode/config actually reaches jobs.
# The flags land in job.json (grading_args) and are forwarded verbatim to
# every `autograder grade` subprocess. No key/secrets are involved here.
# ---------------------------------------------------------------------------
with st.sidebar:
    with st.expander("Grading route", expanded=False):
        _models_toml_path = REPO_ROOT / "models.toml"
        if _models_toml_path.exists():
            grading_mode = st.selectbox(
                "Grading mode", ["legacy", "reliability", "shadow"], index=0,
                help=(
                    "legacy = the validated judge path, unchanged. "
                    "reliability = the evidence/invariant/escalation route through "
                    "the task gateway (models.toml). shadow = both run; the legacy "
                    "grade stays authoritative and the reliability route is "
                    "recorded only (shadow_comparison.json)."
                ),
            )
        else:
            grading_mode = "legacy"
            st.caption(
                "Grading mode: legacy. Create models.toml (copy "
                "models.example.toml) to enable the reliability/shadow routes."
            )
        rag_policy = "RAG_DISABLED"
        if grading_mode != "legacy":
            rag_policy = st.selectbox(
                "Grading RAG policy",
                ["RAG_DISABLED", "RAG_ALWAYS", "RAG_ON_UNCERTAIN", "RAG_ON_ESCALATION"],
                index=0,
                help=(
                    "Course-context policy for the grader. Default RAG_DISABLED — "
                    "the benefit is unmeasured and no unmeasured optional context "
                    "is sent silently. Retrieval itself is always local."
                ),
            )

if grading_mode != "legacy":
    grading_args["--grading-mode"] = grading_mode
    grading_args["--models-config"] = str(_models_toml_path)
    grading_args["--rag-policy"] = rag_policy

job_dir: Path | None = (jobs.jobs_root() / selected_job_name) if selected_job_name else None


# ==========================================================================
# 1. DASHBOARD
# ==========================================================================
if screen == SCREEN_DASHBOARD:
    st.header("Dashboard")
    from autograder.cloudcheck import openrouter_credential_present

    if job_dir is None:
        st.info("No grading batch yet. Start with **Exam setup**: choose the course and exam, "
                "add the answer key and the students' scans, then start grading.")
        _pk = discover_packages()
        try:
            from autograder import courses as _cs
            _nc = len(_cs.list_courses())
        except Exception:  # noqa: BLE001
            _nc = 0
        c = st.columns(4)
        c[0].metric("Courses", _nc)
        c[1].metric("Exam packages", len(_pk))
        c[2].metric("OpenRouter", "configured" if openrouter_credential_present() else "not configured")
        c[3].metric("Grading mode", grading_mode)
        if st.button("➕ Set up exam", type="primary"):
            _nav(SCREEN_SETUP)
    else:
        ov = _job_overview(job_dir)
        job, state = ov["job"], ov["state"]
        pkg = _package_status(job_dir)
        tpl_path = job_dir / "uploads" / "answer_key.template.json"
        exam_name = job.get("key") or "answer key"
        if tpl_path.exists():
            try:
                _t = json.loads(tpl_path.read_text(encoding="utf-8"))
                exam_name = _t.get("name") or _t.get("template_id") or exam_name
            except Exception:  # noqa: BLE001
                pass
        top = st.columns(4)
        top[0].metric("Course", job.get("course_id") or "—")
        top[1].metric("Exam / package", str(exam_name)[:40])
        top[2].metric("Package readiness", pkg["status"])
        top[3].metric("Batch", job_dir.name)
        st.caption(pkg["summary"])

        m = st.columns(6)
        m[0].metric("Students", ov["total"])
        m[1].metric("Auto graded", ov["auto"])
        m[2].metric("Needs review", ov["review"])
        m[3].metric("Failures", ov["counts"]["failed"])
        m[4].metric("Pending", ov["counts"]["pending"])
        status = state.get("status", "created")
        if status == "running" and not ov["alive"]:
            status = "interrupted"
        m[5].metric("Status", status + (" 🟢" if ov["alive"] else ""))
        st.progress(ov["progress"], text=f"Grading progress: {ov['counts']['done']}/{ov['total']} graded")

        sp = _spend(job_dir)
        if (job.get("grading_mode") or "legacy") != "legacy" or sp["job"] or sp["campaign"]:
            st.markdown("**OpenRouter budget / spend**")
            b = st.columns(4)
            b[0].metric("OpenRouter", "configured" if openrouter_credential_present() else "not configured")
            jl = sp["job"]
            b[1].metric("This batch (ledger)", f"${jl['local_ledger']['cumulative_cost']:.4f}" if jl else "$0.0000",
                        help="Persistent usage ledger of this batch: <job>/exams/gateway_ledger/usage.jsonl")
            cl = sp["campaign"]
            b[2].metric("Model-selection campaign", f"${cl['local_ledger']['cumulative_cost']:.4f}" if cl else "$0.0000",
                        help="Campaign ledger (benchmarks). Policy: warning $8 / hard stop $10.")
            _bs = (cl or jl or {}).get("budget") or {}
            b[3].metric("Campaign budget state", _bs.get("state", "OK"),
                        help=f"warn ${_bs.get('warn_usd', 8.0)} / hard ${_bs.get('hard_usd', 10.0)}")

        st.markdown("**Primary actions**")
        a = st.columns(4)
        if a[0].button("🧭 Set up exam"):
            _nav(SCREEN_SETUP)
        if a[1].button("▶ Start grading", type="primary", disabled=ov["alive"] or not ov["total"]):
            jobs.clear_control_requests(job_dir)
            spawn_runner(job_dir)
            _nav(SCREEN_PROGRESS)
        if a[2].button(f"🔍 Review ({ov['review']})"):
            _nav(SCREEN_REVIEW)
        if a[3].button("📄 Results"):
            _nav(SCREEN_RESULTS)


# ==========================================================================
# 2. EXAM SETUP (wizard)
# ==========================================================================
elif screen == SCREEN_SETUP:
    st.header("Exam setup")
    st.caption("Six short steps. Discovery (variants, question alignment, grading policies) is "
               "automatic; you are asked only about what stays unresolved.")

    # ---- step 1: course ---------------------------------------------------
    st.subheader("1 · Course")
    from autograder import courses as course_store

    _courses_avail = [c["course_id"] for c in course_store.list_courses()]
    course_choice = st.selectbox(
        "Course (optional — enables course-context grading when a RAG policy is selected)",
        ["(none)"] + _courses_avail, key="batch_course",
    ) if _courses_avail else "(none)"
    if not _courses_avail:
        st.caption("No courses yet — optional. Create one below to store course material locally.")
    with st.expander("Manage course material (local store; never upload answer keys or rubrics)"):
        existing = course_store.list_courses()
        col_a, col_b = st.columns([2, 3])
        with col_a:
            new_id = st.text_input("New course id (letters/digits/_/-)", key="course_new_id")
            new_name = st.text_input("Course name", key="course_new_name")
            if st.button("Create course") and new_id:
                try:
                    course_store.create_course(new_id, new_name)
                    st.success(f"course {new_id} created")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        with col_b:
            ids = [c["course_id"] for c in existing]
            selected = st.selectbox("Select course", ids) if ids else None
        if selected:
            uploads = st.file_uploader(
                "Upload course material (PDF / TXT / MD / DOCX)",
                accept_multiple_files=True, key=f"course_uploads_{selected}",
                type=["pdf", "txt", "md", "markdown", "docx"],
            )
            allow_flagged = st.toggle(
                "Operator override — ingest files flagged by the content screen "
                "(I verified they contain no answer key / rubric / solutions)",
                value=False, key=f"course_allow_flagged_{selected}",
            )
            for f in uploads or []:
                res = course_store.add_source(selected, f.name, f.getvalue(),
                                              allow_suspicious=allow_flagged)
                if not res["stored"]:
                    (st.error if res.get("suspicious") else st.warning)(f"{f.name}: {res['reason']}")
                elif res.get("suspicious_override"):
                    st.warning(f"{f.name}: ingested via operator override — {res['suspicious_override']}")
            status = course_store.index_status(selected)
            st.markdown(
                f"**Sources:** {status['n_sources']} &nbsp;|&nbsp; "
                f"**Indexed:** {'yes' if status['indexed'] else 'no'} "
                + (f"({status['n_chunks']} chunks, built {status.get('built')}, "
                   f"model `{status.get('embed_model')}`)" if status["indexed"] else "")
                + (" &nbsp;|&nbsp; :red[**STALE — material changed, rebuild needed**]"
                   if status.get("stale") else "")
            )
            for name in status["sources"]:
                cols = st.columns([5, 1])
                cols[0].write(f"• {name}")
                if cols[1].button("remove", key=f"rm_{name}"):
                    course_store.remove_source(selected, name)
                    st.rerun()
            if st.button("Build / rebuild index", type="primary"):
                if status["n_sources"] == 0:
                    st.error("No course files stored yet — upload material above first.")
                else:
                    with st.spinner("Parsing, chunking and embedding (local Ollama)…"):
                        try:
                            manifest = course_store.build_index(selected)
                            st.success(f"index built: {manifest['n_chunks']} chunks, "
                                       f"dim {manifest['dim']}, model {manifest['embed_model']}")
                            st.rerun()
                        except ValueError as e:
                            st.error(f"index build failed: {e}")
                        except Exception as e:  # noqa: BLE001
                            st.error(f"embedding failed: {e} — is Ollama running with the "
                                     f"'{course_store.DEFAULT_EMBED_MODEL}' model pulled? "
                                     f"(ollama pull {course_store.DEFAULT_EMBED_MODEL})")

    # ---- step 2: exam ---------------------------------------------------------
    st.subheader("2 · Exam")
    packages = discover_packages()
    source = st.radio(
        "Exam configuration",
        ["Configured exam package", "Upload key & configuration"],
        horizontal=True,
        help=("A package bundles the answer key with its template (grading modes, "
              "answer-sheet rule) and variant mapping. Upload mode lets you bring a new exam family."),
    )
    key_path = None
    template_obj: ExamTemplate | None = None
    up_key = None
    if source == "Configured exam package":
        if packages:
            choice = st.selectbox("Package", list(packages))
            pkg = packages[choice]
            key_path = pkg["key"]
            template_obj = pkg["template_obj"]
            tpl = template_obj
            st.info(f"**{tpl.name or tpl.template_id}** — mode: `{tpl.mode}`"
                    + (f", answer sheet fixed to page(s) {tpl.answer_sheet_pages}"
                       if tpl.answer_sheet_rule == "fixed_pages" else ", answer sheets detected structurally"))
        else:
            st.warning("No configured packages found (looked for *.template.json).")

    # ---- step 3: key / rubric / solution ----------------------------------------
    st.subheader("3 · Answer key, rubric, solution")
    if source == "Configured exam package":
        if key_path is not None:
            st.caption(f"Answer key from the package: `{key_path.name}` "
                       + ("(parsed JSON)" if key_path.suffix == ".json" else "(PDF — parsed on first grading run)"))
    else:
        up_key = st.file_uploader("Official answer key (PDF or parsed JSON)", type=["pdf", "json"])
        key_path = _stage_dir(up_key, "answer_key")
        up_rubric = st.file_uploader("Rubric / grading configuration / official solution (optional)",
                                     type=["txt", "md", "json"])
        rubric_path = _stage_dir(up_rubric, "rubric")
        st.session_state["rubric_path"] = str(rubric_path) if rubric_path else None
        mode = st.selectbox(
            "Exam mode", ["with_explanation", "multiple_choice", "mixed"],
            help=("multiple_choice: selections only — no explanation transcription or judging model "
                  "calls. mixed: configure per-question modes below."))
        question_modes: dict[str, str] = {}
        if mode == "mixed":
            st.caption('Per-question modes (question id → mode); example: {"1": "multiple_choice", "2": "with_explanation"}')
            qm_text = st.text_area("question_modes JSON", "{}")
            try:
                question_modes = json.loads(qm_text)
            except json.JSONDecodeError as e:
                st.error(f"invalid JSON: {e}")
        sheet_rule = st.selectbox(
            "Answer-sheet rule", ["detected", "fixed_pages"],
            help=("detected: a survey pass locates dedicated answer sheets structurally. fixed_pages: "
                  "this exam family's answer sheet is a fixed page set (e.g. the first page)."))
        sheet_pages: list[int] = []
        if sheet_rule == "fixed_pages":
            pages_text = st.text_input("Answer-sheet page numbers (comma-separated)", "1")
            try:
                sheet_pages = [int(x) for x in pages_text.split(",") if x.strip()]
            except ValueError:
                st.error("page numbers must be integers")

    # ---- step 4: automatic discovery + manual overrides ---------------------------
    st.subheader("4 · Automatic discovery (variants, alignment)")
    st.caption("When grading starts, the pipeline discovers exam variants, their marker symbols and the "
               "question alignment automatically and records every fact with its source. Unresolved "
               "structural issues become ONE package-level review item — never one per student. "
               "Provide mappings here only to override discovery.")
    variants_staged = None
    alignment_staged = None
    if source == "Configured exam package" and key_path is not None:
        variants_path = key_path.with_name(key_path.stem + ".variants.json")
        alignment_path = key_path.with_name(key_path.stem + ".alignment.json")
        d1, d2 = st.columns(2)
        d1.metric("Variant mapping", "provided" if variants_path.exists() else "auto-discover")
        d2.metric("Question alignment", "provided" if alignment_path.exists() else "auto-discover")
        if variants_path.exists():
            vcfg = json.loads(variants_path.read_text(encoding="utf-8"))
            with st.expander("Variant symbol mapping (authoritative)"):
                st.table([{"marker": name, "variant": entry["variant"], "description": entry["description"]}
                          for name, entry in vcfg.get("markers", {}).items()])
    elif source != "Configured exam package":
        up_variants = st.file_uploader("Variant mapping JSON (optional — marker → key column)", type=["json"])
        variants_staged = _stage_dir(up_variants, "variants")
        up_alignment = st.file_uploader("Question alignment JSON (optional — printed id → key id per variant)",
                                        type=["json"], key="up_alignment")
        alignment_staged = _stage_dir(up_alignment, "alignment")
        if key_path is not None:
            template_obj = ExamTemplate(
                template_id=f"uploaded-{datetime.now():%Y%m%d%H%M%S}",
                name=Path(up_key.name).stem if up_key else "uploaded exam",
                mode=mode, question_modes=question_modes,
                answer_sheet_rule=sheet_rule, answer_sheet_pages=sheet_pages,
            )
            # Persist sidecars next to the staged key so the pipeline auto-discovers them.
            tpl_path = key_path.with_name(key_path.stem + ".template.json")
            tpl_path.write_text(template_obj.model_dump_json(indent=1), encoding="utf-8")
            if variants_staged is not None:
                variants_staged.replace(key_path.with_name(key_path.stem + ".variants.json"))
            if alignment_staged is not None:
                alignment_staged.replace(key_path.with_name(key_path.stem + ".alignment.json"))

    # ---- step 5: question policies -----------------------------------------------
    st.subheader("5 · Question grading policies")
    pol_path = key_path.with_name(key_path.stem + ".policies.json") if key_path is not None else None
    if pol_path is not None and pol_path.exists():
        try:
            _pols = json.loads(pol_path.read_text(encoding="utf-8"))
            st.dataframe([{"question": q, "policy": p} for q, p in sorted(_pols.items())], width="stretch")
        except Exception:  # noqa: BLE001
            st.caption("policies file present but unreadable")
    else:
        st.caption("Policies (choice only / explanation required / explanation can rescue / independent) "
                   "are inferred from the rubric automatically at the first grading run and recorded per "
                   "question. To override, place `<key>.policies.json` next to the key.")

    # ---- step 6: preflight ----------------------------------------------------------
    st.subheader("6 · Preflight")
    if key_path is not None and key_path.suffix == ".json":
        try:
            from autograder.key_parser import load_answer_key
            from autograder.preflight import READY, alignment_from_discovery, preflight_package

            _k = load_answer_key(key_path)
            _alp = key_path.with_name(key_path.stem + ".alignment.json")
            _al = json.loads(_alp.read_text(encoding="utf-8")) if _alp.exists() else None
            _rep = preflight_package(key=_k, variants=list(_k.versions),
                                     alignment=alignment_from_discovery(_al, list(_k.versions), _k))
            (st.success if _rep.status == READY else st.error)(f"{_rep.status} — {_rep.summary()}")
            if getattr(_rep, "blocking", None):
                st.dataframe([{"code": f.code, "what": f"{f.subject} {f.subject_id}", "problem": f.message,
                               "needed": f.needed} for f in _rep.blocking], width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Preflight unavailable: {type(exc).__name__}")
    else:
        st.caption("Preflight runs as soon as the answer key is parsed (first grading run for a PDF key) "
                   "and again on the Grading progress screen.")

    # ---- students + create ------------------------------------------------------------
    st.subheader("Students")
    exams_upload = st.file_uploader(
        "Student exams — one/many PDFs or images, or ZIP archive(s)",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "zip"],
        accept_multiple_files=True,
    )
    mask = st.toggle("Mask red instructor annotations before inference", value=False,
                     help="Enable for scans that already carry instructor grading marks.")

    if st.button("Create grading batch", type="primary", disabled=key_path is None or not exams_upload):
        staging = Path(st.session_state.setdefault(
            "staging_dir", str(jobs.jobs_root() / f"_staging-{datetime.now():%Y%m%d-%H%M%S}")))
        staged_exams = []
        for uploaded in exams_upload:
            p = staging / "exams_in" / uploaded.name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(uploaded.getvalue())
            staged_exams.append(p)
        rubric = st.session_state.get("rubric_path")
        _job_grading_args = dict(grading_args)
        if grading_mode != "legacy" and course_choice != "(none)" and rag_policy != "RAG_DISABLED":
            _job_grading_args["--course"] = course_choice
        elif grading_mode != "legacy" and rag_policy != "RAG_DISABLED":
            st.warning(f"RAG policy {rag_policy} is selected but no course is chosen — no retrieval "
                       "will run and grading proceeds without course context.")
        new_job_dir = jobs.create_job(
            course_id=None if course_choice == "(none)" else course_choice,
            key=Path(key_path), exams=staged_exams,
            rubric=Path(rubric) if rubric else None,
            backend_args=backend_args, grading_args=_job_grading_args,
            mask=mask, grading_mode=grading_mode,
        )
        st.session_state["selected_job"] = new_job_dir.name
        try:
            shutil.rmtree(staging, ignore_errors=True)   # raw uploads must not accumulate on disk
        finally:
            st.session_state.pop("staging_dir", None)
        job = jobs.load_job(new_job_dir)
        st.success(f"Batch **{new_job_dir.name}** created with {len(job['exams'])} exam(s). "
                   "Open **Grading progress** to start grading.")
        for issue in job.get("intake_issues", []):
            st.warning(issue)
        if st.button("Go to Grading progress"):
            _nav(SCREEN_PROGRESS)


# ==========================================================================
# 3. GRADING PROGRESS
# ==========================================================================
elif screen == SCREEN_PROGRESS:
    st.header("Grading progress")
    if job_dir is None:
        st.info("No batches yet — create one in Exam setup.")
    else:
        job_name = job_dir.name

        @st.fragment(run_every=3)
        def job_panel(job_dir: Path) -> None:
            ov = _job_overview(job_dir)
            job, state, exams, counts = ov["job"], ov["state"], ov["exams"], ov["counts"]
            alive = ov["alive"]
            rate_limited = sum(1 for e in exams.values()
                               if e.get("error") and ("429" in e["error"] or "rate" in e["error"].lower()))
            status = state.get("status", "created") if alive or state.get("status") != "running" else "interrupted"
            st.subheader(f"Status: {status}" + (" 🟢 runner active" if alive else ""))

            c = st.columns(7)
            c[0].metric("Processed / total", f"{counts['done']}/{len(exams)}")
            c[1].metric("AUTO", ov["auto"], help="graded with no review item")
            c[2].metric("REVIEW", ov["review"], help="graded, at least one item needs a human")
            c[3].metric("FAILED", counts["failed"])
            c[4].metric("Processing", counts["running"])
            c[5].metric("Rate-limited", rate_limited)
            c[6].metric("Elapsed", fmt_elapsed(state))
            current = ov["current"]
            if current:
                stage = (exams.get(current) or {}).get("stage") or "…"
                st.caption(f"Currently processing **{current}** — stage: {stage}")

            # ---- cloud usage from the batch's PERSISTENT ledger (truth, not counters) ----
            lp = _job_ledger_path(job_dir)
            ocr_calls = grader_calls = cache_hits = 0
            cost = 0.0
            if lp.exists():
                from autograder.spend import ledger_summary
                ls = ledger_summary(lp)
                bt = ls["by_task"]
                ocr_calls = sum(v["calls"] for t, v in bt.items() if str(t).startswith("ocr"))
                grader_calls = sum(v["calls"] for t, v in bt.items() if str(t).startswith("grade"))
                cache_hits = ls["cache_hits"]
                cost = ls["cumulative_cost"]
            remaining_est = "—"
            try:
                _kp = job_dir / "uploads" / "answer_key.json"
                if _kp.exists() and counts["done"] and len(exams) > counts["done"]:
                    from autograder.estimate import estimate_job, load_pricing
                    from autograder.key_parser import load_answer_key
                    _gwr = None
                    if _models_toml():
                        from autograder.orchestrator import setup_from_config
                        _gwr = setup_from_config(_models_toml(), job_dir / "exams").gateway
                    _pol_p = job_dir / "uploads" / "answer_key.policies.json"
                    _pol = json.loads(_pol_p.read_text(encoding="utf-8")) if _pol_p.exists() else None
                    _rem = len(exams) - counts["done"]
                    _est = estimate_job(key=load_answer_key(_kp), exams=_rem, policies=_pol, gateway=_gwr,
                                        pricing=load_pricing({"pricing": getattr(_gwr, "pricing_config", None)}
                                                             if getattr(_gwr, "pricing_config", None) else None))
                    if _est.estimated_cost is not None:
                        remaining_est = f"≈ ${_est.estimated_cost:,.2f} (ESTIMATE)"
            except Exception:  # noqa: BLE001 — an estimate never blocks the panel
                remaining_est = "—"
            u = st.columns(5)
            u[0].metric("OCR calls", ocr_calls)
            u[1].metric("Grader calls", grader_calls)
            u[2].metric("Cache hits", cache_hits)
            u[3].metric("Cloud cost (ledger)", f"${cost:.4f}")
            u[4].metric("Estimated remaining cost", remaining_est,
                        help="only when a pricing table is configured; never mixed with the ledger")

            total = max(len(exams), 1)
            st.progress(counts["done"] / total)

            b = st.columns(5)
            if b[0].button("▶ Start / Resume", disabled=alive,
                           help="Runs pending and failed exams; finished work is kept."):
                jobs.clear_control_requests(job_dir)
                spawn_runner(job_dir)
                st.rerun()
            if b[1].button("⏸ Pause", disabled=not alive,
                           help="The current exam finishes safely and stays pending."):
                jobs.request_pause(job_dir)
            if b[2].button("⏹ Stop", disabled=not alive):
                jobs.request_stop(job_dir)
            if b[3].button("🔄 Refresh now"):
                st.rerun()
            if b[4].button("📦 Rebuild combined reports"):
                jobs.combine_outputs(job_dir)
                st.success("combined reports rebuilt")

            st.dataframe(
                [{"exam": anon, "original file": e.get("original_name"), "status": e.get("status"),
                  "stage": e.get("stage"), "variant": e.get("variant"), "score": e.get("predicted"),
                  "review items": e.get("review_items"), "unanswered": e.get("unanswered"),
                  "runtime (s)": e.get("runtime_s"),
                  "error": (e.get("error") or "").splitlines()[0] if e.get("error") else None}
                 for anon, e in sorted(exams.items())],
                width="stretch",
            )

        job_panel(job_dir)
        state = jobs.load_state(job_dir)

        # ---- package preflight: ONE setup warning instead of one review per student ----
        st.divider()
        st.subheader("Package setup")
        pkg = _package_status(job_dir)
        if pkg["status"] == "KEY_NOT_PARSED":
            st.caption("Package checks appear once the answer key has been parsed.")
        elif pkg["report"] is None:
            st.caption(pkg["summary"])
        else:
            from autograder.preflight import READY, reviews_avoided
            _rep = pkg["report"]
            if _rep.status == READY:
                st.success(_rep.summary())
            else:
                st.error(f"**{_rep.status}** — fix these once, at package level. Left unresolved they "
                         f"would produce about {reviews_avoided(_rep, len(state.get('exams', {})))} "
                         "individual review items.")
                st.dataframe([{"code": f.code, "what": f"{f.subject} {f.subject_id}", "problem": f.message,
                               "needed": f.needed} for f in _rep.blocking], width="stretch")
            if _rep.warnings:
                with st.expander(f"Package warnings ({len(_rep.warnings)}) — not blocking"):
                    st.dataframe([{"code": f.code, "what": f"{f.subject} {f.subject_id}", "note": f.message}
                                  for f in _rep.warnings], width="stretch")

        # ---- pre-run ESTIMATE (never mixed with the ledger's actual usage) ----
        st.divider()
        st.subheader("Estimated cloud usage")
        _key_path = job_dir / "uploads" / "answer_key.json"
        if not _key_path.exists():
            st.caption("The estimate appears once the answer key has been parsed (uploads/answer_key.json).")
        else:
            try:
                from autograder.estimate import estimate_job, load_pricing
                from autograder.key_parser import load_answer_key

                _key = load_answer_key(_key_path)
                _pol_path = job_dir / "uploads" / "answer_key.policies.json"
                _pol = json.loads(_pol_path.read_text(encoding="utf-8")) if _pol_path.exists() else None
                # The estimator's gateway comes from models.toml, built against the
                # batch's own state root (no backend is constructed; no call is made).
                _gw = None
                if _models_toml():
                    try:
                        from autograder.orchestrator import setup_from_config
                        _gw = setup_from_config(_models_toml(), job_dir / "exams").gateway
                    except Exception:  # noqa: BLE001 — estimator degrades to call counts
                        _gw = None
                _pricing_cfg = getattr(_gw, "pricing_config", None)
                _est = estimate_job(key=_key, exams=len(state.get("exams", {})), policies=_pol, gateway=_gw,
                                    pricing=load_pricing({"pricing": _pricing_cfg} if _pricing_cfg else None))
                e = st.columns(5)
                e[0].metric("ESTIMATED cloud calls", f"{_est.estimated_cloud_calls:g}")
                e[1].metric("ESTIMATED input tokens", f"{_est.estimated_input_tokens:,}")
                e[2].metric("ESTIMATED output tokens", f"{_est.estimated_output_tokens:,}")
                e[3].metric("ESTIMATED cost", "—" if _est.estimated_cost is None else f"${_est.estimated_cost:,.2f}")
                e[4].metric("Deterministic items", f"{_est.sub_items_deterministic}/{_est.sub_items_total}")
                st.caption("⚠ " + _est.disclaimer
                           + (f" ({_est.cost_unavailable_reason})" if _est.cost_unavailable_reason else "")
                           + f" Escalation assumptions: {_est.assumptions['source']}."
                           + ("" if _gw is not None else " No models.toml — call counts only, no prices."))
                with st.expander("Estimate breakdown (per call type)"):
                    st.dataframe([{"call type": k, "ESTIMATED calls": v} for k, v in _est.estimated_calls.items()],
                                 width="stretch")
            except Exception as exc:  # noqa: BLE001 — an estimate must never block the UI
                st.caption(f"Estimate unavailable: {type(exc).__name__}")

        # ---- batch-level view: systemic warnings first ----
        st.divider()
        st.subheader("Batch checks")
        from autograder import reviewui as _rui

        _results, _extractions = _results_and_extractions(job_dir)
        if not _results:
            st.caption("No graded exams yet — batch checks appear once results exist.")
        else:
            _ov = _rui.batch_overview(_results, _extractions)
            _sev = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
            if _ov["warnings"]:
                st.error(f"{len(_ov['warnings'])} batch-level warning(s) — check these BEFORE working "
                         "through individual reviews: one cause here can explain many of them.")
                st.dataframe([{"severity": _sev.get(w["severity"], "") + " " + w["severity"], "code": w["code"],
                               "scope": f"{w['scope']} {w['scope_id']}", "students": w["affected_students"],
                               "items": w["affected_items"], "what it means": w["explanation"]}
                              for w in _ov["warnings"]], width="stretch")
            else:
                st.success("No batch-level anomaly detected (variant mix, blank/crop/OCR rates, score "
                           "distribution and alignment all look ordinary for this batch).")
            _s = _ov["summary"]
            m = st.columns(4)
            m[0].metric("Review cases", _s["cases"])
            m[1].metric("Decisions needed", _s["decisions_required"],
                        help="Cases sharing an exact mechanical cause are resolved by ONE decision.")
            m[2].metric("Absorbed by grouping", _s["cases_absorbed_by_grouping"])
            m[3].metric("Graded exams", _ov["exams"])
            if _s["cases"] and st.button("Open the review queue"):
                _nav(SCREEN_REVIEW)


# ==========================================================================
# 4. REVIEW QUEUE
# ==========================================================================
elif screen == SCREEN_REVIEW:
    st.header("Review queue")
    if job_dir is None:
        st.info("No batches yet — create one in Exam setup.")
    else:
        from autograder import reviewui as _rui
        from autograder.reviewqueue import PRIORITY_TIERS, apply_scope, group_cases

        _results, _extractions = _results_and_extractions(job_dir)
        if not _results:
            st.caption("No graded exams yet — reviews appear as exams finish.")
        else:
            _ov = _rui.batch_overview(_results, _extractions)
            _warnings = _ov["warnings"]
            _packs = _job_packs(job_dir)
            st.caption("Every item states its typed reason. Priority affects ORDER only — it never changes a "
                       "grade. Batch-level causes come first: one decision there can resolve many items.")
            if _warnings:
                st.error(f"{len(_warnings)} batch-level warning(s) — a systemic cause may explain many "
                         "reviews below.")
                st.dataframe([{"severity": w["severity"], "code": w["code"], "scope": f"{w['scope']} {w['scope_id']}",
                               "students": w["affected_students"], "items": w["affected_items"],
                               "what it means": w["explanation"]} for w in _warnings], width="stretch")
            _s = _ov["summary"]
            m = st.columns(4)
            m[0].metric("Review cases", _s["cases"])
            m[1].metric("Decisions needed", _s["decisions_required"])
            m[2].metric("Absorbed by grouping", _s["cases_absorbed_by_grouping"])
            m[3].metric("Graded exams", _ov["exams"])

            # ---- grouped queue + apply-to-all for exactly-mechanical causes ----
            if _ov["groups"]:
                with st.expander("Queue by cause (priority order, grouped)", expanded=True):
                    st.dataframe([
                        {"priority": (g["explanation"].splitlines()[0] if g["explanation"] else ""),
                         "reason": f"{g['reason_code']} — {_reason_title(g['reason_code'])}",
                         "cases": g["size"], "students": ", ".join(g["students"][:4]),
                         "one decision covers all": g["apply_to_all_eligible"], "scope": g["scope"]}
                        for g in _ov["groups"]], width="stretch")
                    # apply-to-all: wired to ResolutionStore.apply_to_all with the group's scope
                    _all_items = _ov["review_items"]
                    _cases = [_rui.to_case(it, warnings=_warnings) for it in _all_items]
                    for grp in group_cases(_cases):
                        if not grp.apply_to_all_eligible or len(grp.cases) < 2:
                            continue
                        first = grp.cases[0]
                        proto = next((it for it in _all_items if it.exam_id == first.exam_id
                                      and it.question_id == first.question_id
                                      and it.sub_item_id == first.sub_item_id), None)
                        if proto is None or proto.kind not in ("variant", "layout") or not proto.options:
                            continue
                        st.markdown(f"**{grp.reason_code} — {_reason_title(grp.reason_code)}**: "
                                    f"{len(grp.cases)} identical cases ({grp.scope}). One decision applies to all:")
                        cols = st.columns(max(1, min(4, len(proto.options))))
                        for i, opt in enumerate(proto.options[:4]):
                            if cols[i].button(f"Apply '{opt}' to all {len(grp.cases)}",
                                              key=f"ata_{grp.fingerprint}_{i}"):
                                n = _rui.ResolutionStore(job_dir / "exams" / first.exam_id).apply_to_all(
                                    job_dir, proto.kind, proto.question_id, proto.sub_item_id,
                                    decision=opt, scope=apply_scope(grp))
                                st.success(f"applied to {n} exam(s); recorded in apply_to_all.jsonl")
                                st.rerun()

            # ---- per exam: typed reason + evidence + decision ----
            st.divider()
            exam_ids = sorted(e for e, r in _results.items() if r.get("needs_human_review"))
            if not exam_ids:
                st.success("Nothing to review — every graded item settled automatically.")
            else:
                picked = st.selectbox("Student exam", exam_ids, key="review_exam")
                exam_dir = job_dir / "exams" / picked
                result = _results[picked]
                extraction = _extractions.get(picked)
                rstore = _rui.ResolutionStore(exam_dir)
                resolved = rstore.load()
                crop_info = ((result.get("backend_info") or {}).get("evidence_crops") or {})
                items = _rui.build_review_items(picked, result, extraction, packs=_packs, warnings=_warnings)
                st.caption(f"{len(items)} item(s), {len(resolved)} resolved. "
                           + ("Image evidence: available for some items." if any(it.crop_png_b64 for it in items)
                              else "Image evidence: not available for this batch"
                                   + (f" ({crop_info.get('reason')})" if crop_info.get("reason") else
                                      " (no explanation-crop producer in production yet)") + "."))
                for it in items:
                    key = f"{it.question_id}:{it.sub_item_id}"
                    done = resolved.get(key)
                    tier_name = PRIORITY_TIERS.get(it.priority_tier, "")
                    with st.container(border=True):
                        st.markdown(f"**{it.reason_code} — {_reason_title(it.reason_code)}** · item `{key}` · "
                                    f"{it.points_affected:g} pt(s) at stake · priority {it.priority_tier} ({tier_name})"
                                    + (f" · ✅ resolved: *{done['decision']}*" if done else ""))
                        if it.explanation:
                            st.code(it.explanation, language=None)
                        if it.batch_warning_code:
                            st.warning(f"A batch-level issue ({it.batch_warning_code}) affects "
                                       f"{it.batch_warning_students} students and may explain this item — "
                                       "resolve that first.")
                        ev, dec = st.columns([3, 2])
                        with ev:
                            st.markdown("*Evidence*")
                            if it.crop_png_b64:
                                import base64 as _b64
                                st.image(_b64.b64decode(it.crop_png_b64), caption="answer region")
                            else:
                                st.caption("no image evidence available for this item")
                            if it.primary_transcription is not None:
                                st.caption("transcription (immutable — what the grader read):")
                                st.code(it.primary_transcription or "(empty)", language=None)
                            if it.kind == "mc":
                                st.caption(f"deterministic candidates: {it.deterministic_candidate} | "
                                           f"local read: {it.local_candidate} | cloud read: {it.cloud_candidate}")
                            if it.kind == "ocr" and it.secondary_transcription is not None:
                                st.caption(f"second reading: {it.secondary_transcription!r}")
                            if it.kind == "grading":
                                st.caption(f"selected option: {it.selected_option} · proposed "
                                           f"{it.proposed_score}/{it.max_score}")
                                if it.rubric_items:
                                    st.caption(f"rubric items: {', '.join(map(str, it.rubric_items))}")
                                if it.grading_evidence:
                                    st.caption(f"grader note: {it.grading_evidence}")
                                if it.question_context:
                                    with st.expander("question context (answer-free)"):
                                        st.code(it.question_context, language=None)
                            with st.expander("why was this graded this way? (decision trace)"):
                                st.code(_rui.decision_trace_for(exam_dir, result, it.question_id, it.sub_item_id),
                                        language=None)
                        with dec:
                            st.markdown("*Decision*")
                            for i, opt in enumerate(it.options[:4]):
                                if st.button(opt, key=f"rv_{picked}_{key}_{i}"):
                                    rstore.resolve(it.question_id, it.sub_item_id, decision=opt)
                                    st.rerun()
                            if it.apply_to_all_eligible:
                                st.caption("exactly mechanical cause — use the grouped queue above to apply "
                                           "one decision to every identical case")


# ==========================================================================
# 5. RESULTS / EXPORT
# ==========================================================================
elif screen == SCREEN_RESULTS:
    st.header("Results / export")
    if job_dir is None:
        st.info("No batches yet — create one in Exam setup.")
    else:
        job_name = job_dir.name
        state = jobs.load_state(job_dir)
        exam_ids = sorted(state.get("exams", {}))
        if not exam_ids:
            st.caption("This batch has no exams.")
        else:
            picked = st.selectbox("Student exam", exam_ids, key="results_exam")
            exam_dir = job_dir / "exams" / picked
            result_path = exam_dir / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                survey_path = exam_dir / "survey.json"
                sheet_pages = []
                if survey_path.exists():
                    survey = json.loads(survey_path.read_text(encoding="utf-8"))
                    sheet_pages = survey.get("answer_sheet_policy", {}).get("authoritative_pages", [])
                meta = st.columns(4)
                meta[0].metric("Final grade (deterministic)", f"{result['total_awarded']:g}/{result['total_max']:g}")
                meta[1].metric("Variant", result.get("detected_version") or "—")
                meta[2].metric("Answer-sheet pages", ", ".join(map(str, sheet_pages)) or "—")
                meta[3].metric("Review items", len(result.get("needs_human_review", [])))
                vd = result.get("variant_detection")
                if vd:
                    st.caption(f"Variant marker: saw {vd.get('marker_seen')!r} → matched "
                               f"`{vd.get('matched_marker')}` (confident: {vd.get('confident')})")
                st.markdown("**Per-question scores**")
                st.dataframe([{"question": q["question_id"], "type": q.get("question_type"),
                               "points": q.get("points_awarded"), "max": q.get("points_max"),
                               "capped": q.get("capped"), "summary": (q.get("summary") or "")[:120]}
                              for q in result.get("questions", [])], width="stretch")
                with st.expander("Per sub-item detail"):
                    st.dataframe([{"question": q["question_id"], "item": s["sub_item_id"],
                                   "answer": s.get("student_answer"), "correct": s.get("selection_correct"),
                                   "points": s.get("points_total"), "max": s.get("points_max"),
                                   "status": s.get("status"),
                                   "explanation": (s.get("explanation_transcription") or "")[:80] or None,
                                   "needs review": s.get("needs_review"), "reason": (s.get("reason") or "")[:160]}
                                  for q in result.get("questions", []) for s in q.get("sub_results", [])],
                                 width="stretch")
                # ---- review history ----
                st.markdown("**Review history**")
                from autograder import reviewui as _rui
                _res = _rui.ResolutionStore(exam_dir).load()
                _ata = job_dir / "apply_to_all.jsonl"
                rows = [{"item": k, "decision": v.get("decision"), "value": v.get("value"),
                         "by": v.get("by"), "when": v.get("ts")} for k, v in sorted(_res.items())]
                if _ata.exists():
                    for line in _ata.read_text(encoding="utf-8").splitlines():
                        try:
                            r = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        rows.append({"item": f"{r.get('question_id')}:{r.get('sub_item_id')} (apply-to-all)",
                                     "decision": r.get("decision"), "value": r.get("value"),
                                     "by": r.get("by", "lecturer:apply-to-all"), "when": r.get("ts")})
                if rows:
                    st.dataframe(rows, width="stretch")
                else:
                    st.caption("no human decisions recorded for this exam")
                if result.get("needs_human_review"):
                    with st.expander("Open review reasons"):
                        for item in result["needs_human_review"]:
                            st.markdown(f"- **{item['question_id']}.{item['sub_item_id']}** — {item['reason']}")
                # ---- shadow mode: a recorded proposal, never an applied grade ----
                _shadow_path = exam_dir / "shadow_comparison.json"
                if _shadow_path.exists():
                    _cmp = json.loads(_shadow_path.read_text(encoding="utf-8"))
                    _agree = _cmp.get("agreement", {})
                    with st.expander("🧪 Shadow comparison — SHADOW / NON-AUTHORITATIVE (recorded only, NOT applied)"):
                        st.warning("SHADOW / NON-AUTHORITATIVE — every figure below is the shadow route's "
                                   "recorded proposal. The student's actual grade is the final grade above.")
                        st.info(_cmp.get("note", ""))
                        s = st.columns(4)
                        s[0].metric("Exact score agreement", f"{_agree.get('exact_score_agreement', 0)}%")
                        s[1].metric("Mean |delta|", _agree.get("mean_abs_delta"))
                        s[2].metric("Legacy review items", _agree.get("legacy_review_items"))
                        s[3].metric("Reliability review items", _agree.get("reliability_review_items"))
                        st.dataframe([{"question": i["question_id"], "item": i["sub_item_id"],
                                       "legacy": i["legacy_points"], "proposal": i["reliability_points"],
                                       "delta": i["score_delta"], "legacy review": i["legacy_review"],
                                       "reliability state": i["reliability_state"],
                                       "reason": i["reliability_reason_code"], "route": i["route_difference"]}
                                      for i in _cmp.get("items", [])], width="stretch")
                with st.expander("🔍 Why was this graded this way?"):
                    trace_rows = [(q["question_id"], s["sub_item_id"]) for q in result.get("questions", [])
                                  for s in q.get("sub_results", [])]
                    if trace_rows:
                        labels = [f"{q}.{s}" for q, s in trace_rows]
                        chosen = st.selectbox("Item", labels, key=f"trace_{picked}")
                        qid, sid = trace_rows[labels.index(chosen)]
                        st.code(_rui.decision_trace_for(exam_dir, result, qid, sid), language=None)
                d = st.columns(2)
                d[0].download_button("⬇ result.json", result_path.read_bytes(),
                                     file_name=f"{picked}-result.json", mime="application/json")
                report_path = exam_dir / "report.md"
                if report_path.exists():
                    d[1].download_button("⬇ report.md", report_path.read_bytes(),
                                         file_name=f"{picked}-report.md", mime="text/markdown")
            else:
                e = state["exams"][picked]
                st.info(f"No result yet — status: {e.get('status')}")
                if e.get("error"):
                    st.code(e["error"])
                log_path = exam_dir / "grade.log"
                if log_path.exists():
                    with st.expander("Processing log (tail)"):
                        st.code("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))

        st.divider()
        st.subheader("Batch downloads")
        cols = st.columns(4)
        for col, (label, name, mime) in zip(cols, [
            ("⬇ Combined CSV", "combined_results.csv", "text/csv"),
            ("⬇ Combined JSON", "combined_results.json", "application/json"),
            ("⬇ Batch report (MD)", "summary.md", "text/markdown"),
            ("⬇ All reports (ZIP)", "reports.zip", "application/zip"),
        ]):
            p = job_dir / name
            if not p.exists():
                try:
                    jobs.combine_outputs(job_dir)
                except Exception:  # noqa: BLE001 — nothing graded yet
                    pass
            if p.exists():
                col.download_button(label, p.read_bytes(), file_name=f"{job_name}-{name}", mime=mime)


# ==========================================================================
# 6. ADVANCED / DIAGNOSTICS
# ==========================================================================
elif screen == SCREEN_ADVANCED:
    st.header("Advanced / diagnostics")
    from autograder import reviewui as _rui
    from autograder.cloudcheck import openrouter_credential_present
    from autograder.readiness import role_status

    key_present = openrouter_credential_present()
    st.caption("Technical configuration. The OpenRouter key is read only from the OPENROUTER_API_KEY "
               "environment variable and is never shown or stored.")

    # ---- role -> model (UNSELECTED marked) ----
    st.subheader("Model roles")
    _roles = role_status(_models_toml())
    if _roles.get("using_example_as_template"):
        st.warning("models.toml is missing — showing models.example.toml as the template. Cloud roles are "
                   "UNSELECTED until you copy it to models.toml and choose models.")
    _mark = {"UNSELECTED": "⚠ UNSELECTED", "ABSENT": "⚠ UNSELECTED (absent)", "CONFIGURED_CLOUD": "cloud ✓",
             "SELECTED_LOCAL": "local ✓", "DISABLED": "disabled"}
    st.table([{"role / task": t, "status": _mark.get(v.get("status"), v.get("status")),
               "backend": v.get("backend") or "—", "model": v.get("model") or "—"}
              for t, v in (_roles.get("tasks") or {}).items()])
    try:
        from autograder.benchmark.registry import load_registry
        _reg = load_registry()
        st.caption(f"Model selection: {len(_reg.unselected_roles())} role(s) UNSELECTED in the candidate "
                   f"registry (v{_reg.version}, {_reg.updated}). Winners are chosen only from benchmark "
                   "reports (`autograder bench ...`), never by default.")
    except Exception:  # noqa: BLE001
        pass

    # ---- reliability mode / RAG ----
    st.subheader("Grading route & RAG")
    st.write(f"Grading mode for new batches: **{grading_mode}** · RAG policy: **{rag_policy}**")
    try:
        from autograder import courses as _cs
        _clist = _cs.list_courses()
        if _clist:
            st.table([{"course": c["course_id"], "name": c.get("name"),
                       **{k: v for k, v in _cs.index_status(c["course_id"]).items()
                          if k in ("n_sources", "indexed", "stale", "n_chunks", "embed_model")}} for c in _clist])
        else:
            st.caption("course store: no courses (RAG index: none). Embeddings are local (bge-m3 via Ollama).")
    except Exception as e:  # noqa: BLE001
        st.caption(f"course store unavailable: {type(e).__name__}")

    # ---- budget + ledgers (persistent truth) ----
    st.subheader("Budget & spend")
    sp = _spend(job_dir)
    from autograder.spend import EXPERIMENT_HARD_STOP_USD, EXPERIMENT_WARN_USD
    bcols = st.columns(4)
    bcols[0].metric("Campaign warning", f"${EXPERIMENT_WARN_USD:.2f}")
    bcols[1].metric("Campaign hard stop", f"${EXPERIMENT_HARD_STOP_USD:.2f}")
    cl = sp["campaign"]
    bcols[2].metric("Campaign spent (ledger)", f"${cl['local_ledger']['cumulative_cost']:.4f}" if cl else "$0.0000")
    bcols[3].metric("Campaign state", (cl or {}).get("budget", {}).get("state", "OK"))
    if _roles.get("budget_section"):
        st.caption(f"models.toml [budget]: {_roles['budget_section']}")
    for label, view in (("This batch", sp["job"]), ("Model-selection campaign", sp["campaign"])):
        with st.expander(f"{label} — local ledger" + ("" if view else " (no entries yet)")):
            if not view:
                st.caption("no ledger rows yet")
            else:
                L = view["local_ledger"]
                st.write(f"calls {L['cloud_calls']} · cache hits {L['cache_hits']} · tokens in {L['input_tokens']:,} / "
                         f"out {L['output_tokens']:,} · cumulative cost ${L['cumulative_cost']:.4f} · {L['path']}")
                if L["by_task"]:
                    st.table([{"task": t, **v} for t, v in L["by_task"].items()])
                if L["by_model"]:
                    st.table([{"model": m, **v} for m, v in L["by_model"].items()])

    # ---- OpenRouter ----
    st.subheader("OpenRouter")
    o = st.columns(3)
    o[0].metric("Credential in environment", "present" if key_present else "missing")
    o[1].metric("Configured", "YES" if key_present else "NO")
    o[2].metric("Key-usage endpoint", "ready (not called)")
    if st.button("Fetch OpenRouter-reported key usage (GET /api/v1/key)", disabled=not key_present,
                 help="Authenticated metadata only — usage/limit numbers, never the key. Shown next to the "
                      "local ledger; the local ledger stays authoritative for our budget policy."):
        try:
            from autograder.backends import BackendConfig
            from autograder.backends.openrouter import OpenRouterBackend
            from autograder.spend import spend_view
            _meta = OpenRouterBackend(BackendConfig(backend="openrouter", model="openrouter/auto")).key_metadata()
            _camp = sp["campaign"]["local_ledger"]["path"] if sp["campaign"] else None
            if _camp:
                st.json(spend_view(_camp, _meta))
            else:
                st.json(_meta)
        except Exception as e:  # noqa: BLE001
            st.error(f"could not fetch key metadata: {type(e).__name__}: {e}")
    if not key_present:
        st.caption("OpenRouter credential is not configured — set OPENROUTER_API_KEY in the shell that "
                   "starts the app. It is never written to a file.")

    # ---- verifier crops ----
    st.subheader("OCR verifier evidence")
    from autograder.evidencecrops import production_crop_provider
    _vc = production_crop_provider().describe()
    st.write(f"Explanation crop producer: **{_vc.get('status')}** — {_vc.get('reason')}")
    st.caption(_vc.get("fallback", ""))

    # ---- gateway settings summary + connection probe ----
    st.subheader("Gateway")
    _gw = None
    _gw_err = None
    if _models_toml():
        try:
            from autograder.orchestrator import setup_from_config
            _rt = setup_from_config(_models_toml(), (job_dir / "exams") if job_dir else REPO_ROOT)
            _gw = _rt.gateway
        except Exception as e:  # noqa: BLE001
            _gw_err = str(e)
    if _gw_err:
        st.error(f"models.toml could not be loaded: {_gw_err}")
    summary = _rui.settings_summary(gateway=_gw, ledger=getattr(_gw, "ledger", None),
                                    cache=getattr(_gw, "cache", None), budget=getattr(_gw, "budget", None),
                                    openrouter_key_present=key_present)
    if summary["cache"]:
        st.caption(f"request cache: {summary['cache']}")
    if summary["budget"]:
        b = summary["budget"]
        st.caption(f"effective limits: {b['limits']} · paused: {b['paused']}")
    if _gw is not None and st.button("Test connection (minimal-token health probe)"):
        st.write(_rui.test_connection(_gw))

    # ---- readiness (zero-key dry run) ----
    st.subheader("Readiness check (no model or network calls)")
    if st.button("Run readiness check"):
        from autograder.readiness import format_readiness, readiness_report
        _rep = readiness_report(models_config=_models_toml())
        st.code(format_readiness(_rep), language=None)
        with st.expander("readiness JSON"):
            st.json(_rep)
