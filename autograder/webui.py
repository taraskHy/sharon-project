"""Local web interface for the exam autograder (Streamlit).

Start with:  python -m autograder ui        (documented in the README)

Design notes:

- Grading runs in a DETACHED subprocess (``autograder run-job``), so closing
  this app never kills a batch; reopening shows the persisted job state.
- All state lives on disk in the job directory (see autograder/jobs.py);
  the UI only reads/writes those files and spawns/controls the runner.
- Model-visible inputs never contain original filenames or private paths —
  intake anonymizes exam copies; original names appear only in the UI tables.
"""

from __future__ import annotations

import ctypes
import json
import os
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

KNOWN_PACKAGE_DIRS = [REPO_ROOT / "prob_data", REPO_ROOT / "sample_data"]


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
    for d in KNOWN_PACKAGE_DIRS:
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


# --------------------------------------------------------------------------
# sidebar: backend configuration
# --------------------------------------------------------------------------

st.sidebar.title("📝 Exam Autograder")
st.sidebar.caption("Local open-model grading — no cloud APIs required.")

_toml_defaults = {}
_toml_path = REPO_ROOT / "grader.toml"
if _toml_path.exists():
    import tomllib

    _toml_defaults = tomllib.loads(_toml_path.read_text(encoding="utf-8"))

_b = _toml_defaults.get("backend", {})
_g = _toml_defaults.get("grading", {})

with st.sidebar:
    st.subheader("Model backend")
    base_url = st.text_input("Server URL", _b.get("base_url", "http://localhost:11434/v1"))
    model = st.text_input("Model", _b.get("model", "qwen3-vl:8b-instruct"))
    with st.expander("Advanced settings"):
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

tab_new, tab_jobs, tab_courses, tab_settings = st.tabs(
    ["➕ New batch", "📊 Jobs & results", "📚 Courses (experimental RAG)", "⚙ Models & OpenRouter"]
)


# ---------------------------------------------------------------------------
# Models & OpenRouter — settings/admin. The API key is read from the
# environment by the backend and is NEVER displayed, stored, or logged here.
# ---------------------------------------------------------------------------
with tab_settings:
    from autograder import reviewui as _rui

    st.caption(
        "Task → model routing comes from models.toml (copy models.example.toml). "
        "The OpenRouter key is read only from the OPENROUTER_API_KEY environment "
        "variable and is never shown or stored."
    )
    _models_path = REPO_ROOT / "models.toml"
    _gw = None
    _gw_err = None
    if _models_path.exists():
        try:
            from autograder.gateway import ModelGateway
            from autograder.requestcache import RequestCache
            from autograder.usage import UsageLedger

            from autograder.orchestrator import setup_from_config

            _rt = setup_from_config(_models_path, REPO_ROOT)
            _gw = _rt.gateway
        except Exception as e:  # noqa: BLE001
            _gw_err = str(e)
    key_present = bool(os.environ.get("OPENROUTER_API_KEY"))
    summary = _rui.settings_summary(
        gateway=_gw, ledger=getattr(_gw, "ledger", None), cache=getattr(_gw, "cache", None),
        budget=getattr(_gw, "budget", None), openrouter_key_present=key_present,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("OpenRouter", "enabled" if summary["openrouter_enabled"] else "disabled")
    c2.metric("API key in environment", "present" if key_present else "missing")
    c3.metric("models.toml", "loaded" if _gw else ("error" if _gw_err else "absent"))
    if _gw_err:
        st.error(f"models.toml could not be loaded: {_gw_err}")
    if summary["tasks"]:
        st.markdown("**Task → configured model**")
        st.table([{"task": t, **v} for t, v in summary["tasks"].items()])
    if summary["usage"]:
        u = summary["usage"]
        st.markdown("**Usage (ledger)**")
        m = st.columns(5)
        m[0].metric("Cloud requests", u.get("cloud_requests"))
        m[1].metric("Total tokens", u.get("total_tokens"))
        m[2].metric("Reported cost", u.get("reported_cost"))
        m[3].metric("Cache-hit rate", u.get("cache_hit_rate"))
        m[4].metric("% exams fully local", u.get("pct_exams_fully_local"))
    if summary["cache"]:
        st.caption(f"request cache: {summary['cache']}")
    if summary["budget"]:
        b = summary["budget"]
        st.markdown("**Budget (effective limits; — = unlimited)**")
        st.table([{"limit": k, "value": v} for k, v in b["limits"].items()])
        st.caption(f"spent this process: {b['input_tokens']} in / {b['output_tokens']} out tokens, "
                   f"cost {b['cost']}; paused: {b['paused']}" + (f" ({b['pause_reason']})" if b["paused"] else ""))
    if _gw is not None and st.button("Test connection (minimal-token health probe)"):
        st.write(_rui.test_connection(_gw))

# ---------------------------------------------------------------------------
# Courses — persistent course-material store for the EXPERIMENTAL
# qwen_rag_ocr_v1 arm. Material is uploaded once per course, indexed
# locally (Ollama embeddings), and reused across batches. This does NOT
# affect production grading.
# ---------------------------------------------------------------------------
with tab_courses:
    from autograder import courses as course_store

    st.caption(
        "Course summaries power the experimental OCR-repair arm only. "
        "Never upload answer keys or rubrics here — key-like filenames are "
        "refused automatically, and extracted content is screened for "
        "answer-key/rubric indicators (flagged files are refused unless you "
        "explicitly override below)."
    )
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
        # Uploads are ingested IMMEDIATELY (idempotent: same bytes -> same
        # file), so "Build index" always sees what the user just uploaded.
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
                (st.error if res.get("suspicious") else st.warning)(
                    f"{f.name}: {res['reason']}"
                )
            elif res.get("suspicious_override"):
                st.warning(f"{f.name}: ingested via operator override — "
                           f"{res['suspicious_override']}")

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
                        st.success(
                            f"index built: {manifest['n_chunks']} chunks, "
                            f"dim {manifest['dim']}, model {manifest['embed_model']}"
                        )
                        st.rerun()
                    except ValueError as e:
                        st.error(f"index build failed: {e}")
                    except Exception as e:  # noqa: BLE001
                        st.error(
                            f"embedding failed: {e} — is Ollama running with "
                            f"the '{course_store.DEFAULT_EMBED_MODEL}' model "
                            f"pulled? (ollama pull {course_store.DEFAULT_EMBED_MODEL})"
                        )


# --------------------------------------------------------------------------
# New batch
# --------------------------------------------------------------------------

with tab_new:
    st.header("Create a grading batch")

    packages = discover_packages()
    source = st.radio(
        "Exam configuration",
        ["Configured exam package", "Upload key & configuration"],
        horizontal=True,
        help=(
            "A package bundles the answer key with its template (grading modes, "
            "answer-sheet rule) and variant mapping. Upload mode lets you bring "
            "a new exam family."
        ),
    )

    key_path = None
    template_obj: ExamTemplate | None = None
    if source == "Configured exam package":
        if packages:
            choice = st.selectbox("Package", list(packages))
            pkg = packages[choice]
            key_path = pkg["key"]
            template_obj = pkg["template_obj"]
            tpl = template_obj
            st.info(
                f"**{tpl.name or tpl.template_id}** — mode: `{tpl.mode}`"
                + (
                    f", answer sheet fixed to page(s) {tpl.answer_sheet_pages}"
                    if tpl.answer_sheet_rule == "fixed_pages"
                    else ", answer sheets detected structurally"
                )
            )
            variants_path = key_path.with_name(key_path.stem + ".variants.json")
            if variants_path.exists():
                vcfg = json.loads(variants_path.read_text(encoding="utf-8"))
                with st.expander("Variant symbol mapping (authoritative)"):
                    st.table(
                        [
                            {
                                "marker": name,
                                "variant": entry["variant"],
                                "description": entry["description"],
                            }
                            for name, entry in vcfg.get("markers", {}).items()
                        ]
                    )
        else:
            st.warning("No configured packages found (looked for *.template.json).")
    else:
        up_key = st.file_uploader("Official answer key (PDF or parsed JSON)",
                                  type=["pdf", "json"])
        key_path = _stage_dir(up_key, "answer_key")
        up_rubric = st.file_uploader("Rubric / grading configuration (optional)",
                                     type=["txt", "md", "json"])
        rubric_path = _stage_dir(up_rubric, "rubric")
        st.session_state["rubric_path"] = str(rubric_path) if rubric_path else None

        mode = st.selectbox(
            "Exam mode",
            ["with_explanation", "multiple_choice", "mixed"],
            help=(
                "multiple_choice: selections only — no explanation transcription "
                "or judging model calls. mixed: configure per-question modes below."
            ),
        )
        question_modes: dict[str, str] = {}
        if mode == "mixed":
            st.caption("Per-question modes (question id → mode); example: "
                       '{"1": "multiple_choice", "2": "with_explanation"}')
            qm_text = st.text_area("question_modes JSON", "{}")
            try:
                question_modes = json.loads(qm_text)
            except json.JSONDecodeError as e:
                st.error(f"invalid JSON: {e}")
        sheet_rule = st.selectbox(
            "Answer-sheet rule",
            ["detected", "fixed_pages"],
            help=(
                "detected: a survey pass locates dedicated answer sheets "
                "structurally. fixed_pages: this exam family's answer sheet is "
                "a fixed page set (e.g. the first page) — question-page "
                "markings are ignored."
            ),
        )
        sheet_pages: list[int] = []
        if sheet_rule == "fixed_pages":
            pages_text = st.text_input("Answer-sheet page numbers (comma-separated)", "1")
            try:
                sheet_pages = [int(x) for x in pages_text.split(",") if x.strip()]
            except ValueError:
                st.error("page numbers must be integers")
        up_variants = st.file_uploader(
            "Variant mapping JSON (optional — marker → key column)", type=["json"]
        )
        variants_staged = _stage_dir(up_variants, "variants")
        if key_path is not None:
            template_obj = ExamTemplate(
                template_id=f"uploaded-{datetime.now():%Y%m%d%H%M%S}",
                name=Path(up_key.name).stem if up_key else "uploaded exam",
                mode=mode,
                question_modes=question_modes,
                answer_sheet_rule=sheet_rule,
                answer_sheet_pages=sheet_pages,
            )
            # Persist sidecars next to the staged key so the pipeline
            # auto-discovers them.
            tpl_path = key_path.with_name(key_path.stem + ".template.json")
            tpl_path.write_text(template_obj.model_dump_json(indent=1), encoding="utf-8")
            if variants_staged is not None:
                variants_staged.replace(
                    key_path.with_name(key_path.stem + ".variants.json")
                )

    exams_upload = st.file_uploader(
        "Student exams — one/many PDFs or images, or ZIP archive(s)",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "zip"],
        accept_multiple_files=True,
    )
    _courses_avail = [c["course_id"] for c in __import__("autograder.courses", fromlist=["courses"]).list_courses()]
    course_choice = st.selectbox(
        "Course (optional — experimental RAG context)",
        ["(none)"] + _courses_avail, key="batch_course",
    ) if _courses_avail else "(none)"
    mask = st.toggle(
        "Mask red instructor annotations before inference",
        value=False,
        help=(
            "Enable for scans that already carry instructor grading marks. "
            "Leave off for clean student submissions."
        ),
    )

    if st.button("Create job", type="primary", disabled=key_path is None or not exams_upload):
        # Package mode never routes uploads through _stage_dir, so the
        # session staging dir may not exist yet.
        staging = Path(st.session_state.setdefault(
            "staging_dir",
            str(jobs.jobs_root() / f"_staging-{datetime.now():%Y%m%d-%H%M%S}"),
        ))
        staged_exams = []
        for uploaded in exams_upload:
            p = staging / "exams_in" / uploaded.name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(uploaded.getvalue())
            staged_exams.append(p)
        rubric = st.session_state.get("rubric_path")
        job_dir = jobs.create_job(
                course_id=None if course_choice == "(none)" else course_choice,
            key=Path(key_path),
            exams=staged_exams,
            rubric=Path(rubric) if rubric else None,
            backend_args=backend_args,
            grading_args=grading_args,
            mask=mask,
        )
        st.session_state["selected_job"] = job_dir.name
        job = jobs.load_job(job_dir)
        st.success(
            f"Job **{job_dir.name}** created with {len(job['exams'])} exam(s). "
            "Switch to the Jobs tab to start grading."
        )
        for issue in job.get("intake_issues", []):
            st.warning(issue)


# --------------------------------------------------------------------------
# Jobs & results
# --------------------------------------------------------------------------

with tab_jobs:
    all_jobs = list_jobs()
    if not all_jobs:
        st.info("No jobs yet — create one in the New batch tab.")
        st.stop()

    names = [p.name for p in all_jobs]
    default_idx = names.index(st.session_state.get("selected_job")) if st.session_state.get("selected_job") in names else 0
    job_name = st.selectbox("Job", names, index=default_idx)
    job_dir = jobs.jobs_root() / job_name


    @st.fragment(run_every=3)
    def job_panel(job_dir: Path) -> None:
        job = jobs.load_job(job_dir)
        state = jobs.load_state(job_dir)
        alive = runner_alive(job_dir)
        exams = state["exams"]
        counts = {
            s: sum(1 for e in exams.values() if e["status"] == s)
            for s in ("pending", "running", "done", "failed")
        }
        review = sum(1 for e in exams.values() if (e.get("review_items") or 0) > 0)
        rate_limited = sum(
            1 for e in exams.values()
            if e.get("error") and ("429" in e["error"] or "rate" in e["error"].lower())
        )

        status = state["status"] if alive or state["status"] != "running" else "interrupted"
        st.subheader(f"Status: {status}" + (" 🟢 runner active" if alive else ""))

        c = st.columns(8)
        c[0].metric("Discovered", len(exams))
        c[1].metric("Pending", counts["pending"])
        c[2].metric("Processing", counts["running"])
        c[3].metric("Completed", counts["done"])
        c[4].metric("Failed", counts["failed"])
        c[5].metric("Rate-limited", rate_limited)
        c[6].metric("Needs review", review)
        c[7].metric("Elapsed", fmt_elapsed(state))

        current = state.get("current")
        if current:
            stage = exams[current].get("stage") or "…"
            st.caption(f"Currently processing **{current}** — stage: {stage}")
        done_results = [
            json.loads((job_dir / "exams" / a / "result.json").read_text(encoding="utf-8"))
            for a, e in exams.items()
            if e["status"] == "done" and (job_dir / "exams" / a / "result.json").exists()
        ]
        key_sources = {
            (r.get("backend_info") or {}).get("answer_key_source") for r in done_results
        } - {None}
        st.caption(
            f"Backend: `{job['backend_args'].get('--model', '?')}` at "
            f"`{job['backend_args'].get('--base-url', '?')}` · "
            "Answer-key cache: "
            + (
                "reused" if key_sources <= {"cache", "json", "resume"} and key_sources
                else "parsed this batch" if "parsed" in key_sources
                else "—"
            )
        )
        total = max(len(exams), 1)
        st.progress(counts["done"] / total)

        b = st.columns(5)
        if b[0].button("▶ Start / Resume", disabled=alive,
                       help="Runs pending and failed exams; finished work is kept."):
            jobs.clear_control_requests(job_dir)
            spawn_runner(job_dir)
            st.rerun()
        if b[1].button("⏸ Pause", disabled=not alive,
                       help="Finish nothing mid-flight: the current exam is stopped "
                            "safely and stays pending."):
            jobs.request_pause(job_dir)
        if b[2].button("⏹ Stop", disabled=not alive):
            jobs.request_stop(job_dir)
        if b[3].button("🔄 Refresh now"):
            st.rerun()
        if b[4].button("📦 Rebuild combined reports"):
            jobs.combine_outputs(job_dir)
            st.success("combined reports rebuilt")

        st.dataframe(
            [
                {
                    "exam": anon,
                    "original file": e.get("original_name"),
                    "status": e["status"],
                    "stage": e.get("stage"),
                    "variant": e.get("variant"),
                    "score": e.get("predicted"),
                    "review items": e.get("review_items"),
                    "unanswered": e.get("unanswered"),
                    "runtime (s)": e.get("runtime_s"),
                    "error": (e.get("error") or "").splitlines()[0] if e.get("error") else None,
                }
                for anon, e in sorted(exams.items())
            ],
            width="stretch",
        )

    job_panel(job_dir)

    state = jobs.load_state(job_dir)

    # ---- package preflight: ONE setup warning instead of one review per student ----
    st.divider()
    st.subheader("Package setup")
    _pk = job_dir / "uploads" / "answer_key.json"
    if not _pk.exists():
        st.caption("Package checks appear once the answer key has been parsed.")
    else:
        try:
            from autograder.key_parser import load_answer_key
            from autograder.preflight import (READY, alignment_from_discovery,
                                              preflight_package, reviews_avoided)

            _k = load_answer_key(_pk)
            _al_path = job_dir / "uploads" / "answer_key.alignment.json"
            _al = json.loads(_al_path.read_text(encoding="utf-8")) if _al_path.exists() else None
            _rep = preflight_package(
                key=_k, variants=list(_k.versions),
                alignment=alignment_from_discovery(_al, list(_k.versions), _k))
            if _rep.status == READY:
                st.success(_rep.summary())
            else:
                st.error(f"**{_rep.status}** — fix these once, at package level. "
                         f"Left unresolved they would produce about "
                         f"{reviews_avoided(_rep, len(state['exams']))} individual review items.")
                st.dataframe([{"code": f.code, "what": f"{f.subject} {f.subject_id}",
                               "problem": f.message, "needed": f.needed}
                              for f in _rep.blocking], width="stretch")
            if _rep.warnings:
                with st.expander(f"Package warnings ({len(_rep.warnings)}) — not blocking"):
                    st.dataframe([{"code": f.code, "what": f"{f.subject} {f.subject_id}",
                                   "note": f.message} for f in _rep.warnings], width="stretch")
        except Exception as exc:  # noqa: BLE001 — never block the UI on a check
            st.caption(f"Package check unavailable: {type(exc).__name__}")

    # ---- pre-run ESTIMATE (never mixed with the ledger's actual usage) ----
    st.divider()
    st.subheader("Estimated cloud usage")
    _key_path = job_dir / "uploads" / "answer_key.json"
    if not _key_path.exists():
        st.caption("The estimate appears once the answer key has been parsed "
                   "(uploads/answer_key.json).")
    else:
        try:
            from autograder.estimate import estimate_job, load_pricing
            from autograder.key_parser import load_answer_key

            _key = load_answer_key(_key_path)
            _pol_path = job_dir / "uploads" / "answer_key.policies.json"
            _pol = json.loads(_pol_path.read_text(encoding="utf-8")) if _pol_path.exists() else None
            _gw = st.session_state.get("gateway")
            _est = estimate_job(key=_key, exams=len(state["exams"]), policies=_pol, gateway=_gw,
                                pricing=load_pricing(getattr(_gw, "pricing_config", None)
                                                     and {"pricing": _gw.pricing_config}))
            e = st.columns(5)
            e[0].metric("ESTIMATED cloud calls", f"{_est.estimated_cloud_calls:g}")
            e[1].metric("ESTIMATED input tokens", f"{_est.estimated_input_tokens:,}")
            e[2].metric("ESTIMATED output tokens", f"{_est.estimated_output_tokens:,}")
            e[3].metric("ESTIMATED cost",
                        "—" if _est.estimated_cost is None else f"${_est.estimated_cost:,.2f}")
            e[4].metric("Deterministic items",
                        f"{_est.sub_items_deterministic}/{_est.sub_items_total}")
            st.caption("⚠ " + _est.disclaimer
                       + (f" ({_est.cost_unavailable_reason})" if _est.cost_unavailable_reason else "")
                       + f" Escalation assumptions: {_est.assumptions['source']}.")
            with st.expander("Estimate breakdown (per call type)"):
                st.dataframe([{"call type": k, "ESTIMATED calls": v}
                              for k, v in _est.estimated_calls.items()], width="stretch")
        except Exception as exc:  # noqa: BLE001 — an estimate must never block the UI
            st.caption(f"Estimate unavailable: {type(exc).__name__}")

    # ---- batch-level view: systemic warnings first, then the review queue ----
    st.divider()
    st.subheader("Batch checks")
    from autograder import reviewui as _rui

    _results = _rui.load_job_results(job_dir)
    if not _results:
        st.caption("No graded exams yet — batch checks appear once results exist.")
    else:
        _extractions = {}
        for _eid in _results:
            _ep = job_dir / "exams" / _eid / "extraction.json"
            if _ep.exists():
                try:
                    _extractions[_eid] = json.loads(_ep.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    pass
        _ov = _rui.batch_overview(_results, _extractions)
        _sev = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
        if _ov["warnings"]:
            st.error(f"{len(_ov['warnings'])} batch-level warning(s) — check these BEFORE "
                     "working through individual reviews: one cause here can explain many of them.")
            st.dataframe([
                {"severity": _sev.get(w["severity"], "") + " " + w["severity"],
                 "code": w["code"], "scope": f"{w['scope']} {w['scope_id']}",
                 "students": w["affected_students"], "items": w["affected_items"],
                 "what it means": w["explanation"]}
                for w in _ov["warnings"]], width="stretch")
        else:
            st.success("No batch-level anomaly detected (variant mix, blank/crop/OCR rates, "
                       "score distribution and alignment all look ordinary for this batch).")
        _s = _ov["summary"]
        m = st.columns(4)
        m[0].metric("Review cases", _s["cases"])
        m[1].metric("Decisions needed", _s["decisions_required"],
                    help="Cases sharing an exact mechanical cause are resolved by ONE decision.")
        m[2].metric("Absorbed by grouping", _s["cases_absorbed_by_grouping"])
        m[3].metric("Graded exams", _ov["exams"])
        if _s["cases"]:
            with st.expander("Review queue (priority order, grouped)"):
                st.caption("Priority affects ORDER only — it never changes a grade.")
                st.dataframe([
                    {"priority": g["explanation"].splitlines()[0] if g["explanation"] else "",
                     "reason": g["reason_code"], "cases": g["size"],
                     "students": ", ".join(g["students"][:4]),
                     "one decision covers all": g["apply_to_all_eligible"],
                     "scope": g["scope"]}
                    for g in _ov["groups"]], width="stretch")

    st.divider()
    st.subheader("Exam details")
    exam_ids = sorted(state["exams"])
    picked = st.selectbox("Exam", exam_ids)
    exam_dir = job_dir / "exams" / picked
    result_path = exam_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        survey_path = exam_dir / "survey.json"
        sheet_pages = []
        if survey_path.exists():
            survey = json.loads(survey_path.read_text(encoding="utf-8"))
            sheet_pages = survey.get("answer_sheet_policy", {}).get("authoritative_pages", [])
        extraction_conf: dict[tuple[str, str], float] = {}
        extraction_path = exam_dir / "extraction.json"
        if extraction_path.exists():
            extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
            for qx in extraction.get("questions", []):
                for s in qx.get("sub_items", []):
                    extraction_conf[(qx["question_id"], s["sub_item_id"])] = s.get("confidence")

        meta = st.columns(4)
        meta[0].metric("Total", f"{result['total_awarded']:g}/{result['total_max']:g}")
        meta[1].metric("Variant", result.get("detected_version") or "—")
        meta[2].metric("Answer-sheet pages", ", ".join(map(str, sheet_pages)) or "—")
        meta[3].metric("Review items", len(result.get("needs_human_review", [])))
        vd = result.get("variant_detection")
        if vd:
            st.caption(
                f"Variant marker: saw {vd.get('marker_seen')!r} → matched "
                f"`{vd.get('matched_marker')}` (confident: {vd.get('confident')})"
            )

        rows = []
        for q in result.get("questions", []):
            for s in q.get("sub_results", []):
                rows.append(
                    {
                        "question": q["question_id"],
                        "item": s["sub_item_id"],
                        "answer": s.get("student_answer"),
                        "correct": s.get("selection_correct"),
                        "points": s.get("points_total"),
                        "max": s.get("points_max"),
                        "confidence": extraction_conf.get(
                            (q["question_id"], s["sub_item_id"])
                        ),
                        "status": s.get("status"),
                        "explanation": (s.get("explanation_transcription") or "")[:80] or None,
                        "needs review": s.get("needs_review"),
                        "reason": s.get("reason", "")[:160],
                    }
                )
        st.dataframe(rows, width="stretch")

        ambiguous = [r for r in rows if r["status"] == "ambiguous"]
        if ambiguous:
            st.warning(f"Ambiguous items: {[(r['question'], r['item']) for r in ambiguous]}")
        if result.get("needs_human_review"):
            with st.expander("Human-review reasons"):
                for item in result["needs_human_review"]:
                    st.markdown(f"- **{item['question_id']}.{item['sub_item_id']}** — {item['reason']}")
            # Fast REVIEW: one-click resolutions persisted per exam (never edits result.json)
            from autograder import reviewui as _rui2

            ext_path = exam_dir / "extraction.json"
            extraction = json.loads(ext_path.read_text(encoding="utf-8")) if ext_path.exists() else None
            rstore = _rui2.ResolutionStore(exam_dir)
            resolved = rstore.load()
            with st.expander(f"⚡ Quick review ({len(result['needs_human_review'])} items, "
                             f"{len(resolved)} resolved)"):
                for it in _rui2.build_review_items(picked, result, extraction):
                    key = f"{it.question_id}:{it.sub_item_id}"
                    done = resolved.get(key)
                    st.markdown(f"**`{it.reason_code}` {key}** — {it.points_affected:g} pt(s) at "
                                f"stake · priority tier {it.priority_tier}"
                                + (f"  ✅ *{done['decision']}*" if done else ""))
                    if it.explanation:
                        st.code(it.explanation, language=None)
                    if it.batch_warning_code:
                        st.warning(f"A batch-level issue ({it.batch_warning_code}) affects "
                                   f"{it.batch_warning_students} students and may explain this "
                                   "item — resolve that first.")
                    if it.kind == "mc":
                        st.caption(f"deterministic candidates: {it.deterministic_candidate} | "
                                   f"local: {it.local_candidate} | cloud: {it.cloud_candidate}")
                    elif it.kind == "ocr":
                        st.caption(f"primary: {it.primary_transcription!r} | secondary: {it.secondary_transcription!r}")
                    elif it.kind == "grading":
                        st.caption(f"selected {it.selected_option} | proposed {it.proposed_score}/{it.max_score} "
                                   f"| rubric {it.rubric_items} | {it.grading_evidence}")
                    cols = st.columns(max(1, min(4, len(it.options))))
                    for i, opt in enumerate(it.options[:4]):
                        if cols[i].button(opt, key=f"rv_{picked}_{key}_{i}"):
                            rstore.resolve(it.question_id, it.sub_item_id, decision=opt)
                            st.rerun()
                    if it.apply_to_all_eligible:
                        st.caption("this cause is exactly mechanical — one decision can be applied "
                                   "to every identical case (apply-to-all)")

        # ---- shadow mode: a recorded proposal, never an applied grade ----
        _shadow_path = exam_dir / "shadow_comparison.json"
        if _shadow_path.exists():
            _cmp = json.loads(_shadow_path.read_text(encoding="utf-8"))
            _agree = _cmp.get("agreement", {})
            with st.expander("🧪 Shadow comparison (recorded only — NOT applied)"):
                st.info(_cmp.get("note", ""))
                s = st.columns(4)
                s[0].metric("Exact score agreement", f"{_agree.get('exact_score_agreement', 0)}%")
                s[1].metric("Mean |delta|", _agree.get("mean_abs_delta"))
                s[2].metric("Legacy review items", _agree.get("legacy_review_items"))
                s[3].metric("Reliability review items", _agree.get("reliability_review_items"))
                st.caption(f"Authoritative: {_cmp.get('authoritative')} · totals "
                           f"{_cmp.get('totals', {}).get('legacy_total')} (applied) vs "
                           f"{_cmp.get('totals', {}).get('reliability_total')} (proposal)")
                st.dataframe([
                    {"question": i["question_id"], "item": i["sub_item_id"],
                     "legacy": i["legacy_points"], "proposal": i["reliability_points"],
                     "delta": i["score_delta"], "legacy review": i["legacy_review"],
                     "reliability state": i["reliability_state"],
                     "reason": i["reliability_reason_code"], "route": i["route_difference"]}
                    for i in _cmp.get("items", [])], width="stretch")

        # ---- decision trace: why did this item get this grade? ----
        with st.expander("🔍 Why was this graded this way?"):
            from autograder import reviewui as _rui3

            trace_rows = [(q["question_id"], s["sub_item_id"])
                          for q in result.get("questions", [])
                          for s in q.get("sub_results", [])]
            if trace_rows:
                labels = [f"{q}.{s}" for q, s in trace_rows]
                chosen = st.selectbox("Item", labels, key=f"trace_{picked}")
                qid, sid = trace_rows[labels.index(chosen)]
                st.code(_rui3.decision_trace_for(exam_dir, result, qid, sid), language=None)

        d = st.columns(2)
        d[0].download_button(
            "⬇ result.json", result_path.read_bytes(),
            file_name=f"{picked}-result.json", mime="application/json",
        )
        report_path = exam_dir / "report.md"
        if report_path.exists():
            d[1].download_button(
                "⬇ report.md", report_path.read_bytes(),
                file_name=f"{picked}-report.md", mime="text/markdown",
            )
    else:
        e = state["exams"][picked]
        st.info(f"No result yet — status: {e['status']}")
        if e.get("error"):
            st.code(e["error"])
        log_path = exam_dir / "grade.log"
        if log_path.exists():
            with st.expander("Processing log (tail)"):
                st.code("\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                ))

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
            jobs.combine_outputs(job_dir)
        if p.exists():
            col.download_button(label, p.read_bytes(), file_name=f"{job_name}-{name}", mime=mime)
