"""Persistent, resumable batch-grading jobs.

A *job* is a directory holding everything about one grading batch:

    <job_dir>/
        job.json                 configuration (key/template paths, backend
                                 flags, grading options) — no secrets
        state.json               live per-exam status, written atomically
        uploads/exams/<anon>.pdf anonymized copies of the student exams
                                 (model-visible names carry no grades/paths)
        uploads/name_map.json    anon id -> original filename (operator-only)
        exams/<anon>/            per-exam pipeline output (result.json, ...)
        combined_results.csv/json, summary.md, reports.zip
        stop.request             sentinel: finish current exam, then stop
        pause.request            sentinel: same, but state stays 'paused'

Each exam is graded by a **subprocess** (``python -m autograder grade``), so
a stop request can terminate a hung model call safely, a crash loses only
the current exam's unfinished stages (per-stage fingerprints already guard
partial output), and closing the UI never kills a running batch. Resuming a
job re-runs pending/failed/stopped exams with ``--resume``, reusing every
finished stage whose input fingerprint still matches.

Grade-label isolation: the runner never reads expected grades; evaluation
scripts join predictions with labels only after results are saved.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

EXAM_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FINAL_STATES = {"done", "failed"}


def jobs_root() -> Path:
    return Path(os.environ.get("GRADER_JOBS_DIR", "jobs"))


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# intake
# --------------------------------------------------------------------------


@dataclass
class IntakeResult:
    entries: list[dict] = field(default_factory=list)  # {anon_id, original_name}
    issues: list[str] = field(default_factory=list)


def _anonymize_name(original: str, index: int) -> str:
    return f"exam-{index:03d}"


def intake_exams(job_dir: Path, sources: list[Path]) -> IntakeResult:
    """Copy exam files (and ZIP contents) into the job under anonymized
    names. Rejects malformed ZIPs, non-exam files, and duplicate originals.

    Model-visible paths are ONLY the anonymized copies; original filenames
    (which may encode grades or private paths) stay in name_map.json, which
    is never part of any model input.
    """
    result = IntakeResult()
    exams_dir = job_dir / "uploads" / "exams"
    exams_dir.mkdir(parents=True, exist_ok=True)

    collected: list[tuple[str, bytes]] = []
    seen_names: set[str] = set()

    def add(original_name: str, data: bytes) -> None:
        stem_key = Path(original_name).name.lower()
        if stem_key in seen_names:
            result.issues.append(f"duplicate exam file skipped: {original_name}")
            return
        if Path(original_name).suffix.lower() not in EXAM_SUFFIXES:
            result.issues.append(f"unsupported file type skipped: {original_name}")
            return
        if not data:
            result.issues.append(f"empty file skipped: {original_name}")
            return
        seen_names.add(stem_key)
        collected.append((Path(original_name).name, data))

    for src in sources:
        src = Path(src)
        if not src.exists():
            result.issues.append(f"missing input: {src.name}")
            continue
        if src.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(src) as zf:
                    bad = zf.testzip()
                    if bad is not None:
                        result.issues.append(f"ZIP {src.name}: corrupt member {bad!r}")
                        continue
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        # basename only: no zip-slip, no private path segments
                        name = Path(info.filename).name
                        if name.startswith("._") or name.startswith("__MACOSX"):
                            continue
                        add(name, zf.read(info))
            except zipfile.BadZipFile as e:
                result.issues.append(f"malformed ZIP {src.name}: {e}")
        else:
            add(src.name, src.read_bytes())

    collected.sort(key=lambda item: item[0])
    for i, (name, data) in enumerate(collected, start=1):
        anon = _anonymize_name(name, i)
        target = exams_dir / f"{anon}{Path(name).suffix.lower()}"
        target.write_bytes(data)
        result.entries.append({"anon_id": anon, "original_name": name,
                               "file": str(target.relative_to(job_dir)).replace("\\", "/")})

    name_map = {e["anon_id"]: e["original_name"] for e in result.entries}
    _atomic_write(job_dir / "uploads" / "name_map.json", name_map)
    return result


# --------------------------------------------------------------------------
# job lifecycle
# --------------------------------------------------------------------------


def create_job(
    *,
    key: Path,
    exams: list[Path],
    rubric: Path | None = None,
    template: Path | None = None,
    variant_map: Path | None = None,
    alignment_map: Path | None = None,
    backend_args: dict | None = None,
    grading_args: dict | None = None,
    mask: bool = True,
    job_root: Path | None = None,
    job_id: str | None = None,
    course_id: str | None = None,
    grading_mode: str | None = None,
) -> Path:
    """Materialise a new job directory. ``backend_args``/``grading_args`` are
    CLI flag name -> value mappings (e.g. {"--model": "qwen3-vl:8b-instruct"}).
    Uploaded documents are copied INTO the job so later runs don't depend on
    the original upload locations."""
    root = Path(job_root) if job_root else jobs_root()
    job_id = job_id or _dt.datetime.now().strftime("job-%Y%m%d-%H%M%S")
    job_dir = root / job_id
    if job_dir.exists():
        raise FileExistsError(f"job directory already exists: {job_dir}")
    docs_dir = job_dir / "uploads"
    docs_dir.mkdir(parents=True)

    def take(source: Path | None, name: str) -> str | None:
        if source is None:
            return None
        target = docs_dir / f"{name}{Path(source).suffix.lower()}"
        target.write_bytes(Path(source).read_bytes())
        return str(target.relative_to(job_dir)).replace("\\", "/")

    key_rel = take(key, "answer_key")
    # Sidecars auto-discover by key stem, so keep their conventional names
    # next to the copied key.
    for sidecar, explicit in (
        ("variants", variant_map),
        ("alignment", alignment_map),
        ("template", template),
    ):
        src = Path(explicit) if explicit else Path(key).with_name(
            Path(key).stem + f".{sidecar}.json"
        )
        if src.exists():
            (docs_dir / f"answer_key.{sidecar}.json").write_bytes(src.read_bytes())
    rubric_rel = take(rubric, "rubric")

    intake = intake_exams(job_dir, exams)

    job = {
        "job_id": job_id,
        "created_at": _now(),
        "key": key_rel,
        "rubric": rubric_rel,
        "mask": bool(mask),
        "course_id": course_id,  # informational; --course in grading_args is authoritative
        # Informational display field. The AUTHORITATIVE propagation is the
        # flags in grading_args (--grading-mode/--models-config/--rag-policy),
        # forwarded verbatim to every grade subprocess. Old jobs without any
        # of these run exactly as before (legacy defaults).
        "grading_mode": grading_mode or "legacy",
        "backend_args": backend_args or {},
        "grading_args": grading_args or {},
        "intake_issues": intake.issues,
        "exams": intake.entries,
    }
    _atomic_write(job_dir / "job.json", job)

    state = {
        "job_id": job_id,
        "status": "created",
        "created_at": _now(),
        "updated_at": _now(),
        "started_at": None,
        "finished_at": None,
        "current": None,
        "exams": {
            e["anon_id"]: {
                "status": "pending",
                "original_name": e["original_name"],
                "file": e["file"],
                "predicted": None,
                "review_items": None,
                "unanswered": None,
                "variant": None,
                "runtime_s": None,
                "error": None,
            }
            for e in intake.entries
        },
    }
    _atomic_write(job_dir / "state.json", state)
    return job_dir


def load_job(job_dir: Path) -> dict:
    return json.loads((Path(job_dir) / "job.json").read_text(encoding="utf-8"))


def load_state(job_dir: Path) -> dict:
    return json.loads((Path(job_dir) / "state.json").read_text(encoding="utf-8"))


def _save_state(job_dir: Path, state: dict) -> None:
    state["updated_at"] = _now()
    _atomic_write(Path(job_dir) / "state.json", state)


def request_stop(job_dir: Path) -> None:
    (Path(job_dir) / "stop.request").write_text("stop", encoding="utf-8")


def request_pause(job_dir: Path) -> None:
    (Path(job_dir) / "pause.request").write_text("pause", encoding="utf-8")


def clear_control_requests(job_dir: Path) -> None:
    for name in ("stop.request", "pause.request"):
        p = Path(job_dir) / name
        if p.exists():
            p.unlink()


def _control_state(job_dir: Path) -> str | None:
    if (Path(job_dir) / "stop.request").exists():
        return "stopped"
    if (Path(job_dir) / "pause.request").exists():
        return "paused"
    return None


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def _grade_command(job: dict, job_dir: Path, exam_entry: dict) -> list[str]:
    cmd = [
        sys.executable, "-m", "autograder", "grade",
        "--exam", str(job_dir / exam_entry["file"]),
        "--key", str(job_dir / job["key"]),
        "--out", str(job_dir / "exams" / exam_entry["anon_id"]),
        "--resume",
    ]
    if job.get("rubric"):
        cmd += ["--rubric", str(job_dir / job["rubric"])]
    if job.get("mask"):
        cmd += ["--mask"]
    for flag, value in {**job.get("backend_args", {}), **job.get("grading_args", {})}.items():
        if value is None or value is False:
            continue
        if value is True:
            cmd.append(flag)
        else:
            cmd += [flag, str(value)]
    return cmd


_STAGE_PATTERNS = [
    (re.compile(r"parsing answer key|answer key reused|loading structured answer key"), "answer key"),
    (re.compile(r"detecting exam variant|variant:"), "variant detection"),
    (re.compile(r"survey synthesized|surveying the document"), "survey"),
    (re.compile(r"close-reading answer-sheet"), "sheet close-read"),
    (re.compile(r"extracting question"), "extraction"),
    (re.compile(r"judging explanations"), "explanation judging"),
    (re.compile(r"wrote .*report\.md"), "report"),
]


def _stage_from_line(line: str) -> str | None:
    for pattern, stage in _STAGE_PATTERNS:
        if pattern.search(line):
            return stage
    return None


def _package_setup_blocked(log_tail: str) -> bool:
    """A PackageSetupRequired failure is structural: every remaining exam in
    the batch would fail identically, so the job stops after the FIRST one —
    one package blocker, never N per-student failures."""
    return "PackageSetupRequired" in (log_tail or "")


def run_job(job_dir: str | Path, poll_interval: float = 0.5) -> int:
    """Run every pending exam sequentially. Returns 0 when the batch reached
    a terminal state (finished/stopped/paused) without runner-level errors."""
    job_dir = Path(job_dir)
    job = load_job(job_dir)
    state = load_state(job_dir)
    # Starting the runner expresses intent to run: stale control files from a
    # previous stop/pause must not immediately kill the new run.
    clear_control_requests(job_dir)
    state["status"] = "running"
    state["started_at"] = state.get("started_at") or _now()
    _save_state(job_dir, state)

    for entry in job["exams"]:
        anon = entry["anon_id"]
        exam_state = state["exams"][anon]
        if exam_state["status"] == "done":
            continue  # failed/pending/stopped exams are (re)tried
        control = _control_state(job_dir)
        if control:
            state["status"] = control
            state["current"] = None
            _save_state(job_dir, state)
            clear_control_requests(job_dir)
            return 0

        state["current"] = anon
        exam_state["status"] = "running"
        exam_state["stage"] = "starting"
        exam_state["error"] = None
        _save_state(job_dir, state)

        out_dir = job_dir / "exams" / anon
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "grade.log"
        t0 = time.monotonic()
        cmd = _grade_command(job, job_dir, entry)
        interrupted = False
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== {_now()} :: {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            assert proc.stdout is not None
            lines: queue.Queue[str | None] = queue.Queue()

            def _reader(stream, q):
                for line in stream:  # blocking readline in a daemon thread
                    q.put(line)
                q.put(None)

            reader = threading.Thread(target=_reader, args=(proc.stdout, lines), daemon=True)
            reader.start()

            eof = False
            while True:
                try:
                    while True:
                        line = lines.get_nowait()
                        if line is None:
                            eof = True
                            break
                        log.write(line)
                        log.flush()
                        stage = _stage_from_line(line)
                        if stage and stage != exam_state.get("stage"):
                            exam_state["stage"] = stage
                            _save_state(job_dir, state)
                except queue.Empty:
                    pass
                if eof and proc.poll() is not None:
                    break
                control = _control_state(job_dir)
                if control:
                    interrupted = True
                    log.write(f"\n=== {_now()} :: {control} requested — terminating child\n")
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                time.sleep(poll_interval)

        exam_state["runtime_s"] = round(time.monotonic() - t0, 1)
        if interrupted:
            exam_state["status"] = "pending"  # finished stages are fingerprint-safe
            exam_state["stage"] = None
            control = _control_state(job_dir)
            state["status"] = control or "stopped"
            state["current"] = None
            _save_state(job_dir, state)
            clear_control_requests(job_dir)
            return 0

        result_path = out_dir / "result.json"
        if proc.returncode == 0 and result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                exam_state["status"] = "done"
                exam_state["stage"] = None
                exam_state["predicted"] = result.get("total_awarded")
                exam_state["total_max"] = result.get("total_max")
                exam_state["review_items"] = len(result.get("needs_human_review", []))
                exam_state["unanswered"] = len(result.get("unanswered", []))
                exam_state["variant"] = result.get("detected_version")
            except (json.JSONDecodeError, OSError) as e:
                exam_state["status"] = "failed"
                exam_state["error"] = f"result.json unreadable: {e}"
        else:
            tail = ""
            try:
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
                )
            except OSError:
                pass
            exam_state["status"] = "failed"
            exam_state["error"] = (
                f"grade subprocess exited {proc.returncode}; log tail:\n{tail}"
            )
            if _package_setup_blocked(tail):
                state["current"] = None
                state["status"] = "failed"
                state["error"] = (
                    "package setup required — a structural package defect would "
                    "fail every exam identically; fix the package (see "
                    f"exams/{anon}/grade.log), then run the job again"
                )
                _save_state(job_dir, state)
                return 0
        state["current"] = None
        _save_state(job_dir, state)

    combine_outputs(job_dir)
    state = load_state(job_dir)
    state["status"] = "finished"
    state["finished_at"] = _now()
    state["current"] = None
    _save_state(job_dir, state)
    return 0


# --------------------------------------------------------------------------
# combined outputs
# --------------------------------------------------------------------------


def combine_outputs(job_dir: str | Path) -> dict[str, Path]:
    """Write combined CSV/JSON/Markdown + a ZIP of every per-exam report."""
    job_dir = Path(job_dir)
    state = load_state(job_dir)
    rows = []
    for anon, ex in sorted(state["exams"].items()):
        row = {
            "exam": anon,
            "status": ex["status"],
            "variant": ex.get("variant"),
            "predicted": ex.get("predicted"),
            "total_max": ex.get("total_max"),
            "review_items": ex.get("review_items"),
            "unanswered": ex.get("unanswered"),
            "runtime_s": ex.get("runtime_s"),
            "error": (ex.get("error") or "").splitlines()[0] if ex.get("error") else None,
        }
        rows.append(row)

    combined = {
        "job_id": state["job_id"],
        "generated_at": _now(),
        "counts": {
            status: sum(1 for r in rows if r["status"] == status)
            for status in ("done", "failed", "pending", "running")
        },
        "exams": rows,
    }
    json_path = job_dir / "combined_results.json"
    _atomic_write(json_path, combined)

    csv_path = job_dir / "combined_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["exam"])
        writer.writeheader()
        writer.writerows(rows)

    md = [
        f"# Batch report — {state['job_id']}",
        "",
        f"- Generated: {combined['generated_at']}",
        f"- Done: {combined['counts']['done']} | failed: {combined['counts']['failed']} | "
        f"pending: {combined['counts']['pending']}",
        "",
        "| Exam | Status | Variant | Score | Review items | Unanswered | Runtime (s) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        score = (
            f"{r['predicted']:g}/{r['total_max']:g}"
            if r["predicted"] is not None and r.get("total_max")
            else "—"
        )
        md.append(
            f"| {r['exam']} | {r['status']} | {r['variant'] or '—'} | {score} | "
            f"{r['review_items'] if r['review_items'] is not None else '—'} | "
            f"{r['unanswered'] if r['unanswered'] is not None else '—'} | "
            f"{r['runtime_s'] if r['runtime_s'] is not None else '—'} |"
        )
    md_path = job_dir / "summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    zip_path = job_dir / "reports.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(md_path, "summary.md")
        zf.write(json_path, "combined_results.json")
        zf.write(csv_path, "combined_results.csv")
        for anon in sorted(state["exams"]):
            for name in ("result.json", "report.md"):
                p = job_dir / "exams" / anon / name
                if p.exists():
                    zf.write(p, f"{anon}/{name}")
    return {"json": json_path, "csv": csv_path, "summary": md_path, "zip": zip_path}
