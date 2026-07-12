"""Command-line interface.

Usage:

    autograder doctor    --backend openai --base-url http://localhost:11434/v1 --model qwen2.5vl:7b
    autograder parse-key --key sample_data/Exam_solution.pdf --out out/ [backend flags]
    autograder grade     --exam sample_data/student_exam.pdf \
                         --key sample_data/Exam_solution.pdf \
                         [--rubric rubric.txt] [--out out/] [--version auto] [--resume]

Backend selection (provider-independent — see docs/deployment.md):

    --backend openai     any OpenAI-compatible server: Ollama, vLLM, TGI,
                         llama.cpp server, LM Studio, OpenRouter, Groq, ...
    --backend mock       offline fixtures (tests / plumbing checks)
    --backend anthropic  optional, development comparison only

Backend flags may also come from a TOML file via --config (CLI flags win):

    [backend]
    backend = "openai"
    model = "qwen2.5vl:7b"
    base_url = "http://localhost:11434/v1"
    structured_mode = "json_schema"
    timeout_s = 600.0

    [grading]
    max_image_edge = 1600
    max_tokens = 8000

``grade`` runs the full pipeline: key parsing -> survey -> extraction ->
explanation judging -> deterministic scoring. Every intermediate stage is
written to the output directory as JSON, and ``--resume`` reuses stage output
only when an input fingerprint (exam bytes, key bytes, rubric, backend, model,
generation config, render size) is unchanged.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import tomllib
from pathlib import Path

from .backends import BackendConfig, BackendError, VisionBackend, create_backend
from .config import GraderConfig
from .grade import PipelineStateError, detect_version, grade_exam, judge_all
from .ingest import IMAGE_SUFFIXES, load_pages
from .key_parser import load_answer_key, parse_answer_key, save_answer_key
from .report import render_markdown
from .schema import AnswerKey, ExamExtraction, ExamSurvey
from .extract import extract_exam
from .survey import survey_exam


def _log(msg: str) -> None:
    print(f"[autograder] {msg}", flush=True)


def _write_json(obj, path: Path) -> None:
    path.write_text(
        json.dumps(obj.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(f"wrote {path}")


# --------------------------------------------------------------------------
# configuration resolution: CLI flag > TOML file > hard default
# --------------------------------------------------------------------------

_BACKEND_DEFAULTS = {
    "backend": "openai",
    "model": "",
    "base_url": None,
    "api_key_env": "GRADER_API_KEY",
    "structured_mode": "json_schema",
    "temperature": 0.0,
    "timeout_s": 300.0,
    "transport_retries": 2,
    "validation_retries": 2,
    "concurrency": 1,
}
_GRADING_DEFAULTS = {
    "max_image_edge": GraderConfig.max_image_long_edge,
    "max_tokens": 16000,
}


def _load_toml(path: str | None) -> dict:
    if not path:
        return {}
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def resolve_config(args) -> tuple[BackendConfig, int]:
    """Build (BackendConfig, max_image_edge) from CLI + optional TOML file."""
    file_cfg = _load_toml(getattr(args, "config", None))
    b = file_cfg.get("backend", {})
    g = file_cfg.get("grading", {})

    def pick(cli_value, table: dict, key: str, defaults: dict):
        if cli_value is not None:
            return cli_value
        return table.get(key, defaults[key])

    backend_config = BackendConfig(
        backend=pick(args.backend, b, "backend", _BACKEND_DEFAULTS),
        model=pick(args.model, b, "model", _BACKEND_DEFAULTS),
        base_url=pick(args.base_url, b, "base_url", _BACKEND_DEFAULTS),
        api_key_env=pick(args.api_key_env, b, "api_key_env", _BACKEND_DEFAULTS),
        structured_mode=pick(args.structured_mode, b, "structured_mode", _BACKEND_DEFAULTS),
        max_tokens=pick(args.max_tokens, g, "max_tokens", _GRADING_DEFAULTS),
        temperature=pick(args.temperature, b, "temperature", _BACKEND_DEFAULTS),
        timeout_s=pick(args.timeout, b, "timeout_s", _BACKEND_DEFAULTS),
        transport_retries=pick(args.transport_retries, b, "transport_retries", _BACKEND_DEFAULTS),
        validation_retries=pick(args.validation_retries, b, "validation_retries", _BACKEND_DEFAULTS),
        extra_generation=b.get("extra_generation", {}),
        concurrency=pick(getattr(args, "concurrency", None), b, "concurrency", _BACKEND_DEFAULTS),
    )
    max_image_edge = pick(args.max_image_edge, g, "max_image_edge", _GRADING_DEFAULTS)
    return backend_config, max_image_edge


# --------------------------------------------------------------------------
# input fingerprinting (guards --resume against stale intermediates)
# --------------------------------------------------------------------------


def _hash_document(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for f in files:
            h.update(f.name.encode("utf-8"))
            h.update(f.read_bytes())
    else:
        h.update(path.read_bytes())
    return h.hexdigest()


def _fingerprints(
    args, backend: VisionBackend, max_image_edge: int, include_exam: bool,
    exam_path: str | Path | None = None,
) -> dict[str, str]:
    """Fingerprint everything that could change a stage's output.

    ``backend.describe()`` covers backend type, model, base_url, structured
    mode, generation parameters — so resume can never mix results produced by
    different models or configurations.
    """
    backend_desc = json.dumps(backend.describe(), sort_keys=True, ensure_ascii=False)
    key_h = hashlib.sha256()
    key_h.update(_hash_document(Path(args.key)).encode())
    if args.rubric:
        key_h.update(_hash_document(Path(args.rubric)).encode())
    key_h.update(f"|{backend_desc}|{max_image_edge}".encode())
    fp = {"key": key_h.hexdigest()}
    if include_exam:
        exam_h = hashlib.sha256()
        exam_h.update(fp["key"].encode())
        exam_h.update(_hash_document(Path(exam_path or args.exam)).encode())
        fp["exam"] = exam_h.hexdigest()
    return fp


def _stored_fingerprints(out: Path) -> dict[str, str]:
    fp_path = out / "fingerprint.json"
    if fp_path.exists():
        try:
            return json.loads(fp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _reuse(args, stage_file: Path, current_fp: str, stored_fp: str | None) -> bool:
    if not args.resume or not stage_file.exists():
        return False
    if stored_fp != current_fp:
        _log(f"resume: inputs changed since {stage_file.name} was written — recomputing")
        return False
    return True


# --------------------------------------------------------------------------
# pipeline stages
# --------------------------------------------------------------------------


def _get_key(args, backend: VisionBackend, out: Path, max_image_edge: int, reusable: bool) -> AnswerKey:
    key_path = Path(args.key)
    cached = out / "answer_key.json"
    if key_path.suffix.lower() == ".json":
        _log(f"loading structured answer key from {key_path}")
        if args.rubric:
            _log(
                "note: --rubric is ignored when --key is a parsed .json "
                "(edit the JSON directly instead)"
            )
        return load_answer_key(key_path)
    if reusable and cached.exists():
        _log(f"resume: reusing {cached}")
        return load_answer_key(cached)
    _log(f"parsing answer key document {key_path} (one model call)")
    rubric_text = None
    if args.rubric:
        rubric_text = Path(args.rubric).read_text(encoding="utf-8")
    pages = load_pages(key_path, max_image_edge)
    key = parse_answer_key(backend, pages, rubric_text)
    save_answer_key(key, cached)
    _log(f"wrote {cached}")
    return key


def cmd_doctor(args) -> int:
    backend_config, _ = resolve_config(args)
    try:
        backend = create_backend(backend_config)
    except BackendError as e:
        _log(f"FAIL: {e}")
        return 1
    report = backend.health_check()
    _log(f"backend:  {report.backend}")
    _log(f"model:    {report.model}")
    _log(f"config:   {json.dumps(backend.describe(), ensure_ascii=False)}")
    _log(f"status:   {'OK' if report.ok else 'FAIL'}")
    _log(f"detail:   {report.detail}")
    return 0 if report.ok else 1


def cmd_parse_key(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend_config, max_image_edge = resolve_config(args)
    backend = create_backend(backend_config)
    current = _fingerprints(args, backend, max_image_edge, include_exam=False)
    stored = _stored_fingerprints(out)
    key = _get_key(
        args, backend, out, max_image_edge,
        reusable=args.resume and stored.get("key") == current["key"],
    )
    (out / "fingerprint.json").write_text(json.dumps({**stored, **current}), encoding="utf-8")
    _log(
        f"parsed key: {key.exam_title!r}, versions={key.versions}, "
        f"{len(key.questions)} questions, total {key.total_points} points"
    )
    return 0


def run_grade_pipeline(
    args,
    backend: VisionBackend,
    out: Path,
    max_image_edge: int,
    exam_path: str | Path | None = None,
    exam_label: str | None = None,
    pages=None,
):
    """Grade one exam. Factored out so batch evaluation can reuse it.

    ``exam_label`` overrides the exam name recorded in results (used to keep
    grade-bearing filenames away from every downstream artefact), and
    ``pages`` may be pre-loaded (e.g. masked) page images.
    """
    exam_path = exam_path or args.exam
    out.mkdir(parents=True, exist_ok=True)
    config = GraderConfig(max_image_long_edge=max_image_edge, version=args.version)

    current = _fingerprints(args, backend, max_image_edge, include_exam=True, exam_path=exam_path)
    stored = _stored_fingerprints(out)

    key_is_json = Path(args.key).suffix.lower() == ".json"
    key = _get_key(
        args, backend, out, max_image_edge,
        reusable=(not key_is_json) and args.resume and stored.get("key") == current["key"],
    )

    if pages is None:
        _log(f"loading exam scan {exam_path}")
        pages = load_pages(exam_path, max_image_edge)
        _log(f"{len(pages)} pages loaded")

    survey_path = out / "survey.json"
    if _reuse(args, survey_path, current["exam"], stored.get("exam")):
        _log(f"resume: reusing {survey_path}")
        survey = ExamSurvey.model_validate_json(survey_path.read_text(encoding="utf-8"))
    else:
        _log("surveying the document (conventions, grader ink, answer locations)")
        survey = survey_exam(backend, pages, key)
        _write_json(survey, survey_path)

    extraction_path = out / "extraction.json"
    if _reuse(args, extraction_path, current["exam"], stored.get("exam")):
        _log(f"resume: reusing {extraction_path}")
        extraction = ExamExtraction.model_validate_json(
            extraction_path.read_text(encoding="utf-8")
        )
    else:
        extraction = extract_exam(backend, key, survey, pages, progress=_log)
        _write_json(extraction, extraction_path)

    # Stages on disk now correspond to the current inputs.
    (out / "fingerprint.json").write_text(json.dumps(current), encoding="utf-8")

    version_decision = detect_version(key, extraction, config)
    _log(f"version: {version_decision.version} ({version_decision.description})")

    judgements = judge_all(backend, key, extraction, version_decision.version, progress=_log)

    result = grade_exam(
        key,
        extraction,
        judgements,
        version_decision,
        config,
        survey=survey,
        exam_file=exam_label or Path(exam_path).name,
        graded_at=_dt.datetime.now().isoformat(timespec="seconds"),
        model=backend.identity,
    )
    result.backend_info = backend.describe()
    _write_json(result, out / "result.json")
    (out / "report.md").write_text(render_markdown(result), encoding="utf-8")
    _log(f"wrote {out / 'report.md'}")
    return result


def cmd_grade(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend_config, max_image_edge = resolve_config(args)
    backend = create_backend(backend_config)
    result = run_grade_pipeline(args, backend, out, max_image_edge)
    _log(
        f"TOTAL: {result.total_awarded:g}/{result.total_max:g} | "
        f"unanswered: {len(result.unanswered)} | "
        f"needs review: {len(result.needs_human_review)}"
    )
    return 0


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="TOML config file ([backend] and [grading] tables)")
    parser.add_argument(
        "--backend", choices=["openai", "mock", "anthropic"], default=None,
        help="Inference backend (default: openai — any OpenAI-compatible server)",
    )
    parser.add_argument("--model", default=None, help="Model name/tag on the backend")
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 (Ollama)",
    )
    parser.add_argument(
        "--api-key-env", default=None,
        help="Env var holding the API key if the server needs one (default: GRADER_API_KEY)",
    )
    parser.add_argument(
        "--structured-mode", choices=["json_schema", "json_object", "prompt"], default=None,
        help="How to request structured output from the server (default: json_schema)",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens per call")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=None, help="Per-request timeout seconds")
    parser.add_argument("--transport-retries", type=int, default=None)
    parser.add_argument("--validation-retries", type=int, default=None)
    parser.add_argument(
        "--max-image-edge", type=int, default=None,
        help="Long-edge pixel size for rendered pages (default: 2300)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autograder",
        description="Automatic grading of scanned student exams with open models.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="Check that the configured backend is reachable")
    add_backend_args(d)
    d.set_defaults(func=cmd_doctor)

    common = argparse.ArgumentParser(add_help=False)
    add_backend_args(common)
    common.add_argument("--key", required=True, help="Answer key: PDF/image(s) or a parsed .json")
    common.add_argument("--rubric", help="Optional separate rubric text file")
    common.add_argument("--out", default="out", help="Output directory (default: out)")
    common.add_argument(
        "--resume",
        action="store_true",
        help="Reuse stage outputs already in --out (only when inputs are unchanged)",
    )

    pk = sub.add_parser("parse-key", parents=[common], help="Parse the answer key only")
    pk.set_defaults(func=cmd_parse_key)

    g = sub.add_parser("grade", parents=[common], help="Grade a scanned exam end to end")
    g.add_argument("--exam", required=True, help="Student exam: PDF, image, or directory of images")
    g.add_argument(
        "--version",
        default="auto",
        help="Exam version id, or 'auto' to detect from answer agreement (default)",
    )
    g.set_defaults(func=cmd_grade)

    from .evalcli import add_eval_commands  # late import: keeps CLI startup light

    add_eval_commands(sub, common)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BackendError as e:
        _log(f"ERROR: {e}")
        return 2
    except PipelineStateError as e:
        _log(f"ERROR: {e}")
        return 2
    except ValueError as e:
        _log(f"ERROR: {e}")
        return 2
    except FileNotFoundError as e:
        _log(f"ERROR: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
