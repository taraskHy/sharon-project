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

from . import keycache, keyrepair
from .backends import BackendConfig, BackendError, VisionBackend, create_backend
from .config import GraderConfig
from .grade import PipelineStateError, detect_version, grade_exam, judge_all
from .ingest import IMAGE_SUFFIXES, downscale_pages, load_pages
from .prompts import KEY_PARSER_SYSTEM
from .variant import (
    alignment_fingerprint,
    alignment_from_override,
    alignment_override_path,
    config_fingerprint,
    decide_version,
    derive_alignment,
    detect_variant,
    identity_alignment,
    load_alignment_overrides,
    load_cached_alignment,
    load_variant_config,
    store_cached_alignment,
    validate_alignment,
)
from .key_parser import load_answer_key, parse_answer_key, save_answer_key
from .report import render_markdown
from .authority import flag_suspected_sheet_swap
from .schema import AnswerKey, ExamExtraction, ExamSurvey
from .extract import extract_exam
from .survey import candidate_sheet_pages, closeread_sheets, merge_closeread, survey_exam
from .template import (
    apply_template_to_key,
    load_template,
    synthesized_survey,
    template_fingerprint,
)


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
    "survey_image_edge": GraderConfig.survey_image_long_edge,
    "max_tokens": 16000,
}


def _load_toml(path: str | None) -> dict:
    if not path:
        return {}
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def resolve_config(args) -> tuple[BackendConfig, int, int]:
    """Build (BackendConfig, max_image_edge, survey_image_edge) from CLI +
    optional TOML file."""
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
    survey_image_edge = pick(
        getattr(args, "survey_image_edge", None), g, "survey_image_edge", _GRADING_DEFAULTS
    )
    return backend_config, max_image_edge, survey_image_edge


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


def _prompts_version() -> str:
    """Content hash of every pipeline prompt and stage schema: editing a
    prompt (or a stage's output model) in the source must invalidate cached
    stage results, exactly like changing the model would."""
    from . import prompts
    from .schema import ExamExtraction, ExamSurvey

    from .variant import RESOLVER_VERSION

    h = hashlib.sha256()
    for text in (
        prompts.KEY_PARSER_SYSTEM,
        prompts.SURVEY_SYSTEM,
        prompts.SHEET_CLOSEREAD_SYSTEM,
        prompts.VARIANT_DETECT_SYSTEM,
        prompts.ALIGNMENT_SYSTEM,
        prompts.EXTRACTION_SYSTEM,
        prompts.JUDGE_SYSTEM,
        RESOLVER_VERSION,
    ):
        h.update(text.encode("utf-8"))
    for model in (AnswerKey, ExamSurvey, ExamExtraction):
        h.update(json.dumps(model.model_json_schema(), sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def _fingerprints(
    args, backend: VisionBackend, max_image_edge: int, include_exam: bool,
    exam_path: str | Path | None = None, survey_image_edge: int | None = None,
) -> dict[str, str]:
    """Fingerprint everything that could change a stage's output.

    ``backend.describe()`` covers backend type, model, base_url, structured
    mode, generation parameters — so resume can never mix results produced by
    different models or configurations. The prompt/schema version hash does
    the same for source-level changes to the pipeline's prompts.
    """
    backend_desc = json.dumps(backend.describe(), sort_keys=True, ensure_ascii=False)
    key_h = hashlib.sha256()
    key_h.update(_hash_document(Path(args.key)).encode())
    if args.rubric:
        key_h.update(_hash_document(Path(args.rubric)).encode())
    key_h.update(f"|{backend_desc}|{max_image_edge}|{_prompts_version()}".encode())
    if survey_image_edge is not None:
        key_h.update(f"|survey:{survey_image_edge}".encode())
    # Variant context: the marker->variant mapping config, the operator
    # version-override file, and any pinned --version enter the fingerprint,
    # so results graded under one variant interpretation can never be
    # resumed/reused under another.
    variant_cfg = load_variant_config(Path(args.key), getattr(args, "variant_map", None))
    key_h.update(
        f"|variantcfg:{config_fingerprint(variant_cfg) if variant_cfg else 'none'}".encode()
    )
    ov_path = keyrepair.override_path(Path(args.key))
    if ov_path.exists():
        key_h.update(b"|overrides:")
        key_h.update(hashlib.sha256(ov_path.read_bytes()).hexdigest().encode())
    align_path = Path(
        getattr(args, "alignment_map", None) or alignment_override_path(Path(args.key))
    )
    if align_path.exists():
        key_h.update(b"|alignmap:")
        key_h.update(hashlib.sha256(align_path.read_bytes()).hexdigest().encode())
    pin = getattr(args, "version", None)
    if pin and pin != "auto":
        key_h.update(f"|pin:{pin}".encode())
    # The exam template (modes, answer-sheet rule) changes what every stage
    # produces — two exam families can never share stage results or caches.
    tpl = load_template(Path(args.key), getattr(args, "template", None))
    key_h.update(
        f"|template:{template_fingerprint(tpl) if tpl else 'none'}".encode()
    )
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


def _record_stage_fingerprint(out: Path, stage: str, fp: str) -> None:
    """Persist a stage's input fingerprint the moment its output file is
    written, under a per-stage key. A crash later in the pipeline then
    costs only the unfinished stages on the next --resume, while a stage
    file left over from OLDER inputs is still rejected (its recorded
    fingerprint no longer matches)."""
    path = out / "fingerprint.json"
    stored = _stored_fingerprints(out)
    stored[stage] = fp
    path.write_text(json.dumps(stored), encoding="utf-8")


# --------------------------------------------------------------------------
# pipeline stages
# --------------------------------------------------------------------------


def _get_key(
    args, backend: VisionBackend, out: Path, max_image_edge: int, reusable: bool
) -> tuple[AnswerKey, str]:
    """Load/parse the answer key. Returns (key, source) where source is one
    of 'json', 'resume', 'cache', 'parsed' — recorded in results so every
    run states whether it paid for a key parse or reused a validated one."""
    key_path = Path(args.key)
    cached = out / "answer_key.json"
    if key_path.suffix.lower() == ".json":
        _log(f"loading structured answer key from {key_path}")
        if args.rubric:
            _log(
                "note: --rubric is ignored when --key is a parsed .json "
                "(edit the JSON directly instead)"
            )
        return load_answer_key(key_path), "json"
    if reusable and cached.exists():
        _log(f"resume: reusing {cached}")
        return load_answer_key(cached), "resume"

    rubric_text = None
    if args.rubric:
        rubric_text = Path(args.rubric).read_text(encoding="utf-8")

    # When this exam family declares variants, the parser is TOLD which
    # version ids must exist (authoritative context from the marker config —
    # never answers), and the parse is rejected if they don't come back.
    # Parse quality is not deterministic across server restarts even at
    # temperature 0; a key without its versions would silently misgrade
    # every exam, so it must never be cached or used.
    variant_cfg = load_variant_config(key_path, getattr(args, "variant_map", None))
    expected_versions: list[str] = sorted(
        {entry["variant"] for entry in variant_cfg["markers"].values()}
    ) if variant_cfg else []
    hint = None
    if expected_versions:
        hint = (
            "AUTHORITATIVE NOTE: this exam family exists in the versions "
            f"{', '.join(expected_versions)}. The key document encodes "
            "per-version answers (look for its legend, e.g. answer colours). "
            "correct_by_version must use exactly these version ids, and the "
            "'versions' list must equal them."
        )
        rubric_text = f"{rubric_text}\n\n{hint}" if rubric_text else hint

    use_cache = not getattr(args, "no_key_cache", False)
    cache_dir = Path(getattr(args, "key_cache_dir", None) or keycache.default_cache_dir())
    fp = keycache.key_fingerprint(
        key_bytes_hash=_hash_document(key_path),
        rubric_text=rubric_text,
        backend_description=backend.describe(),
        max_image_edge=max_image_edge,
        parser_prompt=KEY_PARSER_SYSTEM,
    )
    if use_cache:
        hit = keycache.load_cached_key(cache_dir, fp)
        if hit is not None:
            hit_text = keyrepair.load_key_text(key_path) if expected_versions else ""
            if not _key_version_problems(hit, expected_versions, hit_text):
                if expected_versions:
                    # Verify/repair even cached entries — entries written by
                    # older code may carry model-decoded columns.
                    keyrepair.repair_key_versions(
                        hit, hit_text, expected_versions,
                        overrides=keyrepair.load_overrides(key_path),
                    )
                _log(f"answer key reused from persistent cache ({fp[:12]}…)")
                save_answer_key(hit, cached)
                return hit, "cache"
            _log("cached answer key failed validation — reparsing")

    pages = load_pages(key_path, max_image_edge)
    key_text = "\n".join(p.text for p in pages if p.text)
    key = None
    candidate = None
    for attempt in (1, 2):  # one bounded re-parse, then the deterministic path
        _log(
            f"parsing answer key document {key_path} (one model call"
            + (f", attempt {attempt})" if attempt > 1 else ")")
        )
        candidate = parse_answer_key(backend, pages, rubric_text)
        problems = _key_version_problems(candidate, expected_versions, key_text)
        if not problems:
            key = candidate
            break
        _log(f"parsed key REJECTED: {'; '.join(problems)}")
        rejected_path = out / f"answer_key.rejected-{attempt}.json"
        save_answer_key(candidate, rejected_path)
        _log(f"rejected candidate preserved for diagnosis: {rejected_path}")

    # Deterministic verify/repair of the per-version columns from the key's
    # born-digital text layer (model decode of the colour/letter-group
    # encoding is unreliable — see docs/validation). Runs on ACCEPTED parses
    # too: deterministic letters override model letters on disagreement.
    # When both model attempts were rejected, this path rescues the last
    # candidate iff the structure is usable — no further model retries.
    repair_report = None
    if expected_versions:
        target = key if key is not None else candidate
        if target is not None and sorted(target.versions) == expected_versions:
            repair_report = keyrepair.repair_key_versions(
                target,
                key_text,
                expected_versions,
                overrides=keyrepair.load_overrides(key_path),
            )
            _log(
                "key version columns: "
                f"{len(repair_report['repaired'])} repaired from text layer, "
                f"{len(repair_report['verified'])} verified, "
                f"{len(repair_report['overridden'])} from operator override, "
                f"{len(repair_report['unverified'])} unverified (flagged for "
                "review on affected versions)"
            )
            for note in repair_report["notes"]:
                _log(f"key repair note: {note}")
            if key is None:
                remaining = _key_version_problems(target, expected_versions, key_text)
                if not remaining:
                    _log(
                        "rejected model parse rescued by deterministic text-layer "
                        "repair — using the repaired key"
                    )
                    key = target
    if key is None:
        raise BackendError(
            "the answer key could not be parsed with the required versions "
            f"{expected_versions} after 2 attempts, and the deterministic "
            "text-layer repair could not rescue it — refusing to grade with a "
            "defective key (nothing was cached)"
        )
    save_answer_key(key, cached)
    _log(f"wrote {cached}")
    if use_cache:
        path = keycache.store_cached_key(
            cache_dir, fp, key,
            components_note={
                "key_file": key_path.name,
                "model": backend.describe().get("model", ""),
                "max_image_edge": max_image_edge,
                "expected_versions": expected_versions,
            },
        )
        _log(f"answer key stored in persistent cache: {path}")
    return key, "parsed"


def _key_version_problems(
    key: AnswerKey, expected_versions: list[str], key_text: str = ""
) -> list[str]:
    """Deterministic acceptance checks for a parsed key when the exam family
    declares variants."""
    if not expected_versions:
        return []
    problems = []
    if sorted(key.versions) != expected_versions:
        problems.append(
            f"key versions {sorted(key.versions)} != required {expected_versions}"
        )
    else:
        for q in key.questions:
            for s in q.sub_items:
                missing = [v for v in expected_versions if not s.correct_by_version.get(v)]
                if missing:
                    problems.append(
                        f"question {q.id} sub-item {s.id}: no answers for {missing}"
                    )
    if not problems and key_text:
        # Flattening detector: the born-digital key text carries per-version
        # letter groups like "F/F/G". If such groups exist with DIFFERING
        # letters but the parsed key gives every version identical answers on
        # every sub-item, the version columns were flattened in decode.
        import re

        n = len(expected_versions)
        raw = re.findall(r"\b[A-Z](?:/[A-Z]){%d}\b" % (n - 1), key_text)
        differing = [g for g in raw if len(set(g.split("/"))) > 1]
        all_uniform = all(
            len({tuple(sorted(s.correct_by_version.get(v, []))) for v in expected_versions}) == 1
            for q in key.questions
            for s in q.sub_items
        )
        if len(differing) >= 3 and all_uniform:
            problems.append(
                f"version columns look FLATTENED: the key text contains "
                f"{len(differing)} per-version letter groups with differing "
                "letters (e.g. "
                + ", ".join(differing[:3])
                + ") but every parsed sub-item has identical answers across "
                "all versions"
            )
    return problems[:8]


def guard_direct_cloud_backend(backend_config: BackendConfig) -> None:
    """The legacy direct-backend path (extraction/survey/variant/key parsing
    and the legacy explanation judge) bypasses the task gateway — no privacy
    scan, no request cache, no budget check, no usage ledger. It must
    therefore never carry OpenRouter traffic: classification uses the
    EFFECTIVE provider (base_url wins over the nominal backend name), so
    `--backend openai --base-url https://openrouter.ai/...` is refused too.

    OpenRouter runs only through --models-config (the task gateway), where
    every call passes privacy -> cache -> budget -> provider -> ledger.
    """
    from .usage import effective_provider

    if effective_provider(backend_config.backend, backend_config.base_url) == "openrouter":
        raise BackendError(
            "refusing to send exam content to OpenRouter over the direct legacy "
            "backend path: it bypasses privacy filtering, the request cache, "
            "budget enforcement and the usage ledger. Configure the OpenRouter "
            "task routes in models.toml and run with --models-config (and "
            "--grading-mode reliability|shadow for graded work) so every call "
            "passes the task gateway."
        )


def cmd_doctor(args) -> int:
    backend_config, _, _ = resolve_config(args)
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
    backend_config, max_image_edge, _ = resolve_config(args)
    guard_direct_cloud_backend(backend_config)
    backend = create_backend(backend_config)
    current = _fingerprints(args, backend, max_image_edge, include_exam=False)
    stored = _stored_fingerprints(out)
    key, key_source = _get_key(
        args, backend, out, max_image_edge,
        reusable=args.resume and stored.get("key") == current["key"],
    )
    (out / "fingerprint.json").write_text(json.dumps({**stored, **current}), encoding="utf-8")
    _log(
        f"parsed key ({key_source}): {key.exam_title!r}, versions={key.versions}, "
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
    survey_image_edge: int | None = None,
    page_loader=None,
):
    """Grade one exam. Factored out so batch evaluation can reuse it.

    ``exam_label`` overrides the exam name recorded in results (used to keep
    grade-bearing filenames away from every downstream artefact), and
    ``pages`` may be pre-loaded (e.g. masked) page images.
    ``survey_image_edge`` caps the render size of the whole-document survey
    pass (defaults to ``GraderConfig.survey_image_long_edge``); extraction
    still reads the authoritative pages at ``max_image_edge``.
    ``page_loader(edge) -> list[PageImage]`` re-renders pages at another
    resolution WITH the caller's preprocessing (e.g. masking) applied — used
    by the close-read, which needs more pixels than the general resolution.
    """
    exam_path = exam_path or args.exam
    out.mkdir(parents=True, exist_ok=True)
    if survey_image_edge is None:
        survey_image_edge = GraderConfig.survey_image_long_edge
    config = GraderConfig(
        max_image_long_edge=max_image_edge,
        survey_image_long_edge=survey_image_edge,
        version=args.version,
    )

    current = _fingerprints(
        args, backend, max_image_edge, include_exam=True, exam_path=exam_path,
        survey_image_edge=survey_image_edge,
    )
    stored = _stored_fingerprints(out)

    key_is_json = Path(args.key).suffix.lower() == ".json"
    key, key_source = _get_key(
        args, backend, out, max_image_edge,
        reusable=(not key_is_json) and args.resume and stored.get("key") == current["key"],
    )
    if not key_is_json:
        _record_stage_fingerprint(out, "key", current["key"])

    # Exam template: enforce per-question grading modes (a multiple-choice-only
    # template structurally disables explanation transcription and judging)
    # and, for fixed-page answer sheets, replace the survey model pass with a
    # deterministic synthesized survey.
    template = load_template(Path(args.key), getattr(args, "template", None))
    if template is not None:
        for note in apply_template_to_key(key, template):
            _log(f"template: {note}")

    # Optional gateway runtime (opt-in): MC resolution chain + policy gate.
    # Without --models-config the validated pipeline runs unchanged.
    _rt = None
    _pols: dict[str, str] = {}
    _mode = (getattr(args, "grading_mode", None) or "legacy").lower()
    _models_cfg = getattr(args, "models_config", None)
    if _models_cfg:
        from . import orchestrator as _orch

        _rt = _orch.setup_from_config(_models_cfg, out.parent if out.parent.name else out)
        _pol_file = getattr(args, "grading_policies", None)
        if _pol_file:
            _pols = json.loads(Path(_pol_file).read_text(encoding="utf-8"))
        else:
            from .discovery import deterministic_policies

            _pols = {q: f.value for q, f in deterministic_policies(key).items() if f.value}
        _orch.install_hooks(_rt, _pols)
        _log(f"gateway runtime enabled ({_models_cfg}); policies for {len(_pols)} question(s)")

    if _mode != "legacy":
        from .preflight import gate_package, package_report_for_key
        from .reliability import GradingModeError

        if _rt is None:
            raise GradingModeError(
                f"--grading-mode {_mode} requires --models-config: the pipeline never "
                "calls a provider directly, only through the task gateway")
        # ONE package-level stop, before any student exam is graded: a
        # structural defect must not become one review per student.
        _pre = gate_package(package_report_for_key(key, args.key, policies=_pols))
        _log(f"package preflight: {_pre.summary().splitlines()[0]}")

    if pages is None:
        _log(f"loading exam scan {exam_path}")
        pages = load_pages(exam_path, max_image_edge)
        _log(f"{len(pages)} pages loaded")
        if getattr(args, "mask", False):
            from .masking import mask_pages

            pages, mask_report = mask_pages(pages)
            (out / "masking.json").write_text(
                json.dumps(mask_report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _log("instructor-annotation masking applied (masking.json written)")
            if page_loader is None:
                page_loader = lambda edge: mask_pages(load_pages(exam_path, edge))[0]  # noqa: E731

    # Marker-based variant detection (BEFORE grading, never answer-based).
    variant_cfg = load_variant_config(Path(args.key), getattr(args, "variant_map", None))
    version_decision = None
    variant_record = None
    if config.version != "auto":
        # Operator override — but a pinned A2/A3 still needs its question
        # alignment, so resolve the decision here rather than post-extraction.
        version_decision = detect_version(key, None, config)
        _log(f"version pinned by operator: {version_decision.version}")
    elif variant_cfg:
        _log(
            f"detecting exam variant from the cover-page "
            f"{variant_cfg.get('marker_kind', 'marker')} (config: {variant_cfg['_path']})"
        )
        detection = detect_variant(backend, pages, variant_cfg)
        version_decision, variant_record = decide_version(detection, variant_cfg, key)
        _log(f"variant: {version_decision.version} ({version_decision.description})")

    survey_path = out / "survey.json"
    if template is not None and template.answer_sheet_rule == "fixed_pages":
        # Template-specific page rule: the answer sheet is a known page set for
        # this exam family. No survey model call, no close-read — the rule is
        # deterministic configuration, and question-page ink is scratch work
        # per the template (never allowed to override the sheet).
        _log(
            f"survey synthesized from template {template.template_id}: answer "
            f"sheet fixed to page(s) {template.answer_sheet_pages} "
            f"(booklet markings {'not graded' if template.booklet_answers_not_graded else 'secondary'})"
        )
        survey = synthesized_survey(template, key, n_pages=len(pages))
        _write_json(survey, survey_path)
        _record_stage_fingerprint(out, "survey", current["exam"])
    elif _reuse(args, survey_path, current["exam"], stored.get("survey") or stored.get("exam")):
        _log(f"resume: reusing {survey_path}")
        survey = ExamSurvey.model_validate_json(survey_path.read_text(encoding="utf-8"))
    else:
        survey_pages = downscale_pages(pages, survey_image_edge)
        _log(
            "surveying the document (page classification, answer-sheet policy, "
            f"conventions, grader ink) at <={survey_image_edge}px"
        )
        survey = survey_exam(backend, survey_pages, key)
        sheet_nums = candidate_sheet_pages(survey)
        if sheet_nums:
            # Fine print (crossed-out title digits, faint swap notes) needs
            # more pixels than the general extraction resolution; re-render
            # just the sheet pages larger. ``page_loader`` keeps masking
            # intact for batch runs (it loads AND masks); plain grade runs
            # re-render from the source document.
            closeread_edge = max(max_image_edge, GraderConfig.closeread_image_long_edge)
            if closeread_edge > max_image_edge:
                loader = page_loader or (lambda edge: load_pages(exam_path, edge))
                closeread_pages = [
                    p for p in loader(closeread_edge) if p.page_number in set(sheet_nums)
                ]
            else:
                closeread_pages = pages
            _log(
                f"close-reading answer-sheet pages {sheet_nums} at "
                f"<={closeread_edge}px (title corrections, conventions, condition)"
            )
            closeread = closeread_sheets(backend, survey, closeread_pages, key)
            if closeread is not None:
                survey = merge_closeread(survey, closeread)
                _write_json(closeread, out / "sheet_closeread.json")
        else:
            _log("no dedicated answer sheets located — booklet answers are authoritative")
        _write_json(survey, survey_path)
        _record_stage_fingerprint(out, "survey", current["exam"])

    # Per-variant question alignment: variants shuffle question/option order,
    # so the printed sub-item numbering must be mapped onto the key's
    # canonical numbering before extraction results can be scored.
    alignment = None
    alignment_note = None
    if variant_cfg and version_decision is not None:
        # 1st choice: the operator-verified mapping shipped next to the key.
        # Model-derived alignments are NEVER silently trusted — two live
        # failures (incomplete map; complete-but-WRONG identity claim with
        # malformed ids) showed a valid bijection proves nothing about
        # correctness. Derived/fallback alignments mark every affected
        # sub-item unresolved_alignment -> human review.
        align_overrides = load_alignment_overrides(
            Path(args.key), getattr(args, "alignment_map", None)
        )
        if align_overrides:
            alignment = alignment_from_override(key, version_decision.version, align_overrides)
            if alignment is not None:
                _log(
                    f"question alignment for {version_decision.version}: "
                    "operator-verified override"
                )
                alignment_note = "operator-override"
        if alignment is None:
            cache_dir = Path(getattr(args, "key_cache_dir", None) or keycache.default_cache_dir())
            align_fp = alignment_fingerprint(current["key"], version_decision.version)
            alignment = load_cached_alignment(cache_dir, align_fp)
            if alignment is not None:
                _log(
                    f"question alignment for {version_decision.version}: reused "
                    "from cache (model-derived — UNVERIFIED, review-flagged)"
                )
                alignment_note = "derived-unverified(cache)"
            else:
                _log(
                    f"deriving question alignment for variant {version_decision.version} "
                    "(model call per question — UNVERIFIED, review-flagged)"
                )
                candidate = derive_alignment(backend, key, survey, pages, version_decision.version)
                problems = validate_alignment(key, candidate)
                if problems:
                    _log(
                        "alignment REJECTED (%s) — using identity numbering and "
                        "flagging every sub-item for review" % "; ".join(problems)
                    )
                    alignment = identity_alignment(key, version_decision.version)
                    alignment_note = "identity-fallback: " + "; ".join(problems)
                else:
                    alignment = candidate
                    alignment_note = "derived-unverified"
                    if not getattr(args, "no_key_cache", False):
                        store_cached_alignment(cache_dir, align_fp, alignment)
        _write_json(alignment, out / "alignment.json")

    # Extraction depends on the EFFECTIVE variant (printed_view relabeling
    # under a non-identity alignment) — a re-detected variant must never
    # reuse another variant's extraction artefacts.
    extraction_fp = current["exam"] + (
        f"|variant:{version_decision.version}" if version_decision is not None else ""
    )
    extraction_path = out / "extraction.json"
    if _reuse(args, extraction_path, extraction_fp, stored.get("extraction") or stored.get("exam")):
        _log(f"resume: reusing {extraction_path}")
        extraction = ExamExtraction.model_validate_json(
            extraction_path.read_text(encoding="utf-8")
        )
    else:
        extraction = extract_exam(
            backend, key, survey, pages, progress=_log, alignment=alignment,
            template=template,
        )
        if alignment_note and alignment_note != "operator-override":
            # Any non-operator alignment (derived, cached-derived, identity
            # fallback) is unresolved: printed-to-key numbering is unverified
            # and no trusted per-item score exists for shuffled questions.
            note = (
                "unresolved_alignment: printed-to-key numbering is not "
                "operator-verified"
                + (
                    " (model-derived mapping)"
                    if alignment_note.startswith("derived")
                    else " (identity fallback after rejected derivation)"
                )
                + "; scores for this item are provisional — human review required"
            )
            # Applies to every question — a derived-identity claim is exactly
            # the observed silent failure mode, so there is no exemption.
            for qx in extraction.questions:
                for s in qx.sub_items:
                    if note not in (s.uncertainty_note or ""):
                        s.uncertainty_note = (
                            f"{s.uncertainty_note}; {note}" if s.uncertainty_note else note
                        )
                    s.confidence = min(s.confidence, 0.5)
        _write_json(extraction, extraction_path)
        _record_stage_fingerprint(out, "extraction", extraction_fp)

    # Every stage on disk now corresponds to the current inputs ("exam" is
    # the whole-pipeline fingerprint eval-batch uses for finished results).
    (out / "fingerprint.json").write_text(
        json.dumps({**_stored_fingerprints(out), **current}), encoding="utf-8"
    )

    if version_decision is None:
        # Operator pin, or an exam family without a marker config (legacy
        # answer-agreement detection with its uncertainty margin).
        version_decision = detect_version(key, extraction, config)
    _log(f"version: {version_decision.version} ({version_decision.description})")

    # Deterministic tripwire: an answer-table mix-up the close-read missed
    # shows up as strongly CROSSED key agreement between sibling matching
    # questions. Review-flag only — never regrade on agreement.
    swap_flags = flag_suspected_sheet_swap(key, extraction, version_decision.version)
    if swap_flags:
        for line in swap_flags:
            _log(f"SWAP SUSPECT: {line}")
        _write_json(extraction, extraction_path)  # persist the added flags

    # ---- the explanation-grading seam: legacy / reliability / shadow ------
    #
    # legacy      the validated ExplanationJudgement path (default, unchanged)
    # reliability the reliability route decides each written answer
    # shadow      BOTH run; legacy stays authoritative and the reliability
    #             route only records what it would have decided
    rel_run = None
    judgements = None
    if _mode in ("legacy", "shadow"):
        judgements = judge_all(backend, key, extraction, version_decision.version, progress=_log)
    if _mode != "legacy":
        from .gradingpack import build_all_packs
        from .reliability import ReliabilityConfig, run_reliability_judging
        from .trace import DecisionTraceStore

        _rel_cfg = ReliabilityConfig(
            mode=_mode, rag_policy=(getattr(args, "rag_policy", None) or "RAG_DISABLED"))
        _packs = build_all_packs(key, _pols, rag_policy=_rel_cfg.rag_policy)
        _exam_id = exam_label or Path(exam_path).stem
        try:
            rel_run = run_reliability_judging(
                key=key, extraction=extraction, version=version_decision.version,
                config=_rel_cfg, gateway=_rt.gateway, packs=_packs, policies=_pols,
                exam_id=_exam_id, trace_store=DecisionTraceStore(out / "decisions.jsonl"),
                variant_source=(variant_record or {}).get("mapping_source"),
                alignment_source=alignment_note, progress=_log)
        except Exception as e:  # noqa: BLE001
            if _mode == "reliability":
                raise
            # shadow must NEVER be able to affect the authoritative grade
            _log(f"shadow route failed ({type(e).__name__}: {e}); legacy result unaffected")
        if _mode == "reliability" and rel_run is not None:
            judgements = rel_run.evaluations
            _log(f"reliability route: {rel_run.by_state()}")

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
    result.backend_info = {**backend.describe(), "answer_key_source": key_source}
    if variant_record is not None:
        result.variant_detection = {
            **variant_record,
            "selected_variant": version_decision.version,
            "uncertain": version_decision.uncertain,
            "question_alignment": alignment_note,
        }
    if _mode == "reliability" and rel_run is not None:
        from .reliability import apply_review_items

        apply_review_items(result, rel_run)
    if _mode == "shadow" and rel_run is not None:
        # The legacy result above is already final. The shadow score is scored
        # by the SAME deterministic scorer into a throwaway object and is only
        # ever written to shadow_comparison.json — never to result.json.
        from .reliability import compare_shadow

        shadow_result = grade_exam(
            key, extraction, rel_run.evaluations, version_decision, config, survey=survey,
            exam_file=exam_label or Path(exam_path).name,
            graded_at=_dt.datetime.now().isoformat(timespec="seconds"),
            model=f"shadow:{backend.identity}")
        for item in rel_run.review_items:
            if (item.question_id, item.sub_item_id) not in {
                    (r.question_id, r.sub_item_id) for r in shadow_result.needs_human_review}:
                shadow_result.needs_human_review.append(item)
        comparison = compare_shadow(legacy_result=result, shadow_result=shadow_result,
                                    run=rel_run)
        (out / "shadow_comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=1), encoding="utf-8")
        _log(f"shadow comparison written ({comparison['agreement']['exact_score_agreement']}% "
             "exact score agreement); the legacy grade is authoritative")

    _write_json(result, out / "result.json")
    (out / "report.md").write_text(render_markdown(result), encoding="utf-8")
    _log(f"wrote {out / 'report.md'}")
    return result


def cmd_grade(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend_config, max_image_edge, survey_image_edge = resolve_config(args)
    guard_direct_cloud_backend(backend_config)
    backend = create_backend(backend_config)
    result = run_grade_pipeline(
        args, backend, out, max_image_edge, survey_image_edge=survey_image_edge
    )
    _log(
        f"TOTAL: {result.total_awarded:g}/{result.total_max:g} | "
        f"unanswered: {len(result.unanswered)} | "
        f"needs review: {len(result.needs_human_review)}"
    )
    return 0


def cmd_run_job(args) -> int:
    from .jobs import run_job

    return run_job(args.job_dir)


def cmd_ui(args) -> int:
    import subprocess

    app = Path(__file__).with_name("webui.py")
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app),
         "--server.port", str(args.port), "--browser.gatherUsageStats", "false"]
    )


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
    parser.add_argument(
        "--survey-image-edge", type=int, default=None,
        help=(
            "Long-edge pixel size for the whole-document SURVEY pass only "
            "(default: 640). Extraction re-reads authoritative pages at "
            "--max-image-edge."
        ),
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
    common.add_argument(
        "--key-cache-dir", default=None,
        help=(
            "Directory of the persistent parsed-answer-key cache (default: "
            "%%LOCALAPPDATA%%/autograder/key_cache or $XDG_CACHE_HOME; also "
            "settable via GRADER_KEY_CACHE)"
        ),
    )
    common.add_argument(
        "--no-key-cache", action="store_true",
        help="Disable the persistent answer-key cache for this run",
    )
    common.add_argument(
        "--variant-map", default=None,
        help=(
            "Path to the marker-to-variant mapping JSON (default: "
            "<key>.variants.json next to the answer key; absent = legacy "
            "answer-agreement version detection)"
        ),
    )
    common.add_argument(
        "--alignment-map", default=None,
        help=(
            "Path to the operator-verified printed-to-key question alignment "
            "JSON (default: <key>.alignment.json next to the answer key). "
            "Without an entry for a variant, model-derived alignment is used "
            "and every affected sub-item is review-flagged as unresolved"
        ),
    )
    common.add_argument(
        "--models-config", default=None,
        help=(
            "Path to models.toml enabling the task gateway (OpenRouter/local "
            "routing, request cache, usage ledger, budgets), the MC resolution "
            "chain and per-question grading-policy early exits. Absent = the "
            "validated legacy pipeline, unchanged."
        ),
    )
    common.add_argument(
        "--grading-policies", default=None,
        help="Optional JSON file {question_id: policy} overriding discovered policies",
    )
    common.add_argument(
        "--grading-mode", default="legacy", choices=["legacy", "reliability", "shadow"],
        help=(
            "How written answers are graded. 'legacy' (default) = the validated "
            "ExplanationJudgement path, unchanged. 'reliability' = the evidence/"
            "invariant/escalation route (needs --models-config). 'shadow' = run "
            "both; the legacy grade stays authoritative and the reliability route "
            "only records what it would have decided, in shadow_comparison.json."
        ),
    )
    common.add_argument(
        "--rag-policy", default="RAG_DISABLED",
        choices=["RAG_DISABLED", "RAG_ALWAYS", "RAG_ON_UNCERTAIN", "RAG_ON_ESCALATION"],
        help=(
            "Grading-side course-context policy. Default RAG_DISABLED: the benefit "
            "is unmeasured and no unmeasured optional context is sent silently."
        ),
    )
    common.add_argument(
        "--course", default=None,
        help=(
            "Course id whose LOCAL persistent index supplies grading-side RAG "
            "context (non-legacy modes, consulted per --rag-policy). Retrieval "
            "is always local — never a cloud call."
        ),
    )
    common.add_argument(
        "--packs-root", default=None,
        help=(
            "Directory where per-question grading packs are persisted once and "
            "reused across all students of a batch (default: <out>/../packs)."
        ),
    )
    common.add_argument(
        "--template", default=None,
        help=(
            "Path to the exam template JSON describing grading modes and the "
            "answer-sheet rule (default: <key>.template.json next to the "
            "answer key; absent = full pipeline with detected answer sheets)"
        ),
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
    g.add_argument(
        "--mask", action="store_true",
        help="Mask red instructor annotations from page images before inference",
    )
    g.set_defaults(func=cmd_grade)

    rj = sub.add_parser(
        "run-job", help="Run (or resume) a batch-grading job directory created by the web UI"
    )
    rj.add_argument("--job-dir", required=True, help="The job directory (jobs/<id>)")
    rj.set_defaults(func=cmd_run_job)

    ui = sub.add_parser("ui", help="Launch the local web interface (Streamlit)")
    ui.add_argument("--port", type=int, default=8501)
    ui.set_defaults(func=cmd_ui)

    from .evalcli import add_eval_commands  # late import: keeps CLI startup light

    add_eval_commands(sub, common)
    return p


def main(argv: list[str] | None = None) -> int:
    # Hebrew in log lines (question titles, review reasons) would crash
    # print() on a cp1252 Windows console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
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
