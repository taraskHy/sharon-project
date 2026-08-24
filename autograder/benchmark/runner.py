"""The benchmark run loop.

    spec -> manifest (hash-verified) -> split/component selection
         -> candidate resolution (UNSELECTED refused; registry-listed)
         -> gateway (ModelGateway: privacy scan, exact-request cache, usage
            ledger, $8/$10 budget) -> per-case request -> leakage check
         -> call (or dry-run plan) -> raw structured output persisted
         -> scoring -> metrics.json / usage.json / run.json

Guarantees:
* **Comparability**: one run = one (role, split, component, candidate,
  backend, route fingerprint, prompt sha, schema sha, adapter version,
  manifest hashes) — hashed into ``config_hash``; the run directory is keyed
  by it, so resuming with a different config is a *different* run.
  Validation-repair round-trips are disabled for benchmark routes
  (``validation_retries=0``): a malformed answer is a schema failure,
  never silently repaired into a success. Transport retries (network/429)
  are recorded, not hidden.
* **Resume**: rows already in outputs.jsonl are skipped; failed rows are
  re-attempted ONLY with ``retry_failed=True`` and the attempt is recorded.
* **Raw outputs**: every structured output (model_dump) or failure is
  appended to outputs.jsonl with usage, latency, cache_hit, fingerprint.
* **Held-out**: ``split == HELD_OUT`` needs ``confirm_held_out=True`` and is
  permanently appended to HELD_OUT_EXECUTIONS.jsonl (also for dry runs).
* **Leakage**: every request is checked against the case's evaluation-side
  label before it can leave the process.
* **Dry run**: builds and checks every request, estimates cost from the
  local pricing table, writes plan.json — ZERO provider calls.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .manifests import (DEFAULT_BENCH_ROOT, DEFAULT_DATASETS_ROOT, REPO_ROOT, ROLE_TASKS,
                        SPLITS, BenchCase, BenchmarkManifest, load_manifest)
from .registry import DEFAULT_REGISTRY_PATH, CandidateRegistry, load_registry
from .roles import Request, adapter_for

DEFAULT_STATE_ROOT = REPO_ROOT / "evaluation" / "model_selection" / "state"
DEFAULT_RUNS_ROOT = REPO_ROOT / "evaluation" / "model_selection" / "runs"
HELD_OUT_LOG = REPO_ROOT / "evaluation" / "model_selection" / "HELD_OUT_EXECUTIONS.jsonl"
UNSELECTED = "UNSELECTED"


class HeldOutRefused(RuntimeError):
    """HELD_OUT requested without explicit confirmation."""


class UnselectedCandidate(RuntimeError):
    """No candidate model for the role (UNSELECTED) or an unlisted slug."""


class LeakageError(RuntimeError):
    """A request would carry evaluation-side information."""


class UnpricedCandidate(RuntimeError):
    """A live cloud run was requested for a model with no local price."""


def require_priced_candidate(model: str, pricing: dict | None) -> None:
    """Refuse a live cloud run for a model this machine cannot price.

    `predicted_call_cost` returns 0.0 for an unknown model, so an unpriced
    candidate would sail through every pre-call budget check and the $10
    ceiling could only react AFTER the provider had already charged. Silence
    there is the whole failure: the run looks free until the ledger says
    otherwise. A campaign must therefore list every cloud candidate it intends
    to call in models.toml [pricing], with prices read from the provider's own
    page."""
    entry = (pricing or {}).get(model)
    have = (isinstance(entry, dict)
            and float(entry.get("input") or 0) > 0 and float(entry.get("output") or 0) > 0)
    if not have:
        raise UnpricedCandidate(
            f"refusing a LIVE run of {model!r}: it has no usable entry in the local [pricing] table, so the "
            "pre-call budget check would estimate $0 and could not refuse a call that crosses the ceiling. "
            "Add `[pricing.\"" + model + "\"] input=<USD per 1M> output=<USD per 1M>` to models.toml "
            "(read the numbers off the provider's model page) and pass --models-config models.toml.")


@dataclass
class RunSpec:
    role: str
    split: str
    candidate: str | None = None            # model slug; None -> resolve from models config (refused if UNSELECTED)
    component: str | None = None            # ocr_verify: REAL | SYNTHETIC | None (both, reported separately)
    backend: str = "openrouter"
    base_url: str | None = None
    models_config: Path | None = None       # for [pricing] (estimator) and default route knobs
    registry_path: Path = DEFAULT_REGISTRY_PATH
    bench_root: Path = DEFAULT_BENCH_ROOT
    datasets_root: Path = DEFAULT_DATASETS_ROOT
    state_root: Path = DEFAULT_STATE_ROOT   # shared cache + ledger + budget for the whole campaign
    runs_root: Path = DEFAULT_RUNS_ROOT
    held_out_log: Path = HELD_OUT_LOG
    limit: int | None = None
    dry_run: bool = True                    # SAFE DEFAULT: no calls unless explicitly False
    confirm_held_out: bool = False
    retry_failed: bool = False
    allow_unlisted: bool = False
    note: str = ""
    max_tokens: int | None = None
    reasoning: dict | None = None
    provider: dict | None = None
    warn_usd: float | None = None           # default: registry [budget] warn_usd (8.00)
    hard_usd: float | None = None           # default: registry [budget] experiment_total_usd (10.00)
    validation_retries: int = 0             # NEVER silently repair malformed output in a benchmark
    subset: str | None = None               # "smoke" -> the frozen pre-registered DEV smoke subset
    skip_key_preflight: bool = False        # tests only: skip the GET /api/v1/key preflight step
    final_evaluation: bool = False          # ONLY the `bench final-eval` path sets this (HELD_OUT live run)
    smoke_root: Path | None = None


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    role: str
    split: str
    component: str | None
    candidate: str
    dry_run: bool
    cases_selected: int
    cases_done: int
    cases_failed: int
    cases_skipped_resume: int
    stopped_reason: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    predicted_cost: float | None = None
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:60]


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=str(REPO_ROOT), timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 — provenance only
        return None


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8",
                    newline="\n")


def _inside_inputs(value: str, input_values: set[str]) -> bool:
    return any(value in iv for iv in input_values)


_SPLIT_TOKEN = re.compile(r"\b(DEV|CALIBRATION|HELD_OUT)\b")
#: label fields that ARE the grading target — a number from one of these reaching
#: the prompt would tell the model the answer it is being measured against
_SCORE_LABEL_FIELDS = frozenset({"score", "label_score", "ground_truth_score", "final_score"})


def leakage_check(case: BenchCase, request: Request, model_visible_fields: tuple[str, ...]) -> None:
    """Refuse any request that carries evaluation-side information:
    * input fields outside the adapter's model-visible whitelist
    * label string values (>= 4 chars) that are not themselves inputs
    * a label field NAME inside a content block, or a split name anywhere
    The system prompt is constant across cases, so it cannot carry per-case
    label information; case-specific leakage can only travel in the content
    blocks (vocabulary overlap such as "substitution" in the verifier's
    instructions is therefore not a leak).
    """
    extra = set(case.inputs) - set(model_visible_fields)
    if extra:
        raise LeakageError(f"{case.case_id}: inputs carry non-whitelisted fields {sorted(extra)}")
    full = request.text_for_inspection()
    content = "\n".join(str(b.get("text", "")) for b in request.content_blocks if b.get("type") == "text")
    input_values: set[str] = set()
    for v in case.inputs.values():
        if isinstance(v, (str, int, float)):
            input_values.add(str(v))
        elif isinstance(v, (list, tuple)):          # e.g. the generic version ids / option letters
            input_values.update(str(x) for x in v if isinstance(x, (str, int, float)))

    # A SEPARATE, deeper collection used only by the numeric check below. The
    # grading pack is a nested dict, so its numbers (max_score, option indices)
    # are legitimately in the prompt; the shallow set above deliberately stays
    # shallow, because widening it would excuse a nested string leak.
    numeric_inputs: set[str] = set()

    def _collect_numbers(v) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            numeric_inputs.add(str(v))
            if float(v).is_integer():
                numeric_inputs.add(str(int(v)))     # 4.0 is written "4" in the prompt
        elif isinstance(v, str):
            numeric_inputs.add(v)
        elif isinstance(v, dict):
            for x in v.values():
                _collect_numbers(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _collect_numbers(x)

    for v in case.inputs.values():
        _collect_numbers(v)

    for k, v in case.label.items():
        # The grading TARGET is a NUMBER, and a string-only comparison left exactly
        # that unchecked. Scoped to the target fields on purpose: bookkeeping counts
        # like `line_count` legitimately share digits with the prompt.
        #
        # Honest limit: a single-character value (a bare 0-4) cannot be told apart
        # from the "Score range: 0..4" line or the pack's own numbers, so only
        # multi-character forms (0.5, 1.5, 2.5, 3.5, 10 ...) are decidable here.
        # The structural guarantee is what actually prevents the leak — build_request
        # is never handed case.label — this is the backstop.
        if k in _SCORE_LABEL_FIELDS and isinstance(v, (int, float)) and not isinstance(v, bool):
            forms = {str(v)} | ({str(int(v))} if float(v).is_integer() else set())
            for form in forms:
                if len(form) < 2 or form in numeric_inputs:
                    continue
                if re.search(r"(?<![\w.])" + re.escape(form) + r"(?![\w.])", content):
                    raise LeakageError(
                        f"{case.case_id}: label field {k!r} value {v!r} appears in the request")
        # A label value that is part of a model-visible input (e.g. the
        # audited reference sitting inside a token-duplication candidate) is
        # visible by construction, not a leak; anything else is.
        if (isinstance(v, str) and len(v) >= 4 and v in content
                and not _inside_inputs(v, input_values)
                and not _fully_enumerated(k, content)):
            raise LeakageError(f"{case.case_id}: label field {k!r} value appears in the request")
        if isinstance(v, (list, tuple)):
            for item in v:
                if (isinstance(item, str) and len(item) >= 4 and item in content
                        and not _inside_inputs(item, input_values)):
                    raise LeakageError(f"{case.case_id}: label field {k!r} item appears in the request")
    if _SPLIT_TOKEN.search(full):
        raise LeakageError(f"{case.case_id}: split name appears in the request")
    # A name the model is REQUIRED TO EMIT cannot be evaluation-side
    # information. `GradeResult.score` collides with the label field `score`,
    # and grade-v3 must be able to say which field it means ("`score` is the
    # EXPLANATION-QUALITY value, not the student's final score"). Exempting
    # exactly the output schema's own property names keeps the guard intact for
    # every genuinely evaluation-only name — explanation_verdict,
    # selection_correct, label_score, rubric_met are not GradeResult fields and
    # stay banned.
    emitted = set(_output_property_names(request))
    for k in case.label:
        if k in emitted:
            continue
        if k in _LABEL_NAMES_NEVER_IN_PROMPT and re.search(r"\b" + re.escape(k) + r"\b", content):
            raise LeakageError(f"{case.case_id}: label field name {k!r} appears in the request")


#: Label fields whose value comes from a small closed vocabulary that a prompt
#: may legitimately ENUMERATE. grade-v3 has to tell the model the three
#: explanation-quality levels it may return, and that list names every class —
#: including whichever one happens to be this case's ground truth.
_ENUMERABLE_LABEL_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "explanation_verdict": ("invalid", "partially_valid", "valid"),
}


def _fully_enumerated(field: str, content: str) -> bool:
    """True when EVERY member of ``field``'s vocabulary appears in the content.

    Seeing one class name is a leak; seeing all of them is a menu. A prompt
    that listed only this case's own verdict would still be caught, because the
    other members would be missing.

    Each member is matched in either rendered form — ``partially_valid`` or
    ``partially valid`` — since prose spells it with a space.
    """
    vocab = _ENUMERABLE_LABEL_VOCABULARIES.get(field)
    if not vocab:
        return False
    low = content.lower()
    return all(v in low or v.replace("_", " ") in low for v in vocab)


def _output_property_names(request: Request) -> set[str]:
    """Top-level and nested property names of the request's output schema."""
    try:
        schema = request.output_model.model_json_schema()
    except Exception:  # noqa: BLE001 — a schema we cannot read exempts nothing
        return set()
    names: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                names.update(props)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return names


_LABEL_NAMES_NEVER_IN_PROMPT = frozenset({
    "expected_verdict", "polarity", "error_kinds", "cer_vs_audited", "audited_reference",
    "corruption_type", "synthetic_group", "reference_status", "rubric_met", "fixed_judge_verdict",
    # the grading TARGET and its neighbours: naming any of these in a prompt would
    # tell the model what it is being measured against
    "score", "label_score", "ground_truth_score", "owner_note", "owner_status", "label_status"})


def resolve_candidate(spec: RunSpec, registry: CandidateRegistry) -> str:
    """The model slug for this run. Never guesses: UNSELECTED or an unlisted
    slug is refused with the candidate list."""
    rc = registry.for_role(spec.role)
    slug = spec.candidate
    if slug is None and spec.models_config is not None and Path(spec.models_config).exists():
        import tomllib
        data = tomllib.loads(Path(spec.models_config).read_text(encoding="utf-8"))
        sec = (data.get("models") or {}).get(ROLE_TASKS[spec.role]) or {}
        raw = str(sec.get("model", "") or "")
        if raw and not raw.startswith("${") and raw != UNSELECTED:
            slug = raw
        else:
            env = re.findall(r"\$\{([A-Z0-9_]+)\}", raw)
            if env and os.environ.get(env[0]):
                slug = os.environ[env[0]]
    if not slug or slug == UNSELECTED:
        raise UnselectedCandidate(
            f"{spec.role} model is not selected (status {rc.status}): pass --candidate <slug>; "
            f"registered candidates: {rc.candidates or '(none registered)'} "
            f"(evaluation/model_selection/candidates.toml)")
    if not spec.allow_unlisted and not registry.is_listed(spec.role, slug):
        raise UnselectedCandidate(
            f"{slug!r} is not a registered candidate for {spec.role}; registered: {rc.candidates}. "
            "Add it to candidates.toml (data, no code change) or pass --allow-unlisted for an exploratory run")
    return slug


#: Decoding parameters a benchmark run must inherit from the PRODUCTION route
#: so the two agree. The model slug is deliberately NOT here: the candidate
#: under test is the one thing the benchmark is allowed to change.
ROUTE_PARITY_FIELDS = ("structured_mode", "max_tokens", "temperature", "timeout_s",
                       "reasoning", "provider", "extra_generation")


def production_route_defaults(models_config, task: str) -> dict:
    """The production decoding configuration for ``task`` from models.toml.

    A benchmark that decodes differently from production measures a model
    nobody will ever run. Before this existed, ``build_route`` hard-coded its
    own knobs and silently dropped ``[models.grade_primary].reasoning``
    ({"effort": "none"}) — so the 2026-08-24 smoke run paid for reasoning
    tokens that production would never have generated, and lost a case to
    truncation because those tokens consumed the 600-token cap.

    Reads decoding knobs ONLY. ``model``/``backend``/``base_url`` are ignored:
    the candidate and its transport come from the run spec.
    """
    if not models_config:
        return {}
    path = Path(models_config)
    if not path.exists():
        return {}
    import tomllib

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    merged = {**(data.get("defaults") or {}), **((data.get("models") or {}).get(task) or {})}
    return {k: merged[k] for k in ROUTE_PARITY_FIELDS if merged.get(k) is not None}


#: route knobs a candidate may override in candidates.toml
CANDIDATE_OVERRIDABLE = ("reasoning", "max_tokens", "provider", "temperature")


def build_route(spec: RunSpec, candidate: str, request_prompt_version: str, default_max_tokens: int,
                registry: "CandidateRegistry | None" = None):
    """Production parity, with declared candidate asymmetry and explicit
    run-spec values winning.

    Precedence: explicit RunSpec field > candidates.toml candidate override >
    models.toml production route > benchmark default.

    The candidate override exists because providers do not offer identical
    inference controls — google/gemini-3.7-flash reports
    ``reasoning.mandatory = true`` and rejects the role's
    ``reasoning={"effort": "none"}`` outright. Benchmarking it in a state it
    cannot run in measures nothing; excluding it discards a deployable model.
    The asymmetry is therefore DECLARED (never inferred at runtime), and lands
    in ``fingerprint_fields`` -> the run's config hash, so a run carries the
    exact configuration it used.
    """
    from ..gateway import TaskRoute

    task = ROLE_TASKS[spec.role]
    prod = production_route_defaults(spec.models_config, task)
    knobs: dict = {
        "structured_mode": "json_schema",
        "max_tokens": default_max_tokens,
        "temperature": 0.0,
        "reasoning": None,
        "provider": None,
    }
    knobs.update(prod)
    # declared per-candidate asymmetry beats the production file
    if registry is not None:
        try:
            over = registry.for_role(spec.role).overrides_for(candidate)
        except KeyError:
            over = {}
        unknown = set(over) - set(CANDIDATE_OVERRIDABLE) - {"why"}
        if unknown:
            raise ValueError(f"{candidate}: candidate_overrides may only set "
                             f"{list(CANDIDATE_OVERRIDABLE)}; got {sorted(unknown)}")
        knobs.update({k: v for k, v in over.items() if k in CANDIDATE_OVERRIDABLE})
    # explicit CLI/spec overrides beat everything
    if spec.max_tokens is not None:
        knobs["max_tokens"] = spec.max_tokens
    if spec.reasoning is not None:
        knobs["reasoning"] = spec.reasoning
    if spec.provider is not None:
        knobs["provider"] = spec.provider
    return TaskRoute(
        task=task, backend=spec.backend, model=candidate, base_url=spec.base_url,
        prompt_version=request_prompt_version, cacheable=True, enabled=True, **knobs)


def build_gateway(spec: RunSpec, route, registry: CandidateRegistry, warn_sink: Callable[[str], None]):
    """Campaign gateway: ONE shared state root (cache + ledger + budget) for
    all benchmark runs, so the $10 ceiling is enforced over the whole campaign."""
    from ..backends import create_backend
    from ..gateway import ModelGateway
    from ..requestcache import RequestCache
    from ..usage import BudgetLimits, BudgetManager, UsageLedger

    hard = spec.hard_usd if spec.hard_usd is not None else (registry.experiment_total_usd or 10.0)
    warn = spec.warn_usd if spec.warn_usd is not None else (registry.warn_usd or 8.0)
    root = Path(spec.state_root)
    cache = RequestCache(root / "gateway_cache")
    ledger = UsageLedger(root / "gateway_ledger" / "usage.jsonl")

    def _factory(cfg):
        # benchmark routes: no validation-repair round-trips (comparability);
        # transport retries stay (network-only, recorded by the backend)
        return create_backend(dataclasses.replace(cfg, validation_retries=spec.validation_retries))

    gw = ModelGateway({route.task: route}, backend_factory=_factory, cache=cache, ledger=ledger, budget=None)
    limits = BudgetLimits(max_cost_total=float(hard), soft_fraction=float(warn) / float(hard) if hard else 0.8)
    gw.budget = BudgetManager(limits, ledger=ledger, warn=warn_sink)
    if spec.models_config is not None and Path(spec.models_config).exists():
        import tomllib
        data = tomllib.loads(Path(spec.models_config).read_text(encoding="utf-8"))
        gw.pricing_config = data.get("pricing") or None
    return gw


def files_root_for(manifest: BenchmarkManifest, bench_root: Path) -> Path:
    """Where a case's relative file paths (crops/images) resolve: the frozen
    hebrew_bench_v2 root for the OCR benchmarks (their crops/ dir), the
    dataset directory for declared datasets."""
    return Path(bench_root) if manifest.role in ("ocr_verify", "ocr_primary") else Path(manifest.root)


def _live_preflight(spec: RunSpec, gw, route, cases, adapter, files_root) -> dict:
    """Part 7 sequence before ANY provider request of a live run. The key
    metadata fetch is explicit and happens HERE only (never automatically
    elsewhere); the local ledger controls the ceiling."""
    from ..cloudcheck import openrouter_credential_present
    from ..spend import campaign_preflight
    from ..usage import predicted_call_cost
    fetcher = None
    backend = gw.backend_for(route.task)
    if hasattr(backend, "key_metadata"):
        fetcher = backend.key_metadata          # GET /api/v1/key, on demand, secret-free parse
    pricing = getattr(gw, "pricing_config", None)
    predicted = 0.0
    for c in cases[: max(1, min(len(cases), 50))]:
        req = adapter.build_request(dict(c.inputs), files_root)
        predicted += predicted_call_cost(route, req.system, req.content_blocks, pricing) or 0.0
    ledger_path = gw.ledger.path if getattr(gw, "ledger", None) is not None else Path(spec.state_root) / "gateway_ledger" / "usage.jsonl"
    hard = spec.hard_usd if spec.hard_usd is not None else 10.0
    warn = spec.warn_usd if spec.warn_usd is not None else 8.0
    return campaign_preflight(credential_present=openrouter_credential_present(), fetch_key_metadata=fetcher,
                              ledger=ledger_path, state_root=Path(spec.state_root), predicted_cost=predicted,
                              warn_usd=warn, hard_usd=hard)


def run_id_for(spec: RunSpec, candidate: str, config_hash: str) -> str:
    comp = (spec.component or "all").lower()
    sub = f"{spec.subset}__" if spec.subset else ""
    return f"{spec.split.lower()}__{sub}{comp}__{_slug(candidate)}__{config_hash[:10]}"


# ----------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------

def run_benchmark(spec: RunSpec, *, gateway=None, registry: CandidateRegistry | None = None,
                  manifest: BenchmarkManifest | None = None, progress: Callable[[str], None] | None = None,
                  now: Callable[[], str] | None = None) -> RunResult:
    log = progress or (lambda m: None)
    ts = now or (lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    split = spec.split.upper()
    if split not in SPLITS:
        raise ValueError(f"unknown split {spec.split!r}; expected one of {SPLITS}")
    registry = registry or load_registry(spec.registry_path)
    manifest = manifest or load_manifest(spec.role, bench_root=spec.bench_root, datasets_root=spec.datasets_root)
    if manifest.status == "NOT_BUILT" or not manifest.cases:
        from .manifests import BenchmarkNotBuilt
        raise BenchmarkNotBuilt(f"{spec.role}: benchmark dataset not built ({manifest.root}); "
                                + "; ".join(manifest.notes[:2]))
    if spec.component is not None and spec.component not in manifest.components:
        raise ValueError(f"{spec.role}: unknown component {spec.component!r}; have {manifest.components}")

    # ---- held-out discipline (before anything else) -------------------------
    if split == "HELD_OUT":
        if spec.dry_run:
            raise HeldOutRefused(
                "HELD_OUT is reserved for final evaluation and cannot be previewed/dry-run. "
                "Develop and select on DEV / CALIBRATION; the final evaluation path is "
                "`bench final-eval` (live, explicitly confirmed, permanently logged)")
        if not (spec.final_evaluation and spec.confirm_held_out):
            raise HeldOutRefused(
                "HELD_OUT can only be executed through the explicit final-evaluation path: "
                "`bench final-eval --role R --candidate SLUG --confirm-held-out "
                "--i-understand-this-spends-money`. The execution is logged permanently in "
                f"{spec.held_out_log}; inspected held-out results can never again be treated as unseen")
    if spec.subset is not None:
        from .subsets import SUBSET_RULES, available_subsets

        known = ["smoke"] + available_subsets(spec.role)
        if spec.subset not in known:
            raise ValueError(f"unknown subset {spec.subset!r}; pre-registered subsets for "
                             f"{spec.role}: {known}")
        if spec.subset == "smoke":
            if split != "DEV":
                raise ValueError("the smoke subset is DEV-only by construction; pass --split dev")
        else:
            want = SUBSET_RULES[(spec.role, spec.subset)][0]
            if split != want:
                raise ValueError(f"subset {spec.subset!r} is {want}-only by construction; "
                                 f"pass --split {want.lower()}")

    candidate = resolve_candidate(spec, registry)
    adapter = adapter_for(spec.role)
    cases = manifest.by_split(split, spec.component)
    if spec.subset == "smoke":
        from .smoke import DEFAULT_SMOKE_ROOT, smoke_case_ids
        ids = set(smoke_case_ids(spec.role, manifest, spec.smoke_root or DEFAULT_SMOKE_ROOT))
        cases = [c for c in cases if c.case_id in ids]
    elif spec.subset is not None:
        # a frozen FULL case list: the pre-registered evaluation population
        from .subsets import subset_case_ids
        ids = set(subset_case_ids(spec.role, spec.subset, manifest))
        cases = [c for c in cases if c.case_id in ids]
        if len(cases) != len(ids):
            missing = sorted(ids - {c.case_id for c in cases})
            raise ValueError(f"subset {spec.subset!r}: {len(missing)} frozen case(s) are not "
                             f"in this split/component selection: {missing[:5]}")
    if spec.limit:
        cases = cases[: spec.limit]
    if spec.role == "ocr_primary":
        # strict OCR scoring: every participating item must have an admissible
        # reference provenance (Part 1) — refuse, never fall back silently
        from .manifests import validate_reference_provenance
        validate_reference_provenance(manifest, [c.case_id for c in cases])
    if not cases:
        raise ValueError(f"{spec.role}: no cases for split {split}"
                         + (f" component {spec.component}" if spec.component else ""))

    # Provenance hashed into the run identity (first request's prompt/schema
    # stands for the adapter's contract; the adapter is deterministic).
    files_root = files_root_for(manifest, spec.bench_root)
    probe = adapter.build_request(dict(cases[0].inputs), files_root)
    prov = probe.provenance()
    route = build_route(spec, candidate, prov["prompt_version"], adapter.default_max_tokens,
                        registry=registry)
    config = {
        "role": spec.role, "task": route.task, "split": split, "component": spec.component,
        "subset": spec.subset,
        "candidate": candidate, "backend": spec.backend, "base_url": spec.base_url,
        "route": route.fingerprint_fields(), "adapter_version": adapter.adapter_version,
        "validation_retries": spec.validation_retries, **prov,
        "manifest_hashes": manifest.hashes,
        "case_ids_sha256": manifest.case_ids_sha256(split, spec.component),
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    run_id = run_id_for(spec, candidate, config_hash)
    run_dir = Path(spec.runs_root) / spec.role / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json = run_dir / "run.json"
    outputs_path = run_dir / "outputs.jsonl"

    existing = json.loads(run_json.read_text(encoding="utf-8")) if run_json.exists() else None
    if existing and existing.get("config_hash") != config_hash:
        raise RuntimeError(f"run directory {run_dir} belongs to a different configuration; refusing to mix")
    record = existing or {
        "run_id": run_id, "config_hash": config_hash, "created_at": ts(), "git_commit": _git_commit(),
        "spec": {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(spec).items()},
        "config": config, "manifest": manifest.summary(), "cases_selected": len(cases),
        "history": [],
    }
    record["history"].append({"ts": ts(), "mode": "dry_run" if spec.dry_run else "live",
                              "note": spec.note, "retry_failed": spec.retry_failed})

    # ---- resume state ---------------------------------------------------------
    prior = _read_jsonl(outputs_path)
    done_ok = {r["case_id"] for r in prior if r.get("ok")}
    done_failed = {r["case_id"] for r in prior if r.get("ok") is False}
    attempts = {}
    for r in prior:
        attempts[r["case_id"]] = max(attempts.get(r["case_id"], 0), int(r.get("attempt", 1)))
    skip = set(done_ok) | (set() if spec.retry_failed else done_failed)

    warnings: list[str] = []
    gw = None
    preflight: dict | None = None
    if not spec.dry_run:
        gw = gateway or build_gateway(spec, route, registry, warnings.append)
        # Cloud readiness is explained in one sentence, never a stack trace.
        from ..cloudcheck import require_cloud_task
        require_cloud_task(gw, route.task)
        ledger_baseline = len(gw.ledger.entries()) if getattr(gw, "ledger", None) is not None else 0
        # ---- live-call preflight (Part 7): credential -> key metadata (explicit)
        # -> checkpoint -> compare with ledger -> budget safe? -> allowed.
        # Only cloud routes are gated; local/mock routes have no account side.
        from ..usage import is_cloud_route
        if is_cloud_route(route.backend, route.base_url):
            # BEFORE any provider request: a model we cannot price cannot be
            # budget-gated, so it does not run at all.
            require_priced_candidate(getattr(route, "model", "") or "", getattr(gw, "pricing_config", None))
        if is_cloud_route(route.backend, route.base_url) and not spec.skip_key_preflight:
            preflight = _live_preflight(spec, gw, route, cases, adapter, files_root)
            if not preflight.get("allowed"):
                raise RuntimeError(f"live preflight refused the run before any provider request: "
                                   f"{preflight.get('reason')}")
    else:
        ledger_baseline = 0

    if split == "HELD_OUT":
        _append_jsonl(spec.held_out_log, {
            "ts": ts(), "role": spec.role, "split": split, "component": spec.component,
            "candidate": candidate, "run_id": run_id, "mode": "final_evaluation_live",
            "cases": len(cases), "config_hash": config_hash, "git_commit": _git_commit(),
            "prompt_version": prov["prompt_version"], "prompt_sha256": prov["prompt_sha256"],
            "schema_sha256": prov["schema_sha256"], "adapter_version": adapter.adapter_version,
            "manifest_hashes": manifest.hashes, "note": spec.note,
            "consequence": "HELD_OUT has been executed; once these results are inspected and used to "
                           "change the system, this split is no longer untouched and must be demoted "
                           "to DEV (docs/model-selection.md §split discipline)"})
        log(f"HELD_OUT final evaluation logged permanently to {spec.held_out_log}")


    predicted_total = 0.0
    plan_rows: list[dict] = []
    n_done = n_failed = n_skipped = 0
    stopped: str | None = None
    pricing = None
    if spec.models_config is not None and Path(spec.models_config).exists():
        import tomllib
        pricing = tomllib.loads(Path(spec.models_config).read_text(encoding="utf-8")).get("pricing")

    bench_root_for_files = files_root
    for case in cases:
        if case.case_id in skip:
            n_skipped += 1
            continue
        request = adapter.build_request(dict(case.inputs), bench_root_for_files)
        leakage_check(case, request, adapter.model_visible_fields)
        attempt = attempts.get(case.case_id, 0) + 1
        if spec.dry_run:
            from ..usage import predicted_call_cost
            est = predicted_call_cost(route, request.system, request.content_blocks, pricing)
            predicted_total += est or 0.0
            plan_rows.append({"case_id": case.case_id, "split": case.split, "component": case.component,
                              "predicted_cost": est, "text_chars": len(request.text_for_inspection()),
                              "images": sum(1 for b in request.content_blocks if b.get("type") == "image")})
            n_done += 1
            continue
        row: dict[str, Any] = {"case_id": case.case_id, "split": case.split, "component": case.component,
                               "attempt": attempt, "ts": ts(), "model": candidate, "task": route.task}
        try:
            res = gw.call(task=route.task, system=request.system, content_blocks=request.content_blocks,
                          output_model=request.output_model, max_tokens=request.max_tokens,
                          meta={"job_id": run_id, "exam_id": case.case_id, "stage": "benchmark",
                                "question_id": case.meta.get("question_id")})
        except Exception as e:  # noqa: BLE001 — classified below, never retried silently
            from ..usage import BudgetExceeded
            name = type(e).__name__
            if isinstance(e, BudgetExceeded):
                stopped = f"budget: {e}"
                row.update({"ok": None, "error_type": name, "error": str(e)[:500], "stopped": True})
                _append_jsonl(outputs_path, row)
                log(f"STOP: {stopped}")
                break
            is_schema = ("validation" in str(e).lower() or "schema" in str(e).lower()
                         or name in ("ValidationError", "SchemaError"))
            row.update({"ok": False, "error_type": name, "error": str(e)[:500],
                        "schema_failure": bool(is_schema), "usage": {}, "latency_s": None})
            _append_jsonl(outputs_path, row)
            n_failed += 1
            log(f"{case.case_id}: FAILED ({name})")
            continue
        value = res.value
        row.update({"ok": True, "output": value.model_dump() if hasattr(value, "model_dump") else value,
                    "usage": dict(res.usage or {}), "latency_s": res.latency_s, "cache_hit": res.cache_hit,
                    "fingerprint": res.fingerprint, "retries": getattr(res, "retries", 0)})
        _append_jsonl(outputs_path, row)
        n_done += 1
        log(f"{case.case_id}: ok ({'cache' if res.cache_hit else f'{res.latency_s}s'})")

    # ---- scoring / metrics --------------------------------------------------
    metrics: dict[str, Any] = {}
    usage_report: dict[str, Any] = {}
    if spec.dry_run:
        _write_json(run_dir / "plan.json", {
            "run_id": run_id, "mode": "dry_run", "cases": len(plan_rows), "skipped_resume": n_skipped,
            "predicted_cost_total": round(predicted_total, 6) if pricing else None,
            "pricing_table_available": bool(pricing), "rows": plan_rows, "leakage_check": "passed"})
        metrics = {"dry_run": True, "cases_planned": len(plan_rows),
                   "predicted_cost_total": round(predicted_total, 6) if pricing else None}
    else:
        all_rows = _read_jsonl(outputs_path)
        latest: dict[str, dict] = {}
        for r in all_rows:
            if r.get("ok") is None:
                continue
            latest[r["case_id"]] = r            # last attempt wins
        by_id = {c.case_id: c for c in cases}
        scored = []
        for cid, r in latest.items():
            c = by_id.get(cid)
            if c is None:
                continue
            scored.append(adapter.score(c, r.get("output") if r.get("ok") else None, r.get("error")))
        metrics = adapter.aggregate(scored, [latest[c] for c in latest])
        metrics["cases_selected"] = len(cases)
        metrics["cases_with_result"] = len(scored)
        _write_json(run_dir / "scored.jsonl.json", scored)
        _write_json(run_dir / "metrics.json", metrics)
        if gw is not None and getattr(gw, "ledger", None) is not None:
            from ..usage import run_cost_report
            usage_report = run_cost_report(gw.ledger, ledger_baseline)
            _write_json(run_dir / "usage.json", usage_report)

    record.update({"updated_at": ts(), "cases_done": n_done, "cases_failed": n_failed,
                   "cases_skipped_resume": n_skipped, "stopped_reason": stopped,
                   "last_mode": "dry_run" if spec.dry_run else "live", "warnings": warnings,
                   "last_preflight": preflight})
    _write_json(run_json, record)
    return RunResult(run_id=run_id, run_dir=run_dir, role=spec.role, split=split, component=spec.component,
                     candidate=candidate, dry_run=spec.dry_run, cases_selected=len(cases), cases_done=n_done,
                     cases_failed=n_failed, cases_skipped_resume=n_skipped, stopped_reason=stopped,
                     metrics=metrics, usage=usage_report,
                     predicted_cost=(round(predicted_total, 6) if (spec.dry_run and pricing) else None),
                     warnings=warnings)


def held_out_executions(path: Path = HELD_OUT_LOG) -> list[dict]:
    return _read_jsonl(Path(path))


__all__ = ["RunSpec", "RunResult", "HeldOutRefused", "UnselectedCandidate", "LeakageError",
           "run_benchmark", "leakage_check", "resolve_candidate", "build_gateway", "build_route", "files_root_for",
           "held_out_executions", "DEFAULT_STATE_ROOT", "DEFAULT_RUNS_ROOT", "HELD_OUT_LOG"]
