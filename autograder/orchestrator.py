"""End-to-end plumbing: turns the new modules into the lecturer flow.

    setup_from_config(models_toml, jobs_root)   -> Runtime (gateway+cache+ledger+budget)
    prepare_exam_package(runtime, key, key_path, ...) -> discovery + policies + packs
                                                     (once per package, persisted)
    install_hooks(runtime, policies)            -> MC chain + policy gate in the
                                                   validated pipeline (per process)
    handle_model_failure(...)                   -> pause semantics for jobs

Nothing here changes behavior unless the runtime is explicitly enabled
(--models-config on the CLI); with no config the pipeline runs exactly as
before. OpenRouter is never mandatory: a config with only local/mock tasks
is valid.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .backends import BackendError
from .discovery import VariantCatalogStore, discover_package, write_sidecars
from .gateway import GatewayConfigError, ModelGateway
from .gradingpack import (DEFAULT_RAG_CHAR_BUDGET, DEFAULT_RAG_TOP_K, PackStore,
                          build_all_packs, source_fingerprint)
from .preflight import alignment_from_discovery, preflight_package
from .requestcache import RequestCache
from .schema import AnswerKey
from .usage import BudgetExceeded, BudgetLimits, BudgetManager, UsageLedger


@dataclass
class Runtime:
    gateway: ModelGateway
    cache: RequestCache
    ledger: UsageLedger
    budget: BudgetManager
    root: Path
    warnings: list[str] = field(default_factory=list)


def setup_from_config(models_config: str | Path, state_root: str | Path,
                      budget: BudgetLimits | None = None,
                      backend_factory: Callable | None = None) -> Runtime:
    """Build the gateway + cache + ledger + budget from models.toml. Local-
    only configs are fine (OpenRouter tasks simply absent/disabled)."""
    root = Path(state_root)
    cache = RequestCache(root / "gateway_cache")
    ledger = UsageLedger(root / "gateway_ledger" / "usage.jsonl")
    warnings: list[str] = []
    # Build the gateway first (parses [budget]); then wire the manager from
    # config unless the caller supplied explicit limits. No [budget] or
    # enabled=false -> unlimited manager (still counts usage).
    gw = ModelGateway.from_file(models_config, cache=cache, ledger=ledger, budget=None,
                                **({"backend_factory": backend_factory} if backend_factory else {}))
    limits = budget or BudgetLimits.from_config(gw.budget_config) or BudgetLimits()
    bm = BudgetManager(limits, ledger=ledger, warn=warnings.append)
    gw.budget = bm
    return Runtime(gw, cache, ledger, bm, root, warnings)


def prepare_exam_package(runtime: Runtime | None, *, key: AnswerKey, key_bytes: bytes,
                         key_path: str | Path, exam_bytes: bytes | None = None,
                         exam_text_layer: str = "", cover_png_b64: str | None = None,
                         rubric_texts: dict[str, str] | None = None,
                         course_id: str | None = None, course_index_hash: str | None = None,
                         retrieve=None, embed_fn=None,
                         rag_top_k: int = DEFAULT_RAG_TOP_K,
                         rag_char_budget: int = DEFAULT_RAG_CHAR_BUDGET,
                         rag_policy: str = "RAG_DISABLED",
                         packages_root: str | Path | None = None,
                         write_missing_sidecars: bool = True) -> dict:
    """Once per exam package: discovery (variants/template/policies) ->
    sidecars in existing contracts -> QuestionGradingPacks. Everything is
    persisted under packages_root/<fingerprint>/ and reused verbatim next time."""
    gw = runtime.gateway if runtime else None
    root = Path(packages_root or (runtime.root if runtime else Path("packages")))
    disc = discover_package(key=key, key_bytes=key_bytes, exam_bytes=exam_bytes,
                            exam_text_layer=exam_text_layer, cover_png_b64=cover_png_b64,
                            rubric_texts=rubric_texts, gateway=gw)
    catalog = VariantCatalogStore(root / "variant_catalog")
    prior = catalog.load(disc.package_fingerprint)
    if prior:
        # human resolutions from an earlier identical package win
        for k in ("variants_config", "alignment", "template"):
            if isinstance(prior.get(k), dict) and prior[k].get("source") == "human":
                setattr(disc, k, type(disc.variants_config)(prior[k]["value"], "human", "reused"))
        for qid, f in (prior.get("policies") or {}).items():
            if isinstance(f, dict) and f.get("source") == "human" and qid in disc.policies:
                disc.policies[qid] = type(disc.variants_config)(f["value"], "human", "reused")
        disc.needs_human = [n for n in disc.needs_human if not (
            (n == "variants" and disc.variants_config.source == "human")
            or (n.startswith("policy:") and disc.policies.get(n.split(":", 1)[1], None)
                and disc.policies[n.split(":", 1)[1]].source == "human"))]
    catalog.save(disc)
    written = write_sidecars(disc, key_path) if write_missing_sidecars else []
    policies = {qid: f.value for qid, f in disc.policies.items() if f.value}
    fp = source_fingerprint(key_bytes, course_index_hash, policies, rag_top_k, rag_char_budget,
                            rag_policy=rag_policy)
    store = PackStore(root / "packs" / disc.package_fingerprint)
    packs = store.load(fp)
    if packs is None:
        packs = build_all_packs(key, policies, course_id=course_id, retrieve=retrieve, embed_fn=embed_fn,
                                rag_top_k=rag_top_k, rag_char_budget=rag_char_budget,
                                rag_policy=rag_policy)
        store.save(packs, fp)
    # Package-level preflight: a structural defect must surface ONCE here, not
    # as one review per student later (see preflight.py / docs).
    versions = list(disc.versions.value or key.versions or [])
    pre = preflight_package(
        key=key, variants=versions,
        alignment=alignment_from_discovery(disc.alignment.value, versions, key),
        policies=policies,
        rubric_question_ids=list((rubric_texts or {}).keys()) or None,
        template=disc.template.value, unresolved=disc.unresolved())
    return {"discovery": disc, "policies": policies, "packs": packs, "sidecars_written": [str(p) for p in written],
            "unresolved": disc.unresolved(), "package_fingerprint": disc.package_fingerprint,
            "pack_fingerprint": fp, "preflight": pre, "package_status": pre.status,
            "setup_required": [f.as_dict() for f in pre.blocking]}


def install_hooks(runtime: Runtime | None, policies: dict[str, str] | None,
                  *, min_confidence: float = 0.9, allow_cloud_mc: bool = True) -> None:
    """Attach the MC resolution chain + policy gate to the validated pipeline
    for THIS process. runtime None removes both (legacy behavior)."""
    from . import extract, grade
    from .mcresolve import resolve_row

    grade.set_grading_policies(policies or None, min_confidence)
    if runtime is None:
        extract.set_mc_resolver(None)
        return

    def chain(**kw):
        return resolve_row(gateway=runtime.gateway, allow_cloud=allow_cloud_mc, **kw)

    extract.set_mc_resolver(chain)


# ------------------------------------------------------ failure semantics ---


def handle_model_failure(exc: Exception, job_dir: str | Path, exam_id: str, stage: str) -> dict:
    """Classify a model-dependent failure for the job runner: cloud/budget
    problems PAUSE (item stays pending, deterministic/local results are
    kept); anything else is a normal exam failure. Never restarts an exam."""
    job_dir = Path(job_dir)
    if isinstance(exc, BudgetExceeded):
        kind, action = "budget", "pause_job"
    elif isinstance(exc, BackendError) and any(k in str(exc).lower() for k in
                                              ("openrouter", "429", "unreachable", "timed out", "rate")):
        kind, action = "provider_unavailable", "pause_item"
    else:
        kind, action = "other", "fail_item"
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "exam_id": exam_id, "stage": stage,
           "kind": kind, "action": action, "error": type(exc).__name__ + ": " + str(exc)[:300]}
    p = job_dir / "model_failures.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if action == "pause_job":
        (job_dir / "pause.request").write_text("budget", encoding="utf-8")
    return rec


def openrouter_configured(runtime: Runtime | None) -> bool:
    from .usage import effective_provider

    if runtime is None:
        return False
    return any(r.enabled and effective_provider(r.backend, r.base_url) == "openrouter"
               for r in runtime.gateway.routes.values()) \
        and bool(os.environ.get("OPENROUTER_API_KEY"))
