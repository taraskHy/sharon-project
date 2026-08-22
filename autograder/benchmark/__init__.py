"""Provider-independent model-selection benchmark harness.

One framework for every cloud model role (docs/model-selection.md):

    ocr_primary       B1  frozen hebrew_bench_v2 items + audited references
    ocr_verify        B2  frozen REAL + SYNTHETIC_NEAR_MISS verifier benchmarks
    grade_primary     B3  frozen transcriptions + grading packs (NO-RAG first)
    grade_escalate    B4  the escalation subset of B3
    mc_resolve_cloud  B5a audited MC rows
    variant_resolve   B5b audited cover/marker reads
    align_resolve     B5c operator-verified alignment permutations

Layout:
    manifests.py  frozen-manifest loading with hash verification, split/
                  component selection (DEV / CALIBRATION / HELD_OUT)
    registry.py   candidates.toml (candidates + campaign budget; DATA only)
    roles.py      per-role adapters: production prompt/schema -> request,
                  model-visible whitelist, per-case scoring, aggregation
    runner.py     the run loop: ModelGateway routing, cache, ledger, budget,
                  resume, raw-output persistence, held-out confirmation,
                  leakage guard, dry-run (zero calls)
    report.py     run/compare reports (REAL and SYNTHETIC always separate)
    cli.py        `autograder bench ...`

Nothing in this package contacts a provider unless ``runner.run_benchmark``
is called with ``dry_run=False`` AND a usable gateway — and even then every
call goes through ModelGateway (privacy scan, request cache, usage ledger,
budget). No winner is chosen here: results are written, never promoted.
"""
from .manifests import (BenchCase, BenchmarkIntegrityError, BenchmarkManifest,
                        BenchmarkNotBuilt, ROLES, ROLE_TASKS, SPLITS, load_manifest)
from .registry import CandidateRegistry, load_registry
from .runner import (HeldOutRefused, LeakageError, RunSpec, RunResult, UnselectedCandidate,
                     run_benchmark)

__all__ = ["BenchCase", "BenchmarkIntegrityError", "BenchmarkManifest", "BenchmarkNotBuilt",
           "ROLES", "ROLE_TASKS", "SPLITS", "load_manifest", "CandidateRegistry",
           "load_registry", "HeldOutRefused", "LeakageError", "RunSpec", "RunResult",
           "UnselectedCandidate", "run_benchmark"]
