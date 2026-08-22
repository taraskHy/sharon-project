"""Frozen benchmark manifests: loading, hash verification, split selection.

A BenchmarkManifest is an in-memory, read-only view over frozen files. Every
loader verifies the on-disk bytes against the hashes recorded at freeze time
and refuses (BenchmarkIntegrityError) on any mismatch — a benchmark that has
drifted from its manifest is not a benchmark. Nothing here writes into a
frozen directory.

Model-visible vs evaluation-side is enforced structurally: ``BenchCase.inputs``
holds ONLY what an adapter may place in a request; ``BenchCase.label`` holds
evaluation-only fields and is never handed to an adapter's request builder
(runner.leakage_check re-verifies on every request).

Splits (docs/generalization.md): DEV may be inspected while developing;
CALIBRATION selects the final model/prompt/config; HELD_OUT is untouched and
needs explicit confirmation to execute (runner.py, permanently logged).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SPLITS = ("DEV", "CALIBRATION", "HELD_OUT")
ROLES = ("ocr_primary", "ocr_verify", "grade_primary", "grade_escalate",
         "mc_resolve_cloud", "variant_resolve", "align_resolve")
#: benchmark role -> ModelGateway task (models.toml [models.<task>])
ROLE_TASKS = {
    "ocr_primary": "ocr_primary",
    "ocr_verify": "ocr_verify",
    "grade_primary": "grade_primary",
    "grade_escalate": "grade_escalate",
    "mc_resolve_cloud": "mc_resolve_cloud",
    "variant_resolve": "variant_resolve_cloud",
    "align_resolve": "align_resolve_cloud",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH_ROOT = REPO_ROOT / "evaluation" / "hebrew_bench_v2"
DEFAULT_DATASETS_ROOT = REPO_ROOT / "evaluation" / "model_selection" / "datasets"

STATUS_FROZEN = "FROZEN"                  # frozen files, hashes verified
STATUS_DERIVED = "DERIVED_FROM_FROZEN"    # built deterministically from frozen, hash-verified inputs
STATUS_NOT_BUILT = "NOT_BUILT"            # declared role; dataset files do not exist yet


class BenchmarkIntegrityError(RuntimeError):
    """On-disk benchmark bytes do not match the recorded hashes."""


class BenchmarkNotBuilt(RuntimeError):
    """The role is declared but its dataset has not been built/frozen."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _require_hash(path: Path, expected: str | None, what: str) -> str:
    actual = sha256_file(path)
    if expected and actual != expected:
        raise BenchmarkIntegrityError(
            f"{what} hash mismatch: {path.name} on disk {actual[:12]}… != recorded {expected[:12]}… "
            "(the frozen benchmark has been modified or the manifest is stale)")
    return actual


@dataclass
class BenchCase:
    case_id: str
    split: str                          # DEV | CALIBRATION | HELD_OUT
    component: str                      # e.g. REAL | SYNTHETIC | ALL
    inputs: dict[str, Any]              # MODEL-VISIBLE ONLY
    label: dict[str, Any]               # evaluation-side only
    meta: dict[str, Any] = field(default_factory=dict)   # neutral descriptors (writer, category)


@dataclass
class BenchmarkManifest:
    role: str
    name: str
    status: str
    root: Path
    hashes: dict[str, str]
    components: list[str]
    cases: list[BenchCase]
    policy: str = ""
    notes: list[str] = field(default_factory=list)
    split_assignment: dict[str, list[str]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    # -- selection -----------------------------------------------------------
    def by_split(self, split: str, component: str | None = None) -> list[BenchCase]:
        s = split.upper()
        if s not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        return [c for c in self.cases
                if c.split == s and (component is None or c.component == component)]

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
        for c in self.cases:
            out.setdefault(c.split, {})
            out[c.split][c.component] = out[c.split].get(c.component, 0) + 1
        return out

    def case_ids_sha256(self, split: str | None = None, component: str | None = None) -> str:
        ids = sorted(c.case_id for c in self.cases
                     if (split is None or c.split == split.upper())
                     and (component is None or c.component == component))
        return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {"role": self.role, "name": self.name, "status": self.status,
                "root": str(self.root), "cases": len(self.cases), "components": self.components,
                "counts": self.counts(), "hashes": self.hashes,
                "split_assignment": self.split_assignment, "notes": self.notes}


# ----------------------------------------------------------------------------
# OCR_VERIFY (B2): frozen REAL + SYNTHETIC_NEAR_MISS
# ----------------------------------------------------------------------------

def _load_verifier_component(root: Path, sub: str, component: str) -> tuple[list[BenchCase], dict, dict]:
    d = root / "verifier_bench" / sub
    man_path = d / "manifest.json"
    if not man_path.exists():
        raise BenchmarkNotBuilt(f"{component} verifier benchmark not found at {d}")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    inputs_p, labels_p = d / "cases_inputs.jsonl", d / "cases_labels.jsonl"
    h_in = _require_hash(inputs_p, man.get("inputs_sha256"), f"{component} inputs")
    h_lab = _require_hash(labels_p, man.get("labels_sha256"), f"{component} labels")
    # CHECKSUMS.sha256 (when present) must agree with the manifest too
    ck = d / "CHECKSUMS.sha256"
    if ck.exists():
        for line in ck.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                h, name = parts[0], parts[-1].lstrip("*")
                target = d / name
                if target.exists() and sha256_file(target) != h:
                    raise BenchmarkIntegrityError(
                        f"{component}: CHECKSUMS.sha256 disagrees with {name} on disk")
    inputs = {r["case_id"]: r for r in _read_jsonl(inputs_p)}
    labels = {r["case_id"]: r for r in _read_jsonl(labels_p)}
    if set(inputs) != set(labels):
        raise BenchmarkIntegrityError(f"{component}: inputs/labels case-id sets differ")
    cases: list[BenchCase] = []
    for cid, inp in inputs.items():
        lab = labels[cid]
        visible = {"case_id": cid, "crop": inp["crop"],
                   "candidate_transcription": inp["candidate_transcription"]}
        extra_input_keys = set(inp) - set(visible)
        if extra_input_keys:
            raise BenchmarkIntegrityError(
                f"{component}: model-visible file carries unexpected fields {sorted(extra_input_keys)}")
        label = {k: v for k, v in lab.items() if k not in ("case_id",)}
        cases.append(BenchCase(
            case_id=cid, split=str(lab.get("split", "")).upper(), component=component,
            inputs=visible, label=label,
            meta={"writer": lab.get("writer"), "item_id": lab.get("item_id")}))
    hashes = {f"{component.lower()}_inputs_sha256": h_in, f"{component.lower()}_labels_sha256": h_lab,
              f"{component.lower()}_manifest_sha256": sha256_file(man_path)}
    return cases, man, hashes


def load_ocr_verify(root: Path = DEFAULT_BENCH_ROOT) -> BenchmarkManifest:
    real, man_r, h_r = _load_verifier_component(root, "selected", "REAL")
    synth, man_s, h_s = _load_verifier_component(root, "synthetic", "SYNTHETIC")
    # The synthetic layer records which REAL benchmark it was derived from:
    # both must be the same frozen object.
    rb = man_s.get("real_benchmark") or {}
    if isinstance(rb, dict) and rb.get("inputs_sha256") and rb["inputs_sha256"] != h_r["real_inputs_sha256"]:
        raise BenchmarkIntegrityError("SYNTHETIC manifest points at a different REAL benchmark")
    split_assignment = man_s.get("split_assignment") or (man_r.get("decision") or {}).get("writer_assignment") or {}
    notes = [
        "REAL and SYNTHETIC are separate components: report separately; COMBINED is secondary only",
        "primary metric: FALSE ACCEPT RATE (incorrect candidate -> SUPPORTED)",
        f"synthetic rules: {man_s.get('rules_version')}; selection: {man_s.get('selection_policy_version')}",
    ]
    return BenchmarkManifest(
        role="ocr_verify", name="hebrew_bench_v2 verifier benchmark (REAL + SYNTHETIC_NEAR_MISS)",
        status=STATUS_FROZEN, root=root / "verifier_bench",
        hashes={**h_r, **h_s, "audit_sha256": man_r.get("audit_sha256", "")},
        components=["REAL", "SYNTHETIC"], cases=real + synth,
        policy=str(man_r.get("_policy", "")), notes=notes,
        split_assignment=split_assignment,
        extra={"real_frozen_at": man_r.get("frozen_at"), "synthetic_frozen_at": man_s.get("frozen_at"),
               "real_report": man_r.get("report"), "synthetic_composition": man_s.get("composition")})


# ----------------------------------------------------------------------------
# OCR_PRIMARY (B1): frozen items + AUDITED references (reference_for_scoring)
# ----------------------------------------------------------------------------

def _load_refaudit():
    """scripts/refaudit.py is the reader of the frozen reference audit. It is
    imported by path (scripts/ is not a package) and cached in sys.modules so
    its dataclasses resolve; the benchmark uses ONLY its read API."""
    if "refaudit" in sys.modules:
        return sys.modules["refaudit"]
    path = REPO_ROOT / "scripts" / "refaudit.py"
    spec = importlib.util.spec_from_file_location("refaudit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refaudit"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


#: The ONLY reference provenance classes admitted to strict OCR scoring.
VALID_REFERENCE_CLASSES = ("audited_confirmed", "audited_corrected", "text_layer_mechanical")


def reference_provenance(ra, store, item: dict, ref_entry: dict, scoring_ref) -> dict:
    """Classify one item's scoring reference explicitly:

        audited_confirmed       owner tier, manual audit CONFIRMED the original text
        audited_corrected       owner tier, manual audit CORRECTED the text (audited text used)
        text_layer_mechanical   born-digital item outside the manual audit scope; reference
                                is the embedded PDF text layer (objective, mechanical provenance)
        INVALID:<why>           anything else (unchecked, ambiguous, unknown provenance,
                                human tier with mechanical provenance, ...)

    Strict model-selection scoring admits ONLY the three valid classes
    (VALID_REFERENCE_CLASSES); the runner refuses otherwise — no silent
    fallback to an unaudited historical reference."""
    iid = item["id"]
    tier = item.get("tier")
    provenance = str(ref_entry.get("provenance", "") or "")
    mechanical = bool(getattr(ra, "_MECHANICAL_PROVENANCE").search(provenance))
    if store.is_eligible(iid):
        status = scoring_ref.status
        if status == "confirmed":
            return {"provenance_class": "audited_confirmed", "valid": True,
                    "detail": f"tier {tier}; manual audit confirmed the original transcription"}
        if status == "corrected":
            return {"provenance_class": "audited_corrected", "valid": True,
                    "detail": f"tier {tier}; manual audit corrected the transcription (audited text used)"}
        return {"provenance_class": f"INVALID:audit_status_{status}", "valid": False,
                "detail": f"tier {tier}; audit status {status!r} is not admissible for strict scoring"}
    # outside the manual audit scope: must be a mechanical (text-layer) provenance
    if tier == "text-layer" and mechanical:
        return {"provenance_class": "text_layer_mechanical", "valid": True,
                "detail": f"tier {tier}; {provenance}"}
    return {"provenance_class": "INVALID:unknown_provenance", "valid": False,
            "detail": f"tier {tier}; provenance {provenance!r} is neither audited nor a text layer"}


def reference_breakdown(manifest: "BenchmarkManifest") -> dict:
    """The explicit 129-item accounting (Part 1): by category x provenance
    class, audit requirement, reference source, frozen/audited status."""
    if manifest.role != "ocr_primary":
        raise ValueError("reference_breakdown is defined for the ocr_primary manifest")
    rows = []
    for c in manifest.cases:
        rows.append({"item_id": c.case_id, "category": c.meta.get("category"), "tier": c.meta.get("tier"),
                     "manual_audit_required": c.meta.get("tier") == "owner",
                     "provenance_class": c.label.get("provenance_class"),
                     "reference_source": c.label.get("reference_source"),
                     "reference_status": c.label.get("reference_status"),
                     "valid_for_strict_scoring": bool(c.label.get("provenance_valid")),
                     "split": c.split})
    by_class: dict[str, int] = {}
    by_cat: dict[str, dict[str, int]] = {}
    for r in rows:
        by_class[r["provenance_class"]] = by_class.get(r["provenance_class"], 0) + 1
        by_cat.setdefault(r["category"], {})
        by_cat[r["category"]][r["provenance_class"]] = by_cat[r["category"]].get(r["provenance_class"], 0) + 1
    hand = [r for r in rows if r["manual_audit_required"]]
    other = [r for r in rows if not r["manual_audit_required"]]
    return {
        "total": len(rows),
        "handwritten_manual_audit": {
            "count": len(hand),
            "confirmed": sum(1 for r in hand if r["provenance_class"] == "audited_confirmed"),
            "corrected": sum(1 for r in hand if r["provenance_class"] == "audited_corrected"),
            "ambiguous": sum(1 for r in hand if r["provenance_class"].startswith("INVALID:audit_status_ambiguous")),
            "invalid": sum(1 for r in hand if not r["valid_for_strict_scoring"]),
            "by_category": {k: v for k, v in by_cat.items() if any(r["category"] == k for r in hand)},
        },
        "other_categories_text_layer": {
            "count": len(other),
            "why_trustworthy": ("born-digital PDFs: the reference is the embedded text layer (mechanical, "
                                "objective provenance recorded per item in references.json); never model output, "
                                "never a manual transcription; outside the manual audit scope by rule"),
            "by_category": {k: v for k, v in by_cat.items() if any(r["category"] == k for r in other)},
            "invalid": sum(1 for r in other if not r["valid_for_strict_scoring"]),
        },
        "by_provenance_class": by_class,
        "all_valid_for_strict_scoring": all(r["valid_for_strict_scoring"] for r in rows),
        "invalid_items": [r["item_id"] for r in rows if not r["valid_for_strict_scoring"]],
        "frozen": {"audit_sha256": manifest.hashes.get("audit_sha256"),
                   "items_sha256": manifest.hashes.get("items_sha256"),
                   "references_sha256": manifest.hashes.get("references_sha256")},
        "rows": rows,
    }


def validate_reference_provenance(manifest: "BenchmarkManifest", case_ids: list[str] | None = None) -> None:
    """Deterministic gate for strict OCR scoring: every participating item
    must carry one of VALID_REFERENCE_CLASSES. Raises BenchmarkIntegrityError
    listing the offenders — never falls back silently."""
    bad = [c.case_id for c in manifest.cases
           if (case_ids is None or c.case_id in case_ids)
           and c.label.get("provenance_class") not in VALID_REFERENCE_CLASSES]
    if bad:
        raise BenchmarkIntegrityError(
            f"{len(bad)} item(s) lack an admissible reference provenance for strict scoring "
            f"(admissible: {VALID_REFERENCE_CLASSES}): {bad[:10]}{'...' if len(bad) > 10 else ''}")


def load_ocr_primary(root: Path = DEFAULT_BENCH_ROOT) -> BenchmarkManifest:
    ra = _load_refaudit()
    store = ra.AuditStore(root)
    man_path = root / "reference_audit_manifest.json"
    if not man_path.exists():
        raise BenchmarkNotBuilt("reference audit is not frozen (no reference_audit_manifest.json)")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if not ra.is_frozen(store):
        raise BenchmarkIntegrityError(
            "reference audit / items.json / references.json no longer match the frozen manifest")
    items_hash = _require_hash(root / "items.json", man.get("items_sha256"), "items.json")
    refs_hash = _require_hash(root / "references.json", man.get("references_sha256"), "references.json")
    items = json.loads((root / "items.json").read_text(encoding="utf-8"))["items"]
    # Writer -> split assignment is the SAME Split A used by the verifier
    # benchmark (same images, same handwriting sources); text-layer items
    # (no writer; born-digital, objective references) are DEV.
    try:
        sel = json.loads((root / "verifier_bench" / "selected" / "manifest.json").read_text(encoding="utf-8"))
        writer_split = (sel.get("decision") or {}).get("writer_assignment") or {}
    except FileNotFoundError:
        writer_split = {}
    w2s = {w: s for s, ws in writer_split.items() for w in ws}
    refs_raw = json.loads((root / "references.json").read_text(encoding="utf-8"))
    cases: list[BenchCase] = []
    eligible = 0
    for it in items:
        iid = it["id"]
        ref = ra.reference_for_scoring(store, iid, "final")
        prov = reference_provenance(ra, store, it, refs_raw.get(iid) or {}, ref)
        split = w2s.get(it.get("writer"), "DEV") if it.get("writer") else "DEV"
        cases.append(BenchCase(
            case_id=iid, split=split, component="ALL",
            inputs={"case_id": iid, "image": it["image"], "category": it["category"],
                    "task": it.get("task", "")},
            label={"reference": ref.reference, "reference_status": ref.status,
                   "reference_source": ref.source, "use_for_strict_cer": ref.use_for_strict_cer,
                   "provenance_class": prov["provenance_class"],
                   "provenance_valid": prov["valid"],
                   "provenance_detail": prov["detail"],
                   "tier": it.get("tier"), "hard": bool(it.get("hard", False))},
            meta={"writer": it.get("writer"), "category": it["category"], "tier": it.get("tier")}))
        eligible += 1 if ref.reference is not None else 0
    return BenchmarkManifest(
        role="ocr_primary", name="hebrew_bench_v2 OCR benchmark (audited references)",
        status=STATUS_FROZEN, root=root,
        hashes={"items_sha256": items_hash, "references_sha256": refs_hash,
                "audit_sha256": man.get("audit_sha256", ""),
                "audit_manifest_sha256": sha256_file(man_path)},
        components=["ALL"], cases=cases, policy=str(man.get("_policy", "")),
        notes=[f"eligibility: {man.get('eligibility_rule')}",
               f"audit summary: {man.get('summary')}",
               "references come from reference_for_scoring(mode='final') — audited text where "
               "corrected, original where confirmed/out-of-scope; never the unaudited text as final truth",
               "historical outputs stay readable via `bench report --role ocr_primary --historical`"],
        split_assignment=writer_split,
        extra={"items_with_reference": eligible})


# ----------------------------------------------------------------------------
# Declared roles (B3/B4/B5): generic frozen-dataset format
# ----------------------------------------------------------------------------
#
#   <datasets_root>/<role>/manifest.json      {name, inputs_sha256, labels_sha256,
#                                              split_assignment?, components?, policy, notes, frozen_at}
#   <datasets_root>/<role>/cases_inputs.jsonl  {case_id, ...model-visible fields...}
#   <datasets_root>/<role>/cases_labels.jsonl  {case_id, split, component?, ...labels...}
#
# Builders for these live in benchmark/datasets.py (they write ONLY into
# datasets_root, never into the frozen hebrew_bench_v2 tree).

DECLARED_ROLE_NOTES = {
    "grade_primary": [
        "B3: frozen transcriptions + NO-RAG grading packs; no OCR model runs during grading selection",
        "label reality: no per-item instructor labels exist yet; until an owner-scored subset exists only "
        "decision/AUTO/REVIEW/validation metrics are computed and accuracy metrics are reported as unavailable",
        "build: `autograder bench build-grading` (benchmark/datasets.py); owner labels via scripts/grade_label_ui.py",
    ],
    "grade_escalate": [
        "B4: the escalation subset of B3 (primary unclean / uncertain / disagreement) + pre-registered G-cells",
        "harvested from grade_primary DEV/CALIBRATION runs: `autograder bench build-escalation --from-run <run_dir>`",
    ],
    "mc_resolve_cloud": [
        "B5a: audited MC rows (evaluation/prob/manual_audit.json) — needs the band crops exported per row",
        "build: `autograder bench build-mc` (benchmark/datasets.py)",
    ],
    "variant_resolve": [
        "B5b: labeled cover crops (prob suits + Stage-A flowers) — build the cover-crop set from existing scans",
        "build: `autograder bench build-variant` (benchmark/datasets.py)",
    ],
    "align_resolve": [
        "B5c: operator-verified permutations sample_data/Exam_solution.alignment.json (A1/A2/A3)",
        "build: `autograder bench build-align` (benchmark/datasets.py)",
    ],
}


def load_declared(role: str, datasets_root: Path = DEFAULT_DATASETS_ROOT) -> BenchmarkManifest:
    d = Path(datasets_root) / role
    man_path = d / "manifest.json"
    notes = list(DECLARED_ROLE_NOTES.get(role, []))
    if not man_path.exists():
        return BenchmarkManifest(role=role, name=f"{role} benchmark (declared, not built)",
                                 status=STATUS_NOT_BUILT, root=d, hashes={}, components=[],
                                 cases=[], notes=notes + [f"expected at {d}"])
    man = json.loads(man_path.read_text(encoding="utf-8"))
    inputs_p, labels_p = d / "cases_inputs.jsonl", d / "cases_labels.jsonl"
    h_in = _require_hash(inputs_p, man.get("inputs_sha256"), f"{role} inputs")
    h_lab = _require_hash(labels_p, man.get("labels_sha256"), f"{role} labels")
    inputs = {r["case_id"]: r for r in _read_jsonl(inputs_p)}
    labels = {r["case_id"]: r for r in _read_jsonl(labels_p)}
    if set(inputs) != set(labels):
        raise BenchmarkIntegrityError(f"{role}: inputs/labels case-id sets differ")
    hashes = {"inputs_sha256": h_in, "labels_sha256": h_lab, "manifest_sha256": sha256_file(man_path)}
    owner_merged = 0
    final_merged = 0
    if role in ("grade_primary", "grade_escalate"):
        # Human labels live in SEPARATE files, never inside the frozen dataset:
        #   final_labels.json  — FINAL ground truth imported from the shared
        #                        labeling app (agreement / adjudicated only)
        #   owner_labels.json  — the local single-owner tool (confirmed only)
        # FINAL wins where both exist; both hashes join the run identity.
        from .finallabels import merge_final_labels
        from .ownerlabels import OwnerLabelStore, merge_owner_labels
        store = OwnerLabelStore(d)
        if store.path.exists():
            owner_merged = merge_owner_labels(labels, store)
            hashes["owner_labels_sha256"] = store.sha256()
        final_merged, final_sha = merge_final_labels(labels, d)
        if final_sha:
            hashes["final_labels_sha256"] = final_sha
    cases = []
    for cid, inp in inputs.items():
        lab = labels[cid]
        cases.append(BenchCase(
            case_id=cid, split=str(lab.get("split", "DEV")).upper(),
            component=str(lab.get("component", "ALL")),
            inputs=dict(inp), label={k: v for k, v in lab.items() if k not in ("case_id", "split", "component")},
            meta={k: lab.get(k) for k in ("writer", "exam_id", "question_id", "sub_item_id") if k in lab}))
    comps = sorted({c.component for c in cases})
    extra = dict(man.get("extra", {}))
    extra["owner_labels_merged"] = owner_merged
    extra["final_labels_merged"] = final_merged
    return BenchmarkManifest(
        role=role, name=man.get("name", f"{role} benchmark"), status=man.get("status", STATUS_FROZEN),
        root=d, hashes=hashes,
        components=comps, cases=cases, policy=str(man.get("policy", "")),
        notes=notes + list(man.get("notes", [])),
        split_assignment=man.get("split_assignment", {}), extra=extra)


def load_manifest(role: str, *, bench_root: Path = DEFAULT_BENCH_ROOT,
                  datasets_root: Path = DEFAULT_DATASETS_ROOT) -> BenchmarkManifest:
    if role not in ROLES:
        raise ValueError(f"unknown benchmark role {role!r}; expected one of {ROLES}")
    if role == "ocr_verify":
        return load_ocr_verify(bench_root)
    if role == "ocr_primary":
        return load_ocr_primary(bench_root)
    return load_declared(role, datasets_root)


def all_manifest_summaries(*, bench_root: Path = DEFAULT_BENCH_ROOT,
                           datasets_root: Path = DEFAULT_DATASETS_ROOT) -> dict[str, dict]:
    """Status of every role's benchmark (readiness / `bench list`). Integrity
    errors are reported, not raised, so one drifted benchmark cannot hide the
    status of the others."""
    out: dict[str, dict] = {}
    for role in ROLES:
        try:
            out[role] = load_manifest(role, bench_root=bench_root, datasets_root=datasets_root).summary()
        except BenchmarkIntegrityError as e:
            out[role] = {"role": role, "status": "INTEGRITY_ERROR", "error": str(e)}
        except BenchmarkNotBuilt as e:
            out[role] = {"role": role, "status": STATUS_NOT_BUILT, "error": str(e)}
    return out


__all__ = ["SPLITS", "ROLES", "ROLE_TASKS", "STATUS_FROZEN", "STATUS_DERIVED", "STATUS_NOT_BUILT",
           "BenchCase", "BenchmarkManifest", "BenchmarkIntegrityError", "BenchmarkNotBuilt",
           "load_manifest", "load_ocr_verify", "load_ocr_primary", "load_declared",
           "reference_provenance", "reference_breakdown", "validate_reference_provenance", "VALID_REFERENCE_CLASSES",
           "all_manifest_summaries", "sha256_file", "DEFAULT_BENCH_ROOT", "DEFAULT_DATASETS_ROOT"]
