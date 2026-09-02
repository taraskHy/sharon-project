"""Frozen FULL case lists for a role — the evaluation population, pre-registered.

A smoke subset is a handful of slot-filling cases for the first paid call. This
module is the other thing: the complete set of cases a comparison is run over,
frozen BEFORE any model output exists so nobody can widen, narrow or reshuffle
the population after seeing a result.

    evaluation/model_selection/subsets/<role>__<name>.json

The selection is a pure function of the frozen dataset and a declared
predicate — never of a model's behaviour. Loading re-verifies the selection
hash, that every id exists in the manifest, and that every id is in the
declared split.

Why this exists rather than "just run the split": DEV holds 32 grade_primary
cases but only 26 carry a derivable explanation verdict. Running the split
would pay for 6 cases with no ground truth and quietly change the denominator
of every metric.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from .manifests import REPO_ROOT, BenchCase, BenchmarkManifest

DEFAULT_SUBSET_ROOT = REPO_ROOT / "evaluation" / "model_selection" / "subsets"
SUBSET_RULES_VERSION = ("subset-rules-v1 (2026-08-25; a frozen full case list is a pure "
                        "function of the frozen dataset and the declared predicate)")


class SubsetError(RuntimeError):
    """Frozen subset missing, tampered, or inconsistent with the manifest."""


def _derivable_supported_verdict(c: BenchCase) -> bool:
    """A case usable as verdict ground truth: the derivation is unique AND the
    class has support in this dataset.

    ``invalid`` is deliberately absent from the accepted set — not because it
    is uninteresting, but because no authoritative example exists in any split
    (every zero-score DEV case turned out to have a WRONG selection, so its
    explanation was never the reason for the zero). Including the class here
    would silently admit cases with no ground truth.
    """
    return (bool(c.label.get("explanation_verdict_derivable"))
            and c.label.get("explanation_verdict") in ("valid", "partially_valid"))


_SEEN46_CAMPAIGN = (REPO_ROOT / "evaluation" / "model_selection" / "experiments"
                    / "OCR_VALIDATION_CAMPAIGN_2026-09-02.json")
_SEEN46_CROPS: set[str] | None = None


def _seen46_campaign_crops() -> set[str]:
    """Crop basenames frozen in the OCR validation campaign (self-hash
    verified on every load). The subset selection stays a pure function of
    frozen artifacts: the campaign freeze + the frozen manifest."""
    global _SEEN46_CROPS
    if _SEEN46_CROPS is None:
        if not _SEEN46_CAMPAIGN.exists():
            raise SubsetError(f"campaign freeze missing: {_SEEN46_CAMPAIGN}")
        doc = json.loads(_SEEN46_CAMPAIGN.read_text(encoding="utf-8"))
        payload = json.dumps({k: v for k, v in doc.items()
                              if k != "experiment_sha256"},
                             ensure_ascii=False, sort_keys=True)
        if hashlib.sha256(payload.encode()).hexdigest() != doc.get("experiment_sha256"):
            raise SubsetError("campaign freeze failed its self-hash check")
        _SEEN46_CROPS = {rel.split("/")[-1] for c in doc["cases"]
                         for rel in c["evidence_crops"]}
    return _SEEN46_CROPS


def _seen46_evidence_crop(c: BenchCase) -> bool:
    """This bench item's crop is one of the frozen SEEN-46 evidence crops.
    (The one human-repaired crop lives in the grade dataset's repair store,
    not in hebrew_bench_v2, so it can never match a bench item — documented
    in the campaign freeze.)"""
    return str(c.inputs.get("image", "")).split("/")[-1] in _seen46_campaign_crops()


#: (role, subset name) -> (split, why, predicate)
SUBSET_RULES: dict[tuple[str, str], tuple[str, str, Callable[[BenchCase], bool]]] = {
    # The OCR-validation population: every ocr_primary bench item whose crop
    # is SEEN-46 grading evidence (campaign freeze 2026-09-02). Split-scoped
    # because writers straddle DEV (e002/e003/e007) and CALIBRATION (e004).
    ("ocr_primary", "seen46_ocr_dev"): (
        "DEV",
        "every DEV ocr_primary item whose crop is frozen SEEN-46 grading "
        "evidence (OCR_VALIDATION_CAMPAIGN_2026-09-02)",
        _seen46_evidence_crop,
    ),
    ("ocr_primary", "seen46_ocr_calibration"): (
        "CALIBRATION",
        "every CALIBRATION ocr_primary item whose crop is frozen SEEN-46 "
        "grading evidence (OCR_VALIDATION_CAMPAIGN_2026-09-02)",
        _seen46_evidence_crop,
    ),
    # Same POPULATION as calibration_verdict, recorded as a separate experiment
    # for the grade-v4-charitable prompt. The v3 freeze is never edited: two
    # prompts are two experiments, and their identical selection_sha256 is the
    # proof that only the prompt changed and not the case list.
    ("grade_primary", "calibration_verdict_v4"): (
        "CALIBRATION",
        "the calibration_verdict population, re-registered for the grade-v4-charitable prompt",
        _derivable_supported_verdict,
    ),
    ("grade_primary", "calibration_verdict"): (
        "CALIBRATION",
        "every CALIBRATION case whose canonical explanation verdict is mathematically derivable "
        "and belongs to a class that has ground-truth support (valid | partially_valid)",
        _derivable_supported_verdict,
    ),
    ("grade_primary", "dev_verdict"): (
        "DEV",
        "every DEV case whose canonical explanation verdict is mathematically derivable "
        "and belongs to a class that has ground-truth support in this dataset "
        "(valid | partially_valid; invalid has zero support)",
        _derivable_supported_verdict,
    ),
}


def subset_path(role: str, name: str, root: Path = DEFAULT_SUBSET_ROOT) -> Path:
    return Path(root) / f"{role}__{name}.json"


def _selection_hash(cases: list[dict]) -> str:
    payload = json.dumps([{"case_id": c["case_id"], "verdict": c.get("verdict")}
                          for c in cases], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _case_row(c: BenchCase) -> dict[str, Any]:
    lab = c.label
    writer, question, sub = c.case_id.split("_")
    return {
        "case_id": c.case_id,
        "split": c.split,
        "component": c.component,
        "writer": writer,
        "question_id": question[1:],
        "sub_item_id": sub[1:],
        "verdict": lab.get("explanation_verdict"),
        "verdict_reason": lab.get("explanation_verdict_reason"),
        "instructor_final_score": lab.get("score"),
        "max_score": (c.inputs.get("pack") or {}).get("max_score"),
        "selection_correct": lab.get("selection_correct"),
        "selection_correct_source": lab.get("selection_correct_source"),
        "transcription_chars": len(c.inputs.get("transcription") or ""),
    }


def _ocr_case_row(c: BenchCase) -> dict[str, Any]:
    """OCR items carry no grade-side fields; the row records identity and
    reference PROVENANCE only — never the reference text itself."""
    return {
        "case_id": c.case_id,
        "split": c.split,
        "component": c.component,
        "writer": c.meta.get("writer"),
        "category": c.meta.get("category"),
        "tier": c.meta.get("tier"),
        "hard": c.label.get("hard"),
        "image": c.inputs.get("image"),
        "reference_status": c.label.get("reference_status"),
        "provenance_class": c.label.get("provenance_class"),
        "provenance_valid": c.label.get("provenance_valid"),
        "verdict": None,     # OCR subsets have no verdict target
    }


def propose_subset(role: str, name: str, manifest: BenchmarkManifest) -> dict[str, Any]:
    """Deterministic, model-blind selection. Pure function of the manifest."""
    key = (role, name)
    if key not in SUBSET_RULES:
        raise SubsetError(f"no subset rule for {role!r}/{name!r} "
                          f"(known: {sorted(k[1] for k in SUBSET_RULES if k[0] == role)})")
    split, why, pred = SUBSET_RULES[key]
    chosen = sorted((c for c in manifest.by_split(split) if pred(c)), key=lambda c: c.case_id)
    excluded = sorted((c.case_id, c.label.get("explanation_verdict_reason"))
                      for c in manifest.by_split(split) if not pred(c))
    row_builder = _ocr_case_row if role.startswith("ocr") else _case_row
    cases = [row_builder(c) for c in chosen]
    dist: dict[str, int] = {}
    for r in cases:
        dist[r["verdict"]] = dist.get(r["verdict"], 0) + 1
    return {
        "role": role, "name": name, "split": split, "why": why,
        "rules_version": SUBSET_RULES_VERSION,
        "manifest_hashes": dict(manifest.hashes),
        "cases": cases,
        "case_count": len(cases),
        "class_distribution": dist,
        "excluded": [{"case_id": cid, "reason": reason} for cid, reason in excluded],
        "selection_sha256": _selection_hash(cases),
    }


def freeze_subset(role: str, name: str, manifest: BenchmarkManifest,
                  root: Path = DEFAULT_SUBSET_ROOT, *, now: str | None = None,
                  git_commit: str | None = None, prompt_version: str | None = None,
                  adapter_version: str | None = None,
                  candidate_configs: dict | None = None,
                  expect_count: int | None = None,
                  expect_distribution: dict | None = None) -> dict[str, Any]:
    """Write the frozen case list once. Refuses to overwrite.

    ``expect_count`` / ``expect_distribution`` are an explicit agreement check:
    the caller states what the population should be, and the freeze refuses if
    the dataset disagrees. A population that silently changed size between the
    plan and the freeze is exactly what a pre-registration exists to prevent.
    """
    p = subset_path(role, name, root)
    if p.exists():
        raise SubsetError(f"{p} already exists: a frozen subset is never re-selected")
    prop = propose_subset(role, name, manifest)
    if expect_count is not None and prop["case_count"] != expect_count:
        raise SubsetError(f"expected {expect_count} cases, the dataset yields "
                          f"{prop['case_count']}; refusing to freeze a population that "
                          "does not match the pre-registered plan")
    if expect_distribution is not None and prop["class_distribution"] != expect_distribution:
        raise SubsetError(f"expected class distribution {expect_distribution}, got "
                          f"{prop['class_distribution']}; refusing to freeze")
    prop.update({
        "frozen_at": now or time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit,
        "prompt_version": prompt_version,
        "adapter_version": adapter_version,
        "candidate_configs": candidate_configs or {},
        "_policy": ("Pre-registered evaluation population. Frozen BEFORE any model output "
                    "for these cases exists; never widened, narrowed or reshuffled "
                    "afterwards. Model-visible inputs remain ONLY the manifest's inputs "
                    "for these case ids."),
    })
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prop, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return prop


def load_subset(role: str, name: str, manifest: BenchmarkManifest | None = None,
                root: Path = DEFAULT_SUBSET_ROOT) -> dict[str, Any]:
    """Load + verify: hash matches, every id exists, every id is in the split."""
    p = subset_path(role, name, root)
    if not p.exists():
        raise SubsetError(f"no frozen subset {name!r} for {role} (expected {p})")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("role") != role or d.get("name") != name:
        raise SubsetError(f"{p}: wrong role/name ({d.get('role')}, {d.get('name')})")
    if _selection_hash(d.get("cases", [])) != d.get("selection_sha256"):
        raise SubsetError(f"{p}: selection hash mismatch — the subset was modified after freezing")
    split = d.get("split")
    if manifest is not None:
        by_id = {c.case_id: c for c in manifest.cases}
        for row in d["cases"]:
            bc = by_id.get(row["case_id"])
            if bc is None:
                raise SubsetError(f"{p}: case {row['case_id']} is not in the frozen manifest")
            if bc.split != split:
                raise SubsetError(f"{p}: case {row['case_id']} is {bc.split}, expected {split}")
            if bc.label.get("explanation_verdict") != row["verdict"]:
                raise SubsetError(f"{p}: case {row['case_id']} verdict changed since freezing "
                                  f"({row['verdict']} -> {bc.label.get('explanation_verdict')})")
    return d


def subset_case_ids(role: str, name: str, manifest: BenchmarkManifest | None = None,
                    root: Path = DEFAULT_SUBSET_ROOT) -> list[str]:
    return [c["case_id"] for c in load_subset(role, name, manifest, root)["cases"]]


def available_subsets(role: str | None = None) -> list[str]:
    return sorted({n for r, n in SUBSET_RULES if role is None or r == role})


__all__ = ["SubsetError", "SUBSET_RULES", "SUBSET_RULES_VERSION", "DEFAULT_SUBSET_ROOT",
           "propose_subset", "freeze_subset", "load_subset", "subset_case_ids",
           "subset_path", "available_subsets"]
