"""Declared-dataset writer for the B3/B4/B5 roles.

The OCR benchmarks are frozen in evaluation/hebrew_bench_v2 and are never
written by this package. The remaining roles (grading, MC, variant,
alignment) use the generic declared format read by manifests.load_declared:

    <datasets_root>/<role>/manifest.json
    <datasets_root>/<role>/cases_inputs.jsonl     model-visible only
    <datasets_root>/<role>/cases_labels.jsonl     evaluation-side only

``write_declared_dataset`` is the ONLY writer: it refuses to overwrite an
existing frozen dataset (a new version is a new directory), keeps inputs
and labels in separate files, records sha256 of both in the manifest, and
stores a split assignment. Builders that turn raw sources into cases
(grading packs + frozen transcriptions; MC band crops + audited answers;
cover crops + suit/flower labels; alignment permutations) call it; they are
added as each raw source is confirmed — no dataset is fabricated here.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from .manifests import SPLITS, STATUS_FROZEN


class DatasetExists(RuntimeError):
    """Refusing to overwrite a frozen declared dataset."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_declared_dataset(out_dir: Path, *, name: str, cases_inputs: Iterable[dict],
                           cases_labels: Iterable[dict], split_assignment: dict | None = None,
                           policy: str = "", notes: list[str] | None = None,
                           extra: dict | None = None, status: str = STATUS_FROZEN,
                           now: str | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    if (out / "manifest.json").exists():
        raise DatasetExists(f"{out} already holds a frozen dataset; write a new directory instead")
    inputs = list(cases_inputs)
    labels = list(cases_labels)
    ids_in = [r["case_id"] for r in inputs]
    ids_lab = [r["case_id"] for r in labels]
    if len(set(ids_in)) != len(ids_in):
        raise ValueError("duplicate case_id in inputs")
    if set(ids_in) != set(ids_lab):
        raise ValueError("inputs/labels case-id sets differ")
    for r in labels:
        if str(r.get("split", "DEV")).upper() not in SPLITS:
            raise ValueError(f"case {r['case_id']}: bad split {r.get('split')!r}")
    label_keys = {k for r in labels for k in r} - {"case_id", "split", "component"}
    for r in inputs:
        leak = set(r) & label_keys
        if leak:
            raise ValueError(f"case {r['case_id']}: input fields overlap label fields {sorted(leak)}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cases_inputs.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in inputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "cases_labels.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {
        "name": name, "status": status,
        "frozen_at": now or time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": len(inputs),
        "inputs_sha256": _sha(out / "cases_inputs.jsonl"),
        "labels_sha256": _sha(out / "cases_labels.jsonl"),
        "split_assignment": split_assignment or {},
        "policy": policy or ("model-visible: cases_inputs.jsonl only; labels are evaluation-side; "
                             "splits DEV/CALIBRATION/HELD_OUT per case"),
        "notes": list(notes or []), "extra": extra or {},
    }
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8",
                                       newline="\n")
    (out / "CHECKSUMS.sha256").write_text(
        f"{man['inputs_sha256']}  cases_inputs.jsonl\n{man['labels_sha256']}  cases_labels.jsonl\n",
        encoding="utf-8", newline="\n")
    return man


__all__ = ["write_declared_dataset", "DatasetExists"]
