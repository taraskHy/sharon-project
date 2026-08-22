"""Import FINAL ground-truth labels (labeling_app export) into a grading
dataset directory as ``final_labels.json`` — the ONLY human-label source the
benchmark treats as ground truth for the shared-labeling workflow.

    final_labels.json  {"schema_version": 1, "source_export": {...}, "labels":
                        {case_id: {score, rubric_decisions, note, source, contributing_graders,
                                   adjudicator, finalized_at}}}

Individual grader labels are never imported as truth; the export's per-item
``labels`` list stays inside the export file for provenance.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

FINAL_LABELS_FILENAME = "final_labels.json"
SUPPORTED_EXPORT_SCHEMA = (1,)


def import_final_labels(export_path: Path, dataset_dir: Path, *, now: str | None = None) -> dict[str, Any]:
    export_path, dataset_dir = Path(export_path), Path(dataset_dir)
    data = json.loads(export_path.read_text(encoding="utf-8"))
    if data.get("schema_version") not in SUPPORTED_EXPORT_SCHEMA or data.get("kind") != "grade_primary_final_labels":
        raise ValueError("not a supported grade_primary final-labels export")
    known = {json.loads(l)["case_id"] for l in (dataset_dir / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    labels: dict[str, dict] = {}
    unknown: list[str] = []
    for row in data.get("items", []):
        cid = row["item_id"]
        if cid not in known:
            unknown.append(cid)
            continue
        if row.get("source") not in ("agreement", "adjudicated"):
            continue
        labels[cid] = {"score": float(row["final_score"]), "rubric_decisions": sorted(row.get("rubric_decisions") or []),
                       "note": row.get("note") or "", "source": row["source"],
                       "contributing_graders": sorted(row.get("contributing_graders") or []),
                       "adjudicator": row.get("adjudicator"), "finalized_at": row.get("finalized_at")}
    out = {"schema_version": 1, "imported_at": now or time.strftime("%Y-%m-%d %H:%M:%S"),
           "source_export": {"path": str(export_path), "content_sha256": data.get("content_sha256"),
                             "exported_at": data.get("exported_at"), "final_count": data.get("final_count"),
                             "bundle_items_sha256": data.get("bundle_items_sha256"),
                             "file_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest()},
           "labels": dict(sorted(labels.items()))}
    (dataset_dir / FINAL_LABELS_FILENAME).write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True),
                                                     encoding="utf-8", newline="\n")
    return {"imported": len(labels), "unknown_case_ids": unknown, "path": str(dataset_dir / FINAL_LABELS_FILENAME)}


def merge_final_labels(labels_by_id: dict[str, dict], dataset_dir: Path) -> tuple[int, str | None]:
    """Merge final_labels.json into evaluation-side labels: score, rubric_met,
    owner_note, label_source='final:<source>'. Returns (count, sha256)."""
    p = Path(dataset_dir) / FINAL_LABELS_FILENAME
    if not p.exists():
        return 0, None
    data = json.loads(p.read_text(encoding="utf-8"))
    n = 0
    for cid, f in (data.get("labels") or {}).items():
        lab = labels_by_id.get(cid)
        if lab is None:
            continue
        lab["score"] = f.get("score")
        lab["rubric_met"] = list(f.get("rubric_decisions") or []) or None
        lab["owner_note"] = f.get("note") or None
        lab["label_source"] = f"final:{f.get('source')}"
        lab["contributing_graders"] = f.get("contributing_graders")
        n += 1
    return n, hashlib.sha256(p.read_bytes()).hexdigest()


__all__ = ["FINAL_LABELS_FILENAME", "import_final_labels", "merge_final_labels"]
