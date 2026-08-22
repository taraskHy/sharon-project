"""Deterministic export of FINAL ground truth (final_labels.json).

Only rows of the ``final_labels`` table are exported — individual grader
labels are attached as provenance, never promoted. The item ordering and
every list/dict are sorted, so two exports of the same state are byte-equal
except for ``exported_at`` (which sits outside ``content_sha256``).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .bundle import Bundle
from .db import LabelDB


def export_final(db: LabelDB, bundle: Bundle, *, now: str | None = None) -> dict[str, Any]:
    items = []
    for f in db.final_rows():
        oid = f["item_id"]
        labels = [{"grader": l["grader"], "score": l["score"], "rubric_decisions": sorted(l["rubric"]),
                   "note": l["note"], "status": l["status"], "revision": l["revision"], "updated_at": l["updated_at"]}
                  for l in db.labels_for_item(oid)]
        labels.sort(key=lambda l: l["grader"])
        elig = bundle.eligibility.get(oid, {})
        items.append({
            "item_id": bundle.id_map.get(oid, oid),       # dataset case id when the private map is present
            "display_id": oid,
            "label_kind": "human_final_label",             # never a deterministic_policy_score
            # False marks an OBSOLETE final (item became policy-decided after the
            # label was written); the importer refuses to promote it to truth.
            "eligible_for_human_label": elig.get("eligible_for_human_label", True) is not False,
            "final_score": f["score"],
            "rubric_decisions": sorted(f["rubric"]),
            "note": f["note"],
            "source": f["source"],                         # agreement | adjudicated
            "adjudicator": f["adjudicator"] or None,
            "contributing_graders": sorted(f["contributing_graders"]),
            "from_revisions": dict(sorted(f["from_revisions"].items())),
            "finalized_at": f["finalized_at"],
            "labels": labels,
        })
    items.sort(key=lambda i: i["item_id"])
    body = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "grade_primary_final_labels",
        "bundle_items_sha256": bundle.meta.get("items_sha256"),
        "dataset_inputs_sha256": (bundle.meta.get("source") or {}).get("dataset_inputs_sha256"),
        "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "final_count": len(items),
        "obsolete_ineligible_count": sum(1 for i in items if not i["eligible_for_human_label"]),
        "eligibility": bundle.meta.get("eligibility"),
        "exported_at": now or time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": items,
    }


def write_export(db: LabelDB, bundle: Bundle, path: Path, *, now: str | None = None) -> dict[str, Any]:
    data = export_final(db, bundle, now=now)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    return data


__all__ = ["export_final", "write_export"]
