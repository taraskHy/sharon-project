"""Anonymized, self-contained labeling bundle.

    <bundle>/bundle.json        bundle_sha256, counts, source hashes, schema
    <bundle>/items.json         [{item_id (opaque), question_text, rubric, scoring_rules,
                                 official_solution, transcription, max_score,
                                 rubric_items [{id,text}], images [relative paths]}]
    <bundle>/images/<item_id>_<n>.png
    <bundle>/private/id_map.json   opaque item_id -> dataset case_id  (NEVER served;
                                   used only by export / admin on the owner's PC)

The bundle is built ONCE from the frozen grade_primary dataset by
``build_bundle`` (an offline step that may import the repository's benchmark
helpers); the web app at runtime reads only these files. Nothing in the
served payload carries repository paths, writer codes, splits, labels,
provenance notes or model outputs.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION

#: keys of the dataset's evaluation-side label rows that must NEVER reach the bundle
FORBIDDEN_IN_BUNDLE = ("split", "writer", "label_status", "transcription_items", "transcription_provenance",
                       "transcription_source", "evidence_images", "score", "rubric_met", "owner_note",
                       "owner_status", "question_id", "sub_item_id")


def opaque_id(case_id: str, salt: str) -> str:
    return "g" + hashlib.sha256(f"{salt}:{case_id}".encode("utf-8")).hexdigest()[:10]


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_bundle(dataset_dir: Path, out_dir: Path, *, evaluation_root: Path, salt: str | None = None,
                 now: str | None = None) -> dict[str, Any]:
    """Build the anonymized bundle from a frozen grade_primary dataset
    directory (cases_inputs.jsonl + cases_labels.jsonl + manifest.json).
    ``evaluation_root`` resolves the dataset's `evidence_images` paths."""
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    man = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    inputs = [json.loads(l) for l in (dataset_dir / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = {r["case_id"]: r for r in (json.loads(l) for l in (dataset_dir / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    salt = salt or man.get("inputs_sha256", "bundle")
    if (out_dir / "items.json").exists():
        raise FileExistsError(f"{out_dir} already holds a bundle; remove it or choose another directory")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "private").mkdir(parents=True, exist_ok=True)
    items, id_map = [], {}
    for row in sorted(inputs, key=lambda r: r["case_id"]):
        cid = row["case_id"]
        oid = opaque_id(cid, salt)
        pack = row["pack"]
        imgs = []
        for n, rel in enumerate(labels.get(cid, {}).get("evidence_images") or [], start=1):
            src = Path(evaluation_root) / rel
            if not src.exists():
                continue
            dst_rel = f"images/{oid}_{n}.png"
            shutil.copyfile(src, out_dir / dst_rel)
            imgs.append(dst_rel)
        sol = pack.get("official_solution") or {}
        items.append({
            "item_id": oid,
            "question_text": pack.get("question_text", ""),
            "rubric": list(pack.get("rubric") or []),
            "scoring_rules": list(pack.get("scoring_rules") or []),
            "official_solution": "\n".join(str(v) for v in sol.values()) if isinstance(sol, dict) else str(sol),
            "transcription": row.get("transcription", ""),
            "max_score": float(pack.get("max_score") or labels.get(cid, {}).get("max_score") or 0),
            "rubric_items": [{"id": ri.get("id"), "text": ri.get("text", "")} for ri in (pack.get("rubric_items") or [])],
            "images": imgs,
        })
        id_map[oid] = cid
    # never let an evaluation-side field slip through
    for it in items:
        leak = set(it) & set(FORBIDDEN_IN_BUNDLE)
        if leak:
            raise RuntimeError(f"bundle item carries forbidden field(s) {sorted(leak)}")
    (out_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1, sort_keys=True),
                                        encoding="utf-8", newline="\n")
    (out_dir / "private" / "id_map.json").write_text(json.dumps(id_map, ensure_ascii=False, indent=1, sort_keys=True),
                                                     encoding="utf-8", newline="\n")
    meta = {
        "schema_version": SCHEMA_VERSION, "built_at": now or time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": len(items), "images": sum(len(i["images"]) for i in items),
        "items_sha256": _sha_file(out_dir / "items.json"),
        "source": {"dataset_dir_name": dataset_dir.name, "dataset_inputs_sha256": man.get("inputs_sha256"),
                   "dataset_labels_sha256": man.get("labels_sha256"), "dataset_manifest_sha256": _sha_file(dataset_dir / "manifest.json")},
        "policy": ("opaque item ids; question/rubric/solution/transcription/max score/images only; no split, writer, "
                   "provenance, labels or model output; private/id_map.json is never served"),
    }
    (out_dir / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True),
                                         encoding="utf-8", newline="\n")
    return meta


class Bundle:
    """Read-only view used by the running app."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "bundle.json").read_text(encoding="utf-8"))
        self.items: list[dict] = json.loads((self.root / "items.json").read_text(encoding="utf-8"))
        self.by_id: dict[str, dict] = {i["item_id"]: i for i in self.items}
        p = self.root / "private" / "id_map.json"
        self.id_map: dict[str, str] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def item(self, item_id: str) -> dict | None:
        return self.by_id.get(item_id)

    def image_path(self, item_id: str, n: int) -> Path | None:
        it = self.by_id.get(item_id)
        if not it or n < 1 or n > len(it["images"]):
            return None
        p = (self.root / it["images"][n - 1]).resolve()
        if self.root.resolve() not in p.parents:
            return None
        return p if p.exists() else None

    def grader_payload(self, item_id: str) -> dict | None:
        """EXACTLY what a grader may see — no labels, no provenance."""
        it = self.by_id.get(item_id)
        if it is None:
            return None
        return {"item_id": it["item_id"], "question_text": it["question_text"], "rubric": it["rubric"],
                "scoring_rules": it["scoring_rules"], "official_solution": it["official_solution"],
                "transcription": it["transcription"], "max_score": it["max_score"],
                "rubric_items": it["rubric_items"],
                "images": [f"/api/images/{it['item_id']}/{n}" for n in range(1, len(it["images"]) + 1)]}


__all__ = ["build_bundle", "Bundle", "opaque_id", "FORBIDDEN_IN_BUNDLE"]
