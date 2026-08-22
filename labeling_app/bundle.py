"""Anonymized, self-contained labeling bundle — WITH explicit source provenance.

    <bundle>/bundle.json            bundle_sha256, counts, source hashes, schema
    <bundle>/items.json             [{item_id (opaque), question_text, rubric, scoring_rules,
                                     official_solution, transcription, max_score,
                                     rubric_items [{id,text}], images [rel paths],
                                     provenance {exam, case_id, question_id, part, row, line_count,
                                                 page, page_available, page_image (rel path|null),
                                                 unavailable [..]}}]
    <bundle>/images/<item_id>_<n>.png   the answer crops (primary grading evidence)
    <bundle>/pages/exam<NNN>_p<P>.png   the full source page, instructor red ink MASKED
    <bundle>/private/id_map.json        opaque item_id -> dataset case_id        (never served)
    <bundle>/private/provenance.json    item_id -> {source_file, crop_files, mask report, ...} (never served)

Provenance is CARRIED EXPLICITLY from the upstream records (never
reconstructed from opaque ids):

* e002 cells  -> evaluation/hebrew_bench/crops_manifest.json (source pdf, page, row, crop size)
* e003..e007 lines -> evaluation/htr_pilot_sources.json (source pdf; answer-sheet page per question)
* exam id = the anonymized writer code's number (e003 -> exam 003); the source
  PDF filename carries the instructor's TOTAL GRADE in its name (test/003_70.pdf)
  and is therefore kept PRIVATE (admin/export only), never shown to graders.

Full source pages carry the instructor's red ink (per-row ticks/crosses and
the question total) — an expected-label leak. Pages are rendered locally and
the red ink is masked with a dilated red-dominance mask; a page is served ONLY
when the residual strict-red pixel count is below RESIDUAL_RED_MAX, otherwise
``page_available`` is false ("unavailable", never guessed).
Per-line bounding boxes are not recorded upstream -> reported unavailable.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION

#: keys of the dataset's evaluation-side label rows that must NEVER reach the bundle
FORBIDDEN_IN_BUNDLE = ("split", "writer", "label_status", "transcription_items", "transcription_provenance",
                       "transcription_source", "evidence_images", "score", "rubric_met", "owner_note",
                       "owner_status", "question_id", "sub_item_id")
#: provenance fields a grader may see (everything else stays in private/provenance.json)
GRADER_PROVENANCE_FIELDS = ("exam", "case_id", "question_id", "part", "row", "line_count", "page",
                            "page_available", "unavailable")

PAGE_MAX_EDGE = 1400
#: strict red-ink rule (scripts/m2_bench_build.has_red_ink): after masking, a page with
#: more than this many strict-red pixels is NOT served
RESIDUAL_RED_MAX = 60
MASK_DILATE_PX = 5

_CELL_RE = re.compile(r"^(e\d{3})_q(\d+)_r(\d+)$")
_LINE_ITEM_RE = re.compile(r"^hl_(e\d{3})_q(\d+)_r(\d+)__l(\d+)$")


def opaque_id(case_id: str, salt: str) -> str:
    return "g" + hashlib.sha256(f"{salt}:{case_id}".encode("utf-8")).hexdigest()[:10]


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------- provenance --

def load_provenance_sources(repo_root: Path) -> dict[str, Any]:
    """The upstream provenance records (read-only)."""
    repo_root = Path(repo_root)
    crops_manifest = json.loads((repo_root / "evaluation" / "hebrew_bench" / "crops_manifest.json").read_text(encoding="utf-8"))
    htr_sources = json.loads((repo_root / "evaluation" / "htr_pilot_sources.json").read_text(encoding="utf-8"))
    return {"cells": {c["id"]: c for c in crops_manifest}, "writers": htr_sources}


def case_provenance(case_id: str, label_row: dict, sources: dict[str, Any]) -> dict[str, Any]:
    """Explicit provenance for one grading case. Fields that the upstream
    records do not carry are listed in ``unavailable`` — never guessed."""
    m = _CELL_RE.match(case_id)
    if not m:
        raise ValueError(f"unexpected case id {case_id!r}")
    writer, q, r = m.group(1), m.group(2), m.group(3)
    items = list(label_row.get("transcription_items") or [])
    prov: dict[str, Any] = {
        "exam": writer[1:],                       # e003 -> "003" (anonymized exam number)
        "writer": writer,
        "case_id": case_id,
        "question_id": q,
        "part": f"r{r}",
        "row": int(r),
        "line_count": len(items),
        "source_items": items,                    # hebrew_bench_v2 item ids (crop provenance)
        "page": None, "page_source": None, "source_file": None, "unavailable": [],
    }
    cell = sources["cells"].get(case_id)
    if cell is not None:                          # e002 cells: crops_manifest carries pdf + page + row
        prov["page"] = int(cell["page"])
        prov["page_source"] = "evaluation/hebrew_bench/crops_manifest.json"
        prov["source_file"] = cell["source"]
        prov["crop_size"] = [cell.get("width"), cell.get("height")]
        if int(cell.get("row", r)) != int(r):
            raise ValueError(f"{case_id}: crops_manifest row {cell.get('row')} != case row {r}")
    else:
        w = sources["writers"].get(writer)
        sheet = (w or {}).get("sheets", {}).get(q)
        if w and sheet and sheet.get("page"):
            prov["page"] = int(sheet["page"])
            prov["page_source"] = "evaluation/htr_pilot_sources.json"
            prov["source_file"] = w["pdf"]
        else:
            prov["unavailable"].append("page number (no upstream record for this writer/question)")
    prov["unavailable"].append("line bounding box on the page (not recorded upstream)")
    return prov


# ------------------------------------------------------------ page render --

def _np():
    import numpy as np
    return np


def strong_red_mask(arr) -> Any:
    """Red-dominant pixels (looser than the production masker, so anti-aliased
    pink halos are caught), dilated by MASK_DILATE_PX."""
    np = _np()
    r = arr[:, :, 0].astype(np.int16); g = arr[:, :, 1].astype(np.int16); b = arr[:, :, 2].astype(np.int16)
    dom = r - np.maximum(g, b)
    mask = (dom > 22) & (r > 60)
    if not mask.any():
        return mask
    k = MASK_DILATE_PX
    pad = np.pad(mask, k)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out |= pad[k + dy: k + dy + h, k + dx: k + dx + w]
    return out


def strict_red_count(arr) -> int:
    np = _np()
    r = arr[:, :, 0].astype(int); g = arr[:, :, 1].astype(int); b = arr[:, :, 2].astype(int)
    return int(((r > 140) & (r - g > 50) & (r - b > 50)).sum())


def render_masked_page(pdf: Path, page_no: int, *, max_edge: int = PAGE_MAX_EDGE) -> tuple[bytes, dict[str, Any]]:
    """Render page ``page_no`` (1-based) locally (PyMuPDF, no model), whiten the
    instructor's red ink (dilated mask) and report the residual strict-red
    count. Returns (png_bytes, report)."""
    import fitz
    np = _np()
    doc = fitz.open(str(pdf))
    try:
        page = doc[page_no - 1]
        r = page.rect
        zoom = max_edge / max(r.width, r.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    finally:
        doc.close()
    before = strict_red_count(arr)
    mask = strong_red_mask(arr)
    masked = int(mask.sum())
    if masked:
        arr[mask] = 255
    residual = strict_red_count(arr)
    pix2 = fitz.Pixmap(fitz.csRGB, arr.shape[1], arr.shape[0], arr.tobytes(), False)
    png = pix2.tobytes("png")
    return png, {"page": page_no, "width": int(arr.shape[1]), "height": int(arr.shape[0]),
                 "strict_red_before": before, "masked_pixels": masked, "strict_red_after": residual,
                 "ok": residual <= RESIDUAL_RED_MAX, "max_edge": max_edge,
                 "method": f"red-dominance (r-max(g,b)>22, r>60) dilated {MASK_DILATE_PX}px -> white; "
                           f"served only if strict-red residual <= {RESIDUAL_RED_MAX}"}


# ------------------------------------------------------------------ build --

def build_bundle(dataset_dir: Path, out_dir: Path, *, evaluation_root: Path, repo_root: Path | None = None,
                 salt: str | None = None, render_pages: bool = True, page_max_edge: int = PAGE_MAX_EDGE,
                 now: str | None = None) -> dict[str, Any]:
    """Build the anonymized bundle from a frozen grade_primary dataset directory.
    ``evaluation_root`` resolves `evidence_images`; ``repo_root`` (default:
    evaluation_root's parent) resolves the upstream provenance records and PDFs."""
    from autograder.eligibility import eligibility_counts, split_cases
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    repo_root = Path(repo_root) if repo_root else Path(evaluation_root).parent
    man = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    inputs = [json.loads(l) for l in (dataset_dir / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = {r["case_id"]: r for r in (json.loads(l) for l in (dataset_dir / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    salt = salt or man.get("inputs_sha256", "bundle")
    if (out_dir / "items.json").exists():
        raise FileExistsError(f"{out_dir} already holds a bundle; remove it or choose another directory")
    # Eligibility gate (autograder.eligibility, wrapping the production policy
    # machinery): only cases whose explanation genuinely requires a HUMAN score
    # become labeling items. Policy-decided cases (confidently wrong MC under a
    # zero rule, or choice_only) go to private/excluded.json — they are
    # policy/early-exit provenance, never human workload.
    labelable, policy_decided = split_cases(inputs, labels)
    eligibility_by_case = {row["case_id"]: e for row, e in labelable + policy_decided}
    counts = eligibility_counts([e for _, e in labelable + policy_decided])
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    (out_dir / "private").mkdir(parents=True, exist_ok=True)
    sources = load_provenance_sources(repo_root)
    items, id_map, private_prov = [], {}, {}
    page_cache: dict[tuple[str, int], dict] = {}          # (source_file, page) -> {rel, report}
    for row in sorted((r for r, _ in labelable), key=lambda r: r["case_id"]):
        cid = row["case_id"]
        oid = opaque_id(cid, salt)
        pack = row["pack"]
        lab = labels.get(cid, {})
        imgs, crop_files = [], []
        for n, rel in enumerate(lab.get("evidence_images") or [], start=1):
            src = Path(evaluation_root) / rel
            if not src.exists():
                continue
            dst_rel = f"images/{oid}_{n}.png"
            shutil.copyfile(src, out_dir / dst_rel)
            imgs.append(dst_rel)
            crop_files.append(rel)
        prov = case_provenance(cid, lab, sources)
        page_rel, page_report = None, None
        if render_pages and prov["page"] and prov["source_file"]:
            key = (prov["source_file"], prov["page"])
            if key not in page_cache:
                pdf = repo_root / prov["source_file"]
                if pdf.exists():
                    png, rep = render_masked_page(pdf, prov["page"], max_edge=page_max_edge)
                    rel = f"pages/exam{prov['exam']}_p{prov['page']}.png"
                    if rep["ok"]:
                        (out_dir / rel).write_bytes(png)
                        page_cache[key] = {"rel": rel, "report": rep}
                    else:
                        page_cache[key] = {"rel": None, "report": rep}
                else:
                    page_cache[key] = {"rel": None, "report": {"ok": False, "error": "source pdf missing"}}
            page_rel, page_report = page_cache[key]["rel"], page_cache[key]["report"]
        if prov["page"] and not page_rel:
            prov["unavailable"].append("full source page image (not rendered or red-ink masking not trusted)")
        visible = {k: prov[k] for k in GRADER_PROVENANCE_FIELDS if k in prov}
        visible["page_available"] = bool(page_rel)
        visible["page_image"] = page_rel
        sol = pack.get("official_solution") or {}
        items.append({
            "item_id": oid,
            "question_text": pack.get("question_text", ""),
            "rubric": list(pack.get("rubric") or []),
            "scoring_rules": list(pack.get("scoring_rules") or []),
            "official_solution": "\n".join(str(v) for v in sol.values()) if isinstance(sol, dict) else str(sol),
            "transcription": row.get("transcription", ""),
            "max_score": float(pack.get("max_score") or lab.get("max_score") or 0),
            "rubric_items": [{"id": ri.get("id"), "text": ri.get("text", "")} for ri in (pack.get("rubric_items") or [])],
            "images": imgs,
            "provenance": visible,
            "eligible_for_human_label": True,
            "eligibility_reason": eligibility_by_case[cid].reason,
        })
        id_map[oid] = cid
        private_prov[oid] = {**prov, "crop_files": crop_files, "page_image": page_rel, "page_report": page_report}
    for it in items:
        leak = set(it) & set(FORBIDDEN_IN_BUNDLE)
        if leak:
            raise RuntimeError(f"bundle item carries forbidden field(s) {sorted(leak)}")
        assert "source_file" not in it["provenance"] and "writer" not in it["provenance"]
    (out_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1, sort_keys=True),
                                        encoding="utf-8", newline="\n")
    (out_dir / "private" / "id_map.json").write_text(json.dumps(id_map, ensure_ascii=False, indent=1, sort_keys=True),
                                                     encoding="utf-8", newline="\n")
    (out_dir / "private" / "provenance.json").write_text(json.dumps(private_prov, ensure_ascii=False, indent=1, sort_keys=True),
                                                         encoding="utf-8", newline="\n")
    excluded_records = [{"case_id": row["case_id"], **e.to_dict()} for row, e in
                        sorted(policy_decided, key=lambda t: t[0]["case_id"])]
    (out_dir / "private" / "excluded.json").write_text(
        json.dumps(excluded_records, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    pages = sorted({i["provenance"]["page_image"] for i in items if i["provenance"].get("page_image")})
    meta = {
        "schema_version": SCHEMA_VERSION, "built_at": now or time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": len(items), "images": sum(len(i["images"]) for i in items), "pages": len(pages),
        "items_with_page": sum(1 for i in items if i["provenance"].get("page_available")),
        "items_sha256": _sha_file(out_dir / "items.json"),
        "eligibility": counts,
        "source": {"dataset_dir_name": dataset_dir.name, "dataset_inputs_sha256": man.get("inputs_sha256"),
                   "dataset_labels_sha256": man.get("labels_sha256"), "dataset_manifest_sha256": _sha_file(dataset_dir / "manifest.json"),
                   "provenance_records": ["evaluation/hebrew_bench/crops_manifest.json", "evaluation/htr_pilot_sources.json"]},
        "page_policy": ("full source pages are rendered locally with the instructor's red ink masked "
                        f"(dilated red-dominance mask); a page is served only if <= {RESIDUAL_RED_MAX} strict-red "
                        "pixels remain; the source filename (carries the instructor total) stays private"),
        "policy": ("opaque item ids; question/rubric/solution/transcription/max score/images + explicit source "
                   "provenance (exam, case id, question, part, page) only; no split/provenance notes/labels/model "
                   "output; private/*.json is never served"),
    }
    (out_dir / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True),
                                         encoding="utf-8", newline="\n")
    return meta


# ------------------------------------------------------------------- read --

class Bundle:
    """Read-only view used by the running app."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "bundle.json").read_text(encoding="utf-8"))
        self.items: list[dict] = json.loads((self.root / "items.json").read_text(encoding="utf-8"))
        self.by_id: dict[str, dict] = {i["item_id"]: i for i in self.items}
        p = self.root / "private" / "id_map.json"
        self.id_map: dict[str, str] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        pp = self.root / "private" / "provenance.json"
        self.private_provenance: dict[str, dict] = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else {}
        # Per-item human-label eligibility. New bundles carry it explicitly;
        # a pre-eligibility bundle has no flag (None = unknown until
        # apply_dataset_eligibility recomputes from the dataset).
        self.eligibility: dict[str, dict] = {
            i["item_id"]: {"eligible_for_human_label": (bool(i["eligible_for_human_label"])
                                                        if "eligible_for_human_label" in i else None),
                           "reason": i.get("eligibility_reason")}
            for i in self.items}

    def apply_dataset_eligibility(self, dataset_dir: Path | str) -> dict:
        """Recompute eligibility for every bundle item straight from the
        dataset (the single source of truth) — this is how a STALE bundle,
        built before eligibility filtering existed, still fails safely.

        Refuses (``applied: False`` + reason) instead of guessing when the
        dataset files or the bundle's private id map are missing, or when the
        dataset is not the one this bundle was built from
        (``inputs_sha256`` mismatch). Callers must treat a refusal loudly —
        never as "everything is eligible"."""
        ds = Path(dataset_dir)
        inputs_p, labels_p = ds / "cases_inputs.jsonl", ds / "cases_labels.jsonl"
        if not inputs_p.exists() or not labels_p.exists():
            return {"applied": False, "reason": f"dataset files missing under {ds}"}
        if not self.id_map:
            return {"applied": False,
                    "reason": "bundle has no private/id_map.json; items cannot be joined to dataset cases"}
        man_p = ds / "manifest.json"
        ds_sha = None
        if man_p.exists():
            ds_sha = json.loads(man_p.read_text(encoding="utf-8")).get("inputs_sha256")
        bundle_sha = (self.meta.get("source") or {}).get("dataset_inputs_sha256")
        if ds_sha and bundle_sha and ds_sha != bundle_sha:
            return {"applied": False,
                    "reason": (f"dataset inputs_sha256 {ds_sha[:12]}… does not match the bundle's source "
                               f"dataset {str(bundle_sha)[:12]}… — wrong dataset for this bundle")}
        from autograder.eligibility import eligibility_for_case
        inputs = {r["case_id"]: r for r in (json.loads(l) for l in
                  inputs_p.read_text(encoding="utf-8").splitlines() if l.strip())}
        labels = {r["case_id"]: r for r in (json.loads(l) for l in
                  labels_p.read_text(encoding="utf-8").splitlines() if l.strip())}
        unmatched = []
        for oid, cid in self.id_map.items():
            if oid not in self.by_id:
                continue
            if cid not in inputs:
                unmatched.append(oid)                     # stays unknown -> never silently eligible
                continue
            e = eligibility_for_case(inputs[cid], labels.get(cid))
            self.eligibility[oid] = {"eligible_for_human_label": e.eligible_for_human_label,
                                     "reason": e.reason,
                                     "deterministic_score": e.deterministic_score}
        return {"applied": True, "reason": "", "ineligible": self.ineligible_item_ids(),
                "unmatched_items": sorted(unmatched)}

    def eligibility_known(self) -> bool:
        """True when EVERY item's eligibility is explicit (bundle flags or a
        dataset recompute). Only then may the DB flip flags on bundle items."""
        return all(e.get("eligible_for_human_label") is not None for e in self.eligibility.values())

    def ineligible_item_ids(self) -> list[str]:
        """Items explicitly marked NOT human-labelable (unknown counts as
        eligible only because a pre-eligibility bundle carries no flag; the
        dataset recompute in apply_dataset_eligibility settles those)."""
        return sorted(i for i, e in self.eligibility.items()
                      if e.get("eligible_for_human_label") is False)

    def item(self, item_id: str) -> dict | None:
        return self.by_id.get(item_id)

    def _inside(self, p: Path) -> bool:
        return self.root.resolve() in p.resolve().parents

    def image_path(self, item_id: str, n: int) -> Path | None:
        it = self.by_id.get(item_id)
        if not it or n < 1 or n > len(it["images"]):
            return None
        p = (self.root / it["images"][n - 1]).resolve()
        return p if self._inside(p) and p.exists() else None

    def page_path(self, item_id: str) -> Path | None:
        it = self.by_id.get(item_id)
        rel = (it or {}).get("provenance", {}).get("page_image")
        if not rel:
            return None
        p = (self.root / rel).resolve()
        return p if self._inside(p) and p.exists() and rel.startswith("pages/") else None

    def grader_payload(self, item_id: str) -> dict | None:
        """EXACTLY what a grader may see — no labels, no private provenance."""
        it = self.by_id.get(item_id)
        if it is None:
            return None
        prov = dict(it.get("provenance") or {})
        prov.pop("page_image", None)
        prov["page_url"] = f"/api/pages/{it['item_id']}" if it.get("provenance", {}).get("page_available") else None
        return {"item_id": it["item_id"], "question_text": it["question_text"], "rubric": it["rubric"],
                "scoring_rules": it["scoring_rules"], "official_solution": it["official_solution"],
                "transcription": it["transcription"], "max_score": it["max_score"],
                "rubric_items": it["rubric_items"],
                "images": [f"/api/images/{it['item_id']}/{n}" for n in range(1, len(it["images"]) + 1)],
                "provenance": prov}


__all__ = ["build_bundle", "Bundle", "opaque_id", "FORBIDDEN_IN_BUNDLE", "GRADER_PROVENANCE_FIELDS",
           "case_provenance", "load_provenance_sources", "render_masked_page", "strong_red_mask",
           "strict_red_count", "RESIDUAL_RED_MAX", "PAGE_MAX_EDGE"]
