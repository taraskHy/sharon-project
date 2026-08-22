"""Manual GRADE_PRIMARY evidence repairs — human transcription of the lines the
frozen OCR benchmark never audited, stored OUTSIDE that benchmark.

Some student answer lines exist in the authoritative line inventory but carry no
audited transcription, because their upstream line crop was tagged
``bad_segmentation`` (the crop geometry is wrong — a sliver, or two lines in one
image). The grading model reads the transcription only, so such a case cannot be
scored for accuracy. Repairing that is GRADE_PRIMARY dataset preparation, NOT a
change to the frozen OCR benchmark:

* ``evaluation/hebrew_bench_v2`` (129 items, 102 audited references, its
  checksums and every historical OCR result) is never read-modified here and
  never written. ``assert_frozen_bench_unchanged`` proves it around every apply.
* repairs live next to the grading dataset, in the same place as the other
  human-label side files (``owner_labels.json``, ``final_labels.json``):

      <dataset>/manual_evidence_repairs.jsonl          one record per repaired line
      <dataset>/manual_evidence_repairs/crops/<line_id>.png   the repaired crop

Geometry is DERIVED, never guessed: every upstream line crop of a cell is a
pixel-exact sub-image of that cell's ``*_cell_clean.png`` crop, so each line's
band is found by exact match (``locate_exact``) and the region a mis-segmented
line should have covered is the part of the cell no VALID line covers. A repair
records the crop rectangle it was made from, that crop's sha256, and the cell it
came from — so the crop can be re-derived and re-verified at any time.

A repair may legitimately conclude that the mis-segmented sliver is NOT a
distinct line of writing (``disposition="no_text_segmentation_artifact"``): the
line is then resolved with no text rather than inventing one.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .datasets import (DEFAULT_BENCH_ROOT, DEFAULT_HTR_ROOT, DatasetBuildError, evidence_inventory_summary,
                       load_line_inventory)
from .manifests import REPO_ROOT

REPAIRS_FILENAME = "manual_evidence_repairs.jsonl"
REPAIRS_CROPS_DIRNAME = "manual_evidence_repairs"
REPAIR_SOURCE = "manual_grade_evidence_repair"
REPAIR_SCHEMA_VERSION = 1
DISPOSITIONS = ("transcribed", "no_text_segmentation_artifact")
#: the frozen OCR benchmark files this workflow must never change
FROZEN_BENCH_FILES = ("items.json", "references.json", "reference_audit.json", "reference_audit_manifest.json")

_CELL_RE = re.compile(r"^(e\d{3})_q(\d+)_r(\d+)$")
_LINE_RE = re.compile(r"^(e\d{3})_q(\d+)_r(\d+)__l(\d+)$")


class RepairError(RuntimeError):
    """A repair record or its crop is not admissible."""


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p: Path) -> str:
    return _sha_bytes(Path(p).read_bytes())


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- geometry --

def _png_array(path: Path):
    import fitz
    import numpy as np
    pix = fitz.Pixmap(str(path))
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return a[:, :, :3].astype(np.int16)


def locate_exact(cell, line) -> int | None:
    """Row offset at which ``line`` occurs EXACTLY inside ``cell`` (same width),
    or None. Exact means zero pixel difference — the line crop and the cell crop
    are two crops of the same render, so a real line always matches exactly."""
    import numpy as np
    if line.shape[1] != cell.shape[1] or line.shape[0] > cell.shape[0]:
        return None
    h = line.shape[0]
    for y in range(cell.shape[0] - h + 1):
        if not np.any(cell[y:y + h] - line):
            return y
    return None


def cell_crop_path(case_id: str, *, evaluation_root: Path) -> Path:
    m = _CELL_RE.match(case_id)
    if not m:
        raise RepairError(f"unexpected case id {case_id!r}")
    w, q, r = m.group(1), m.group(2), m.group(3)
    return Path(evaluation_root) / "htr_pilot" / "images" / w / f"q{q}_r{r}_cell_clean.png"


def case_geometry(case_id: str, *, evaluation_root: Path | None = None,
                  htr_root: Path = DEFAULT_HTR_ROOT) -> dict[str, Any]:
    """Deterministic geometry of one cell: the cell crop, every recorded line's
    exact band inside it, and the bands no VALID line covers.

    The cell crop is only accepted when EVERY recorded line crop of the case is
    a pixel-exact sub-image of it — the file is proven to be this case's cell,
    not assumed from its name."""
    evaluation_root = Path(evaluation_root) if evaluation_root else REPO_ROOT / "evaluation"
    inv = load_line_inventory(htr_root).get(case_id)
    if inv is None:
        raise RepairError(f"{case_id}: no upstream line inventory record")
    cell_p = cell_crop_path(case_id, evaluation_root=evaluation_root)
    if not cell_p.exists():
        raise RepairError(f"{case_id}: no cell crop at {cell_p}")
    cell = _png_array(cell_p)
    lines: list[dict] = []
    for line in inv["lines"]:
        img = Path(evaluation_root) / line["image"]
        if not img.exists():
            raise RepairError(f"{line['sample_id']}: line image missing ({line['image']})")
        arr = _png_array(img)
        y = locate_exact(cell, arr)
        if y is None:
            raise RepairError(f"{line['sample_id']}: its crop is not a pixel-exact region of the cell crop "
                              f"{cell_p.name} — the geometry cannot be derived honestly")
        lines.append({**line, "y0": y, "y1": y + arr.shape[0], "height": arr.shape[0],
                      "image_sha256": _sha_file(img),
                      "audited": bool(line["human_verified"]) and line["annotation_status"] == "ok"})
    covered = [(l["y0"], l["y1"]) for l in lines if l["audited"]]
    uncovered = _gaps(sorted(covered), cell.shape[0])
    return {"case_id": case_id, "n_lines": inv["n_lines"], "line_inventory_source": inv["source"],
            "cell_image": str(cell_p.relative_to(Path(evaluation_root).parent)).replace(os.sep, "/"),
            "cell_image_abs": str(cell_p), "cell_sha256": _sha_file(cell_p),
            "cell_width": cell.shape[1], "cell_height": cell.shape[0],
            "lines": lines, "uncovered_bands": uncovered}


def _gaps(spans: list[tuple[int, int]], height: int) -> list[dict]:
    """Row bands of [0, height) not covered by any span, largest first."""
    out, cur = [], 0
    for a, b in spans:
        if a > cur:
            out.append({"y0": cur, "y1": a})
        cur = max(cur, b)
    if cur < height:
        out.append({"y0": cur, "y1": height})
    for g in out:
        g["height"] = g["y1"] - g["y0"]
    return sorted(out, key=lambda g: -g["height"])


def suggested_band(geo: dict[str, Any], line_id: str) -> dict[str, int]:
    """The default repair rectangle for a line: the largest cell region no
    audited line covers, falling back to the mis-segmented crop's own band."""
    recorded = next((l for l in geo["lines"] if l["sample_id"] == line_id), None)
    if recorded is None:
        raise RepairError(f"{line_id}: not a recorded line of {geo['case_id']}")
    big = geo["uncovered_bands"][0] if geo["uncovered_bands"] else None
    if big and big["height"] >= recorded["height"]:
        return {"y0": big["y0"], "y1": big["y1"]}
    return {"y0": recorded["y0"], "y1": recorded["y1"]}


def render_band(geo: dict[str, Any], y0: int, y1: int, x0: int | None = None, x1: int | None = None) -> bytes:
    """PNG bytes of a rectangle of the cell crop (deterministic re-crop)."""
    import fitz
    import numpy as np
    cell = _png_array(Path(geo["cell_image_abs"]))
    h, w = cell.shape[0], cell.shape[1]
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    x0 = 0 if x0 is None else max(0, int(x0))
    x1 = w if x1 is None else min(w, int(x1))
    if y1 - y0 < 2 or x1 - x0 < 2:
        raise RepairError(f"empty crop rectangle ({x0},{y0})-({x1},{y1})")
    sub = np.ascontiguousarray(cell[y0:y1, x0:x1].astype(np.uint8))
    pix = fitz.Pixmap(fitz.csRGB, sub.shape[1], sub.shape[0], sub.tobytes(), False)
    return pix.tobytes("png")


# ------------------------------------------------------------------- store --

@dataclass
class RepairStore:
    """Append/replace-by-line_id JSONL store + the repaired crop files."""

    dataset_dir: Path

    @property
    def path(self) -> Path:
        return Path(self.dataset_dir) / REPAIRS_FILENAME

    @property
    def crops_dir(self) -> Path:
        return Path(self.dataset_dir) / REPAIRS_CROPS_DIRNAME / "crops"

    def records(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        out: dict[str, dict] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["line_id"]] = r          # last write wins; the file is append-then-rewrite
        return dict(sorted(out.items()))

    def get(self, line_id: str) -> dict | None:
        return self.records().get(line_id)

    def sha256(self) -> str | None:
        return _sha_file(self.path) if self.path.exists() else None

    def save(self, *, case_id: str, line_id: str, transcription: str, verified_by: str,
             disposition: str = "transcribed", crop_png: bytes | None = None,
             crop_geometry: dict | None = None, source_pdf: str | None = None, source_page: int | None = None,
             original_crop: dict | None = None, note: str = "", line_index: int | None = None,
             line_count: int | None = None, now: str | None = None) -> dict:
        """Write one human repair. Validates the line belongs to the case, the
        disposition/transcription agree, and the crop is persisted + hashed."""
        if disposition not in DISPOSITIONS:
            raise RepairError(f"unknown disposition {disposition!r}")
        m = _LINE_RE.match(line_id)
        if not m or f"{m.group(1)}_q{m.group(2)}_r{m.group(3)}" != case_id:
            raise RepairError(f"line {line_id!r} does not belong to case {case_id!r}")
        text = (transcription or "").strip()
        if disposition == "transcribed" and not text:
            raise RepairError("a transcribed repair needs the handwritten text (or mark it a segmentation artifact)")
        if disposition == "no_text_segmentation_artifact" and text:
            raise RepairError("a segmentation-artifact repair carries no transcription")
        if not (verified_by or "").strip():
            raise RepairError("a repair must record who verified it")
        rec = {
            "schema_version": REPAIR_SCHEMA_VERSION,
            "case_id": case_id, "line_id": line_id, "line_index": line_index, "line_count": line_count,
            "source": REPAIR_SOURCE, "disposition": disposition,
            "transcription": text, "human_verified": True, "verified_by": verified_by.strip(),
            "crop_path": None, "crop_sha256": None, "crop_geometry": crop_geometry,
            "source_pdf": source_pdf, "source_page": source_page,
            "original_crop": original_crop, "note": note or "",
            "created_at": now or _now(), "updated_at": now or _now(),
        }
        existing = self.get(line_id)
        if existing:
            rec["created_at"] = existing.get("created_at", rec["created_at"])
        if crop_png:
            self.crops_dir.mkdir(parents=True, exist_ok=True)
            cp = self.crops_dir / f"{line_id}.png"
            cp.write_bytes(crop_png)
            rec["crop_path"] = f"{REPAIRS_CROPS_DIRNAME}/crops/{cp.name}"
            rec["crop_sha256"] = _sha_bytes(crop_png)
        elif existing and existing.get("crop_path"):
            rec["crop_path"], rec["crop_sha256"] = existing["crop_path"], existing["crop_sha256"]
        if not rec["crop_path"]:
            raise RepairError("a repair must persist the crop it was made from")
        self._rewrite({**self.records(), line_id: rec})
        return rec

    def delete(self, line_id: str) -> bool:
        recs = self.records()
        if line_id not in recs:
            return False
        recs.pop(line_id)
        self._rewrite(recs)
        return True

    def _rewrite(self, recs: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with tmp.open("w", encoding="utf-8", newline="\n") as f:
                for _, r in sorted(recs.items()):
                    f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(tmp, self.path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


# ------------------------------------------------------------ expectations --

def expected_repairs(dataset_dir: Path) -> list[dict]:
    """The lines the DATASET itself says need a human transcription — derived
    from cases_labels.jsonl, never hardcoded.

    A line is expected while it is still listed in
    ``lines_without_audited_transcription``, and stays expected (as
    ``applied=True``) once ``apply_repairs`` has folded it in, so the store keeps
    verifying after the repair lands instead of looking like stray records."""
    d = Path(dataset_dir)
    rows = [json.loads(l) for l in (d / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        pending = list(r.get("lines_without_audited_transcription") or [])
        applied = [s for s in (r.get("evidence_repairs") or []) if s not in pending]
        for sid in pending + applied:
            ev = next((e for e in (r.get("evidence_lines") or []) if e.get("sample_id") == sid), {})
            out.append({"case_id": r["case_id"], "line_id": sid, "split": r.get("split"),
                        "applied": sid in applied,
                        "line_index": ev.get("index"), "line_count": r.get("line_count"),
                        "image": ev.get("original_image") or ev.get("image"),
                        "transcription_status": ev.get("original_transcription_status")
                        or ev.get("transcription_status"),
                        "question_id": r.get("question_id"), "sub_item_id": r.get("sub_item_id")})
    return sorted(out, key=lambda r: r["line_id"])


def repair_status(dataset_dir: Path) -> dict[str, Any]:
    exp = expected_repairs(dataset_dir)
    store = RepairStore(Path(dataset_dir))
    recs = store.records()
    remaining = [e["line_id"] for e in exp if e["line_id"] not in recs]
    return {"expected": len(exp), "repaired": len(exp) - len(remaining),
            "remaining": remaining,
            "applied": [e["line_id"] for e in exp if e["applied"]],
            "unexpected_records": sorted(set(recs) - {e["line_id"] for e in exp}),
            "complete": not remaining and len(exp) > 0,
            "by_disposition": {dp: sum(1 for r in recs.values() if r.get("disposition") == dp) for dp in DISPOSITIONS},
            "store_path": str(store.path), "store_sha256": store.sha256(),
            "cases": sorted({e["case_id"] for e in exp})}


# --------------------------------------------------------------- integrity --

def frozen_bench_hashes(bench_root: Path = DEFAULT_BENCH_ROOT) -> dict[str, str]:
    """sha256 of the frozen OCR-benchmark files this workflow must never change."""
    b = Path(bench_root)
    return {name: _sha_file(b / name) for name in FROZEN_BENCH_FILES if (b / name).exists()}


def assert_frozen_bench_unchanged(before: dict[str, str], bench_root: Path = DEFAULT_BENCH_ROOT) -> dict[str, str]:
    after = frozen_bench_hashes(bench_root)
    if after != before:
        moved = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise RepairError(f"the frozen OCR benchmark changed ({moved}) — refusing; hebrew_bench_v2 is immutable here")
    return after


def verify_repairs(dataset_dir: Path, *, evaluation_root: Path | None = None,
                   htr_root: Path = DEFAULT_HTR_ROOT) -> dict[str, Any]:
    """Every integrity rule, as data. ``ok`` is True only when all pass."""
    d = Path(dataset_dir)
    store = RepairStore(d)
    recs = store.records()
    exp = {e["line_id"]: e for e in expected_repairs(d)}
    problems: list[dict] = []
    raw_ids = []
    if store.path.exists():
        raw_ids = [json.loads(l)["line_id"] for l in store.path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for dup in sorted({i for i in raw_ids if raw_ids.count(i) > 1}):
        problems.append({"line_id": dup, "problem": "duplicate line id in the repair store"})
    for lid, r in recs.items():
        e = exp.get(lid)
        if e is None:
            problems.append({"line_id": lid, "problem": "not an expected repair for this dataset"})
            continue
        if r.get("case_id") != e["case_id"]:
            problems.append({"line_id": lid, "problem": f"case mismatch ({r.get('case_id')} != {e['case_id']})"})
        m = _LINE_RE.match(lid)
        if not m or f"{m.group(1)}_q{m.group(2)}_r{m.group(3)}" != r.get("case_id"):
            problems.append({"line_id": lid, "problem": "line id does not belong to its case"})
        if not r.get("human_verified"):
            problems.append({"line_id": lid, "problem": "not human_verified"})
        if r.get("source") != REPAIR_SOURCE:
            problems.append({"line_id": lid, "problem": f"unexpected source {r.get('source')!r}"})
        if r.get("disposition") not in DISPOSITIONS:
            problems.append({"line_id": lid, "problem": f"unknown disposition {r.get('disposition')!r}"})
        if r.get("disposition") == "transcribed" and not (r.get("transcription") or "").strip():
            problems.append({"line_id": lid, "problem": "transcribed repair without text"})
        cp = d / (r.get("crop_path") or "")
        if not r.get("crop_path") or not cp.exists():
            problems.append({"line_id": lid, "problem": "repaired crop file missing"})
        elif _sha_file(cp) != r.get("crop_sha256"):
            problems.append({"line_id": lid, "problem": "repaired crop hash does not match the record"})
        g = r.get("crop_geometry") or {}
        if not all(k in g for k in ("y0", "y1", "cell_sha256")):
            problems.append({"line_id": lid, "problem": "no recorded crop geometry"})
        else:
            try:
                geo = case_geometry(r["case_id"], evaluation_root=evaluation_root, htr_root=htr_root)
                if geo["cell_sha256"] != g["cell_sha256"]:
                    problems.append({"line_id": lid, "problem": "cell crop changed since the repair was made"})
                elif cp.exists() and _sha_bytes(render_band(geo, g["y0"], g["y1"], g.get("x0"), g.get("x1"))) != r.get("crop_sha256"):
                    problems.append({"line_id": lid, "problem": "recorded geometry does not re-derive the stored crop"})
            except (RepairError, DatasetBuildError) as ex:
                problems.append({"line_id": lid, "problem": f"geometry could not be re-derived: {ex}"})
    st = repair_status(d)
    return {"ok": not problems and st["complete"], "problems": problems, **st,
            "frozen_bench_sha256": frozen_bench_hashes()}


# ----------------------------------------------------------------- applying --

def _repaired_transcription(row_label: dict, input_row: dict, recs: dict[str, dict],
                            *, evaluation_root: Path, dataset_dir: Path) -> tuple[str, list[dict], list[str]]:
    """The complete transcription in AUTHORITATIVE LINE ORDER, plus the updated
    evidence lines and the repaired line ids used.

    The frozen text holds the audited lines in line order joined by newlines
    (``audited_cells``); each repaired line is spliced into its own position, so
    the result reads top-to-bottom exactly as the student wrote it. The historic
    per-line record is preserved: the original crop stays in ``original_image``
    and the repair provenance is attached, never overwritten in silence."""
    lines = sorted(row_label.get("evidence_lines") or [], key=lambda x: x["index"])
    missing = set(row_label.get("lines_without_audited_transcription") or [])
    frozen = input_row.get("transcription") or ""
    audited_texts = frozen.split("\n") if frozen else []
    if len(audited_texts) != len([e for e in lines if e.get("sample_id") not in missing]):
        raise RepairError(f"{row_label['case_id']}: the frozen transcription does not have one text line per "
                          f"audited evidence line — refusing to guess where a repaired line belongs")
    audited_iter = iter(audited_texts)
    parts: list[str] = []
    new_lines: list[dict] = []
    used: list[str] = []
    for e in lines:
        sid = e.get("sample_id")
        if sid in missing:
            r = recs.get(sid)
            if r is None:
                raise RepairError(f"{sid}: no repair record")
            used.append(sid)
            if r["disposition"] == "transcribed":
                parts.append(r["transcription"])
            crop_abs = Path(dataset_dir) / r["crop_path"]
            try:                                    # the dataset lives under evaluation/, so the repaired crop
                rel = str(crop_abs.resolve().relative_to(Path(evaluation_root).resolve()))
                image = rel.replace(os.sep, "/")    # resolves like every other evidence image
            except ValueError:
                image = e["image"]                  # outside the evidence root: keep the historic path
            new_lines.append({**e, "image": image, "original_image": e["image"],
                              "transcription_status": f"{REPAIR_SOURCE}:{r['disposition']}",
                              "original_transcription_status": e.get("transcription_status"),
                              "repair": {"source": REPAIR_SOURCE, "disposition": r["disposition"],
                                         "verified_by": r["verified_by"], "created_at": r["created_at"],
                                         "crop_sha256": r["crop_sha256"], "crop_path": r["crop_path"],
                                         "crop_geometry": r.get("crop_geometry"),
                                         "source_pdf": r.get("source_pdf"), "source_page": r.get("source_page")}})
        else:
            parts.append(next(audited_iter))
            new_lines.append(dict(e))
    return "\n".join(p for p in parts if p != ""), new_lines, used


def apply_repairs(dataset_dir: Path, *, bench_root: Path = DEFAULT_BENCH_ROOT,
                  htr_root: Path = DEFAULT_HTR_ROOT, evaluation_root: Path | None = None,
                  now: str | None = None, dry_run: bool = False, allow_partial: bool = False) -> dict[str, Any]:
    """Fold the human repairs into the grading dataset: the model-visible
    transcription gains the repaired lines IN AUTHORITATIVE LINE ORDER, the case
    becomes ``transcription_complete``, and the manifest records the revision
    (previous/new inputs+labels sha256, which lines, the repair store hash).

    Case ids never change, scores are never touched, and the frozen OCR
    benchmark is hash-checked before and after."""
    d = Path(dataset_dir)
    man_p, inputs_p, labels_p = d / "manifest.json", d / "cases_inputs.jsonl", d / "cases_labels.jsonl"
    if not man_p.exists():
        raise DatasetBuildError(f"{d} holds no frozen dataset")
    ev_root = Path(evaluation_root) if evaluation_root else REPO_ROOT / "evaluation"
    frozen_before = frozen_bench_hashes(bench_root)
    man = json.loads(man_p.read_text(encoding="utf-8"))
    old_inputs_sha, old_labels_sha = _sha_file(inputs_p), _sha_file(labels_p)
    if old_inputs_sha != man.get("inputs_sha256") or old_labels_sha != man.get("labels_sha256"):
        raise DatasetBuildError("dataset files do not match the manifest hashes; refusing to touch a drifted dataset")
    ver = verify_repairs(d, evaluation_root=evaluation_root, htr_root=htr_root)
    if ver["problems"]:
        raise RepairError(f"repair store is not admissible: {ver['problems'][:3]}")
    if not ver["complete"] and not allow_partial:
        raise RepairError(f"{len(ver['remaining'])} of {ver['expected']} repairs are still missing "
                          f"({ver['remaining'][:3]}…); complete them or pass allow_partial")
    recs = RepairStore(d).records()
    inputs = [json.loads(l) for l in inputs_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = [json.loads(l) for l in labels_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_case_input = {r["case_id"]: r for r in inputs}
    changed_cases, used_lines = [], []
    new_labels = []
    for row in labels:
        cid = row["case_id"]
        missing = list(row.get("lines_without_audited_transcription") or [])
        if not missing or not all(sid in recs for sid in missing):
            new_labels.append(row)
            continue
        inp = by_case_input[cid]
        before_text = inp.get("transcription") or ""
        text, new_lines, used = _repaired_transcription(
            row, inp, recs, evaluation_root=ev_root, dataset_dir=d)
        inp["transcription"] = text
        new_row = {**row, "evidence_lines": new_lines, "transcription_complete": True,
                   "lines_without_audited_transcription": [],
                   "evidence_repairs": sorted(used),
                   "transcription_source": (row.get("transcription_source") or "")
                   + f" + {REPAIR_SOURCE} for {', '.join(sorted(used))}"}
        new_labels.append(new_row)
        used_lines.extend(used)
        changed_cases.append({"case_id": cid, "lines_repaired": sorted(used),
                              "transcription_chars_before": len(before_text), "transcription_chars_after": len(text),
                              "transcription_complete_before": bool(row.get("transcription_complete")),
                              "transcription_complete_after": True})
    inputs_body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inputs)
    labels_body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_labels)
    new_inputs_sha = _sha_bytes(inputs_body.encode("utf-8"))
    new_labels_sha = _sha_bytes(labels_body.encode("utf-8"))
    summary = {"dataset": str(d), "cases": len(new_labels), "cases_changed": changed_cases,
               "lines_repaired": sorted(used_lines), "dry_run": dry_run, "written": False,
               "previous_inputs_sha256": old_inputs_sha, "inputs_sha256": new_inputs_sha,
               "previous_labels_sha256": old_labels_sha, "labels_sha256": new_labels_sha,
               "inputs_changed": new_inputs_sha != old_inputs_sha,
               "repair_store_sha256": ver["store_sha256"],
               "evidence_inventory": evidence_inventory_summary(new_labels),
               "frozen_bench_sha256": frozen_before}
    if dry_run or (new_inputs_sha == old_inputs_sha and new_labels_sha == old_labels_sha):
        assert_frozen_bench_unchanged(frozen_before, bench_root)
        return summary
    inputs_p.write_text(inputs_body, encoding="utf-8", newline="\n")
    labels_p.write_text(labels_body, encoding="utf-8", newline="\n")
    if _sha_file(inputs_p) != new_inputs_sha or _sha_file(labels_p) != new_labels_sha:
        raise DatasetBuildError("post-write verification failed")
    man["inputs_sha256"], man["labels_sha256"] = new_inputs_sha, new_labels_sha
    man.setdefault("revisions", []).append({
        "at": now or _now(), "kind": REPAIR_SOURCE,
        "why": ("lines recorded in the authoritative line inventory but never audited by the OCR benchmark "
                "(their upstream crop was tagged bad_segmentation) were re-cropped and transcribed by a human "
                "for GRADE_PRIMARY only; evaluation/hebrew_bench_v2 is unchanged"),
        "previous_inputs_sha256": old_inputs_sha, "inputs_sha256": new_inputs_sha, "inputs_changed": True,
        "previous_labels_sha256": old_labels_sha, "labels_sha256": new_labels_sha,
        "lines_repaired": sorted(used_lines), "cases_changed": [c["case_id"] for c in changed_cases],
        "repair_store": REPAIRS_FILENAME, "repair_store_sha256": ver["store_sha256"],
        "frozen_bench_sha256": frozen_before,
        "note": "model input changed: any downstream run/bundle made against the previous inputs_sha256 is a "
                "different evidence version and must be re-registered"})
    man.setdefault("extra", {})["evidence_inventory"] = summary["evidence_inventory"]
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (d / "CHECKSUMS.sha256").write_text(
        f"{new_inputs_sha}  cases_inputs.jsonl\n{new_labels_sha}  cases_labels.jsonl\n",
        encoding="utf-8", newline="\n")
    assert_frozen_bench_unchanged(frozen_before, bench_root)
    summary["written"] = True
    return summary


__all__ = ["REPAIRS_FILENAME", "REPAIRS_CROPS_DIRNAME", "REPAIR_SOURCE", "REPAIR_SCHEMA_VERSION", "DISPOSITIONS",
           "FROZEN_BENCH_FILES", "RepairError", "RepairStore", "case_geometry", "cell_crop_path", "locate_exact",
           "render_band", "suggested_band", "expected_repairs", "repair_status", "verify_repairs", "apply_repairs",
           "frozen_bench_hashes", "assert_frozen_bench_unchanged"]
