"""Declared-dataset writer + builders for the B3/B4/B5 roles.

The OCR benchmarks are frozen in evaluation/hebrew_bench_v2 and are never
written by this package. The remaining roles use the generic declared format
read by manifests.load_declared:

    <datasets_root>/<role>/manifest.json
    <datasets_root>/<role>/cases_inputs.jsonl     model-visible only
    <datasets_root>/<role>/cases_labels.jsonl     evaluation-side only
    (+ role assets: bands/*.png, covers/*.png; + owner_labels.json for grading)

``write_declared_dataset`` is the ONLY writer: it refuses to overwrite an
existing frozen dataset, keeps inputs and labels in separate files, refuses
input/label field overlap, records sha256 of both in the manifest, and stores
the split assignment.

Builders (NO model calls; local deterministic processing only):
    build_grading_dataset   audited cell transcriptions + NO-RAG grading packs from the
                            frozen image-processing key; labels = owner scores (separate file)
    build_mc_dataset        deterministic answer-table band crops of the prob scans
                            (ingest.load_pages + tablecrop), ambiguous rows only;
                            labels = agent-audited answers (provenance recorded)
    build_variant_dataset   marker-region crops (bottom third of page 1) of prob
                            scans + operator-verified Stage-A covers; labels = variant id
    build_escalation_dataset harvest genuinely unclean cases from a grade_primary run
No builder fabricates labels; provenance strings travel with every label.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

from .manifests import DEFAULT_BENCH_ROOT, REPO_ROOT, SPLITS, STATUS_FROZEN

#: writer -> split, the SAME Split A as the verifier/OCR benchmarks
WRITER_SPLIT_A = {"e002": "DEV", "e003": "DEV", "e007": "DEV", "e004": "CALIBRATION",
                  "e005": "HELD_OUT", "e006": "HELD_OUT"}


class DatasetExists(RuntimeError):
    """Refusing to overwrite a frozen declared dataset."""


class DatasetBuildError(RuntimeError):
    """A builder could not produce an honest dataset from the existing artifacts."""


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


# ----------------------------------------------------------------------------
# helpers shared by builders
# ----------------------------------------------------------------------------

def _load_key(key_json: Path):
    from ..schema import AnswerKey
    kd = json.loads(Path(key_json).read_text(encoding="utf-8"))
    return AnswerKey.model_validate(kd.get("answer_key", kd)), kd


def default_grading_key_path() -> Path | None:
    """The frozen image-processing key parse the hebrew_bench cells come from:
    prefer the in-tree repaired copy (Q2.8 per versions-override / the
    representative-exam audit), else the legacy key cache entry used by
    scripts/grading_rag_ab.py."""
    import os
    candidates = [
        REPO_ROOT / "eval_out" / "shared" / "answer_key.json",
        Path(os.environ.get("LOCALAPPDATA", "")) / "autograder" / "key_cache"
        / "0758cd7fa39b5949d86d2b7c21d6ecd380d089a451a9c09d5e980cf39344d9c3.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


_LINE_RE = re.compile(r"^hl_(e\d{3})_q(\d+)_r(\d+)__l(\d+)$")
_CELL_RE = re.compile(r"^hc_(e\d{3})_q(\d+)_r(\d+)$")


def audited_cells(bench_root: Path = DEFAULT_BENCH_ROOT) -> dict[str, dict]:
    """Group the owner-tier hebrew_bench_v2 items into student-answer cells
    keyed `e00X_qY_rZ`, with the audited (final) transcription assembled from
    the cell item or the ordered line items. Cells whose line set is not
    contiguous from l1 are marked incomplete."""
    from .manifests import _load_refaudit, reference_provenance
    ra = _load_refaudit()
    store = ra.AuditStore(bench_root)
    items = json.loads((bench_root / "items.json").read_text(encoding="utf-8"))["items"]
    refs_raw = json.loads((bench_root / "references.json").read_text(encoding="utf-8"))
    cells: dict[str, dict] = {}
    for it in items:
        iid = it["id"]
        m_line, m_cell = _LINE_RE.match(iid), _CELL_RE.match(iid)
        if not (m_line or m_cell):
            continue
        w, q, r = (m_line or m_cell).group(1), (m_line or m_cell).group(2), (m_line or m_cell).group(3)
        cid = f"{w}_q{q}_r{r}"
        ref = ra.reference_for_scoring(store, iid, "final")
        prov = reference_provenance(ra, store, it, refs_raw.get(iid) or {}, ref)
        c = cells.setdefault(cid, {"cell_id": cid, "writer": w, "question_id": q, "sub_item_id": r,
                                   "parts": [], "images": [], "provenance_classes": set()})
        c["parts"].append((int(m_line.group(4)) if m_line else 0, ref.reference or "", iid))
        c["images"].append(it["image"])
        c["provenance_classes"].add(prov["provenance_class"])
    for c in cells.values():
        parts = sorted(c["parts"])
        idx = [n for n, _, _ in parts]
        c["items"] = [i for _, _, i in parts]
        c["complete"] = (idx == [0]) or (idx == list(range(1, len(idx) + 1)))
        c["transcription"] = "\n".join(t for _, t, _ in parts)
        c["provenance_classes"] = sorted(c["provenance_classes"])
        c["provenance_valid"] = all(pc in ("audited_confirmed", "audited_corrected") for pc in c["provenance_classes"])
        del c["parts"]
    return cells


# ----------------------------------------------------------------------------
# 4A GRADE_PRIMARY
# ----------------------------------------------------------------------------

def build_grading_dataset(out_dir: Path, *, key_json: Path | None = None,
                          bench_root: Path = DEFAULT_BENCH_ROOT,
                          grading_policy: str = "choice_and_explanation_independent",
                          now: str | None = None) -> dict[str, Any]:
    """Inputs only (owner labels come later via OwnerLabelStore). Each case:
    the sub-item's grading pack (question text, rubric, official solution,
    max points, policy; NO RAG), selected=None (explanation-only cells),
    the FROZEN audited transcription; version=None."""
    from dataclasses import asdict
    from ..gradingpack import build_pack
    key_path = Path(key_json) if key_json else default_grading_key_path()
    if key_path is None or not key_path.exists():
        raise DatasetBuildError("no frozen answer-key JSON found (eval_out/shared/answer_key.json or the "
                                "0758cd7f key-cache entry); pass --key-json")
    key, _ = _load_key(key_path)
    qs = {q.id: q for q in key.questions}
    cells = audited_cells(bench_root)
    inputs, labels, excluded = [], [], []
    for cid in sorted(cells):
        c = cells[cid]
        if not c["complete"]:
            excluded.append({"cell": cid, "why": f"incomplete line set {c['items']}"})
            continue
        if not c["provenance_valid"]:
            excluded.append({"cell": cid, "why": f"reference provenance {c['provenance_classes']}"})
            continue
        q = qs.get(c["question_id"])
        sub = next((s for s in (q.sub_items if q else []) if s.id == c["sub_item_id"]), None)
        if q is None or sub is None:
            excluded.append({"cell": cid, "why": "no matching key sub-item"})
            continue
        pack = build_pack(key, q, grading_policy=grading_policy)
        # narrow to THIS sub-item (scripts/grading_rag_ab.py convention)
        pack.correct_by_version = {sub.id: dict(pack.correct_by_version.get(sub.id, {}))}
        pack.official_solution = {sub.id: pack.official_solution.get(sub.id, sub.reference_explanation or "")}
        pack.question_text = f"{q.title}\n- ({sub.id}) {sub.prompt}"
        pack.max_score = float(sub.points)
        visible_pack = {
            "question_id": pack.question_id, "question_text": pack.question_text,
            "question_type": pack.question_type, "max_score": pack.max_score,
            "correct_by_version": pack.correct_by_version, "rubric": list(pack.rubric),
            "scoring_rules": list(pack.scoring_rules), "grading_policy": pack.grading_policy,
            "official_solution": pack.official_solution,
            "rubric_items": [asdict(ri) for ri in pack.rubric_items],
            "evidence_policy": pack.evidence_policy, "score_granularity": pack.score_granularity,
        }
        inputs.append({"case_id": cid, "pack": visible_pack, "selected": None,
                       "transcription": c["transcription"], "version": None})
        labels.append({"case_id": cid, "split": WRITER_SPLIT_A.get(c["writer"], "DEV"),
                       "writer": c["writer"], "question_id": c["question_id"], "sub_item_id": c["sub_item_id"],
                       "score": None, "rubric_met": None,
                       "label_status": "NEEDS_OWNER_LABEL",
                       "transcription_source": "audited human reference (reference_for_scoring mode=final)",
                       "transcription_items": c["items"], "transcription_provenance": c["provenance_classes"],
                       "evidence_images": [f"hebrew_bench_v2/{p}" for p in c["images"]],
                       "max_score": pack.max_score})
    split_assignment = {s: sorted({l["writer"] for l in labels if l["split"] == s}) for s in SPLITS}
    man = write_declared_dataset(
        Path(out_dir), name="grade_primary — frozen audited cell transcriptions + NO-RAG grading packs",
        cases_inputs=inputs, cases_labels=labels, split_assignment=split_assignment,
        policy=("model-visible: pack (question text, rubric, official solution, max points, policy), "
                "selected=None, frozen audited transcription, version=None. NO RAG. Evaluation-side: owner "
                "final score / rubric decisions (owner_labels.json, written by scripts/grade_label_ui.py); "
                "no label is ever derived from a model."),
        notes=["first model-selection grading benchmark is NO-RAG by construction",
               "evidence_images are for the OWNER's labeling only — never a model input of this role",
               "label reality: human per-item scores did not exist before this dataset; the owner "
               "labeling tool creates them"],
        extra={"key_path": str(key_path), "key_sha256": _sha(key_path),
               "key_provenance": ("image-processing exam key parsed locally by qwen3-vl:8b-instruct from "
                                  "sample_data/Exam_solution.pdf (2026-07-12); eval_out copy carries the Q2.8 "
                                  "G/F/F repair from versions-override.json"),
               "grading_policy": grading_policy,
               "pack_builder": "gradingpack.build_pack narrowed per sub-item; no course/retrieval",
               "excluded_cells": excluded, "cells_total": len(cells), "cases": len(inputs)},
        now=now)
    man["excluded_cells"] = excluded
    return man


# ----------------------------------------------------------------------------
# 4B GRADE_ESCALATE (harvested from a grade_primary run)
# ----------------------------------------------------------------------------

def build_escalation_dataset(out_dir: Path, *, from_run_dir: Path, grade_dataset_dir: Path,
                             now: str | None = None) -> dict[str, Any]:
    """The escalation benchmark = the genuinely unclean cases of a
    grade_primary run (validation failure / uncertain / REVIEW decision).
    Until such a run exists the role stays PENDING_PRIMARY_RESULTS."""
    run = Path(from_run_dir)
    scored_p = run / "scored.jsonl.json"
    if not scored_p.exists():
        raise DatasetBuildError(f"{run} has no scored results (run grade_primary first)")
    scored = json.loads(scored_p.read_text(encoding="utf-8"))
    unclean = {r["case_id"] for r in scored
               if r.get("decision") == "REVIEW" or r.get("validation_ok") is False or r.get("uncertain")}
    if not unclean:
        raise DatasetBuildError("the grade_primary run has no unclean cases to escalate")
    gd = Path(grade_dataset_dir)
    inputs = [json.loads(l) for l in (gd / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = [json.loads(l) for l in (gd / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    inputs = [r for r in inputs if r["case_id"] in unclean]
    labels = [r for r in labels if r["case_id"] in unclean]
    return write_declared_dataset(
        Path(out_dir), name="grade_escalate — genuinely unclean cases harvested from a grade_primary run",
        cases_inputs=inputs, cases_labels=labels,
        split_assignment=json.loads((gd / "manifest.json").read_text(encoding="utf-8")).get("split_assignment", {}),
        notes=["escalation cases are harvested, never chosen by hand; difficulty is not invented",
               f"harvested from {run.name}: decision REVIEW / validation failure / uncertain"],
        extra={"harvested_from_run": str(run), "criteria": "decision==REVIEW or validation_ok==False or uncertain",
               "grade_primary_manifest_sha256": _sha(gd / "manifest.json")}, now=now)


# ----------------------------------------------------------------------------
# 4C MC_RESOLVE (prob answer-table bands; ambiguous rows only)
# ----------------------------------------------------------------------------

AUDIT_PROVENANCE = ("evaluation/prob/manual_audit.json — dev-assistant AGENT visual audit 2026-08-07 (two agents, "
                    "130/130 rows unanimous); NOT human ground truth; owner verified only the three disputed "
                    "exam TOTALS (05, 06, 13); four key columns pinned by 10 exactly matching totals")


def build_mc_dataset(out_dir: Path, *, prob_root: Path = REPO_ROOT / "prob_data",
                     audit_path: Path = REPO_ROOT / "evaluation" / "prob" / "manual_audit.json",
                     max_image_edge: int = 1400, now: str | None = None) -> dict[str, Any]:
    from ..ingest import load_pages
    from ..tablecrop import analyze_answer_table, answer_table_row_bands
    from ..template import load_template
    key, _ = _load_key(prob_root / "sol.answer_key.json")
    template = load_template(prob_root / "sol.answer_key.json")
    letters = list(template.answer_table_columns_rtl)
    n_rows = len(key.questions[0].sub_items)
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    by_src = {e["source_index"]: e for e in audit["exams"]}
    out = Path(out_dir)
    if (out / "manifest.json").exists():
        raise DatasetExists(f"{out} already holds a frozen dataset")
    (out / "bands").mkdir(parents=True, exist_ok=True)
    inputs, labels, per_exam = [], [], {}
    for src in sorted(by_src):
        pdf = prob_root / f"{src}.pdf"
        if not pdf.exists():
            continue
        pages = load_pages(pdf, max_image_edge)
        sheet = next((p for p in pages if p.page_number == template.answer_sheet_pages[0]), pages[0])
        bands = answer_table_row_bands(sheet, n_rows)
        masses = analyze_answer_table(sheet, n_rows, n_options=len(letters))
        per_exam[src] = 0
        for rm in masses:
            if len(rm.marked) <= 1:
                continue                      # deterministic rows never reach the resolver
            rid = rm.row_index + 1
            cand = [letters[i] for i in rm.marked]
            rel = f"bands/prob{src}_r{rid}.png"
            (out / rel).write_bytes(bands[rm.row_index].png_bytes)
            cid = f"prob{src}_r{rid}"
            inputs.append({"case_id": cid, "band_png": rel, "letters": letters, "candidates": cand})
            ex = by_src[src]
            labels.append({"case_id": cid, "split": "DEV", "answer": ex["answers"][str(rid)],
                           "state": ex["statuses"][str(rid)], "marks_description": ex["marks"].get(str(rid)),
                           "source_exam": ex["exam"], "source_index": src,
                           "label_provenance": AUDIT_PROVENANCE,
                           "deterministic_candidates_contain_answer": ex["answers"][str(rid)] in cand})
            per_exam[src] += 1
    if not inputs:
        raise DatasetBuildError("no ambiguous rows found in the prob scans")
    return write_declared_dataset(
        out, name="mc_resolve_cloud — deterministically ambiguous prob answer-table rows (band crops)",
        cases_inputs=inputs, cases_labels=labels, split_assignment={"DEV": sorted(per_exam)},
        policy=("model-visible: band crop + option letters + deterministic candidate letters (exactly "
                "production's mcresolve._prompt_blocks); evaluation-side: audited selected answer + audit "
                "text. DEV only: too small for CALIBRATION/HELD_OUT — expand from further audited jobs "
                "before selecting a cloud resolver."),
        notes=["band crops regenerated deterministically (ingest.load_pages + tablecrop) — no model",
               "labels are agent-audited (provenance recorded on every label), not human ground truth",
               "historical local-Qwen reads live in evaluation/mc_fallback/ as OUTPUTS, never labels"],
        extra={"prob_root": str(prob_root), "max_image_edge": max_image_edge, "letters_rtl": letters,
               "ambiguous_rows_per_exam": per_exam, "audit_sha256": _sha(Path(audit_path))}, now=now)


# ----------------------------------------------------------------------------
# 4D VARIANT_RESOLVE (marker-region crops)
# ----------------------------------------------------------------------------

#: vertical fraction of page 1 kept (the marker sits "in the bottom third, next
#: to 'בהצלחה!'" for both exam families); the header/identity area is excluded.
MARKER_REGION = (0.66, 1.0)
STAGE_A_COVERS = (("sample_data/student_exam.pdf", "A1"), ("test/003_70.pdf", "A2"), ("test/002_76.pdf", "A3"))
STAGE_A_PROVENANCE = ("sample_data/Exam_solution.alignment.json _source — printed booklet content matched to the key "
                      "by the operator (2026-07-13); marker->variant mapping confirmed by the exam owner")


def render_marker_region(pdf: Path, *, page_index: int = 0, region: tuple[float, float] = MARKER_REGION,
                         max_edge: int = 1400) -> bytes:
    """Deterministic local render of the marker region of a page (PyMuPDF
    clip; no model). Returns PNG bytes."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf))
    try:
        page = doc[page_index]
        r = page.rect
        clip = fitz.Rect(r.x0, r.y0 + r.height * region[0], r.x1, r.y0 + r.height * region[1])
        zoom = max_edge / max(r.width, r.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
        return pix.tobytes("png")
    finally:
        doc.close()


def build_variant_dataset(out_dir: Path, *, prob_root: Path = REPO_ROOT / "prob_data",
                          audit_path: Path = REPO_ROOT / "evaluation" / "prob" / "manual_audit.json",
                          stage_a: tuple = STAGE_A_COVERS, max_edge: int = 1400,
                          now: str | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    if (out / "manifest.json").exists():
        raise DatasetExists(f"{out} already holds a frozen dataset")
    (out / "covers").mkdir(parents=True, exist_ok=True)
    inputs, labels = [], []
    # prob card suits
    key, _ = _load_key(prob_root / "sol.answer_key.json")
    versions_prob = list(key.versions)
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    pinned = {"02", "30", "28", "15", "21", "24", "29", "32", "36", "37"}   # totals-pinned exams
    for ex in sorted(audit["exams"], key=lambda e: e["source_index"]):
        src = ex["source_index"]
        pdf = prob_root / f"{src}.pdf"
        if not pdf.exists():
            continue
        rel = f"covers/prob{src}_marker.png"
        (out / rel).write_bytes(render_marker_region(pdf, max_edge=max_edge))
        cid = f"prob{src}"
        inputs.append({"case_id": cid, "versions": versions_prob, "cover_png": rel})
        labels.append({"case_id": cid, "split": "DEV", "variant": ex["suit"], "variants": [ex["suit"]],
                       "n_variants": 1, "marker_kind": "card-suit symbol", "source_exam": ex["exam"],
                       "source_index": src,
                       "label_provenance": AUDIT_PROVENANCE + ("; suit independently pinned by the exam total"
                                                               if src in pinned else "; suit agent-read only")})
    # Stage-A flowers (operator content-verified)
    for rel_pdf, variant in stage_a:
        pdf = REPO_ROOT / rel_pdf
        if not pdf.exists():
            continue
        stem = Path(rel_pdf).stem.replace(" ", "_")
        rel = f"covers/flower_{stem}_marker.png"
        (out / rel).write_bytes(render_marker_region(pdf, max_edge=max_edge))
        cid = f"flower_{stem}"
        inputs.append({"case_id": cid, "versions": ["A1", "A2", "A3"], "cover_png": rel})
        labels.append({"case_id": cid, "split": "DEV", "variant": variant, "variants": [variant], "n_variants": 1,
                       "marker_kind": "flower", "source_file": rel_pdf, "label_provenance": STAGE_A_PROVENANCE})
    if not inputs:
        raise DatasetBuildError("no cover sources found")
    return write_declared_dataset(
        out, name="variant_resolve — marker-region crops (bottom third of page 1) with audited variant ids",
        cases_inputs=inputs, cases_labels=labels, split_assignment={"DEV": ["prob", "stage_a"]},
        policy=("model-visible: the page-1 marker region (y 66%-100%; header/identity area excluded) + the "
                "generic variant ids of that exam family; evaluation-side: the audited variant. Production "
                "currently sends the FULL page 1 — this benchmark is the marker-region variant; a production-"
                "parity run can render full pages locally without committing them. DEV only (16 cases)."),
        notes=["no symbol is assumed universal: variant ids and marker kinds come from each exam family's data",
               "prob labels are agent-audited (10/13 pinned by totals); flowers are operator content-verified"],
        extra={"marker_region": MARKER_REGION, "max_edge": max_edge, "stage_a": list(stage_a),
               "audit_sha256": _sha(Path(audit_path))}, now=now)


__all__ = ["write_declared_dataset", "DatasetExists", "DatasetBuildError", "WRITER_SPLIT_A",
           "audited_cells", "default_grading_key_path", "build_grading_dataset", "build_escalation_dataset",
           "build_mc_dataset", "build_variant_dataset", "render_marker_region", "MARKER_REGION",
           "AUDIT_PROVENANCE", "STAGE_A_COVERS"]
