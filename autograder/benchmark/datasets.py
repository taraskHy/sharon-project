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
    repair_grading_evidence re-freeze ONLY the label-side evidence inventory of an
                            existing grade_primary dataset from the upstream line
                            records (inputs byte-identical; revision recorded)
No builder fabricates labels; provenance strings travel with every label.

Evidence inventory (grade_primary): the answer crops a HUMAN grades are taken
from the AUTHORITATIVE upstream line inventory (evaluation/htr_pilot/splits/*.json:
one record per handwritten line, ``n_lines`` per cell), never from the subset of
lines that happen to carry an audited OCR transcription — one image per recorded
line, in line order. A line without an audited transcription is still evidence;
the case is then marked ``transcription_complete: false`` (the model-visible
transcription covers only the audited lines) and is NOT an accuracy case.
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
_SAMPLE_RE = re.compile(r"^(e\d{3})_q(\d+)_r(\d+)__l(\d+)$")

#: the HTR pilot package — the AUTHORITATIVE per-cell line inventory of the
#: handwritten answers (one record per line crop, ``n_lines`` per cell)
DEFAULT_HTR_ROOT = REPO_ROOT / "evaluation" / "htr_pilot"
HTR_SPLIT_FILES = ("train", "val", "internal_test")
#: exam-002 cells are whole-cell crops (one image per cell) recorded here
CELL_CROP_MANIFEST = "evaluation/hebrew_bench/crops_manifest.json"
#: the evaluation-side evidence block of a grading label row (written as ONE
#: contiguous group, in this order, by the builder and by the repair)
EVIDENCE_LABEL_FIELDS = ("evidence_images", "evidence_kind", "line_count", "line_inventory_source",
                         "evidence_lines", "transcription_complete", "lines_without_audited_transcription")


def _annotation_record(htr_root: Path, split: str, sample_id: str) -> dict | None:
    p = Path(htr_root) / "annotations" / split / f"{sample_id}.json"
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return rec if isinstance(rec, dict) else None


def load_line_inventory(htr_root: Path = DEFAULT_HTR_ROOT) -> dict[str, dict]:
    """The authoritative per-cell line inventory: ``<htr_root>/splits/*.json``
    (the HTR pilot package's declared sample manifests — writer, question, row,
    line_index, n_lines, line-crop image per record). The per-line annotation
    status is joined from ``annotations/<split>/<sample_id>.json`` as
    information only (an unannotated or badly segmented line is STILL a line).

    Returns ``{cell_id: {"n_lines", "source", "lines": [{line_index, sample_id,
    image (evaluation-relative), annotation_status, human_verified, split}, ...]}}``
    with lines sorted by line_index and verified to be exactly 1..n_lines.
    Raises DatasetBuildError when the package is missing or inconsistent —
    never guesses a line count."""
    htr_root = Path(htr_root)
    splits_dir = htr_root / "splits"
    if not splits_dir.is_dir():
        raise DatasetBuildError(f"upstream line inventory missing: {splits_dir} (HTR pilot package)")
    pkg = htr_root.name                                   # image paths are package-relative
    cells: dict[str, dict] = {}
    for split in HTR_SPLIT_FILES:
        sp = splits_dir / f"{split}.json"
        if not sp.exists():
            continue
        source = f"evaluation/{pkg}/splits/{split}.json"
        for rec in json.loads(sp.read_text(encoding="utf-8")):
            sid = rec["sample_id"]
            m = _SAMPLE_RE.match(sid)
            if not m:
                raise DatasetBuildError(f"{source}: unexpected sample id {sid!r}")
            w, q, r, li = m.group(1), m.group(2), m.group(3), int(m.group(4))
            if (rec.get("writer"), int(rec.get("question")), int(rec.get("row")),
                    int(rec.get("line_index"))) != (w, int(q), int(r), li):
                raise DatasetBuildError(f"{source}: record fields disagree with sample id {sid!r}")
            cid = f"{w}_q{q}_r{r}"
            c = cells.setdefault(cid, {"cell_id": cid, "n_lines": int(rec["n_lines"]), "source": source,
                                       "lines": []})
            if c["n_lines"] != int(rec["n_lines"]) or c["source"] != source:
                raise DatasetBuildError(f"{cid}: inconsistent n_lines/split across its upstream line records")
            ann = _annotation_record(htr_root, rec.get("split") or split, sid)
            c["lines"].append({"line_index": li, "sample_id": sid,
                               "image": f"{pkg}/{rec['images']['line']}",
                               "annotation_status": (ann or {}).get("status"),
                               "human_verified": bool((ann or {}).get("human_verified")),
                               "split": rec.get("split") or split})
    for cid, c in cells.items():
        c["lines"].sort(key=lambda l: l["line_index"])
        idx = [l["line_index"] for l in c["lines"]]
        if idx != list(range(1, c["n_lines"] + 1)):
            raise DatasetBuildError(f"{cid}: upstream line records {idx} are not exactly 1..{c['n_lines']}")
    return cells


def _cell_evidence(cid: str, parts: list[tuple], inventory: dict[str, dict], *, bench_root: Path,
                   evaluation_root: Path) -> dict[str, Any]:
    """The evidence block of one cell: one image per recorded line, in line
    order. A line WITH an audited transcription is represented by its bench
    crop (verified byte-identical to the upstream line image); a line WITHOUT
    one by the upstream line image itself. Nothing is inferred from file
    names: the inventory record is the only source of the line list."""
    bench_rel = Path(bench_root).name                      # "hebrew_bench_v2"
    if [n for n, *_ in parts] == [0]:                      # exam-002 whole-cell crop
        _, _, iid, image, pclass = parts[0]
        rel = f"{bench_rel}/{image}"
        return {"evidence_kind": "cell_crop", "line_count": 1, "line_inventory_source": CELL_CROP_MANIFEST,
                "evidence": [{"index": 1, "sample_id": None, "bench_item": iid, "image": rel,
                              "transcription_status": pclass}],
                "evidence_images": [rel], "transcription_complete": True,
                "lines_without_audited_transcription": []}
    inv = inventory.get(cid)
    if inv is None:
        raise DatasetBuildError(f"{cid}: no upstream line record (htr_pilot splits) for this cell — "
                                "the line count cannot be determined honestly")
    audited = {n: (iid, image, pclass) for n, _, iid, image, pclass in parts}
    extra = sorted(set(audited) - {l["line_index"] for l in inv["lines"]})
    if extra:
        raise DatasetBuildError(f"{cid}: audited line(s) {extra} are not in the upstream inventory "
                                f"(n_lines {inv['n_lines']})")
    evidence, missing = [], []
    for line in inv["lines"]:
        n = line["line_index"]
        up = Path(evaluation_root) / line["image"]
        if not up.exists():
            raise DatasetBuildError(f"{cid}: upstream line image missing: {line['image']}")
        if n in audited:
            iid, image, pclass = audited[n]
            bench_png = Path(bench_root) / image
            if _sha(bench_png) != _sha(up):
                raise DatasetBuildError(f"{iid}: bench crop {image} differs from the upstream line image "
                                        f"{line['image']}; evidence provenance is ambiguous — refusing")
            evidence.append({"index": n, "sample_id": line["sample_id"], "bench_item": iid,
                             "image": f"{bench_rel}/{image}", "transcription_status": pclass})
        else:
            status = line["annotation_status"] or "unannotated"
            evidence.append({"index": n, "sample_id": line["sample_id"], "bench_item": None,
                             "image": line["image"], "transcription_status": f"no_audited_transcription:{status}"})
            missing.append(line["sample_id"])
    return {"evidence_kind": "line_crops", "line_count": int(inv["n_lines"]),
            "line_inventory_source": inv["source"], "evidence": evidence,
            "evidence_images": [e["image"] for e in evidence],
            "transcription_complete": not missing, "lines_without_audited_transcription": missing}


def evidence_label_fields(cell: dict) -> dict[str, Any]:
    """The evaluation-side evidence block of a label row (EVIDENCE_LABEL_FIELDS order)."""
    return {"evidence_images": list(cell["evidence_images"]), "evidence_kind": cell["evidence_kind"],
            "line_count": int(cell["line_count"]), "line_inventory_source": cell["line_inventory_source"],
            "evidence_lines": [dict(e) for e in cell["evidence"]],
            "transcription_complete": bool(cell["transcription_complete"]),
            "lines_without_audited_transcription": list(cell["lines_without_audited_transcription"])}


def audited_cells(bench_root: Path = DEFAULT_BENCH_ROOT, *, htr_root: Path = DEFAULT_HTR_ROOT,
                  evaluation_root: Path | None = None) -> dict[str, dict]:
    """Group the owner-tier hebrew_bench_v2 items into student-answer cells
    keyed `e00X_qY_rZ`, with the audited (final) transcription assembled from
    the cell item or the ordered line items, and the cell's EVIDENCE taken
    from the authoritative upstream line inventory (``load_line_inventory``):
    one image per recorded line, in line order. Cells whose audited line set
    is not contiguous from l1 are marked incomplete (the membership rule);
    ``transcription_complete`` tells whether EVERY recorded line is audited."""
    from .manifests import _load_refaudit, reference_provenance
    ra = _load_refaudit()
    bench_root = Path(bench_root)
    evaluation_root = Path(evaluation_root) if evaluation_root else bench_root.parent
    store = ra.AuditStore(bench_root)
    items = json.loads((bench_root / "items.json").read_text(encoding="utf-8"))["items"]
    refs_raw = json.loads((bench_root / "references.json").read_text(encoding="utf-8"))
    inventory = load_line_inventory(htr_root)
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
                                   "parts": [], "provenance_classes": set()})
        c["parts"].append((int(m_line.group(4)) if m_line else 0, ref.reference or "", iid, it["image"],
                           prov["provenance_class"]))
        c["provenance_classes"].add(prov["provenance_class"])
    for cid, c in cells.items():
        parts = sorted(c["parts"])
        idx = [n for n, *_ in parts]
        c["items"] = [i for _, _, i, _, _ in parts]
        c["bench_images"] = [img for _, _, _, img, _ in parts]
        c["complete"] = (idx == [0]) or (idx == list(range(1, len(idx) + 1)))
        c["transcription"] = "\n".join(t for _, t, _, _, _ in parts)
        c["provenance_classes"] = sorted(c["provenance_classes"])
        c["provenance_valid"] = all(pc in ("audited_confirmed", "audited_corrected") for pc in c["provenance_classes"])
        c.update(_cell_evidence(cid, parts, inventory, bench_root=bench_root, evaluation_root=evaluation_root))
        del c["parts"]
    return cells


def evidence_inventory_summary(label_rows: list[dict]) -> dict[str, Any]:
    """Manifest accounting for the evidence block across a dataset."""
    rows = list(label_rows)
    incomplete = sorted(r["case_id"] for r in rows if r.get("transcription_complete") is False)
    by_kind: dict[str, int] = {}
    for r in rows:
        by_kind[r.get("evidence_kind", "?")] = by_kind.get(r.get("evidence_kind", "?"), 0) + 1
    return {"source": "evaluation/htr_pilot/splits/*.json (line cells: one image per recorded line, in line order); "
                      + CELL_CROP_MANIFEST + " (exam-002 whole-cell crops)",
            "cases": len(rows), "evidence_images": sum(len(r.get("evidence_images") or []) for r in rows),
            "by_kind": dict(sorted(by_kind.items())),
            "transcription_incomplete_cases": incomplete,
            "policy": ("every recorded line is evidence for the HUMAN label; a case whose model-visible "
                       "transcription does not cover every recorded line is transcription_complete=false and "
                       "is excluded from model ACCURACY metrics (decision metrics still apply)")}


# ----------------------------------------------------------------------------
# 4A GRADE_PRIMARY
# ----------------------------------------------------------------------------

def route_case_by_eligibility(inp: dict, lab: dict):
    """Route one would-be grading case through the eligibility gate
    (autograder.eligibility — the single source of truth). Returns
    ``(eligibility, early_exit_record_or_None)``: a policy-decided case yields
    a provenance record for policy_early_exit.jsonl instead of a benchmark
    case; every case lands in exactly one of the two outcomes."""
    from ..eligibility import eligibility_for_case
    elig = eligibility_for_case(inp, lab)
    if elig.eligible_for_human_label:
        return elig, None
    return elig, {"case_id": inp["case_id"], "final_score": elig.deterministic_score,
                  "source": ("deterministic_mc_wrong" if elig.reason == "wrong_mc_deterministic_zero"
                             else "policy_no_explanation_component"),
                  "policy": elig.policy, "mc_correct": elig.mc_state == "correct",
                  "mc_state": elig.mc_state, "reason": elig.reason,
                  "selected_option": elig.selected_option,
                  "accepted_options": list(elig.accepted_options),
                  "split": lab["split"], "writer": lab["writer"],
                  "question_id": lab["question_id"], "sub_item_id": lab["sub_item_id"],
                  "max_score": lab["max_score"]}


def build_grading_dataset(out_dir: Path, *, key_json: Path | None = None,
                          bench_root: Path = DEFAULT_BENCH_ROOT, htr_root: Path = DEFAULT_HTR_ROOT,
                          grading_policy: str = "choice_and_explanation_independent",
                          now: str | None = None) -> dict[str, Any]:
    """Inputs only (owner labels come later via OwnerLabelStore). Each case:
    the sub-item's grading pack (question text, rubric, official solution,
    max points, policy; NO RAG), selected=None (explanation-only cells),
    the FROZEN audited transcription; version=None.

    Eligibility gate (autograder.eligibility — the single source of truth):
    a case whose score the grading policy already decides deterministically
    (confidently wrong MC under wrong_choice_zero, or under
    explanation_required_if_correct with a zero/selection wrong-answer rule)
    is NOT a model-accuracy case. It is routed to policy_early_exit.jsonl
    beside the dataset and never enters cases_inputs.jsonl. Unresolved or
    absent MC never triggers the gate."""
    from dataclasses import asdict
    from ..eligibility import eligibility_counts
    from ..gradingpack import build_pack
    key_path = Path(key_json) if key_json else default_grading_key_path()
    if key_path is None or not key_path.exists():
        raise DatasetBuildError("no frozen answer-key JSON found (eval_out/shared/answer_key.json or the "
                                "0758cd7f key-cache entry); pass --key-json")
    key, _ = _load_key(key_path)
    qs = {q.id: q for q in key.questions}
    cells = audited_cells(bench_root, htr_root=htr_root)
    inputs, labels, excluded, early_exit, eligibilities = [], [], [], [], []
    for cid in sorted(cells):
        c = cells[cid]
        if not c["complete"]:
            excluded.append({"cell": cid, "why": f"incomplete line set {c['items']}",
                             "line_count": c["line_count"]})
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
        if pack.wrong_answer_rule is not None:      # emitted only when known: keeps old rebuilds byte-identical
            visible_pack["wrong_answer_rule"] = pack.wrong_answer_rule
        inp = {"case_id": cid, "pack": visible_pack, "selected": None,
               "transcription": c["transcription"], "version": None}
        lab = {"case_id": cid, "split": WRITER_SPLIT_A.get(c["writer"], "DEV"),
               "writer": c["writer"], "question_id": c["question_id"], "sub_item_id": c["sub_item_id"],
               "score": None, "rubric_met": None,
               "label_status": "NEEDS_OWNER_LABEL",
               "transcription_source": "audited human reference (reference_for_scoring mode=final)",
               "transcription_items": c["items"], "transcription_provenance": c["provenance_classes"],
               **evidence_label_fields(c),
               "max_score": pack.max_score}
        elig, exit_record = route_case_by_eligibility(inp, lab)
        eligibilities.append(elig)
        if exit_record is not None:
            early_exit.append(exit_record)
            continue
        lab["eligibility"] = elig.to_dict()
        inputs.append(inp)
        labels.append(lab)
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
               "excluded_cells": excluded, "cells_total": len(cells), "cases": len(inputs),
               "eligibility": eligibility_counts(eligibilities),
               "policy_early_exit_cases": len(early_exit),
               "evidence_inventory": evidence_inventory_summary(labels)},
        now=now)
    if early_exit:
        with (Path(out_dir) / "policy_early_exit.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in early_exit:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man["excluded_cells"] = excluded
    man["policy_early_exit"] = early_exit
    return man


def _with_evidence_block(row: dict, block: dict) -> dict:
    """Replace the row's evidence block in place (same position as the
    builder writes it); rows without any evidence key get it before max_score."""
    out: dict = {}
    inserted = False
    for k, v in row.items():
        if k in EVIDENCE_LABEL_FIELDS:
            if not inserted:
                out.update(block)
                inserted = True
            continue
        if k == "max_score" and not inserted:
            out.update(block)
            inserted = True
        out[k] = v
    if not inserted:
        out.update(block)
    return out


def repair_grading_evidence(dataset_dir: Path, *, bench_root: Path = DEFAULT_BENCH_ROOT,
                            htr_root: Path = DEFAULT_HTR_ROOT, evaluation_root: Path | None = None,
                            now: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Re-freeze ONLY the evidence block of an existing grade_primary dataset's
    label rows from the authoritative upstream line inventory. The
    model-visible ``cases_inputs.jsonl`` is never touched (its sha256 must still
    match the manifest before AND after), case membership is never changed,
    and the frozen transcription items/provenance must agree with the OCR
    benchmark — any disagreement refuses instead of guessing. The manifest
    records the revision (previous/new labels sha256, cases whose evidence
    changed); ``CHECKSUMS.sha256`` is rewritten. ``dry_run`` computes the same
    summary without writing."""
    d = Path(dataset_dir)
    man_p, inputs_p, labels_p = d / "manifest.json", d / "cases_inputs.jsonl", d / "cases_labels.jsonl"
    if not man_p.exists():
        raise DatasetBuildError(f"{d} holds no frozen dataset")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    if _sha(inputs_p) != man.get("inputs_sha256"):
        raise DatasetBuildError("cases_inputs.jsonl does not match the manifest hash; refusing to touch a drifted dataset")
    old_labels_sha = _sha(labels_p)
    if old_labels_sha != man.get("labels_sha256"):
        raise DatasetBuildError("cases_labels.jsonl does not match the manifest hash; refusing to touch a drifted dataset")
    inputs = {r["case_id"]: r for r in (json.loads(l) for l in inputs_p.read_text(encoding="utf-8").splitlines() if l.strip())}
    rows = [json.loads(l) for l in labels_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    cells = audited_cells(bench_root, htr_root=htr_root, evaluation_root=evaluation_root)
    new_rows, updated, changed = [], [], []
    for row in rows:
        cid = row["case_id"]
        c = cells.get(cid)
        if c is None:
            raise DatasetBuildError(f"{cid}: no longer derivable from the frozen OCR benchmark; refusing")
        if not c["complete"] or not c["provenance_valid"]:
            raise DatasetBuildError(f"{cid}: case membership would change (complete={c['complete']}, "
                                    f"provenance_valid={c['provenance_valid']}); refusing")
        if list(row.get("transcription_items") or []) != c["items"] \
                or sorted(row.get("transcription_provenance") or []) != c["provenance_classes"]:
            raise DatasetBuildError(f"{cid}: frozen transcription items/provenance disagree with the OCR benchmark; refusing")
        if inputs.get(cid, {}).get("transcription") != c["transcription"]:
            raise DatasetBuildError(f"{cid}: frozen model-visible transcription disagrees with the OCR benchmark; refusing")
        new = _with_evidence_block(row, evidence_label_fields(c))
        if new != row:
            updated.append(cid)
            before = list(row.get("evidence_images") or [])
            if before != new["evidence_images"]:
                changed.append({"case_id": cid, "evidence_images_before": before,
                                "evidence_images_after": list(new["evidence_images"]),
                                "line_count": new["line_count"],
                                "transcription_complete": new["transcription_complete"],
                                "lines_without_audited_transcription": list(new["lines_without_audited_transcription"])})
        new_rows.append(new)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_rows)
    new_labels_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    summary = {"dataset": str(d), "cases": len(new_rows), "inputs_sha256": man["inputs_sha256"],
               "inputs_changed": False, "previous_labels_sha256": old_labels_sha, "labels_sha256": new_labels_sha,
               "rows_updated": updated, "cases_evidence_changed": changed,
               "evidence_inventory": evidence_inventory_summary(new_rows), "dry_run": dry_run,
               "written": False}
    if dry_run or new_labels_sha == old_labels_sha:
        return summary
    with labels_p.open("w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    if _sha(labels_p) != new_labels_sha or _sha(inputs_p) != man["inputs_sha256"]:
        raise DatasetBuildError("post-write verification failed")
    man["labels_sha256"] = new_labels_sha
    man.setdefault("revisions", []).append({
        "at": now or time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "evidence_inventory_repair",
        "why": ("evidence images are taken from the authoritative upstream line inventory (one image per "
                "recorded line, in line order); the original build took them from the OCR-benchmark subset, "
                "which silently dropped lines without an audited transcription"),
        "inputs_sha256": man["inputs_sha256"], "inputs_changed": False,
        "previous_labels_sha256": old_labels_sha, "labels_sha256": new_labels_sha,
        "rows_updated": len(updated), "cases_evidence_changed": [c["case_id"] for c in changed],
        "line_inventory_source": "evaluation/htr_pilot/splits/*.json"})
    man.setdefault("extra", {})["evidence_inventory"] = summary["evidence_inventory"]
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (d / "CHECKSUMS.sha256").write_text(
        f"{man['inputs_sha256']}  cases_inputs.jsonl\n{man['labels_sha256']}  cases_labels.jsonl\n",
        encoding="utf-8", newline="\n")
    summary["written"] = True
    return summary


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
           "audited_cells", "load_line_inventory", "evidence_label_fields", "evidence_inventory_summary",
           "repair_grading_evidence", "EVIDENCE_LABEL_FIELDS", "DEFAULT_HTR_ROOT", "CELL_CROP_MANIFEST",
           "default_grading_key_path", "build_grading_dataset", "build_escalation_dataset",
           "build_mc_dataset", "build_variant_dataset", "render_marker_region", "MARKER_REGION",
           "AUDIT_PROVENANCE", "STAGE_A_COVERS"]
