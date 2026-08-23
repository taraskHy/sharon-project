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
#: how a FINAL was REACHED. "authoritative" is a single instructor-copied grade
#: promoted verbatim — see labeling_app.db.finalize_authoritative. An export
#: carrying any other value is recorded and skipped, never silently dropped.
FINAL_SOURCES_ACCEPTED = ("agreement", "adjudicated", "authoritative")
#: schema 3 adds ground-truth PROVENANCE (ground_truth_source) alongside the
#: evidence version; schema 1/2 exports stay importable (source treated as unknown)
SUPPORTED_EXPORT_SCHEMA = (1, 2, 3)


def import_final_labels(export_path: Path, dataset_dir: Path, *, now: str | None = None) -> dict[str, Any]:
    from ..eligibility import eligibility_for_case
    export_path, dataset_dir = Path(export_path), Path(dataset_dir)
    data = json.loads(export_path.read_text(encoding="utf-8"))
    if data.get("schema_version") not in SUPPORTED_EXPORT_SCHEMA or data.get("kind") != "grade_primary_final_labels":
        raise ValueError("not a supported grade_primary final-labels export")
    label_rows = {json.loads(l)["case_id"]: json.loads(l)
                  for l in (dataset_dir / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    input_rows = {json.loads(l)["case_id"]: json.loads(l)
                  for l in (dataset_dir / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    known = set(label_rows)
    labels: dict[str, dict] = {}
    unknown: list[str] = []
    ignored_ineligible: dict[str, dict] = {}
    ignored_stale_evidence: dict[str, dict] = {}
    ignored_unknown_source: dict[str, dict] = {}
    for row in data.get("items", []):
        cid = row["item_id"]
        if cid not in known:
            unknown.append(cid)
            continue
        if row.get("source") not in FINAL_SOURCES_ACCEPTED:
            # Recorded, never silent: an unrecognised FINAL source used to drop the
            # row without a trace, so a whole export could import zero labels and
            # still look successful.
            ignored_unknown_source[cid] = {"reason": "unrecognised FINAL source",
                                           "source": row.get("source"),
                                           "accepted": list(FINAL_SOURCES_ACCEPTED),
                                           "ignored_human_final_label": float(row["final_score"])}
            continue
        if row.get("evidence_stale") is True:
            # the FINAL was made against evidence the labeling bundle has since
            # corrected; it is provenance, never ground truth, until re-reviewed
            ignored_stale_evidence[cid] = {"reason": "evidence_changed_after_final",
                                           "evidence_sha256": row.get("evidence_sha256"),
                                           "ignored_human_final_label": float(row["final_score"])}
            continue
        if row.get("eligible_for_human_label") is False:      # export already marked it obsolete
            ignored_ineligible[cid] = {"reason": "export_marked_ineligible", "policy": None,
                                       "deterministic_policy_score": None,
                                       "ignored_human_final_label": float(row["final_score"])}
            continue
        # Eligibility is recomputed HERE from the dataset itself (the single
        # source of truth) — a human label for a case whose score the grading
        # policy already decides deterministically is never promoted to
        # benchmark ground truth, whatever the export claims.
        if cid in input_rows:
            elig = eligibility_for_case(input_rows[cid], label_rows.get(cid))
            if not elig.eligible_for_human_label:
                ignored_ineligible[cid] = {"reason": elig.reason, "policy": elig.policy,
                                           "deterministic_policy_score": elig.deterministic_score,
                                           "ignored_human_final_label": float(row["final_score"])}
                continue
        labels[cid] = {"score": float(row["final_score"]),
                       "rubric_decisions": sorted(row.get("rubric_decisions") or []),
                       "note": row.get("note") or "", "source": row["source"],
                       "contributing_graders": sorted(row.get("contributing_graders") or []),
                       "adjudicator": row.get("adjudicator"), "finalized_at": row.get("finalized_at"),
                       # ground-truth PROVENANCE and evidence VERSION are separate facts
                       "ground_truth_source": row.get("ground_truth_source"),
                       "evidence_sha256": row.get("evidence_sha256"),
                       "evidence_repaired": bool(row.get("evidence_repaired"))}
    out = {"schema_version": 1, "imported_at": now or time.strftime("%Y-%m-%d %H:%M:%S"),
           "source_export": {"path": str(export_path), "content_sha256": data.get("content_sha256"),
                             "exported_at": data.get("exported_at"), "final_count": data.get("final_count"),
                             "bundle_items_sha256": data.get("bundle_items_sha256"),
                             "file_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest()},
           "labels": dict(sorted(labels.items()))}
    if ignored_ineligible:
        out["ignored_ineligible"] = dict(sorted(ignored_ineligible.items()))
    if ignored_stale_evidence:
        out["ignored_stale_evidence"] = dict(sorted(ignored_stale_evidence.items()))
    if ignored_unknown_source:
        out["ignored_unknown_source"] = dict(sorted(ignored_unknown_source.items()))
    (dataset_dir / FINAL_LABELS_FILENAME).write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True),
                                                     encoding="utf-8", newline="\n")
    return {"imported": len(labels), "unknown_case_ids": unknown,
            "ignored_unknown_source": dict(sorted(ignored_unknown_source.items())),
            "ignored_ineligible": sorted(ignored_ineligible), "ignored_stale_evidence": sorted(ignored_stale_evidence),
            "path": str(dataset_dir / FINAL_LABELS_FILENAME)}


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
        # The benchmark keeps these as SEPARATE facts: what the ground-truth score
        # is and how it was derived, versus which evidence version the model sees
        # and whether that evidence was ever repaired.
        lab["ground_truth_score"] = f.get("score")
        lab["ground_truth_source"] = f.get("ground_truth_source")
        lab["evidence_sha256"] = f.get("evidence_sha256")
        lab["evidence_repaired"] = bool(f.get("evidence_repaired"))
        n += 1
    return n, hashlib.sha256(p.read_bytes()).hexdigest()


__all__ = ["FINAL_LABELS_FILENAME", "import_final_labels", "merge_final_labels"]
