"""Confirmed row-transposition repair: e004_q2_r6 <-> e004_q2_r8.

Owner source-page inspection (2026-09-01) CONFIRMED that the ROW-ATTACHED
evidence of these two cases is transposed: the crop/transcription filed under
r6 (the echo-mask sub-item) is the student's answer to r8 (multiply-by-2),
and vice versa — corroborated independently by the answer TEXTS themselves
and by the neighbor-fit swap probe's single reciprocal signature
(SWAP_PROBE_2026-08-30).

Repairs the SOURCE MAPPING, not the grades:

    swapped between the two cases (row-attached only)
        inputs.transcription
        labels: evidence_images, transcription_items, transcription_provenance,
                evidence_lines, evidence_kind, line_count, line_inventory_source,
                lines_no_text_artifact, lines_resolved, lines_transcribed,
                lines_without_audited_transcription, transcription_complete,
                transcription_source
    NOT swapped (each case id keeps its logical sub-item)
        case_id, pack (question/rubric/official solution), question_id,
        sub_item_id, split, writer, max_score, label_status, rubric_met,
        selection fields, derived explanation-verdict fields,
        original instructor score (final_labels.json is untouched)

Recorded as manifest revision kind ``confirmed_row_transposition`` with the
old/new mapping, hashes, and owner-confirmed provenance. Refuses to run
twice, refuses drifted hashes, verifies the other 44+ rows byte-identical.
Zero model calls; HELD_OUT untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DS = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
CASE_A, CASE_B = "e004_q2_r6", "e004_q2_r8"
REVISION_KIND = "confirmed_row_transposition"

ROW_ATTACHED_LABEL_FIELDS = (
    "evidence_images", "transcription_items", "transcription_provenance",
    "evidence_lines", "evidence_kind", "line_count", "line_inventory_source",
    "lines_no_text_artifact", "lines_resolved", "lines_transcribed",
    "lines_without_audited_transcription", "transcription_complete",
    "transcription_source",
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _dump(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def apply(dry_run: bool = False) -> dict:
    man_p, in_p, lab_p = DS / "manifest.json", DS / "cases_inputs.jsonl", DS / "cases_labels.jsonl"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    if any(r.get("kind") == REVISION_KIND for r in man.get("revisions", [])):
        raise SystemExit("REFUSED: the confirmed_row_transposition revision is already applied")
    if _sha(in_p) != man["inputs_sha256"] or _sha(lab_p) != man["labels_sha256"]:
        raise SystemExit("REFUSED: dataset does not match its manifest hashes")

    inputs, labels = _rows(in_p), _rows(lab_p)
    idx_i = {r["case_id"]: n for n, r in enumerate(inputs)}
    idx_l = {r["case_id"]: n for n, r in enumerate(labels)}
    ia, ib = inputs[idx_i[CASE_A]], inputs[idx_i[CASE_B]]
    la, lb = labels[idx_l[CASE_A]], labels[idx_l[CASE_B]]
    assert ia.get("selected") is None and ib.get("selected") is None, \
        "selected is expected to be None (frozen policy); a row-attached selection would need review"

    before = {
        CASE_A: {"transcription_sha256": hashlib.sha256(ia["transcription"].encode()).hexdigest(),
                 "evidence_images": list(la["evidence_images"])},
        CASE_B: {"transcription_sha256": hashlib.sha256(ib["transcription"].encode()).hexdigest(),
                 "evidence_images": list(lb["evidence_images"])},
    }
    # the swap: model-visible transcription + every row-attached label field
    ia["transcription"], ib["transcription"] = ib["transcription"], ia["transcription"]
    for f in ROW_ATTACHED_LABEL_FIELDS:
        if (f in la) != (f in lb):
            raise SystemExit(f"REFUSED: field {f!r} present on only one row")
        if f in la:
            la[f], lb[f] = lb[f], la[f]
    after = {
        CASE_A: {"transcription_sha256": hashlib.sha256(ia["transcription"].encode()).hexdigest(),
                 "evidence_images": list(la["evidence_images"])},
        CASE_B: {"transcription_sha256": hashlib.sha256(ib["transcription"].encode()).hexdigest(),
                 "evidence_images": list(lb["evidence_images"])},
    }
    assert before[CASE_A]["transcription_sha256"] == after[CASE_B]["transcription_sha256"]
    assert before[CASE_B]["transcription_sha256"] == after[CASE_A]["transcription_sha256"]

    new_inputs, new_labels = _dump(inputs), _dump(labels)
    rev = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": REVISION_KIND,
        "why": ("owner source-page inspection (2026-09-01) confirmed the row-attached "
                "evidence of e004_q2_r6 and e004_q2_r8 is transposed; corroborated by the "
                "answer texts and the SWAP_PROBE_2026-08-30 reciprocal signature. The "
                "source mapping is repaired; grades, packs, logical sub-item identity and "
                "the original instructor scores are untouched"),
        "owner_confirmed": True,
        "authority": "original exam page (test/004_58.pdf, question 2 rows 6/8)",
        "cases_changed": [CASE_A, CASE_B],
        "mapping_before": before,
        "mapping_after": after,
        "previous_inputs_sha256": man["inputs_sha256"],
        "inputs_sha256": hashlib.sha256(new_inputs.encode("utf-8")).hexdigest(),
        "inputs_changed": True,
        "previous_labels_sha256": man["labels_sha256"],
        "labels_sha256": hashlib.sha256(new_labels.encode("utf-8")).hexdigest(),
        "rows_updated": 2,
        "model_involved": False,
        "note": ("model outputs and human reviews recorded against the pre-repair evidence "
                 "are preserved historically and marked stale/invalid downstream "
                 "(review bundle fingerprints; STALE_MODEL_OUTPUTS artifact); any run or "
                 "bundle made against the previous hashes is a different evidence version"),
    }
    result = {"revision": rev, "dry_run": dry_run}
    if dry_run:
        return result

    # verify the OTHER rows are byte-identical before writing
    old_inputs, old_labels = _rows(in_p), _rows(lab_p)
    for n, r in enumerate(inputs):
        if r["case_id"] not in (CASE_A, CASE_B):
            assert r == old_inputs[n], r["case_id"]
    for n, r in enumerate(labels):
        if r["case_id"] not in (CASE_A, CASE_B):
            assert r == old_labels[n], r["case_id"]

    in_p.write_text(new_inputs, encoding="utf-8", newline="\n")
    lab_p.write_text(new_labels, encoding="utf-8", newline="\n")
    if _sha(in_p) != rev["inputs_sha256"] or _sha(lab_p) != rev["labels_sha256"]:
        raise SystemExit("post-write verification failed")
    man["inputs_sha256"] = rev["inputs_sha256"]
    man["labels_sha256"] = rev["labels_sha256"]
    man.setdefault("revisions", []).append(rev)
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (DS / "CHECKSUMS.sha256").write_text(
        f"{rev['inputs_sha256']}  cases_inputs.jsonl\n{rev['labels_sha256']}  cases_labels.jsonl\n",
        encoding="utf-8", newline="\n")
    result["written"] = True
    return result


def replay_swaps(inputs_rows: list[dict], labels_rows: list[dict],
                 dataset_dir: Path = DS) -> list[str]:
    """Apply — or equivalently UN-apply, the swap is self-inverse — every
    owner-confirmed ``confirmed_row_transposition`` revision recorded in the
    dataset manifest to the given row lists, IN PLACE. Derivation-chain
    verifiers use this to walk between historical states and the live one.
    Returns the list of case-pair swaps performed."""
    man = json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))
    idx_i = {r["case_id"]: n for n, r in enumerate(inputs_rows)}
    idx_l = {r["case_id"]: n for n, r in enumerate(labels_rows)}
    done: list[str] = []
    for rev in man.get("revisions", []):
        if rev.get("kind") != REVISION_KIND or not rev.get("owner_confirmed"):
            continue
        a, b = rev["cases_changed"]
        ia, ib = inputs_rows[idx_i[a]], inputs_rows[idx_i[b]]
        la, lb = labels_rows[idx_l[a]], labels_rows[idx_l[b]]
        ia["transcription"], ib["transcription"] = ib["transcription"], ia["transcription"]
        for f in ROW_ATTACHED_LABEL_FIELDS:
            if f in la and f in lb:
                la[f], lb[f] = lb[f], la[f]
        done.append(f"{a}<->{b}")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    out = apply(dry_run=args.dry_run)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
