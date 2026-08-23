"""The PRE-manual-repair grade_primary dataset, reconstructed deterministically.

The checked-in `evaluation/model_selection/datasets/grade_primary` is now
POST-repair: the owner transcribed one line by hand and ruled the other eight
mis-segmented slivers to be segmentation artifacts, and
`bench apply-evidence-repairs` folded those nine decisions in. That is the
authoritative dataset and tests must never write to it.

Tests that exercise PRE-repair behaviour (deriving the expected repairs, partial
and complete application, the dry run, the revision entry, PARTIALLY_READY ->
READY) still need a dataset in the earlier state. Reconstructing it is exact,
not a guess, because `apply_repairs` is invertible from what it recorded:

* which lines it touched          -> `evidence_repairs` on the label row
* what each line looked like before -> `original_image` /
  `original_transcription_status`, kept beside the new `repair` block
* what text it inserted           -> the owner's own repair record
  (`manual_evidence_repairs.jsonl`), so nothing is invented or re-derived
* the note it appended            -> a fixed suffix on `transcription_source`

and above all, the manifest revision records the sha256 of BOTH files as they
were immediately before the repair. `build_pre_repair_dataset` re-hashes its own
reconstruction against those recorded values and refuses if they disagree, so a
test can never silently run against a fictional "before" state.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from autograder.benchmark.evidence_repairs import REPAIR_SOURCE, REPAIRS_CROPS_DIRNAME, REPAIRS_FILENAME

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
DATASET_FILES = ("cases_inputs.jsonl", "cases_labels.jsonl", "manifest.json", "CHECKSUMS.sha256")
#: the suffix apply_repairs appends to `transcription_source`
SOURCE_SUFFIX = f" + {REPAIR_SOURCE} for "
#: fields apply_repairs adds to a repaired evidence line
REPAIR_LINE_FIELDS = ("repair", "original_image", "original_transcription_status")
#: line dimensions apply_repairs states explicitly on every row (the effective-evidence layer)
RESOLUTION_FIELDS = ("lines_transcribed", "lines_no_text_artifact", "lines_resolved")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(Path(p).read_bytes())


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def body(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


def manifest(dataset: Path = DATASET) -> dict:
    return json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))


def repair_revision(dataset: Path = DATASET) -> dict:
    """The manifest revision that recorded the manual repair."""
    revs = [r for r in manifest(dataset).get("revisions") or [] if r.get("kind") == REPAIR_SOURCE]
    if not revs:
        raise AssertionError(f"{dataset} has no {REPAIR_SOURCE} revision — is it really the repaired dataset?")
    return revs[-1]


def repair_store(dataset: Path = DATASET) -> dict[str, dict]:
    """The owner's nine manual decisions, keyed by line id (read-only)."""
    p = Path(dataset) / REPAIRS_FILENAME
    if not p.exists():
        return {}
    return {r["line_id"]: r for r in rows(p)}


def repaired_cases(dataset: Path = DATASET) -> list[str]:
    """The cases carrying a manual repair — derived from the dataset, not listed."""
    return sorted({r["case_id"] for r in rows(Path(dataset) / "cases_labels.jsonl") if r.get("evidence_repairs")})


def repaired_line_ids(dataset: Path = DATASET) -> list[str]:
    return sorted(sid for r in rows(Path(dataset) / "cases_labels.jsonl")
                  for sid in (r.get("evidence_repairs") or []))


# ------------------------------------------------- the inverse of apply_repairs --

def _unapply_label(row: dict) -> dict:
    """Undo BOTH layers apply_repairs writes: the effective-evidence view (the
    explicit line dimensions, and `evidence_images` narrowed to real answer
    evidence) and, on a repaired row, the repair itself."""
    reps = list(row.get("evidence_repairs") or [])
    out: dict = {}
    for k, v in row.items():
        if k == "evidence_repairs" or k in RESOLUTION_FIELDS:
            continue                                   # neither field existed before the repair
        if k == "evidence_lines" and reps:
            v = [({kk: vv for kk, vv in e.items() if kk not in REPAIR_LINE_FIELDS}
                  | {"image": e["original_image"], "transcription_status": e["original_transcription_status"]})
                 if e.get("sample_id") in reps else dict(e)
                 for e in v]
        elif k == "transcription_complete" and reps:
            v = False
        elif k == "lines_without_audited_transcription" and reps:
            v = sorted(reps)
        elif k == "transcription_source" and reps:
            v = v.split(SOURCE_SUFFIX)[0]
        out[k] = v
    # before the effective-evidence layer, evidence_images was simply every
    # recorded line's crop, in line order — artifact slivers included
    out["evidence_images"] = [e["image"] for e in out["evidence_lines"]]
    return out


def _unapply_input(row: dict, label: dict, store: dict[str, dict]) -> dict:
    """Remove the line(s) the owner typed — using her own recorded text, never
    a re-derivation of it. A `no_text_segmentation_artifact` decision added no
    text, so those rows come back unchanged."""
    typed = {store[sid]["transcription"] for sid in (label.get("evidence_repairs") or [])
             if store.get(sid, {}).get("disposition") == "transcribed"}
    if not typed:
        return dict(row)
    kept = [t for t in (row.get("transcription") or "").split("\n") if t not in typed]
    return {**row, "transcription": "\n".join(kept)}


def pre_repair_rows(dataset: Path = DATASET) -> tuple[list[dict], list[dict]]:
    """(inputs, labels) as they were immediately before the manual repair."""
    d = Path(dataset)
    store = repair_store(d)
    labels = rows(d / "cases_labels.jsonl")
    by_case = {r["case_id"]: r for r in labels}
    inputs = [_unapply_input(r, by_case[r["case_id"]], store) for r in rows(d / "cases_inputs.jsonl")]
    return inputs, [_unapply_label(r) for r in labels]


def build_pre_repair_dataset(dest: Path, dataset: Path = DATASET) -> Path:
    """Write the reconstructed pre-repair dataset to `dest` and PROVE it is the
    real earlier state by matching the sha256 pair the manifest recorded."""
    d, dest = Path(dataset), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rev, man = repair_revision(d), manifest(d)
    inputs, labels = pre_repair_rows(d)
    ib, lb = body(inputs), body(labels)
    if sha_bytes(ib) != rev["previous_inputs_sha256"] or sha_bytes(lb) != rev["previous_labels_sha256"]:
        raise AssertionError(
            "the pre-repair reconstruction does not match the hashes the manifest recorded "
            f"(inputs {sha_bytes(ib)} vs {rev['previous_inputs_sha256']}, "
            f"labels {sha_bytes(lb)} vs {rev['previous_labels_sha256']}) — refusing to test against a "
            "state that never existed")
    (dest / "cases_inputs.jsonl").write_bytes(ib)
    (dest / "cases_labels.jsonl").write_bytes(lb)
    old = {k: v for k, v in man.items() if k != "revisions"}
    old["inputs_sha256"], old["labels_sha256"] = sha_bytes(ib), sha_bytes(lb)
    revs = list(man.get("revisions") or [])
    cut = next(i for i, r in enumerate(revs) if r.get("kind") == REPAIR_SOURCE)
    old["revisions"] = revs[:cut]                  # everything from the repair onward had not happened yet
    extra = dict(old.get("extra") or {})
    inv = dict(extra.get("evidence_inventory") or {})
    if inv:
        inv["transcription_incomplete_cases"] = repaired_cases(d)
        extra["evidence_inventory"] = inv
        old["extra"] = extra
    (dest / "manifest.json").write_text(json.dumps(old, ensure_ascii=False, indent=1),
                                        encoding="utf-8", newline="\n")
    (dest / "CHECKSUMS.sha256").write_text(
        f"{old['inputs_sha256']}  cases_inputs.jsonl\n{old['labels_sha256']}  cases_labels.jsonl\n",
        encoding="utf-8", newline="\n")
    return dest


def copy_live_dataset(dest: Path, dataset: Path = DATASET, *, with_repairs: bool = False) -> Path:
    """A writable copy of the CURRENT (repaired) dataset — for tests that must
    exercise apply/verify against the real post-repair state without touching it."""
    d, dest = Path(dataset), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in DATASET_FILES:
        shutil.copy2(d / name, dest / name)
    if with_repairs:
        copy_repair_store(dest, d)
    return dest


def copy_repair_store(dest: Path, dataset: Path = DATASET) -> Path:
    """Copy the owner's real repair records and crops into `dest` (read-only source)."""
    d, dest = Path(dataset), Path(dest)
    shutil.copy2(d / REPAIRS_FILENAME, dest / REPAIRS_FILENAME)
    src_crops = d / REPAIRS_CROPS_DIRNAME
    if src_crops.exists():
        shutil.copytree(src_crops, dest / REPAIRS_CROPS_DIRNAME, dirs_exist_ok=True)
    return dest
