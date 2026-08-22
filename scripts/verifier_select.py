"""Verifier benchmark — SELECTED candidate layer on top of the raw pool.

Dataset construction only (no model calls). Reads:

- the frozen manual audit (scripts/refaudit.py; must be frozen)
- the RAW harvested pool emitted by ``refaudit.py verifier-prep --emit``
  (verifier_bench/cases_inputs.jsonl + cases_labels.jsonl + manifest.json,
  kept byte-for-byte unchanged here)

and builds a compact, balanced, leakage-safe benchmark:

POSITIVES   exactly one per audited handwriting item: crop + the audited
            reference as the candidate transcription, expected SUPPORTED.
            The model-visible row is indistinguishable from a negative
            (opaque id, crop, candidate).
NEGATIVES   real historical OCR errors from the raw pool, deduplicated per
            image on (canonical-normalized text, digit/sign signature), then
            at most 2 per image chosen for coverage: the most SUBTLE
            low-distance error first (the dangerous false-accept class),
            then the candidate adding the most new error kinds (severe /
            number-sign preferred). One documented exception: a 3rd
            candidate when it is the image's only number/sign/formula error.
SPLITS      writer-level DEV / CALIBRATION / HELD_OUT; every case of one
            image lands in one split (zero image overlap by construction).

``propose`` writes verifier_bench/selection_proposal.json and prints the
composition report; ``freeze`` (only after the owner approves the split)
writes verifier_bench/selected/{cases_inputs.jsonl,cases_labels.jsonl,
manifest.json}. The raw pool is never modified.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("refaudit", _HERE / "refaudit.py")
refaudit = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("refaudit", refaudit)
_spec.loader.exec_module(refaudit)

RAW_DIRNAME = refaudit.VERIFIER_DIRNAME           # verifier_bench/ (flat raw pool)
SELECTED_DIRNAME = "selected"
PROPOSAL_FILENAME = "selection_proposal.json"

MAX_NEGATIVES_PER_IMAGE = 2
SUBTLE_CER = 0.20    # descriptive coverage buckets, NOT model thresholds
SEVERE_CER = 0.50
ERROR_KINDS = ("omission", "substitution", "unsupported_addition",
               "number_sign_formula", "other_divergence")

# Writer-level split proposals (owner decides; nothing is frozen silently).
SPLIT_PROPOSALS = {
    "A": {"DEV": ["e002", "e003", "e007"], "CALIBRATION": ["e004"],
          "HELD_OUT": ["e005", "e006"]},
    "B": {"DEV": ["e002", "e003", "e006"], "CALIBRATION": ["e004"],
          "HELD_OUT": ["e005", "e007"]},
}


class SelectionError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------- loading ----

def load_raw_pool(store: "refaudit.AuditStore") -> dict:
    raw_dir = store.bench_dir / RAW_DIRNAME
    manifest_path = raw_dir / "manifest.json"
    inputs_path = raw_dir / "cases_inputs.jsonl"
    labels_path = raw_dir / "cases_labels.jsonl"
    for p in (manifest_path, inputs_path, labels_path):
        if not p.exists():
            raise SelectionError(f"raw verifier pool missing: {p} (run refaudit.py "
                                 "verifier-prep --emit after freezing the audit)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs_bytes, labels_bytes = inputs_path.read_bytes(), labels_path.read_bytes()
    if (manifest.get("inputs_sha256") != _sha(inputs_bytes)
            or manifest.get("labels_sha256") != _sha(labels_bytes)):
        raise SelectionError("raw pool files do not match their manifest hashes — "
                             "the raw pool must stay byte-identical to its emission")
    inputs = {r["case_id"]: r for r in _read_jsonl(inputs_path)}
    labels = _read_jsonl(labels_path)
    return {"dir": raw_dir, "manifest": manifest, "inputs": inputs, "labels": labels,
            "inputs_sha256": _sha(inputs_bytes), "labels_sha256": _sha(labels_bytes)}


# -------------------------------------------------------------- selection ----

def _severity(cer: float) -> str:
    if cer <= SUBTLE_CER:
        return "subtle"
    if cer >= SEVERE_CER:
        return "severe"
    return "moderate"


def build_selection(store: "refaudit.AuditStore", raw: dict,
                    split_name: str = "A") -> dict:
    """Deterministic construction of the selected benchmark (not persisted)."""
    if not refaudit.is_frozen(store):
        raise SelectionError("the manual reference audit is not frozen/current; "
                             "freeze it before building the selected benchmark")
    normalize, lev, _wa = refaudit._load_metric_fns()
    sig = refaudit.digit_op_signature

    def cer(ref: str, hyp: str) -> float:
        r, h = normalize(ref), normalize(hyp)
        if not r:
            return 0.0 if not h else 1.0
        return lev(r, h) / len(r)

    split_def = SPLIT_PROPOSALS[split_name]
    writer_split = {w: s for s, ws in split_def.items() for w in ws}

    cases_inputs: list[dict] = []
    cases_labels: list[dict] = []
    per_item_counts: dict[str, int] = {}
    exceptions: list[dict] = []
    unique_negatives_total = 0
    real_correct_candidates = 0

    # index raw negatives per item
    raw_by_item: dict[str, list[dict]] = collections.defaultdict(list)
    for lab in raw["labels"]:
        if lab["expected_verdict"] == "supported":
            real_correct_candidates += 1
            continue
        raw_by_item[lab["item_id"]].append(lab)

    for item_id in store.eligible_ids:
        entry = store.entry(item_id)
        if entry["status"] not in ("confirmed", "corrected"):
            continue                       # ambiguous items: no trustworthy label
        item = store.item(item_id)
        writer = item.get("writer") or "unknown"
        split = writer_split.get(writer)
        if split is None:
            raise SelectionError(f"writer {writer!r} of {item_id} is not assigned in "
                                 f"split proposal {split_name}")
        reference = entry["audited_reference"]
        crop = item["image"]

        # --- positive: the audited reference as the candidate ---------------
        pos_id = _sha(f"positive::{item_id}::{refaudit._sha256_json(entry)}".encode())[:12]
        cases_inputs.append({"case_id": pos_id, "crop": crop,
                             "candidate_transcription": reference})
        cases_labels.append({"case_id": pos_id, "item_id": item_id, "writer": writer,
                             "split": split, "expected_verdict": "supported",
                             "polarity": "positive", "error_kinds": [],
                             "severity": None, "cer_vs_audited": 0.0,
                             "source_configs": ["audited_reference"],
                             "raw_case_ids": []})
        per_item_counts[item_id] = 1

        # --- negatives: dedup per image, then coverage-driven picks ----------
        groups: dict[tuple, dict] = {}
        for lab in raw_by_item.get(item_id, []):
            cand = raw["inputs"][lab["case_id"]]["candidate_transcription"]
            key = (normalize(cand), sig(cand))
            if key == (normalize(reference), sig(reference)):
                continue                   # equivalent to the positive
            g = groups.setdefault(key, {"candidate": cand, "kinds": set(),
                                        "configs": [], "raw_ids": []})
            g["kinds"].update(lab.get("error_kinds") or [])
            g["configs"].append(lab["source_config"])
            g["raw_ids"].append(lab["case_id"])
        negatives = []
        for g in groups.values():
            c = cer(reference, g["candidate"])
            negatives.append({**g, "cer": c, "severity": _severity(c)})
        unique_negatives_total += len(negatives)
        # deterministic ordering before any pick
        negatives.sort(key=lambda n: (n["cer"], n["candidate"]))

        picks: list[dict] = []
        if negatives:
            subtle = [n for n in negatives if n["severity"] == "subtle" and n["cer"] > 0]
            first = subtle[0] if subtle else negatives[0]     # closest miss first
            picks.append(first)
            rest = [n for n in negatives if n is not first]
            if rest:
                def score(n):
                    new_kinds = len(n["kinds"] - first["kinds"])
                    return (new_kinds,
                            1 if "number_sign_formula" in n["kinds"] else 0,
                            1 if n["severity"] != first["severity"] else 0,
                            n["cer"], n["candidate"])
                rest.sort(key=score, reverse=True)
                picks.append(rest[0])
            covered = set().union(*(p["kinds"] for p in picks))
            if "number_sign_formula" not in covered:
                extra = [n for n in negatives if n not in picks
                         and "number_sign_formula" in n["kinds"]]
                if extra:
                    picks.append(extra[0])
                    exceptions.append({"item_id": item_id,
                                       "reason": "number_sign_formula coverage: the "
                                                 "image's only number/sign error was "
                                                 "not among the two coverage picks"})
        for n in picks:
            neg_id = _sha(f"negative::{item_id}::{n['candidate']}".encode())[:12]
            cases_inputs.append({"case_id": neg_id, "crop": crop,
                                 "candidate_transcription": n["candidate"]})
            cases_labels.append({"case_id": neg_id, "item_id": item_id, "writer": writer,
                                 "split": split, "expected_verdict": "review",
                                 "polarity": "negative",
                                 "error_kinds": sorted(n["kinds"]),
                                 "severity": n["severity"],
                                 "cer_vs_audited": round(n["cer"], 4),
                                 "source_configs": sorted(set(n["configs"])),
                                 "raw_case_ids": sorted(n["raw_ids"])})
            per_item_counts[item_id] += 1

    # Opaque ordering: sort by case id so positives/negatives interleave.
    cases_inputs.sort(key=lambda r: r["case_id"])
    label_by_id = {l["case_id"]: l for l in cases_labels}
    cases_labels = [label_by_id[r["case_id"]] for r in cases_inputs]

    report = _report(store, cases_labels, per_item_counts, exceptions,
                     unique_negatives_total, real_correct_candidates, split_name, raw)
    return {"inputs": cases_inputs, "labels": cases_labels, "report": report}


def _report(store, labels, per_item_counts, exceptions, unique_negs, real_correct,
            split_name, raw) -> dict:
    items = {l["item_id"] for l in labels}
    writers = collections.Counter(store.item(i).get("writer") for i in store.eligible_ids)
    pos = [l for l in labels if l["polarity"] == "positive"]
    neg = [l for l in labels if l["polarity"] == "negative"]
    kind_counts = collections.Counter(k for l in neg for k in l["error_kinds"])
    sev_counts = collections.Counter(l["severity"] for l in neg)
    per_image = collections.Counter(per_item_counts.values())
    by_split: dict[str, dict] = {}
    for s in ("DEV", "CALIBRATION", "HELD_OUT"):
        rows = [l for l in labels if l["split"] == s]
        by_split[s] = {
            "cases": len(rows),
            "positives": sum(1 for l in rows if l["polarity"] == "positive"),
            "negatives": sum(1 for l in rows if l["polarity"] == "negative"),
            "unique_images": len({l["item_id"] for l in rows}),
            "writers": sorted({l["writer"] for l in rows}),
            "number_sign_cases": sum(1 for l in rows
                                     if "number_sign_formula" in l["error_kinds"]),
        }
    image_splits = collections.defaultdict(set)
    for l in labels:
        image_splits[l["item_id"]].add(l["split"])
    overlap = [i for i, s in image_splits.items() if len(s) > 1]
    return {
        "split_proposal": split_name,
        "split_definition": SPLIT_PROPOSALS[split_name],
        "unique_handwriting_items": len(items),
        "writers_represented": dict(sorted(writers.items())),
        "positive_cases": len(pos),
        "real_correct_raw_candidates_not_added": real_correct,
        "unique_negatives_after_dedup": unique_negs,
        "selected_negatives": len(neg),
        "total_selected_cases": len(labels),
        "cases_per_image_distribution": dict(sorted(per_image.items())),
        "max_negatives_per_image_rule": MAX_NEGATIVES_PER_IMAGE,
        "documented_exceptions": exceptions,
        "error_kind_counts_overlapping": dict(kind_counts),
        "number_sign_formula_coverage": {
            "negative_cases": kind_counts.get("number_sign_formula", 0),
            "images_with_number_sign_case": len({l["item_id"] for l in neg
                                                 if "number_sign_formula" in l["error_kinds"]}),
        },
        "severity_counts": dict(sev_counts),
        "by_split": by_split,
        "images_in_multiple_splits": overlap,
        "zero_image_overlap_between_splits": not overlap,
        "raw_pool_provenance": {"inputs_sha256": raw["inputs_sha256"],
                                "labels_sha256": raw["labels_sha256"],
                                "cases": len(raw["labels"])},
    }


# ------------------------------------------------------------- persistence ----

def write_proposal(store, selection: dict, raw: dict | None = None) -> Path:
    """Persist the proposal (primary split) plus every alternative split's
    per-split composition, so the owner can decide from one file."""
    out = store.bench_dir / RAW_DIRNAME / PROPOSAL_FILENAME
    alternatives = {}
    if raw is not None:
        for name in SPLIT_PROPOSALS:
            if name == selection["report"]["split_proposal"]:
                continue
            alt = build_selection(store, raw, name)["report"]
            alternatives[name] = {"split_definition": alt["split_definition"],
                                  "by_split": alt["by_split"],
                                  "zero_image_overlap_between_splits":
                                      alt["zero_image_overlap_between_splits"]}
    doc = {"_policy": ("PROPOSAL ONLY — not a frozen benchmark. Review the "
                       "composition and the split, then run freeze."),
           "generated_at": refaudit._now(), **selection["report"],
           "alternative_splits": alternatives}
    refaudit._atomic_write_json(out, doc)
    return out


def freeze_selected(store, selection: dict, split_name: str,
                    rationale: str = "", expect_counts: tuple | None = None) -> dict:
    """Write verifier_bench/selected/ (inputs, labels, manifest, checksums).
    Only after the owner approved the composition; the raw pool is untouched.
    ``expect_counts=(positives, negatives, total)`` refuses a freeze whose
    composition differs from what was proposed and approved."""
    report = selection["report"]
    if not report["zero_image_overlap_between_splits"]:
        raise SelectionError("refusing to freeze: image overlap between splits")
    if expect_counts is not None:
        got = (report["positive_cases"], report["selected_negatives"],
               report["total_selected_cases"])
        if tuple(expect_counts) != got:
            raise SelectionError(f"composition {got} differs from the approved "
                                 f"{tuple(expect_counts)} — not freezing")
    out_dir = store.bench_dir / RAW_DIRNAME / SELECTED_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = out_dir / "cases_inputs.jsonl"
    labels_path = out_dir / "cases_labels.jsonl"
    tmps = []
    try:
        for path, rows in ((inputs_path, selection["inputs"]), (labels_path, selection["labels"])):
            tmp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmps.append((tmp, path))
        for tmp, path in tmps:
            os.replace(tmp, path)
    except BaseException:
        for tmp, _ in tmps:
            tmp.unlink(missing_ok=True)
        raise
    image_ids_per_split: dict[str, list[str]] = {}
    for lab in selection["labels"]:
        image_ids_per_split.setdefault(lab["split"], set()).add(lab["item_id"])
    image_ids_per_split = {s: sorted(v) for s, v in sorted(image_ids_per_split.items())}
    manifest = {
        "_policy": ("Frozen SELECTED verifier benchmark (REAL historical OCR errors "
                    "+ one positive per audited image). cases_inputs.jsonl is the "
                    "ONLY model-visible file (opaque id, crop, candidate); "
                    "cases_labels.jsonl is evaluation-side only. Primary model-"
                    "selection metric: FALSE ACCEPT RATE (incorrect transcription "
                    "classified SUPPORTED); never overall accuracy."),
        "frozen_at": refaudit._now(),
        "decision": {"split": split_name,
                     "writer_assignment": SPLIT_PROPOSALS[split_name],
                     "rationale": rationale},
        "image_ids_per_split": image_ids_per_split,
        "zero_image_overlap_between_splits": True,   # asserted above
        "report": report,                            # incl. error-kind metadata
        "raw_pool": report["raw_pool_provenance"],
        "inputs_sha256": _sha(inputs_path.read_bytes()),
        "labels_sha256": _sha(labels_path.read_bytes()),
        "audit_sha256": refaudit._sha256_json(store.entries_canonical()),
    }
    manifest_path = out_dir / "manifest.json"
    refaudit._atomic_write_json(manifest_path, manifest)
    # Persist the manifest's own hash alongside (a manifest cannot contain it).
    checksums = "\n".join(
        f"{_sha((out_dir / name).read_bytes())}  {name}"
        for name in ("cases_inputs.jsonl", "cases_labels.jsonl", "manifest.json")) + "\n"
    tmp = out_dir / f"CHECKSUMS.{os.getpid()}.tmp"
    tmp.write_text(checksums, encoding="utf-8")
    os.replace(tmp, out_dir / "CHECKSUMS.sha256")
    return manifest


# -------------------------------------------------------------------- CLI ----

def _print_report(r: dict) -> None:
    print(f"split proposal {r['split_proposal']}: {r['split_definition']}")
    print(f"unique handwriting items: {r['unique_handwriting_items']}")
    print(f"writers represented: {r['writers_represented']}")
    print(f"positives: {r['positive_cases']}   (real correct raw candidates not added: "
          f"{r['real_correct_raw_candidates_not_added']})")
    print(f"unique negatives after dedup: {r['unique_negatives_after_dedup']}")
    print(f"selected negatives: {r['selected_negatives']}   total selected: "
          f"{r['total_selected_cases']}")
    print(f"cases per image: {r['cases_per_image_distribution']}   "
          f"exceptions: {len(r['documented_exceptions'])}")
    print(f"error kinds (overlapping): {r['error_kind_counts_overlapping']}")
    print(f"number/sign coverage: {r['number_sign_formula_coverage']}")
    print(f"severity: {r['severity_counts']}")
    for s, v in r["by_split"].items():
        print(f"  {s:12s} cases={v['cases']:4d} pos={v['positives']:3d} neg={v['negatives']:3d} "
              f"images={v['unique_images']:3d} writers={v['writers']} "
              f"num/sign={v['number_sign_cases']}")
    print(f"zero image overlap between splits: {r['zero_image_overlap_between_splits']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="verifier benchmark selection layer (no model calls)")
    ap.add_argument("--bench-dir", default=None)
    ap.add_argument("--split", default="A", choices=sorted(SPLIT_PROPOSALS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("propose", help="build + report the composition; writes selection_proposal.json")
    fr = sub.add_parser("freeze", help="write verifier_bench/selected/ (after owner approval)")
    fr.add_argument("--rationale", default="", help="owner's split rationale (persisted)")
    fr.add_argument("--expect-counts", default=None,
                    help="positives,negatives,total approved by the owner; refuses on mismatch")
    args = ap.parse_args(argv)
    store = refaudit.AuditStore(Path(args.bench_dir) if args.bench_dir else None)
    try:
        raw = load_raw_pool(store)
        selection = build_selection(store, raw, args.split)
    except SelectionError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if args.cmd == "propose":
        out = write_proposal(store, selection, raw)
        _print_report(selection["report"])
        print(f"proposal written: {out}")
        return 0
    expect = (tuple(int(x) for x in args.expect_counts.split(","))
              if args.expect_counts else None)
    try:
        manifest = freeze_selected(store, selection, args.split,
                                   rationale=args.rationale, expect_counts=expect)
    except SelectionError as exc:
        print(f"REFUSED: {exc}")
        return 2
    _print_report(selection["report"])
    print(f"frozen: {store.bench_dir / RAW_DIRNAME / SELECTED_DIRNAME}  inputs_sha256={manifest['inputs_sha256'][:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
