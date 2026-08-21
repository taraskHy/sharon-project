"""Manual reference-audit toolchain for evaluation/hebrew_bench_v2.

Benchmark/developer tooling — NOT production grading, and NOT a model tool:
nothing in this module (or the UI on top of it) ever invokes a model,
backend, or network. The human auditor is the only source of judgments.

Guarantees:

- ``references.json`` / ``items.json`` / ``outputs/`` are READ-ONLY here.
  Audit state lives in a separate ``reference_audit.json`` beside them.
- Every write is atomic (tmp file + os.replace); each recorded decision is
  persisted immediately, so a closed/rerun UI resumes exactly where the
  auditor stopped.
- The frozen audit manifest can only be written when zero items remain
  unchecked.

Scoring rule (``reference_for_scoring``):

    confirmed / corrected -> audited_reference (strict CER eligible)
    ambiguous             -> flagged; never silently ordinary ground truth
    unchecked             -> mode="final" REFUSES (UncheckedReferenceError);
                             mode="preview" returns the original reference
                             marked not-audited (strict CER ineligible)

CLI:

    python scripts/refaudit.py summary
    python scripts/refaudit.py preview        # historical CER preview (read-only)
    python scripts/refaudit.py freeze         # refuses while unchecked items remain
    python scripts/refaudit.py verifier-prep  # dry counts; emits only after freeze
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_DIR = REPO_ROOT / "evaluation" / "hebrew_bench_v2"

AUDIT_FILENAME = "reference_audit.json"
MANIFEST_FILENAME = "reference_audit_manifest.json"
PREVIEW_FILENAME = "audit_metrics_preview.json"
VERIFIER_DIRNAME = "verifier_bench"

STATUSES = ("unchecked", "confirmed", "corrected", "ambiguous")
_DIGITS_OPS = re.compile(r"[0-9+\-*/=<>%^]")
# A hyphen directly after a Hebrew letter is a prefix connector ("ב-High",
# "מ-1"), not a minus sign — it must not count as a math operator.
_HEB_PREFIX_HYPHEN = re.compile(r"(?<=[א-ת])-")


def digit_op_signature(text: str) -> str:
    """The ordered digit/operator sequence of *text*, used to detect
    number/sign/formula corruption that CER normalization would hide (the
    canonical ``normalize`` deletes '-', so 'x = -3' equals 'x = 3' there)."""
    return "".join(_DIGITS_OPS.findall(_HEB_PREFIX_HYPHEN.sub("", text or "")))

_AUDIT_POLICY = (
    "Human reference audit for hebrew_bench_v2. This file NEVER overwrites "
    "references.json: original_reference is a frozen copy, audited_reference "
    "is the human auditor's verdict. Statuses: unchecked|confirmed|corrected|"
    "ambiguous. Written atomically after every decision."
)


class RefAuditError(RuntimeError):
    """Generic audit-tooling error."""


class UncheckedReferenceError(RefAuditError):
    """A final-model-selection consumer asked for an unaudited reference."""


class FreezeError(RefAuditError):
    """The audit manifest cannot be frozen yet."""


def bench_dir_from_env() -> Path:
    return Path(os.environ.get("REFAUDIT_BENCH_DIR", str(DEFAULT_BENCH_DIR)))


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique tmp per writer: two concurrent processes must never interleave
    # writes through one shared tmp file and then promote a torn result.
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)  # never leave a partial tmp behind
        raise


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj) -> str:
    return _sha256_bytes(json.dumps(obj, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8"))


# ---------------------------------------------------------------- metrics ----

def _load_metric_fns():
    """normalize/lev/word_align from scripts/hebrew_bench_eval.py — the SAME
    canonical metric definitions the frozen benchmark evaluator uses."""
    spec = importlib.util.spec_from_file_location(
        "hebrew_bench_eval", Path(__file__).resolve().parent / "hebrew_bench_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize, mod.lev, mod.word_align


# ------------------------------------------------------------- audit store ----

class AuditStore:
    """The separate, non-destructive audit state for one benchmark directory."""

    def __init__(self, bench_dir: Path | None = None):
        self.bench_dir = Path(bench_dir) if bench_dir else bench_dir_from_env()
        items_doc = json.loads((self.bench_dir / "items.json").read_text(encoding="utf-8"))
        refs_doc = json.loads((self.bench_dir / "references.json").read_text(encoding="utf-8"))
        self.items: list[dict] = list(items_doc["items"])
        self.references: dict[str, dict] = {
            k: v for k, v in refs_doc.items() if not k.startswith("_")}
        self.audit_path = self.bench_dir / AUDIT_FILENAME
        self.manifest_path = self.bench_dir / MANIFEST_FILENAME
        self._lock = threading.Lock()
        self._doc: dict = {}
        self._reload_from_disk()

    def _reload_from_disk(self) -> None:
        if self.audit_path.exists():
            self._doc = json.loads(self.audit_path.read_text(encoding="utf-8"))
        else:
            self._doc = {"_policy": _AUDIT_POLICY, "version": 1, "entries": {}}
        self._doc.setdefault("entries", {})

    # -- reads --------------------------------------------------------------
    @property
    def item_ids(self) -> list[str]:
        return [it["id"] for it in self.items]

    def item(self, item_id: str) -> dict:
        for it in self.items:
            if it["id"] == item_id:
                return it
        raise KeyError(item_id)

    def original_reference(self, item_id: str) -> str:
        ref = self.references.get(item_id)
        if ref is None:
            raise KeyError(f"no reference for item {item_id!r}")
        return ref["text"]

    def entry(self, item_id: str) -> dict:
        """The audit entry (a fresh default when the item is unchecked)."""
        stored = self._doc["entries"].get(item_id)
        if stored is not None:
            return dict(stored)
        return {"item_id": item_id,
                "original_reference": self.original_reference(item_id),
                "audited_reference": None,
                "status": "unchecked", "note": "", "audited_at": None}

    def status(self, item_id: str) -> str:
        return self.entry(item_id)["status"]

    # -- writes (each one persists atomically) ------------------------------
    def record(self, item_id: str, status: str, audited_text: str | None = None,
               note: str = "") -> dict:
        if status not in ("confirmed", "corrected", "ambiguous"):
            raise RefAuditError(f"record() takes confirmed|corrected|ambiguous, got {status!r}")
        original = self.original_reference(item_id)
        if status == "confirmed":
            audited = original                    # by contract, always the original
        elif status == "corrected":
            if audited_text is None or not audited_text.strip():
                raise RefAuditError(
                    "corrected requires a non-empty audited transcription "
                    "(every benchmark crop contains text; an empty correction "
                    "would silently poison strict CER)")
            audited = audited_text
        else:  # ambiguous — preserve entered text when there is any
            audited = (audited_text
                       if audited_text and audited_text != original else None)
        entry = {"item_id": item_id, "original_reference": original,
                 "audited_reference": audited, "status": status,
                 "note": note or "", "audited_at": _now()}
        with self._lock:
            # Merge into the LATEST on-disk state before writing: a stale
            # in-memory copy (another session/process decided items since we
            # loaded) must never clobber or resurrect other decisions.
            self._reload_from_disk()
            self._doc["entries"][item_id] = entry
            self._save_locked()
        return dict(entry)

    def reset_item(self, item_id: str) -> None:
        """Reset ONE item back to unchecked (the only reset the UI offers
        without an explicit typed confirmation)."""
        with self._lock:
            self._reload_from_disk()
            if self._doc["entries"].pop(item_id, None) is not None:
                self._save_locked()

    def reset_all(self, confirm: str = "") -> None:
        """Dangerous: wipes every decision. Requires confirm='RESET'."""
        if confirm != "RESET":
            raise RefAuditError("global reset requires confirm='RESET'")
        with self._lock:
            self._doc["entries"] = {}
            self._save_locked()

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        _atomic_write_json(self.audit_path, self._doc)

    # -- aggregates ----------------------------------------------------------
    def summary(self) -> dict:
        counts = {"confirmed": 0, "corrected": 0, "ambiguous": 0}
        for item_id in self.item_ids:
            status = self.status(item_id)
            if status in counts:
                counts[status] += 1
        total = len(self.item_ids)
        checked = sum(counts.values())
        return {"total": total, "checked": checked, **counts,
                "unchecked": total - checked, "remaining": total - checked}

    def entries_canonical(self) -> dict:
        """All 129 entries (defaults included) keyed by id — the hash basis."""
        return {item_id: self.entry(item_id) for item_id in self.item_ids}


# ---------------------------------------------------- scoring resolution ----

@dataclass
class ScoringReference:
    item_id: str
    status: str
    reference: str | None
    use_for_strict_cer: bool
    source: str  # audited | ambiguous_audit | original_unaudited


def reference_for_scoring(store: AuditStore, item_id: str,
                          mode: str = "final") -> ScoringReference:
    """Deterministic reference resolution for every future benchmark.

    ``mode="final"`` (final model selection): unchecked items REFUSE.
    ``mode="preview"``: unchecked items resolve to the original reference,
    explicitly marked not-audited and ineligible for strict CER claims.
    """
    if mode not in ("final", "preview"):
        raise RefAuditError(f"unknown mode {mode!r}")
    entry = store.entry(item_id)
    status = entry["status"]
    if status in ("confirmed", "corrected"):
        return ScoringReference(item_id, status, entry["audited_reference"],
                                use_for_strict_cer=True, source="audited")
    if status == "ambiguous":
        # Never silently ordinary ground truth: the caller must branch on
        # status/use_for_strict_cer (exclude from strict CER or report apart).
        return ScoringReference(item_id, status, entry["audited_reference"],
                                use_for_strict_cer=False, source="ambiguous_audit")
    if mode == "final":
        raise UncheckedReferenceError(
            f"item {item_id!r} is unchecked: final model selection must not "
            "treat it as audited ground truth (finish the manual audit or "
            "run in preview mode)")
    return ScoringReference(item_id, "unchecked", entry["original_reference"],
                            use_for_strict_cer=False, source="original_unaudited")


# ------------------------------------------------------------- freezing ----

def freeze_manifest(store: AuditStore) -> dict:
    """Write the frozen audit manifest. Refuses while unchecked items remain."""
    summary = store.summary()
    if summary["unchecked"] > 0:
        raise FreezeError(
            f"cannot freeze: {summary['unchecked']} of {summary['total']} items "
            "are still unchecked")
    entries = store.entries_canonical()
    manifest = {
        "_policy": ("Frozen manifest of the human reference audit. The hash "
                    "binds the audit content; original references remain in "
                    "references.json untouched."),
        "frozen_at": _now(),
        "summary": summary,
        "audit_sha256": _sha256_json(entries),
        "items_sha256": _sha256_bytes((store.bench_dir / "items.json").read_bytes()),
        "references_sha256": _sha256_bytes(
            (store.bench_dir / "references.json").read_bytes()),
        "audit_file": AUDIT_FILENAME,
    }
    _atomic_write_json(store.manifest_path, manifest)
    return manifest


def is_frozen(store: AuditStore) -> bool:
    """True when a manifest exists AND still matches the current audit state
    AND the benchmark files it was frozen against are byte-identical."""
    if not store.manifest_path.exists():
        return False
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    return (
        manifest.get("audit_sha256") == _sha256_json(store.entries_canonical())
        and manifest.get("items_sha256")
            == _sha256_bytes((store.bench_dir / "items.json").read_bytes())
        and manifest.get("references_sha256")
            == _sha256_bytes((store.bench_dir / "references.json").read_bytes())
    )


# ------------------------------------------------- historical CER preview ----

def _iter_output_configs(store: AuditStore) -> list[str]:
    outputs = store.bench_dir / "outputs"
    if not outputs.is_dir():
        return []
    return sorted(d.name for d in outputs.iterdir() if (d / "run1").is_dir())


def _load_output(store: AuditStore, config: str, item_id: str) -> dict | None:
    path = store.bench_dir / "outputs" / config / "run1" / f"{item_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def preview_metrics(store: AuditStore, configs: list[str] | None = None) -> dict:
    """Which historical results WOULD change under audited references.

    Read-only over persisted outputs; never touches m2_bench_results.csv or
    any file under outputs/. Ambiguous items are excluded from the paired
    aggregates and reported separately; unchecked items are not compared
    (the preview is meaningful mid-audit and complete after it)."""
    normalize, lev, _word_align = _load_metric_fns()

    def cer(gt: str, hyp: str) -> float:
        gt_n, hyp_n = normalize(gt), normalize(hyp)
        if not gt_n:
            return 0.0 if not hyp_n else 1.0
        return lev(gt_n, hyp_n) / len(gt_n)

    corrected_ids = [i for i in store.item_ids if store.status(i) == "corrected"]
    per_config = []
    for config in (configs or _iter_output_configs(store)):
        compared, ambiguous_ids, unchecked_with_output = [], [], 0
        association_excluded = hard_excluded = 0
        for item_id in store.item_ids:
            output = _load_output(store, config, item_id)
            if output is None or not isinstance(output.get("transcription"), str):
                continue
            item = store.item(item_id)
            # Mirror the canonical evaluator (scripts/m2_bench_eval.py):
            # association rows are pair-accuracy scored (never CER) and hard
            # items are excluded from CER aggregates — a preview that CER'd
            # them would match no number in m2_bench_results.csv.
            if item.get("category") == "option_row_association":
                association_excluded += 1
                continue
            if item.get("hard"):
                hard_excluded += 1
                continue
            status = store.status(item_id)
            if status == "ambiguous":
                ambiguous_ids.append(item_id)
                continue
            if status == "unchecked":
                unchecked_with_output += 1
                continue
            entry = store.entry(item_id)
            old_cer = cer(entry["original_reference"], output["transcription"])
            new_cer = cer(entry["audited_reference"], output["transcription"])
            compared.append({"item_id": item_id, "status": status,
                             "old_cer": round(old_cer, 4),
                             "audited_cer": round(new_cer, 4),
                             "reference_changed":
                                 entry["audited_reference"] != entry["original_reference"]})
        n = len(compared)
        per_config.append({
            "config": config,
            "items_with_output_compared": n,
            "old_cer_mean": round(sum(c["old_cer"] for c in compared) / n, 4) if n else None,
            "audited_cer_mean": round(sum(c["audited_cer"] for c in compared) / n, 4) if n else None,
            "affected_items": sum(1 for c in compared if c["old_cer"] != c["audited_cer"]),
            "reference_corrections_in_compared":
                sum(1 for c in compared if c["reference_changed"]),
            "ambiguous_excluded": len(ambiguous_ids),
            "ambiguous_item_ids": ambiguous_ids,
            "unchecked_not_compared": unchecked_with_output,
            "association_excluded_from_cer": association_excluded,
            "hard_excluded_from_cer": hard_excluded,
            "items": compared,
        })
    return {
        "_policy": ("PREVIEW ONLY: shows what historical OCR results would "
                    "look like under audited references. Historical result "
                    "files and m2_bench_results.csv are never modified."),
        "generated_at": _now(),
        "audit_summary": store.summary(),
        "reference_corrections_total": len(corrected_ids),
        "corrected_item_ids": corrected_ids,
        "historical_results_untouched": True,
        "configs": per_config,
    }


def write_preview(store: AuditStore, report: dict | None = None) -> Path:
    report = report or preview_metrics(store)
    out = store.bench_dir / PREVIEW_FILENAME
    _atomic_write_json(out, report)
    return out


# ------------------------------------------------ verifier benchmark prep ----

def _error_kinds(normalize, word_align, reference: str, candidate: str) -> list[str]:
    """Deterministic characterization of a real OCR error vs the audited
    reference (evaluation-side only; the verifier never sees any of this)."""
    kinds: list[str] = []
    subs, deletions, insertions = word_align(normalize(reference).split(),
                                             normalize(candidate).split())
    if deletions:
        kinds.append("omission")
    if insertions:
        kinds.append("unsupported_addition")
    if subs:
        kinds.append("substitution")
    if digit_op_signature(reference) != digit_op_signature(candidate):
        kinds.append("number_sign_formula")
    return kinds or ["other_divergence"]


def verifier_prep(store: AuditStore, emit: bool = False) -> dict:
    """Harvest verifier-benchmark cases from audited references + persisted
    historical OCR outputs. NO synthetic corruptions (post-freeze, separate
    step, so the case set cannot lock in before the audit finishes).

    The MODEL-VISIBLE case file carries ONLY {case_id, item_id, crop,
    candidate_transcription}; every label (expected verdict, error kinds,
    references) goes to a separate evaluation-only file. Emission refuses
    until the audit manifest is frozen and current."""
    normalize, lev, word_align = _load_metric_fns()
    inputs, labels = [], []
    counts = {"correct_candidate": 0, "error_candidate": 0}
    kind_counts: dict[str, int] = {}
    association_excluded = 0
    for config in _iter_output_configs(store):
        for item_id in store.item_ids:
            if store.status(item_id) not in ("confirmed", "corrected"):
                continue
            output = _load_output(store, config, item_id)
            if output is None or not isinstance(output.get("transcription"), str):
                continue
            item = store.item(item_id)
            if item.get("category") == "option_row_association":
                # Association references are stored as the glued RTL text
                # layer; a semantically correct pair reading normalizes
                # differently, so equality labels would be wrong. Pair-scored
                # separately, never a fidelity case.
                association_excluded += 1
                continue
            candidate = output["transcription"]
            entry = store.entry(item_id)
            reference = entry["audited_reference"]
            # Fidelity equality must be sign-aware: normalize() deletes '-',
            # so a dropped minus would otherwise be labeled 'supported'.
            ok = (normalize(candidate) == normalize(reference)
                  and digit_op_signature(candidate) == digit_op_signature(reference))
            # The model-visible case id must be OPAQUE: the producing config
            # name correlates with the expected verdict and would hand the
            # verifier a label shortcut. The mapping lives in labels only.
            case_id = _sha256_bytes(f"{item_id}::{config}".encode("utf-8"))[:12]
            inputs.append({"case_id": case_id,
                           "crop": item["image"],
                           "candidate_transcription": candidate})
            kinds = [] if ok else _error_kinds(normalize, word_align,
                                               reference, candidate)
            labels.append({"case_id": case_id,
                           "item_id": item_id,
                           "expected_verdict": "supported" if ok else "review",
                           "error_kinds": kinds,
                           "source_config": config,
                           "audited_reference": reference})
            counts["correct_candidate" if ok else "error_candidate"] += 1
            for kind in kinds:
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
    report = {"cases": len(inputs), **counts, "error_kind_counts": kind_counts,
              "association_excluded": association_excluded,
              "note_hard_items": ("hard items are kept: after the human audit "
                                  "their references are verified ground truth"),
              "audit_summary": store.summary(), "emitted": False}
    if not emit:
        return report
    if not is_frozen(store):
        raise FreezeError(
            "verifier case emission requires a frozen, up-to-date audit "
            "manifest — finish the manual audit and run freeze first "
            "(dry counts are available without emitting)")
    out_dir = store.bench_dir / VERIFIER_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = out_dir / "cases_inputs.jsonl"
    labels_path = out_dir / "cases_labels.jsonl"
    # Transactional pair: write BOTH tmp files first, then promote both
    # back-to-back, cleaning up on any failure — a new inputs file must never
    # sit beside a stale labels file.
    pending: list[tuple[Path, Path]] = []
    try:
        for path, rows in ((inputs_path, inputs), (labels_path, labels)):
            tmp = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            pending.append((tmp, path))
        for tmp, path in pending:
            os.replace(tmp, path)
    except BaseException:
        for tmp, _path in pending:
            tmp.unlink(missing_ok=True)
        raise
    manifest = {
        "_policy": ("Verifier benchmark cases. cases_inputs.jsonl is the ONLY "
                    "file whose content may reach the verifier model (crop + "
                    "candidate transcription). cases_labels.jsonl is "
                    "evaluation-side only: expected verdicts, error kinds and "
                    "audited references must never enter a verifier prompt."),
        "generated_at": _now(),
        "cases": len(inputs),
        "inputs_sha256": _sha256_bytes(inputs_path.read_bytes()),
        "labels_sha256": _sha256_bytes(labels_path.read_bytes()),
        "audit_sha256": _sha256_json(store.entries_canonical()),
    }
    _atomic_write_json(out_dir / "manifest.json", manifest)
    report.update({"emitted": True, "out_dir": str(out_dir)})
    return report


# ------------------------------------------------------------------- CLI ----

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="hebrew_bench_v2 manual reference-audit tooling (no model calls)")
    parser.add_argument("--bench-dir", default=None,
                        help="benchmark directory (default: evaluation/hebrew_bench_v2)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary", help="audit progress counts")
    sub.add_parser("preview", help="historical CER preview under audited references")
    sub.add_parser("freeze", help="write the frozen audit manifest (refuses if incomplete)")
    prep = sub.add_parser("verifier-prep", help="verifier benchmark case harvest")
    prep.add_argument("--emit", action="store_true",
                      help="write case files (requires a frozen audit manifest)")
    args = parser.parse_args(argv)

    store = AuditStore(Path(args.bench_dir) if args.bench_dir else None)
    if args.cmd == "summary":
        print(json.dumps(store.summary(), indent=1))
        return 0
    if args.cmd == "preview":
        report = preview_metrics(store)
        out = write_preview(store, report)
        for cfg in report["configs"]:
            print(f"{cfg['config']:32s} compared={cfg['items_with_output_compared']:4d} "
                  f"old={cfg['old_cer_mean']} audited={cfg['audited_cer_mean']} "
                  f"affected={cfg['affected_items']} ambiguous={cfg['ambiguous_excluded']}")
        print(f"written: {out}")
        return 0
    if args.cmd == "freeze":
        try:
            manifest = freeze_manifest(store)
        except FreezeError as exc:
            print(f"REFUSED: {exc}")
            return 2
        print(json.dumps({k: manifest[k] for k in ("frozen_at", "summary", "audit_sha256")},
                         indent=1))
        return 0
    if args.cmd == "verifier-prep":
        try:
            report = verifier_prep(store, emit=args.emit)
        except FreezeError as exc:
            print(f"REFUSED: {exc}")
            return 2
        print(json.dumps({k: v for k, v in report.items() if k != "audit_summary"},
                         ensure_ascii=False, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
