"""Manual selection-correctness audit for the ambiguous GRADE_PRIMARY cases.

Benchmark/developer tooling — NOT production grading, and NOT a model tool:
nothing in this module (or the UI on top of it) ever invokes a model, an OCR
backend, or the network. The human auditor is the only source of judgments.

WHY THIS EXISTS
---------------
The instructor's final sub-item score inverts to an explanation verdict
uniquely for 4 (``valid``) and 2 (``partially_valid``), but a 0 is produced by
six distinct (selection, verdict) states. Resolving a 0 needs exactly ONE bit
that no artifact currently holds authoritatively: was the student's SELECTION
correct?

    selection wrong   -> the explanation was never the reason for the 0;
                         the case carries no explanation ground truth (EXCLUDED)
    selection correct -> the factor was 0 and, given an audited non-empty
                         transcription, the verdict is `invalid` uniquely

SCOPE: only the zero-score cases outside HELD_OUT. HELD_OUT is never listed,
never rendered, and never audited here.

EVIDENCE POLICY (why the model-derived extractions are not used)
----------------------------------------------------------------
``eval_out/exams/*/extraction.json`` carries a ``final_answer`` per sub-item,
but those files were produced by ``qwen3-vl:8b-instruct`` (see the sibling
``result.json`` -> ``model``). A model's reading of the answer table cannot be
the ground truth a model is later benchmarked against. Those files are used
ONLY to locate the page and row for the human — never for the letter itself,
which is deliberately not shown in the UI so it cannot anchor the auditor.

The exam VERSION (A1/A2/A3) does come from an authoritative source: the
operator-confirmed marker->variant mapping in the audited variant_resolve
dataset. It is used to show the expected correct option as a convenience.

Guarantees
----------
- The frozen dataset (``cases_inputs.jsonl``, ``cases_labels.jsonl``,
  ``final_labels.json``, ``manifest.json``) is READ-ONLY here. Audit state
  lives in a separate ``selection_audit.json`` beside it.
- Every write is atomic (tmp file + os.replace) and immediate, so a closed or
  rerun UI resumes exactly where the auditor stopped.
- ``unresolved`` is a first-class outcome. An unresolved case is NEVER treated
  as ``selection_correct=False``.

CLI
---
    python scripts/selaudit.py cases        # the ambiguous case ids + status
    python scripts/selaudit.py summary      # audit progress
    python scripts/selaudit.py derive       # full 67-row verdict derivation
    python scripts/selaudit.py freeze       # refuses while any case is unaudited
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "evaluation" / "model_selection" / "datasets" / "grade_primary"
AUDIT_FILENAME = "selection_audit.json"

#: exam scans, by writer id. Source: datasets/*_manifest.json (anon_id ->
#: original_path). Deterministic mapping, no model involved.
EXAM_FILE_BY_WRITER = {
    "e002": "test/002_76.pdf",
    "e003": "test/003_70.pdf",
    "e004": "test/004_58.pdf",
    "e005": "test/005_48.pdf",
    "e006": "test/006_86.pdf",
    "e007": "test/007_48.pdf",
}

#: exam VERSION per writer, from the operator-confirmed marker->variant
#: mapping in evaluation/model_selection/datasets/variant_resolve
#: (label_provenance: printed booklet content matched to the key by the
#: operator, confirmed by the exam owner). Absent = not audited.
AUDITED_VERSION_BY_WRITER = {
    "e002": "A3",
    "e003": "A2",
}

_LOCK = threading.Lock()

OUTCOMES = ("unaudited", "correct", "incorrect", "unresolved")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ------------------------------------------------------------------ dataset --


@dataclass
class Dataset:
    root: Path

    @property
    def manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    @property
    def final_labels(self) -> dict:
        p = self.root / "final_labels.json"
        return json.loads(p.read_text(encoding="utf-8"))["labels"] if p.exists() else {}

    @property
    def inputs(self) -> dict:
        return {r["case_id"]: r for r in _read_jsonl(self.root / "cases_inputs.jsonl")}

    def split_of(self, case_id: str) -> str:
        writer = case_id.split("_")[0]
        for split, writers in self.manifest["split_assignment"].items():
            if writer in writers:
                return split
        return "?"

    def max_points(self, case_id: str) -> float:
        pack = (self.inputs.get(case_id) or {}).get("pack") or {}
        return float(pack.get("max_score") or 4.0)

    def transcription(self, case_id: str) -> str:
        return (self.inputs.get(case_id) or {}).get("transcription") or ""

    def correct_options(self, case_id: str) -> list[str] | None:
        """Accepted option(s) for this sub-item under the writer's AUDITED
        version. None when the version was never audited — in that case the
        UI shows nothing rather than a guess."""
        writer, _, sub = case_id.split("_")
        version = AUDITED_VERSION_BY_WRITER.get(writer)
        if not version:
            return None
        pack = (self.inputs.get(case_id) or {}).get("pack") or {}
        entry = (pack.get("correct_by_version") or {}).get(sub[1:])
        if not isinstance(entry, dict):
            return None
        return entry.get(version)

    def ambiguous_case_ids(self) -> list[str]:
        """Zero-score cases outside HELD_OUT — exactly the rows whose verdict
        the instructor score cannot determine. Derived, never hardcoded."""
        out = []
        for case_id, label in self.final_labels.items():
            if label.get("score") != 0.0:
                continue
            if self.split_of(case_id) == "HELD_OUT":
                continue
            out.append(case_id)
        return sorted(out)


# -------------------------------------------------------------------- store --


class SelectionAuditStore:
    """Human decisions about selection correctness. Separate file; the frozen
    dataset is never modified."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path | None = None):
        self.ds = Dataset(Path(root) if root else DATASET_DIR)
        self.path = self.ds.root / AUDIT_FILENAME

    # -- state ---------------------------------------------------------------

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema_version": self.SCHEMA_VERSION, "decisions": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data.setdefault("decisions", {})
        return data

    @property
    def case_ids(self) -> list[str]:
        return self.ds.ambiguous_case_ids()

    def decision(self, case_id: str) -> dict:
        return self._load()["decisions"].get(case_id, {})

    def outcome(self, case_id: str) -> str:
        return self.decision(case_id).get("outcome", "unaudited")

    def version_confirmed(self, case_id: str) -> bool:
        """Whether this writer's exam VERSION comes from the authoritative
        operator-confirmed mapping. Without it there is no key to compare a
        marked letter against."""
        return case_id.split("_")[0] in AUDITED_VERSION_BY_WRITER

    def selection_correct(self, case_id: str) -> bool | None:
        """True / False / None. ``unresolved`` and ``unaudited`` are both None
        — an unresolved case is never silently False.

        CORRECTNESS ALSO REQUIRES A CONFIRMED EXAM VERSION. The audit records
        what the student MARKED, which is a fact a human can read off the
        page; whether that mark is CORRECT is a comparison against the key,
        and the key is selected by the exam version. Where the version was
        never audited (e004), the marked letter is preserved and correctness
        stays unresolved — deriving it from an unverified version would put a
        guess into ground truth.
        """
        o = self.outcome(case_id)
        if o not in ("correct", "incorrect"):
            return None
        if not self.version_confirmed(case_id):
            return None
        return o == "correct"

    def marked_option(self, case_id: str) -> str | None:
        """The letter the human read off the answer table, always preserved
        even when correctness cannot be settled."""
        return self.decision(case_id).get("selected_option")

    # -- writes --------------------------------------------------------------

    def record(self, case_id: str, *, outcome: str, selected_option: str | None = None,
               auditor: str = "owner", note: str = "", evidence: str = "") -> dict:
        if outcome not in OUTCOMES or outcome == "unaudited":
            raise ValueError(f"outcome must be one of {OUTCOMES[1:]}, got {outcome!r}")
        if case_id not in self.case_ids:
            raise ValueError(f"{case_id!r} is not an ambiguous case of this dataset")
        entry = {
            "case_id": case_id,
            "outcome": outcome,
            "selected_option": (selected_option or "").strip().upper() or None,
            "auditor": auditor,
            "note": note,
            "evidence": evidence or "human inspection of the exam answer table",
            "source": "human_visual_audit",
            "model_involved": False,
        }
        with _LOCK:
            data = self._load()                 # re-read: never write back a stale copy
            data["decisions"][case_id] = entry
            _atomic_write_json(self.path, data)
        return entry

    def clear(self, case_id: str) -> None:
        with _LOCK:
            data = self._load()
            data["decisions"].pop(case_id, None)
            _atomic_write_json(self.path, data)

    # -- progress ------------------------------------------------------------

    def remaining(self) -> list[str]:
        return [c for c in self.case_ids if self.outcome(c) == "unaudited"]

    def complete(self) -> bool:
        return not self.remaining()

    def summary(self) -> dict:
        counts = {o: 0 for o in OUTCOMES}
        for c in self.case_ids:
            counts[self.outcome(c)] += 1
        return {"ambiguous_cases": len(self.case_ids), "counts": counts,
                "complete": self.complete(), "audit_file": str(self.path)}

    # -- context for the UI --------------------------------------------------

    def context(self, case_id: str) -> dict:
        """Everything the human needs, and nothing that could anchor them.

        Deliberately ABSENT: the model's extracted letter, and the instructor
        score (the audit must not be reasoned backwards from the grade).
        """
        writer, question, sub = case_id.split("_")
        return {
            "case_id": case_id,
            "writer": writer,
            "question_id": question[1:],
            "sub_item_id": sub[1:],
            "split": self.ds.split_of(case_id),
            "exam_file": EXAM_FILE_BY_WRITER.get(writer),
            "audited_version": AUDITED_VERSION_BY_WRITER.get(writer),
            "correct_options": self.ds.correct_options(case_id),
            "answer_table_page": 11 if question == "q1" else 12,
            "answer_table_row": sub[1:],
            "max_points": self.ds.max_points(case_id),
            "outcome": self.outcome(case_id),
            "decision": self.decision(case_id),
        }


# --------------------------------------------------------------- derivation --


def derive_all(store: SelectionAuditStore | None = None) -> list[dict]:
    """Every case of the dataset with its verdict ground truth or an explicit
    refusal. HELD_OUT rows are derived too (the arithmetic is split-blind and
    reads only the instructor label) but no HELD_OUT content is inspected."""
    from autograder.benchmark.verdicts import derive_verdict

    store = store or SelectionAuditStore()
    ds = store.ds
    labels = ds.final_labels
    rows = []
    for case_id in sorted(labels):
        split = ds.split_of(case_id)
        sel = store.selection_correct(case_id) if case_id in store.case_ids else None
        d = derive_verdict(
            case_id=case_id,
            instructor_final_score=labels[case_id].get("score"),
            selection_correct=sel,
            max_points=ds.max_points(case_id),
            transcription=ds.transcription(case_id),
        )
        row = d.as_row()
        row["split"] = split
        audited = case_id in store.case_ids and store.outcome(case_id) != "unaudited"
        row["marked_option"] = store.marked_option(case_id) if audited else None
        row["exam_version_confirmed"] = store.version_confirmed(case_id) if audited else None
        if sel is not None:
            row["selection_correct_source"] = (
                "human_visual_audit + operator-confirmed exam version")
        elif audited:
            # the human read the mark, but no confirmed version means no key to
            # compare it against. This is NOT "nobody has looked yet": no
            # further selection audit can resolve it, only a version audit.
            from autograder.benchmark.verdicts import UNRESOLVED_VERSION_UNCONFIRMED

            row["selection_correct_source"] = None
            row["derivation_reason"] = UNRESOLVED_VERSION_UNCONFIRMED
            row["selection_unresolved_reason"] = (
                "exam version not audited for this writer; marked option recorded, "
                "correctness not derivable")
        elif d.derivable and row["instructor_final_score"]:
            row["selection_correct_source"] = "implied_by_full_or_partial_credit"
        else:
            row["selection_correct_source"] = None
        rows.append(row)
    return rows


def derivation_summary(rows: list[dict]) -> dict:
    from autograder.benchmark.verdicts import summarize, VerdictDerivation

    pairs = []
    for r in rows:
        pairs.append((r["split"], VerdictDerivation(
            case_id=r["case_id"], instructor_final_score=r["instructor_final_score"],
            selection_correct=r["selection_correct"],
            derived_explanation_verdict=r["derived_explanation_verdict"],
            derivable=r["derivable"], derivation_reason=r["derivation_reason"],
            max_points=r["max_points"], implied_final_score=r["implied_final_score"])))
    return summarize(pairs).as_dict()


# --------------------------------------------------------------------- CLI ---


def _cmd_cases(store: SelectionAuditStore, args) -> None:
    print(f"ambiguous (zero-score, non-HELD_OUT) cases: {len(store.case_ids)}")
    for c in store.case_ids:
        ctx = store.context(c)
        ver = ctx["audited_version"] or "NOT AUDITED"
        opts = ",".join(ctx["correct_options"]) if ctx["correct_options"] else "unknown"
        print(f"  {c:14s} {ctx['split']:12s} {ctx['exam_file']:18s} "
              f"page {ctx['answer_table_page']} row {ctx['answer_table_row']:2s} "
              f"version {ver:11s} correct={opts:8s} -> {store.outcome(c)}")


def _cmd_summary(store: SelectionAuditStore, args) -> None:
    print(json.dumps(store.summary(), indent=1))


def _cmd_derive(store: SelectionAuditStore, args) -> None:
    rows = derive_all(store)
    summary = derivation_summary(rows)
    if args.json:
        print(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
        return
    print(json.dumps(summary, indent=1))
    print()
    print(f"{'case_id':14s} {'split':12s} {'score':>5s} {'sel':>6s} {'verdict':16s} reason")
    for r in rows:
        if args.split and r["split"] != args.split:
            continue
        print(f"{r['case_id']:14s} {r['split']:12s} {str(r['instructor_final_score']):>5s} "
              f"{str(r['selection_correct']):>6s} "
              f"{str(r['derived_explanation_verdict']):16s} {r['derivation_reason']}")


def _cmd_freeze(store: SelectionAuditStore, args) -> None:
    left = store.remaining()
    if left:
        raise SystemExit(f"REFUSED: {len(left)} case(s) still unaudited: {', '.join(left)}\n"
                         "Run the UI:  .venv\\Scripts\\python.exe -m streamlit run "
                         "scripts\\selection_audit_ui.py -- --browser.gatherUsageStats false")
    print("all ambiguous cases audited")
    print(json.dumps(derivation_summary(derive_all(store)), indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=None, help="grade_primary dataset directory")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cases", help="the ambiguous case ids + status")
    sub.add_parser("summary", help="audit progress")
    d = sub.add_parser("derive", help="full verdict derivation for every case")
    d.add_argument("--json", action="store_true")
    d.add_argument("--split", default=None)
    sub.add_parser("freeze", help="refuses while any ambiguous case is unaudited")
    args = p.parse_args(argv)
    store = SelectionAuditStore(args.root)
    {"cases": _cmd_cases, "summary": _cmd_summary,
     "derive": _cmd_derive, "freeze": _cmd_freeze}[args.cmd](store, args)


if __name__ == "__main__":
    main()
