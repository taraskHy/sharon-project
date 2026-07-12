"""Dataset discovery, anonymization, and deterministic splitting.

The graded-exam corpus lives in a directory (``test/`` in this repository)
whose filenames encode the instructor's final grade: ``<index>_<grade>.pdf``
(e.g. ``02_78.pdf``). That grade is a LABEL — it must never reach the model.

This module:

- parses and validates the filename convention, reporting malformed names and
  duplicate indices;
- assigns each exam an anonymized identifier that carries no grade
  information (``exam-<index>``);
- produces a deterministic, seed-fixed train/validation split;
- writes version-controlled manifest files under ``datasets/``.

The manifests themselves contain the expected grades (they are the label
store) — they are consumed by evaluation code only AFTER grading completes,
and are never part of any model input.
"""

from __future__ import annotations

import datetime as _dt
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz

SPLIT_SEED = 42
TRAIN_FRACTION = 0.6  # small corpus: keep a meaningful validation set

# The known exam form in this corpus is 13 pages; deviations are flagged.
EXPECTED_PAGE_COUNT = 13

_FILENAME_RE = re.compile(
    r"^(?P<index>\d+)_(?P<grade>\d{1,3})\.(?P<ext>pdf|png|jpg|jpeg|tif|tiff)$",
    re.IGNORECASE,
)


@dataclass
class ExamRecord:
    anon_id: str
    original_path: str  # relative to the repository root
    expected_grade: int
    split: str  # train | validation | final_test
    has_detailed_labels: bool = False  # only exam-level grades exist today
    instructor_annotations: str = "assumed_present"  # assumed_present | detected | none
    masking_status: str = "not_masked"  # not_masked | masked | failed
    warnings: list[str] = field(default_factory=list)


@dataclass
class DiscoveryReport:
    records: list[ExamRecord]
    malformed: list[str]
    duplicate_indices: list[str]


def parse_exam_filename(name: str) -> tuple[str, int] | None:
    """Return (index, grade) for a valid ``<index>_<grade>.<ext>`` name."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    grade = int(m.group("grade"))
    if grade > 100:
        return None
    return m.group("index"), grade


def anon_id_for(index: str) -> str:
    """Anonymized identifier: keeps the index (audit trail), drops the grade."""
    return f"exam-{int(index):03d}"


def discover_exams(root: str | Path, repo_root: str | Path = ".") -> DiscoveryReport:
    root = Path(root)
    repo_root = Path(repo_root)
    records: list[ExamRecord] = []
    malformed: list[str] = []
    seen_indices: dict[str, str] = {}
    duplicates: list[str] = []

    for f in sorted(root.iterdir()):
        if not f.is_file():
            continue
        parsed = parse_exam_filename(f.name)
        if parsed is None:
            malformed.append(f.name)
            continue
        index, grade = parsed
        canonical_index = str(int(index))
        if canonical_index in seen_indices:
            duplicates.append(
                f"index {canonical_index}: {seen_indices[canonical_index]} and {f.name}"
            )
            continue
        seen_indices[canonical_index] = f.name

        warnings: list[str] = []
        if f.suffix.lower() == ".pdf":
            try:
                with fitz.open(f) as doc:
                    n_pages = len(doc)
                if n_pages != EXPECTED_PAGE_COUNT:
                    warnings.append(
                        f"page count {n_pages} differs from the expected exam form "
                        f"({EXPECTED_PAGE_COUNT} pages) — possibly a partial scan"
                    )
            except Exception as e:  # noqa: BLE001 - surface as data-quality warning
                warnings.append(f"could not open PDF: {e}")

        try:
            rel = f.relative_to(repo_root)
        except ValueError:
            rel = f  # dataset lives outside the repo: record the absolute path
        records.append(
            ExamRecord(
                anon_id=anon_id_for(index),
                original_path=str(rel).replace("\\", "/"),
                expected_grade=grade,
                split="",  # assigned later
                warnings=warnings,
            )
        )
    return DiscoveryReport(records=records, malformed=malformed, duplicate_indices=duplicates)


def assign_split(records: list[ExamRecord], seed: int = SPLIT_SEED) -> None:
    """Deterministic split on sorted anonymized IDs with a fixed seed.

    Each exam index appears exactly once (duplicates were rejected during
    discovery), so no exam or variant of it can land in both splits.
    """
    ordered = sorted(records, key=lambda r: r.anon_id)
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    n_train = round(len(shuffled) * TRAIN_FRACTION)
    train_ids = {r.anon_id for r in shuffled[:n_train]}
    for r in records:
        r.split = "train" if r.anon_id in train_ids else "validation"


def _manifest_payload(records: list[ExamRecord], split: str, seed: int) -> dict:
    return {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "seed": seed,
        "train_fraction": TRAIN_FRACTION,
        "count": len(records),
        "entries": [asdict(r) for r in records],
    }


def write_manifests(
    report: DiscoveryReport,
    datasets_dir: str | Path = "datasets",
    seed: int = SPLIT_SEED,
) -> dict[str, Path]:
    datasets_dir = Path(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[ExamRecord]] = {"train": [], "validation": []}
    for r in report.records:
        by_split.setdefault(r.split, []).append(r)

    paths: dict[str, Path] = {}
    for split in ("train", "validation"):
        path = datasets_dir / f"{split}_manifest.json"
        path.write_text(
            json.dumps(_manifest_payload(by_split[split], split, seed), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths[split] = path

    final_path = datasets_dir / "final_test_manifest.json"
    if not final_path.exists():
        final_path.write_text(
            json.dumps(
                {
                    "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                    "split": "final_test",
                    "note": (
                        "HELD-OUT final test set (48 exams, not yet present in the "
                        "repository). These exams must remain unseen during all "
                        "development: no training, validation, model selection, "
                        "prompt tuning, calibration, debugging, or preprocessing "
                        "decisions may use them. Populate this manifest only after "
                        "the full grading configuration has been frozen (see "
                        "docs/evaluation.md), and do not modify the system based "
                        "on results obtained from them."
                    ),
                    "frozen_configuration": None,
                    "entries": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    paths["final_test"] = final_path

    issues = datasets_dir / "discovery_issues.json"
    issues.write_text(
        json.dumps(
            {"malformed_filenames": report.malformed, "duplicate_indices": report.duplicate_indices},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["issues"] = issues
    return paths


def load_manifest(path: str | Path) -> list[ExamRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ExamRecord(**e) for e in data.get("entries", [])]
