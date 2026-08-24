"""Pre-registered SMOKE subsets — the first thing a new model ever sees.

A smoke subset is a tiny, FROZEN list of DEV case ids per role, chosen by
deterministic rules BEFORE any new model output exists, so the first paid
execution is ~10 calls instead of 100+ and nobody can tune the subset to a
model afterwards. The frozen file records every case, the slot (what it
represents) and the rule that picked it, plus a selection hash; loading
re-verifies the hash, that every id exists, and that every id is DEV.

    evaluation/model_selection/smoke/<role>_smoke.json

Rules are data (SMOKE_RULES below, version-stamped). Each slot picks the
FIRST qualifying case id in the role's slot order — by default the smallest
case id, so the choice carries no judgement. A role may override that order
(SMOKE_ORDER) and may require each slot to introduce a new value of some
attribute (SMOKE_DIVERSITY, e.g. a different writer per slot). Both are still
pure functions of the frozen manifest: no peeking at model output, and the
result is reproducible from the rules alone.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from .manifests import REPO_ROOT, BenchCase, BenchmarkManifest

DEFAULT_SMOKE_ROOT = REPO_ROOT / "evaluation" / "model_selection" / "smoke"
SMOKE_RULES_VERSION = ("smoke-rules-v3 (2026-08-24, DEV only; min-id per slot unless the role "
                       "declares SMOKE_ORDER / SMOKE_DIVERSITY; grade_primary slots are the "
                       "three canonical explanation-verdict classes and require a derivable "
                       "verdict, replacing the v2 final-score buckets)")


class SmokeError(RuntimeError):
    """Frozen smoke subset missing, tampered, or not DEV-only."""


_DIGIT = re.compile(r"\d")
_LATIN = re.compile(r"[A-Za-z]{2,}")
_OPS = re.compile(r"[=+*/^<>]|(?<![א-ת])-")


def _ref(c: BenchCase) -> str:
    return c.label.get("reference") or ""


#: role -> ordered list of (slot, why, predicate over BenchCase)
SMOKE_RULES: dict[str, list[tuple[str, str, Callable[[BenchCase], bool]]]] = {
    "ocr_primary": [
        ("ordinary_handwritten_line", "ordinary Hebrew handwriting: a non-hard handwritten line without digits",
         lambda c: c.meta.get("category") == "handwritten_line" and not c.label.get("hard") and not _DIGIT.search(_ref(c))),
        ("difficult_handwriting", "difficult handwriting: an item the benchmark marks hard=True",
         lambda c: c.meta.get("tier") == "owner" and bool(c.label.get("hard"))),
        ("short_answer", "short answer: the shortest human-transcribed reference (<= 25 chars)",
         lambda c: c.meta.get("tier") == "owner" and 0 < len(_ref(c)) <= 25),
        ("longer_answer", "longer answer: a human-transcribed reference of >= 100 chars",
         lambda c: c.meta.get("tier") == "owner" and len(_ref(c)) >= 100),
        ("formula_numeric", "formula / numeric content: a printed formula item containing operators",
         lambda c: c.meta.get("category") == "formula_printed" and bool(_OPS.search(_ref(c)))),
        ("mixed_hebrew_english", "mixed Hebrew / English technical text",
         lambda c: c.meta.get("category") == "mixed_he_en" and bool(_LATIN.search(_ref(c)))),
        ("handwritten_cell_numeric", "multi-line handwritten cell with digits (non-hard)",
         lambda c: c.meta.get("category") == "handwritten_cell" and not c.label.get("hard") and bool(_DIGIT.search(_ref(c)))),
        ("option_row_association", "RTL option-letter/value association row (structure, not prose)",
         lambda c: c.meta.get("category") == "option_row_association"),
    ],
    "ocr_verify": [
        ("supported_1", "REAL positive: the audited transcription itself (must be SUPPORTED)",
         lambda c: c.component == "REAL" and c.label.get("polarity") == "positive"),
        ("supported_2", "REAL positive #2",
         lambda c: c.component == "REAL" and c.label.get("polarity") == "positive"),
        ("supported_3", "REAL positive #3",
         lambda c: c.component == "REAL" and c.label.get("polarity") == "positive"),
        ("real_omission", "REAL historical error: omission only",
         lambda c: c.component == "REAL" and c.label.get("error_kinds") == ["omission"]),
        ("real_substitution", "REAL historical error: substitution only",
         lambda c: c.component == "REAL" and c.label.get("error_kinds") == ["substitution"]),
        ("real_unsupported_addition", "REAL historical error including an unsupported addition",
         lambda c: c.component == "REAL" and "unsupported_addition" in (c.label.get("error_kinds") or [])),
        ("real_number_sign_formula", "REAL historical error touching a number/sign/formula token",
         lambda c: c.component == "REAL" and "number_sign_formula" in (c.label.get("error_kinds") or [])),
        ("real_subtle", "REAL subtle error (severity 'subtle') — the hard false-accept case",
         lambda c: c.component == "REAL" and c.label.get("severity") == "subtle"),
        ("synthetic_digit_substitution", "SYNTHETIC near miss: digit substitution (numeric group)",
         lambda c: c.component == "SYNTHETIC" and c.label.get("corruption_type") == "digit_substitution"),
        ("synthetic_short_token_omission", "SYNTHETIC near miss: short token omission",
         lambda c: c.component == "SYNTHETIC" and c.label.get("corruption_type") == "short_token_omission"),
        ("synthetic_char_deletion", "SYNTHETIC near miss: single character deletion",
         lambda c: c.component == "SYNTHETIC" and c.label.get("corruption_type") == "char_deletion"),
        ("synthetic_token_duplication", "SYNTHETIC near miss: token duplication / addition",
         lambda c: c.component == "SYNTHETIC" and c.label.get("corruption_type") == "token_duplication_addition"),
    ],
    # The grading model's responsibility is the explanation VERDICT, not the
    # final number (docs/grade-primary-benchmark-target.md), so the slots are
    # the three canonical verdict classes — one per production factor.
    #
    # A slot only accepts a case whose verdict is MATHEMATICALLY DERIVABLE.
    # That matters most for `invalid`: a zero-score case is only an invalid
    # EXPLANATION when the selection was correct. A zero from a wrong
    # selection carries no explanation ground truth, and letting it fill this
    # slot would put a fabricated label in front of the first paid call.
    "grade_primary": [
        ("verdict_invalid", "factor 0: the grader must be able to withhold all credit "
                            "(requires a confirmed-correct selection, else the zero says "
                            "nothing about the explanation)",
         lambda c: _verdict(c) == "invalid"),
        ("verdict_partially_valid", "factor 0.5: the hardest judgement — some credit, not all",
         lambda c: _verdict(c) == "partially_valid"),
        ("verdict_valid", "factor 1: a sufficient explanation must not be marked down",
         lambda c: _verdict(c) == "valid"),
    ],
}

#: role -> sort key over DEV cases; the first qualifying case per slot wins.
#: Default (absent) = lexicographic case id.
SMOKE_ORDER: dict[str, Callable[[BenchCase], Any]] = {
    # LONGEST answer first. "Do not cherry-pick easy cases": the shortest answer
    # in a bucket is usually a fragment that tests nothing, and min-id would pick
    # exactly that. Ties fall back to the case id, so this stays deterministic.
    "grade_primary": lambda c: (-len((c.inputs or {}).get("transcription") or ""), c.case_id),
}

#: role -> attribute each slot must introduce a NEW value of (or None).
#: Spreads the subset across students instead of grading one person three times.
SMOKE_DIVERSITY: dict[str, Callable[[BenchCase], Any]] = {
    "grade_primary": lambda c: c.meta.get("writer") or c.label.get("writer"),
}


def _verdict(c: BenchCase) -> str | None:
    """The DERIVED canonical explanation verdict of a case, or None.

    Only a case whose verdict is mathematically unique can fill a slot: an
    undecidable case would otherwise land in whichever bucket ``None``
    compares into, which is exactly how a fabricated label reaches a paid run.
    """
    if not c.label.get("explanation_verdict_derivable"):
        return None
    return c.label.get("explanation_verdict")


def _score(c: BenchCase) -> float | None:
    """The authoritative FINAL score merged into the manifest, or None.

    Only a case with real ground truth can fill a score-bucket slot: an
    unlabeled case would silently land in whichever bucket None compares into.
    """
    s = c.label.get("score")
    return None if s is None else float(s)


def _max_score(c: BenchCase) -> float:
    return float(c.label.get("max_score") or 0.0)


def _selection_hash(cases: list[dict]) -> str:
    payload = json.dumps([{"case_id": c["case_id"], "slot": c["slot"]} for c in cases],
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def propose_smoke(role: str, manifest: BenchmarkManifest) -> dict[str, Any]:
    """Deterministic selection over DEV only: each slot takes the smallest
    qualifying case id not already taken. Pure function of the frozen
    manifest + SMOKE_RULES — no model output involved."""
    rules = SMOKE_RULES.get(role)
    if not rules:
        raise SmokeError(f"no smoke rules for role {role!r}")
    order = SMOKE_ORDER.get(role) or (lambda c: c.case_id)
    diversity = SMOKE_DIVERSITY.get(role)
    dev = sorted(manifest.by_split("DEV"), key=order)
    taken: set[str] = set()
    seen_diverse: set[Any] = set()
    chosen: list[dict] = []
    for slot, why, pred in rules:
        def _ok(c):
            if c.case_id in taken or not pred(c):
                return False
            return not (diversity and diversity(c) in seen_diverse)
        pick = next((c for c in dev if _ok(c)), None)
        if pick is None:
            chosen.append({"slot": slot, "why": why, "case_id": None, "component": None,
                           "note": "no DEV case satisfies this rule"})
            continue
        taken.add(pick.case_id)
        if diversity:
            seen_diverse.add(diversity(pick))
        chosen.append({"slot": slot, "why": why, "case_id": pick.case_id, "component": pick.component,
                       "split": pick.split, **({"category": pick.meta.get("category")} if pick.meta.get("category") else {})})
    cases = [c for c in chosen if c["case_id"]]
    return {"role": role, "split": "DEV", "rules_version": SMOKE_RULES_VERSION,
            "manifest_hashes": manifest.hashes, "cases": cases,
            "unfilled_slots": [c["slot"] for c in chosen if not c["case_id"]],
            "selection_sha256": _selection_hash(cases)}


def smoke_path(role: str, root: Path = DEFAULT_SMOKE_ROOT) -> Path:
    return Path(root) / f"{role}_smoke.json"


def freeze_smoke(role: str, manifest: BenchmarkManifest, root: Path = DEFAULT_SMOKE_ROOT, *,
                 now: str | None = None, allow_unfilled: bool = False) -> dict[str, Any]:
    """Write the frozen subset once. Refuses to overwrite an existing file
    (a smoke subset is never re-selected after it exists), and refuses to
    freeze a subset that does not cover every slot.

    An unfilled slot is a hole in what the first paid run can observe, and the
    holes are never random: for grade_primary the slot most likely to be empty
    is ``verdict_invalid``, whose ground truth needs the selection-correctness
    audit. Freezing without it would ship a subset that cannot tell whether a
    grader can WITHHOLD credit — the one behaviour a grading benchmark must
    not leave unmeasured. Pass ``allow_unfilled`` to accept the gap knowingly;
    the frozen file records it either way.
    """
    p = smoke_path(role, root)
    if p.exists():
        raise SmokeError(f"{p} already exists: smoke subsets are frozen and never re-selected")
    prop = propose_smoke(role, manifest)
    unfilled = list(prop.get("unfilled_slots") or [])
    if unfilled and not allow_unfilled:
        raise SmokeError(
            f"{role}: {len(unfilled)} slot(s) could not be filled from the frozen dataset: "
            + ", ".join(unfilled)
            + ". Freezing now would hide that gap behind a hash. Supply the missing ground "
              "truth (for grade_primary's verdict slots: scripts/selection_audit_ui.py, then "
              "re-apply the verdict-target revision), or pass allow_unfilled to accept it.")
    prop["frozen_at"] = now or time.strftime("%Y-%m-%d %H:%M:%S")
    prop["_policy"] = ("Pre-registered DEV smoke subset for the first live execution of a candidate. "
                       "Frozen before any new model output; never modified or optimized after results. "
                       "Model-visible inputs are still ONLY the manifest's inputs for these case ids.")
    if unfilled:
        # A gap that is only a slot name in a list is a gap nobody reads. Say
        # what is NOT observable, so a later reader cannot mistake this subset
        # for full coverage of the role's slots.
        prop["_unfilled_slots_why"] = (
            "Frozen deliberately with allow_unfilled. These slots had NO qualifying case in "
            "the frozen dataset, so this subset does not observe them at all: "
            + ", ".join(unfilled)
            + ". For grade_primary/verdict_invalid this is structural — an `invalid` label "
              "requires an instructor score of 0 together with a CORRECT selection, and the "
              "2026-08-25 selection audit (8/8 human-audited) found every zero-score DEV case "
              "had a WRONG selection. Credit-withholding on a correct choice is therefore not "
              "measurable on this dataset. Do not read a result from this subset as evidence "
              "about the missing class.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prop, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return prop


def load_smoke(role: str, manifest: BenchmarkManifest | None = None, root: Path = DEFAULT_SMOKE_ROOT) -> dict[str, Any]:
    """Load + verify: hash matches, every id exists, every id is DEV."""
    p = smoke_path(role, root)
    if not p.exists():
        raise SmokeError(f"no frozen smoke subset for {role} (expected {p}); freeze it with `bench smoke freeze`")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("role") != role or d.get("split") != "DEV":
        raise SmokeError(f"{p}: wrong role/split ({d.get('role')}, {d.get('split')})")
    if _selection_hash(d.get("cases", [])) != d.get("selection_sha256"):
        raise SmokeError(f"{p}: selection hash mismatch — the smoke subset was modified after freezing")
    if manifest is not None:
        by_id = {c.case_id: c for c in manifest.cases}
        for c in d["cases"]:
            bc = by_id.get(c["case_id"])
            if bc is None:
                raise SmokeError(f"{p}: case {c['case_id']} is not in the frozen manifest")
            if bc.split != "DEV":
                raise SmokeError(f"{p}: case {c['case_id']} is {bc.split}, smoke subsets must be DEV only")
    return d


def smoke_case_ids(role: str, manifest: BenchmarkManifest, root: Path = DEFAULT_SMOKE_ROOT) -> list[str]:
    return [c["case_id"] for c in load_smoke(role, manifest, root)["cases"]]


def smoke_status(role: str, manifest: BenchmarkManifest | None, root: Path = DEFAULT_SMOKE_ROOT) -> dict[str, Any]:
    p = smoke_path(role, root)
    if not p.exists():
        return {"role": role, "frozen": False, "path": str(p)}
    try:
        d = load_smoke(role, manifest, root)
        return {"role": role, "frozen": True, "path": str(p), "cases": len(d["cases"]),
                "selection_sha256": d["selection_sha256"], "rules_version": d.get("rules_version"),
                "frozen_at": d.get("frozen_at"), "valid": True}
    except SmokeError as e:
        return {"role": role, "frozen": True, "path": str(p), "valid": False, "error": str(e)}


__all__ = ["DEFAULT_SMOKE_ROOT", "SMOKE_RULES", "SMOKE_ORDER", "SMOKE_DIVERSITY", "SMOKE_RULES_VERSION", "SmokeError", "propose_smoke",
           "freeze_smoke", "load_smoke", "smoke_case_ids", "smoke_status", "smoke_path"]
