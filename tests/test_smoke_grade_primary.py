"""The frozen GRADE_PRIMARY smoke subset — the first three cases any grading
model is ever paid to see.

The point of freezing is that nobody can pick the cases after seeing a result.
These tests check the properties that make the subset worth trusting: it is
DEV-only, it spans the three score buckets, it spans three students, it takes
the HARDEST (longest) answer in each bucket rather than the most convenient
one, and it is reproducible from the rules alone.

Offline: no model, network or OCR calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.smoke import (DEFAULT_SMOKE_ROOT, SMOKE_DIVERSITY, SMOKE_ORDER,
                                        SMOKE_RULES, SmokeError, load_smoke, propose_smoke,
                                        smoke_case_ids)

ROLE = "grade_primary"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(ROLE)


@pytest.fixture(scope="module")
def frozen(manifest):
    return load_smoke(ROLE, manifest)


def _by_id(manifest):
    return {c.case_id: c for c in manifest.cases}


def test_the_frozen_subset_is_exactly_what_the_rules_produce(manifest, frozen):
    """If this fails, either the rules or the frozen file changed — and a smoke
    subset is never re-selected after freezing."""
    assert [c["case_id"] for c in propose_smoke(ROLE, manifest)["cases"]] == \
           [c["case_id"] for c in frozen["cases"]]
    assert frozen["selection_sha256"] == propose_smoke(ROLE, manifest)["selection_sha256"]


def test_it_is_three_dev_cases_and_no_held_out(manifest, frozen):
    cases = [_by_id(manifest)[c["case_id"]] for c in frozen["cases"]]
    assert len(cases) == 3
    assert {c.split for c in cases} == {"DEV"}, "a paid smoke run must never touch HELD_OUT"


def test_it_covers_no_partial_and_full_credit(manifest, frozen):
    """A grader that only ever awards full marks would pass a subset of 4s."""
    cases = [_by_id(manifest)[c["case_id"]] for c in frozen["cases"]]
    scores = [float(c.label["score"]) for c in cases]
    maxes = [float(c.label["max_score"]) for c in cases]
    assert any(s == 0 for s in scores), "no zero: withholding credit is untested"
    assert any(0 < s < m for s, m in zip(scores, maxes)), "no partial credit"
    assert any(s == m for s, m in zip(scores, maxes)), "no full-credit case"
    assert [c["slot"] for c in frozen["cases"]] == ["no_credit", "partial_credit", "full_credit"]


def test_every_case_has_real_ground_truth(manifest, frozen):
    """Without a FINAL label the run measures nothing."""
    for c in (_by_id(manifest)[x["case_id"]] for x in frozen["cases"]):
        assert c.label.get("score") is not None
        assert c.label.get("transcription_complete") is not False, (
            "the model would not see the whole answer the human graded")


def test_it_spans_three_students(manifest, frozen):
    cases = [_by_id(manifest)[c["case_id"]] for c in frozen["cases"]]
    writers = [SMOKE_DIVERSITY[ROLE](c) for c in cases]
    assert len(set(writers)) == 3, f"the subset grades the same student twice: {writers}"
    assert len({c.label.get("question_id") for c in cases}) >= 2, "only one question is covered"


def test_it_takes_the_hardest_case_in_each_bucket_not_the_easiest(manifest, frozen):
    """'Do not cherry-pick easy cases': each pick must be the LONGEST answer
    available in its bucket among still-eligible students."""
    dev = sorted(manifest.by_split("DEV"), key=SMOKE_ORDER[ROLE])
    picked, used_writers = [], set()
    for (slot, _why, pred), rec in zip(SMOKE_RULES[ROLE], frozen["cases"]):
        eligible = [c for c in dev if pred(c) and c.case_id not in picked
                    and SMOKE_DIVERSITY[ROLE](c) not in used_writers]
        best = eligible[0]
        assert rec["case_id"] == best.case_id, (
            f"slot {slot}: frozen {rec['case_id']} is not the longest eligible answer "
            f"({best.case_id}, {len(best.inputs['transcription'])} chars)")
        picked.append(best.case_id)
        used_writers.add(SMOKE_DIVERSITY[ROLE](best))


def test_a_tampered_frozen_file_is_refused(tmp_path, manifest, frozen):
    root = tmp_path / "smoke"
    root.mkdir()
    d = json.loads((DEFAULT_SMOKE_ROOT / f"{ROLE}_smoke.json").read_text(encoding="utf-8"))
    d["cases"][0]["case_id"] = d["cases"][0]["case_id"].replace("e003", "e002")
    (root / f"{ROLE}_smoke.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SmokeError, match="hash mismatch"):
        load_smoke(ROLE, manifest, root)


def test_the_runner_resolves_the_same_ids(manifest, frozen):
    assert smoke_case_ids(ROLE, manifest) == [c["case_id"] for c in frozen["cases"]]


def test_the_diversity_hook_does_not_disturb_the_roles_that_do_not_use_it(manifest):
    """ocr_primary / ocr_verify were frozen under the min-id rule; adding the
    hooks must not change what those rules select."""
    for role in ("ocr_primary", "ocr_verify"):
        assert role not in SMOKE_ORDER and role not in SMOKE_DIVERSITY
        m = load_manifest(role)
        assert load_smoke(role, m)["selection_sha256"] == propose_smoke(role, m)["selection_sha256"]
