"""Deterministic verification/repair of per-version answer-key columns.

The key encodes per-version answers as slash letter-groups ("F/F/G") whose
positions follow the key's legend. Model decoding of that encoding proved
unreliable (columns flattened) — the text layer is born-digital, so the
groups are decoded deterministically; colour-only information without a
text-layer group is either supplied by a one-time operator override or
flagged unverified → human review for affected versions.
"""

from autograder.cli import _key_version_problems
from autograder.config import GraderConfig
from autograder.grade import VersionDecision, grade_exam
from autograder.keyrepair import question_segments, repair_key_versions
from autograder.schema import (
    ExamExtraction,
    QuestionExtraction,
    SubItemExtraction,
)
from tests.test_grade import make_key

# Text-layer fixture mirroring the real key's structure: question headers and
# one letter-group per matching sub-item, in print order.
KEY_TEXT = """
שאלה מספר 1: (32 נקודות)
1. הפעלת טשטוש F/F/G – Motion Blur
2. הפעלת G/G/F - High Pass Filter
3. חידוד D/E/I – High Frequency Emphasis
4. הפעלת H/A/E - Low Pass Filter
5. הוספת רעש C/I/C
6. הרעשה I/C/D
7. הגברת בהירות A/H/B
8. הכפלת E/B/H
שאלה מספר 2: (32 נקודות)
1. פעולה F/G/D
2. פעולה G/F/E
3. פעולה D/D/F
4. פעולה H/E/G
5. פעולה C/C/H
6. פעולה I/I/A
7. פעולה A/B/C
8. פעולה E/A/B
שאלה מספר 3: (36 נקודות)
16. נשתמש ב-Hough Transform
הצבעים הם R,B,G לפי A1, A2, A3 בהתאם.
"""


def flattened_key():
    """A key whose versions were flattened to A1's letter everywhere."""
    key = make_key()
    for q in key.questions:
        for s in q.sub_items:
            a1 = s.correct_by_version.get("A1", ["X"])
            s.correct_by_version = {"A1": list(a1), "A2": list(a1), "A3": list(a1)}
    return key


def test_question_segments_split_by_headers():
    segs = question_segments(KEY_TEXT)
    assert set(segs) == {"1", "2", "3"}
    assert "Motion Blur" in segs["1"]
    assert "Hough" in segs["3"]


def test_repair_decodes_triplets_and_flags_colour_only_items():
    key = flattened_key()
    report = repair_key_versions(key, KEY_TEXT, ["A1", "A2", "A3"])

    q1 = key.question("1")
    item1 = next(s for s in q1.sub_items if s.id == "1")
    assert item1.correct_by_version == {"A1": ["F"], "A2": ["F"], "A3": ["G"]}
    item4 = next(s for s in q1.sub_items if s.id == "4")
    assert item4.correct_by_version == {"A1": ["H"], "A2": ["A"], "A3": ["E"]}
    assert item1.versions_unverified == []

    # The fixture key has two questions: matching Q1 (8 items, all decodable
    # from the text groups) and MC Q3 (20 items, colour-only). Every Q1 item
    # must be decoded one way or the other.
    assert len(report["repaired"]) + len(report["verified"]) == 8

    # Q3 has no letter groups: colour-only -> unverified for every version.
    q3 = key.question("3")
    assert all(s.versions_unverified == ["A1", "A2", "A3"] for s in q3.sub_items)
    assert len(report["unverified"]) == 20


def test_repair_preserves_accepted_alternatives_when_primary_matches():
    key = flattened_key()
    q1 = key.question("1")
    item1 = next(s for s in q1.sub_items if s.id == "1")
    item1.correct_by_version["A1"] = ["F", "B"]  # accepted alternative from a note
    repair_key_versions(key, KEY_TEXT, ["A1", "A2", "A3"])
    assert sorted(item1.correct_by_version["A1"]) == ["B", "F"], (
        "alternatives survive when the deterministic primary letter agrees"
    )
    assert item1.correct_by_version["A3"] == ["G"]


def test_repair_skips_questions_with_group_count_mismatch():
    key = flattened_key()
    text = KEY_TEXT.replace("8. הכפלת E/B/H\n", "")  # Q1 now has 7 groups for 8 items
    report = repair_key_versions(key, text, ["A1", "A2", "A3"])
    q1 = key.question("1")
    assert all(s.versions_unverified == ["A1", "A2", "A3"] for s in q1.sub_items), (
        "positional decode with a count mismatch would be a guess — flagged instead"
    )
    assert any("positional decode unsafe" in n for n in report["notes"])


def test_operator_override_applies_and_clears_flag():
    key = flattened_key()
    overrides = {"3": {"16": {"A2": ["B"]}}}
    repair_key_versions(key, KEY_TEXT, ["A1", "A2", "A3"], overrides=overrides)
    q3 = key.question("3")
    item16 = next(s for s in q3.sub_items if s.id == "16")
    assert item16.correct_by_version["A2"] == ["B"]
    assert item16.versions_unverified == []
    other = next(s for s in q3.sub_items if s.id == "17")
    assert other.versions_unverified == ["A1", "A2", "A3"]


def test_grading_flags_unverified_version_items_for_review():
    key = make_key()
    q3 = key.question("3")
    for s in q3.sub_items:
        s.versions_unverified = ["A2"]  # only A2's column is unverified
    extraction = ExamExtraction(
        questions=[
            QuestionExtraction(
                question_id=q.id,
                source_pages=[1],
                authoritative_source="test",
                sub_items=[
                    SubItemExtraction(
                        sub_item_id=s.id,
                        status="answered" if q.id == "3" else "unanswered",
                        final_answer="B" if q.id == "3" else None,
                        answer_origin="answer_sheet",
                        interpretation_rationale="t",
                        confidence=1.0,
                    )
                    for s in q.sub_items
                ],
            )
            for q in key.questions
        ]
    )
    for version, expect_flagged in [("A2", True), ("A1", False)]:
        result = grade_exam(
            key, extraction, {}, VersionDecision(version, "pinned", False),
            GraderConfig(), exam_file="e.pdf", graded_at="t", model="mock:m",
        )
        q3_res = next(q for q in result.questions if q.question_id == "3")
        flagged = [s for s in q3_res.sub_results if s.needs_review]
        if expect_flagged:
            assert len(flagged) == len(q3_res.sub_results), version
            assert any("deterministically unverified" in s.reason for s in q3_res.sub_results)
        else:
            assert not any(
                "deterministically unverified" in (s.reason or "") for s in q3_res.sub_results
            ), "A1's column is not flagged when only A2 is unverified"


def test_fragmented_group_recovered_via_override_consuming_position():
    """A letter group fragmented across text-layer lines cannot be decoded
    positionally; supplying THAT item via the operator override lets the
    remaining items decode deterministically instead of flagging the whole
    question."""
    key = flattened_key()
    # Remove item 8's group: 7 groups remain for 8 sub-items.
    text = KEY_TEXT.replace("8. הכפלת E/B/H\n", "8. הכפלת\n")
    overrides = {"1": {"8": {"A1": ["E"], "A2": ["B"], "A3": ["H"]}}}
    report = repair_key_versions(key, text, ["A1", "A2", "A3"], overrides=overrides)
    q1 = key.question("1")
    item7 = next(s for s in q1.sub_items if s.id == "7")
    assert item7.correct_by_version == {"A1": ["A"], "A2": ["H"], "A3": ["B"]}
    item8 = next(s for s in q1.sub_items if s.id == "8")
    assert item8.correct_by_version["A3"] == ["H"]
    assert item8.versions_unverified == []
    assert len(report["repaired"]) + len(report["verified"]) == 7
    assert report["overridden"] == ["1.8"]


def test_group_split_across_whitespace_is_still_matched():
    from autograder.keyrepair import _group_pattern, _normalize_group

    text = "1. פעולה F  /  F\n/G אחרי"
    got = [_normalize_group(g) for g in _group_pattern(3).findall(text)]
    assert got == ["F/F/G"]
    # Axis labels like X/Y never match (letters restricted to A-I).
    assert _group_pattern(3).findall("בציר X/Y/Z") == []


def test_flattening_detector_rejects_uniform_key_when_text_differs():
    key = flattened_key()
    problems = _key_version_problems(key, ["A1", "A2", "A3"], KEY_TEXT)
    assert any("FLATTENED" in p for p in problems)
    # After repair the same key passes.
    repair_key_versions(key, KEY_TEXT, ["A1", "A2", "A3"])
    assert _key_version_problems(key, ["A1", "A2", "A3"], KEY_TEXT) == []
