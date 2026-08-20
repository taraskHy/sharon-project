"""Privacy minimisation for provider requests (§9) and exact-only reuse (§10)."""

from __future__ import annotations

import json

import pytest

from autograder.privacy import (IDENTITY_KEYS, PrivacyError, ProviderRequest, anonymous_item_id,
                                assert_anonymous, build_grading_request, build_ocr_request,
                                safe_ledger_entry, safe_log_name, scan_for_identifiers, scrub_meta)
from autograder.reuse import (EXACT_KINDS, ExactReuseStore, SemanticReuseRefused,
                              exact_fingerprint, image_fingerprint, reuse_grade_by_similarity)

# Fabricated identifying metadata that a caller might be holding nearby.
IDENTITY = {
    "student_name": "שרון כהן",
    "student_id": "312456789",
    "email": "sharon@example.ac.il",
    "original_name": "Cohen_Sharon_final_exam_87.pdf",
    "path": r"C:\Users\ethan\exams\Cohen_Sharon_final_exam_87.pdf",
    "school": "Example University",
}
IDENTIFIERS = list(IDENTITY.values())


# ---------------------------------------------------------------- §9 OCR -----


def test_ocr_request_carries_only_the_crop_and_an_anonymous_id():
    item = anonymous_item_id("job-1", "exam-003", "1", "4")
    req = build_ocr_request(item_id=item, crop_png_b64="QUJD",
                            meta={**IDENTITY, "job_id": "job-1", "question_id": "1"})
    payload = req.as_dict()
    assert scan_for_identifiers(payload, IDENTIFIERS) == []
    assert payload["meta"] == {"job_id": "job-1", "question_id": "1", "item_id": item}
    kinds = [b["type"] for b in payload["content_blocks"]]
    assert kinds == ["image", "text"] and item in payload["content_blocks"][1]["text"]
    assert len(payload["content_blocks"]) == 2          # crop + instruction, nothing else


def test_grading_request_carries_only_grading_context():
    item = anonymous_item_id("job-1", "exam-003", "1", "4")
    req = build_grading_request(item_id=item, question_context="Question 1 (max 4 pts)\nRubric:\n R1: ...",
                                transcription="התדרים הגבוהים נשמרים", selected_option="C",
                                evidence="התדרים הגבוהים",
                                meta={**IDENTITY, "job_id": "job-1", "pack_hash": "abc"})
    payload = req.as_dict()
    assert scan_for_identifiers(payload, IDENTIFIERS) == []
    assert set(payload["meta"]) == {"job_id", "pack_hash", "item_id"}
    text = payload["content_blocks"][0]["text"]
    assert "התדרים הגבוהים נשמרים" in text and "Student selected option: C" in text


def test_identifying_metadata_is_dropped_not_renamed():
    assert scrub_meta(IDENTITY) == {}
    assert scrub_meta({**IDENTITY, "stage": "grade"}) == {"stage": "grade"}


def test_the_audit_scanner_actually_catches_leaks():
    leaky = ProviderRequest("grade_primary", [{"type": "text", "text":
                                               "Grade this exam by שרון כהן"}],
                            {"student_id": "312456789"}).as_dict()
    problems = scan_for_identifiers(leaky, IDENTIFIERS)
    assert any("student_id" in p for p in problems)
    assert any("שרון כהן" in p for p in problems)
    with pytest.raises(PrivacyError):
        assert_anonymous(leaky, IDENTIFIERS)


def test_filesystem_paths_are_caught_even_without_a_known_identifier():
    payload = {"content_blocks": [{"type": "text", "text": r"see C:\Users\ethan\exams\scan.pdf"}]}
    assert any("filesystem path" in p for p in scan_for_identifiers(payload))


def test_anonymous_ids_are_stable_and_non_reversible():
    a = anonymous_item_id("job-1", "exam-003", "1", "4")
    b = anonymous_item_id("job-1", "exam-003", "1", "4")
    c = anonymous_item_id("job-1", "exam-004", "1", "4")
    assert a == b and a != c and a.startswith("item-")
    assert "exam-003" not in a


def test_logs_use_internal_ids_not_filenames_or_paths():
    assert safe_log_name("exam-003.pdf") == "exam-003"
    assert safe_log_name("Cohen_Sharon_final_87.pdf").startswith("file-")
    assert "Cohen" not in safe_log_name(r"C:\Users\ethan\Cohen_Sharon.pdf")
    entry = safe_ledger_entry({"task": "grade_primary", "exam_file": IDENTITY["original_name"],
                               "student_name": IDENTITY["student_name"], "input_tokens": 120})
    blob = json.dumps(entry, ensure_ascii=False)
    assert "Cohen" not in blob and "שרון" not in blob
    assert entry["input_tokens"] == 120 and entry["item_ref"].startswith("file-")


def test_every_identity_key_is_covered_by_the_whitelist():
    for k in IDENTITY:
        assert k in IDENTITY_KEYS
    assert not (IDENTITY_KEYS & set(scrub_meta({k: "x" for k in IDENTITY_KEYS})))


# --------------------------------------------------------------- §10 reuse ---


def test_exact_mechanical_decisions_are_reused(tmp_path):
    store = ExactReuseStore(tmp_path / "reuse.json")
    fp = exact_fingerprint("variant_marker", marker="four-petal icon", page=1, region="bottom third")
    store.put("variant_marker", fp, {"variant": "variant_1"}, by="lecturer")
    hit = store.lookup("variant_marker", marker="four-petal icon", page=1, region="bottom third")
    assert hit and hit.decision == {"variant": "variant_1"} and hit.by == "lecturer"
    assert store.stats()["hits"] == 1


def test_any_difference_in_the_mechanical_facts_is_a_miss(tmp_path):
    store = ExactReuseStore(tmp_path / "reuse.json")
    store.put("alignment", exact_fingerprint("alignment", printed="16", canonical="20"),
              {"map": {"16": "20"}})
    assert store.lookup("alignment", printed="16", canonical="20") is not None
    assert store.lookup("alignment", printed="16", canonical="21") is None
    assert store.lookup("alignment", printed="17", canonical="20") is None


def test_image_reuse_is_byte_exact():
    a = image_fingerprint(b"\x89PNG-crop-A")
    assert a == image_fingerprint(b"\x89PNG-crop-A")
    assert a != image_fingerprint(b"\x89PNG-crop-B")


def test_reuse_survives_a_restart(tmp_path):
    p = tmp_path / "reuse.json"
    fp = exact_fingerprint("question_pack", question="1", rubric_hash="r1")
    ExactReuseStore(p).put("question_pack", fp, {"pack_hash": "abc"})
    assert ExactReuseStore(p).get("question_pack", fp).decision == {"pack_hash": "abc"}


def test_semantic_reuse_is_refused_at_every_entry_point(tmp_path):
    store = ExactReuseStore(tmp_path / "reuse.json")
    with pytest.raises(SemanticReuseRefused):
        exact_fingerprint("answer_similarity", text="התדרים הגבוהים נשמרים")
    with pytest.raises(SemanticReuseRefused):
        store.put("score_neighbour", "score_neighbour:abc", {"score": 7})
    with pytest.raises(SemanticReuseRefused):
        reuse_grade_by_similarity(answer="התדרים הגבוהים נשמרים", neighbour_score=7)


def test_identical_student_answers_do_not_share_a_grade(tmp_path):
    """Two students writing the exact same sentence is NOT a mechanical
    identity — nothing in this module can turn it into a shared score."""
    store = ExactReuseStore(tmp_path / "reuse.json")
    same_text = "התדרים הגבוהים נשמרים בתמונה"
    with pytest.raises(SemanticReuseRefused):
        exact_fingerprint("student_answer", text=same_text)
    assert all(k not in EXACT_KINDS for k in ("student_answer", "answer_similarity"))
    # the only text-derived exact key is the question pack itself, not an answer
    assert store.lookup("question_pack", question="1", rubric_hash="r1") is None


def test_unknown_kinds_are_rejected():
    with pytest.raises(ValueError):
        exact_fingerprint("vibes", x=1)


# ------------------------------------------------ integration: the gateway ---


def test_gateway_blocks_a_request_carrying_identifying_keys():
    from autograder.backends.mock import MockBackend
    from autograder.escalation import GradeResult
    from autograder.gateway import ModelGateway

    gw = ModelGateway.from_dict({"models": {"grade_primary": {"backend": "mock", "model": "m"}}},
                                backend_factory=lambda c: MockBackend(
                                    config=c, responder=lambda *a: GradeResult(score=1)))
    leaky = [{"type": "text", "text": "grade this", "student_name": IDENTITY["student_name"]}]
    with pytest.raises(PrivacyError):
        gw.call(task="grade_primary", system="s", content_blocks=leaky, output_model=GradeResult)
    clean = [{"type": "text", "text": "[item-abc] grade this"}]
    assert gw.call(task="grade_primary", system="s", content_blocks=clean,
                   output_model=GradeResult).value.score == 1


def test_gateway_ledger_never_persists_a_filename_or_path(tmp_path):
    from autograder.backends.mock import MockBackend
    from autograder.escalation import GradeResult
    from autograder.gateway import ModelGateway
    from autograder.usage import UsageLedger

    ledger = UsageLedger(tmp_path / "usage.jsonl")
    gw = ModelGateway.from_dict({"models": {"grade_primary": {"backend": "mock", "model": "m"}}},
                                backend_factory=lambda c: MockBackend(
                                    config=c, responder=lambda *a: GradeResult(score=1)),
                                ledger=ledger)
    gw.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "x"}],
            output_model=GradeResult,
            meta={"job_id": "job-1", "exam_id": "exam-003", "question_id": "1",
                  "exam_file": IDENTITY["original_name"], "student_name": IDENTITY["student_name"]})
    blob = (tmp_path / "usage.jsonl").read_text(encoding="utf-8")
    assert "Cohen" not in blob and "שרון" not in blob
    row = json.loads(blob.splitlines()[0])
    # the gateway whitelists what it records: identity keys never even appear
    assert row["exam_id"] == "exam-003" and row["question_id"] == "1"
    assert not (set(row) & IDENTITY_KEYS)


def test_path_like_strings_are_warned_about_not_silently_sent():
    from autograder.backends.mock import MockBackend
    from autograder.escalation import GradeResult
    from autograder.gateway import ModelGateway

    gw = ModelGateway.from_dict({"models": {"grade_primary": {"backend": "mock", "model": "m"}}},
                                backend_factory=lambda c: MockBackend(
                                    config=c, responder=lambda *a: GradeResult(score=1)))
    gw.call(task="grade_primary", system="s",
            content_blocks=[{"type": "text", "text": r"student wrote C:\Users\x\scan.pdf"}],
            output_model=GradeResult)
    assert gw.privacy_warnings and "path-like" in gw.privacy_warnings[0]
