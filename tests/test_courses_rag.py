"""Course-material store + qwen_rag_ocr_v1 arm: offline tests.

Embeddings are injected (deterministic character-ngram bag vectors) so no
model or network is needed; the injected embedder still produces genuine
lexical similarity, which is enough to test Hebrew retrieval plumbing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from autograder import courses

spec = importlib.util.spec_from_file_location(
    "m2_rag_ocr", Path(__file__).resolve().parents[1] / "scripts" / "m2_rag_ocr.py"
)
rag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag)


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic char-trigram hash embedding — real lexical similarity."""
    dim = 256
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = t.lower()
        for j in range(len(t) - 2):
            h = int(hashlib.sha1(t[j:j + 3].encode()).hexdigest()[:6], 16) % dim
            out[i, h] += 1.0
    return out


fake_embed.model_name = "fake-trigram-256"

HEBREW_MD = """# התמרת פורייה

התמרת פורייה מפרקת אות לרכיבי תדר. התדרים הגבוהים מייצגים שינויים חדים
בתמונה כמו שפות וקצוות.

# מסנן מעביר גבוהים

High Pass Filter מסיר את התדרים הנמוכים ואת רכיב ה-DC ומשאיר את
התדרים הגבוהים בלבד.

# היסטוגרמה

היסטוגרמה מתארת את התפלגות עוצמות האפור בתמונה.
"""


@pytest.fixture()
def course(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADER_COURSES_DIR", str(tmp_path / "courses"))
    courses.create_course("imgproc", "עיבוד תמונה")
    res = courses.add_source("imgproc", "summary.md", HEBREW_MD.encode("utf-8"))
    assert res["stored"]
    courses.build_index("imgproc", embed_fn=fake_embed)
    return "imgproc"


def test_chunk_ids_deterministic():
    parsed = {"source": "s.md", "sha256": "abc",
              "blocks": [{"text": "פסקה ראשונה על תדרים גבוהים ונמוכים בתמונה",
                          "page": None, "section": "A"}]}
    a = courses.chunk_parsed(parsed)
    b = courses.chunk_parsed(parsed)
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]
    parsed2 = {**parsed, "blocks": [{**parsed["blocks"][0], "text": "טקסט אחר לגמרי בפסקה הזאת"}]}
    assert courses.chunk_parsed(parsed2)[0]["chunk_id"] != a[0]["chunk_id"]


def test_hebrew_survives_ingestion(course):
    chunks = [json.loads(l) for l in
              (courses.course_dir(course) / "chunks" / "chunks.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    joined = " ".join(c["text"] for c in chunks)
    assert "התמרת פורייה" in joined
    assert "High Pass Filter" in joined
    assert all(c["source"] == "summary.md" for c in chunks)
    assert any(c.get("section") for c in chunks)  # heading metadata preserved


def test_index_persists_and_reloads(course):
    # retrieval works from disk without rebuilding (fresh call, no build)
    hits = courses.retrieve(course, "מהו מסנן מעביר גבוהים High Pass", top_k=2,
                            embed_fn=fake_embed)
    assert hits and all("chunk_id" in h and "similarity" in h and "source" in h
                        for h in hits)


def test_retrieval_finds_intended_hebrew_concept(course):
    hits = courses.retrieve(course, "התדרים הנמוכים ורכיב DC מוסרים על ידי המסנן",
                            top_k=1, embed_fn=fake_embed)
    assert "High Pass" in hits[0]["text"] or "מעביר גבוהים" in (hits[0].get("section") or "") + hits[0]["text"]


def test_material_change_marks_stale_and_rebuild_fixes(course):
    assert courses.index_status(course)["stale"] is False
    courses.add_source(course, "extra.txt", "חומר חדש על קונבולוציה ומסכות".encode())
    assert courses.index_status(course)["stale"] is True
    courses.build_index(course, embed_fn=fake_embed)
    st = courses.index_status(course)
    assert st["stale"] is False and st["n_chunks"] >= 3


def test_key_like_files_refused(course):
    for bad in ("answer_key.pdf", "rubric_final.txt", "מחוון.txt", "solution.md"):
        res = courses.add_source(course, bad, b"secret answers")
        assert not res["stored"], bad
    # and none of that content can reach the corpus
    chunks_text = (courses.course_dir(course) / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")
    assert "secret answers" not in chunks_text


def test_repair_prompt_contains_only_allowed_inputs():
    chunks = [{"chunk_id": "c1", "text": "קטע קורס", "source": "s.md",
               "page": None, "similarity": 0.9}]
    prompt = rag.build_repair_prompt("שאלה מודפסת", "טקסט OCR גולמי", chunks)
    assert "שאלה מודפסת" in prompt and "טקסט OCR גולמי" in prompt and "קטע קורס" in prompt
    # canary: a key/reference string that exists elsewhere in the repo must
    # not appear — the builder has no path to keys by construction
    assert "correct_by_version" not in prompt and "verified_ground_truth" not in prompt
    for forbidden in ("answer key", "rubric", "reference"):
        assert forbidden not in prompt.lower()
    # the system prompt carries the non-correction contract
    assert "Do NOT make the student's answer more correct" in rag.REPAIR_SYSTEM
    assert "Preserve student mistakes" in rag.REPAIR_SYSTEM


def test_raw_text_preserved_and_safe_degradation():
    raw = "טקסט מקורי עם טעות סטודנט"
    rec = rag.assemble_record("it1", raw, None, [], err="model down")
    assert rec["raw_text"] == raw  # byte-for-byte
    assert rec["suggested_text"] == raw  # degrade to raw, never invent
    assert rec["retrieval_empty"] is True
    rec2 = rag.assemble_record("it2", raw,
                               {"suggested_text": "אחר", "edits": [],
                                "semantic_change_risk": False}, [])
    assert rec2["raw_text"] == raw and rec2["suggested_text"] == "אחר"


def test_semantic_risk_flags_review_and_metadata_persisted():
    chunks = [{"chunk_id": "c9", "text": "t", "source": "doc.pdf", "page": 3,
               "similarity": 0.77}]
    rep = {"suggested_text": "מתוקן", "semantic_change_risk": False,
           "edits": [{"raw": "pc", "suggested": "DC", "reason": "course term",
                      "risk": "semantic"}]}
    rec = rag.assemble_record("it3", "raw", rep, chunks)
    assert rec["needs_review"] is True  # semantic edit forces review
    assert rec["retrieved"] == [{"chunk_id": "c9", "source": "doc.pdf",
                                 "page": 3, "similarity": 0.77}]
    rep2 = {"suggested_text": "x", "edits": [], "semantic_change_risk": True}
    assert rag.assemble_record("it4", "raw", rep2, [])["needs_review"] is True


def test_rag_arm_scored_from_suggested_text():
    """Evaluator contract: the arm's record exposes the repaired text as
    "transcription" (the field m2_bench_eval/m2_grading_eval score) while
    raw_text stays byte-for-byte and every audit field survives."""
    raw = "טקסט גולמי עם שגיאת OCR"
    rep = {"suggested_text": "טקסט גולמי עם שגיאה", "edits": [
        {"raw": "שגיאת OCR", "suggested": "שגיאה", "reason": "r",
         "risk": "low", "supporting_chunk_ids": ["c1"]}],
        "semantic_change_risk": False}
    chunks = [{"chunk_id": "c1", "text": "t", "source": "s.md", "page": None,
               "similarity": 0.5}]
    rec = rag.assemble_record("it5", raw, rep, chunks)
    assert rec["transcription"] == rec["suggested_text"] == rep["suggested_text"]
    assert rec["raw_text"] == raw  # never overwritten by the adapter
    assert rec["edits"] and rec["retrieved"][0]["chunk_id"] == "c1"
    assert rec["semantic_change_risk"] is False
    # repair failure: scored text degrades to raw, never invents
    rec2 = rag.assemble_record("it6", raw, None, [], err="down")
    assert rec2["transcription"] == rec2["suggested_text"] == raw


def test_question_context_fails_closed(tmp_path):
    """No file -> no question text; sample_data/ paths and non-JSON refused;
    a curated operator file loads."""
    assert rag.load_question_context(None) == {}
    with pytest.raises(SystemExit):
        rag.load_question_context(
            str(rag.REPO / "sample_data" / "anything.json"))
    with pytest.raises(SystemExit):
        rag.load_question_context(str(tmp_path / "questions.txt"))
    good = tmp_path / "questions.json"
    good.write_text(json.dumps({"1": "נוסח שאלה נקי"}, ensure_ascii=False),
                    encoding="utf-8")
    assert rag.load_question_context(str(good)) == {"1": "נוסח שאלה נקי"}


def test_no_solved_booklet_in_rag_source():
    """Invariant: the RAG arm has NO code path into the solved booklet or
    the grading-eval module it used to slice question text from."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "m2_rag_ocr.py")
    text = src.read_text(encoding="utf-8")
    assert "m2_grading_eval" not in text
    assert "question_context()" not in text
    assert text.count("Exam_solution") == 1  # docstring warning only
    assert "fail" in text.lower() and "sample_data" in text


def test_prompt_without_question_context_fails_closed():
    prompt = rag.build_repair_prompt("", "טקסט OCR", [])
    assert "(not provided" in prompt
    assert "טקסט OCR" in prompt


def test_filename_alias_hardening(course):
    for bad in ("key.pdf", "answers.docx", "תשובות.pdf", "מפתח.md",
                "grades.txt", "references.txt", "grading_notes.md"):
        res = courses.add_source(course, bad, "טקסט כלשהו".encode())
        assert not res["stored"], bad
    # word-bounded English: legitimate lecture names still pass
    for ok in ("keynote_lecture.md", "monkey_vision_notes.txt"):
        res = courses.add_source(course, ok, "סיכום הרצאה על ראייה".encode())
        assert res["stored"], ok


def test_content_screen_rejects_and_operator_override(course):
    # dense numbered option-letter list = classic MC answer key
    key_like = "\n".join(f"{i}. א" for i in range(1, 9))
    res = courses.add_source(course, "sikum_last_lecture.txt",
                             key_like.encode())
    assert res["stored"] is False and res.get("suspicious") is True
    assert not (courses.course_dir(course) / "sources"
                / "sikum_last_lecture.txt").exists()
    # marker phrases fire too, even with a harmless filename
    marked = "פרק ראשון\nמחוון בדיקה לשאלה 1\nמחוון בדיקה לשאלה 2".encode()
    assert courses.add_source(course, "notes2.txt", marked)["stored"] is False
    # a short exercise list in real notes must NOT trip the screen
    benign = "תרגול:\n1. א\n2. ב\n3. ג\nהסבר מפורט על כל סעיף".encode()
    assert courses.add_source(course, "targul.txt", benign)["stored"] is True
    # explicit operator override ingests, with the reason recorded
    res2 = courses.add_source(course, "sikum_last_lecture.txt",
                              key_like.encode(), allow_suspicious=True)
    assert res2["stored"] is True and res2.get("suspicious_override")
