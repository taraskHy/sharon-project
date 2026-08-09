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
