"""Anti-overfitting guards (2026-08 generalization audit, docs/generalization.md).

Pins: configurable package discovery (no dataset dirs in code), removal of
exam-specific literals from model-visible prompts, build-time re-screen of
the course RAG corpus with persisted operator overrides, and the fresh-state
guarantee that a rerun with all derived state deleted reproduces the result.
All offline; mock backends only.
"""

from __future__ import annotations

import json
from pathlib import Path

from autograder import courses
from autograder.reviewui import package_dirs
from tests.test_courses_rag import HEBREW_MD, fake_embed

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------- configurable package dirs --


def test_package_dirs_default_has_no_dataset_names(tmp_path, monkeypatch):
    monkeypatch.delenv("GRADER_PACKAGE_DIRS", raising=False)
    dirs = package_dirs(repo_root=tmp_path)  # no grader.toml in tmp root
    names = [d.name for d in dirs]
    assert names == ["packages", "sample_data"]
    assert "prob_data" not in names  # historical dataset dir must be config-only


def test_package_dirs_env_override(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("GRADER_PACKAGE_DIRS",
                       os.pathsep.join(["my_packages", str(tmp_path / "abs")]))
    dirs = package_dirs(repo_root=tmp_path)
    assert dirs == [tmp_path / "my_packages", tmp_path / "abs"]


def test_package_dirs_grader_toml(tmp_path, monkeypatch):
    monkeypatch.delenv("GRADER_PACKAGE_DIRS", raising=False)
    (tmp_path / "grader.toml").write_text(
        '[ui]\npackage_dirs = ["prob_data", "sample_data"]\n', encoding="utf-8")
    dirs = package_dirs(repo_root=tmp_path)
    assert dirs == [tmp_path / "prob_data", tmp_path / "sample_data"]


# ------------------------------------- prompt-leakage removal regression ---


def test_model_visible_prompts_carry_no_current_exam_content():
    """The removed exam-specific literals must never return to model-visible
    prompt/schema text (real key answers, legend, student note, instructor
    score, rubric attribution). keyrepair.py's guarded decoder is documented
    separately and is not model-visible."""
    banned = [
        "F/F/G", "A/H/B",                        # real per-version answer groups
        "R,B,G for versions A1",                 # real key legend
        "28/32",                                 # real instructor score
        "mixed up questions 1 and 2",            # real student note (EN)
        "התבלבלתי",                              # real student note (HE)
        "exam's own rubric",                     # false per-exam rubric claim
        "in version 2 the answer is 3",          # real version-note answers
        "20 items x 2",                          # real Q3 cap structure
    ]
    for name in ("prompts.py", "schema.py", "escalation.py", "grade.py",
                 "survey.py", "alignment.py", "mcresolve.py", "discovery.py"):
        text = (REPO_ROOT / "autograder" / name).read_text(encoding="utf-8")
        for literal in banned:
            assert literal not in text, f"{literal!r} found in autograder/{name}"


# --------------------------------------- RAG corpus build-time re-screen ---


def test_out_of_band_key_file_is_excluded_at_build_time(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADER_COURSES_DIR", str(tmp_path / "courses"))
    d = courses.create_course("cv")
    res = courses.add_source("cv", "lecture.md", HEBREW_MD.encode("utf-8"))
    assert res["stored"]
    # Bypass the UI door entirely: drop files straight into sources/.
    (d / "sources" / "answer_key.md").write_text("1. A\n2. B\n", encoding="utf-8")
    (d / "sources" / "notes.md").write_text(
        "מחוון בדיקה לשאלות\nanswer key for the exam\n" + "1. A\n2. B\n3. C\n"
        "4. D\n5. A\n6. B\n", encoding="utf-8")
    (d / "sources" / "grades_export.csv").write_text("id,grade\n1,90\n",
                                                     encoding="utf-8")
    manifest = courses.build_index("cv", embed_fn=fake_embed)
    excluded = {e["file"] for e in manifest["excluded_sources"]}
    assert {"answer_key.md", "notes.md", "grades_export.csv"} <= excluded
    assert set(manifest["source_hashes"]) == {"lecture.md"}
    chunks = (d / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")
    assert "answer key" not in chunks and "מחוון" not in chunks


def test_operator_override_is_persisted_and_honored_at_build(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADER_COURSES_DIR", str(tmp_path / "courses"))
    d = courses.create_course("cv2")
    courses.add_source("cv2", "lecture.md", HEBREW_MD.encode("utf-8"))
    flagged = ("סיכום מחוון ותרגול\nthe answer key discussion follows\n"
               "worked example content\n")
    res = courses.add_source("cv2", "worked_examples.md",
                            flagged.encode("utf-8"), allow_suspicious=True)
    assert res["stored"] and res.get("suspicious_override")
    meta = json.loads((d / "course.json").read_text(encoding="utf-8"))
    assert any(o["filename"] == "worked_examples.md"
               for o in meta["suspicious_overrides"])  # override persisted on disk
    manifest = courses.build_index("cv2", embed_fn=fake_embed)
    assert "worked_examples.md" in manifest["source_hashes"]  # override honored
    assert "worked_examples.md" in manifest["suspicious_overrides"]  # ...and audited


# ------------------------------------------------------- fresh state (0.4) --


def test_fresh_state_rerun_reproduces_the_result(tmp_path, monkeypatch):
    """Grading a package with NO previous derived state, deleting every
    derived artifact, and grading again must give the same result — the
    outcome may never depend on having processed the exam before."""
    import shutil

    from tests.test_grading_modes import FakeRuntime, _grade_responses, _run

    result_a, _backend, out_a = _run(tmp_path, monkeypatch, mode="reliability",
                                     runtime=FakeRuntime(tmp_path / "rt-a",
                                                         _grade_responses()),
                                     out_name="out-a")
    # wipe ALL derived/disposable state from the first run
    shutil.rmtree(out_a, ignore_errors=True)
    shutil.rmtree(tmp_path / "rt-a", ignore_errors=True)
    shutil.rmtree(tmp_path / "packs", ignore_errors=True)
    result_b, _backend2, _out_b = _run(tmp_path, monkeypatch, mode="reliability",
                                       runtime=FakeRuntime(tmp_path / "rt-b",
                                                           _grade_responses()),
                                       out_name="out-b")
    assert result_b.total_awarded == result_a.total_awarded
    assert result_b.total_max == result_a.total_max
    assert (len(result_b.needs_human_review) == len(result_a.needs_human_review))
