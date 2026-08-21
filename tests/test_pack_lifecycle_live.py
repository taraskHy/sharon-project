"""Persistent QuestionGradingPack lifecycle AT THE LIVE SEAM.

The audit found PackStore/persistence had zero production callers — the
reliability seam rebuilt packs in memory per run. These tests prove the live
``cli.run_grade_pipeline`` now builds packs ONCE per exam package, persists
them, reuses them across students, and invalidates on every relevant input
change. Mock backend + mock gateway; no provider, no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autograder import gradingpack as gp
from autograder import orchestrator as orch
from autograder.cli import run_grade_pipeline
from autograder.key_parser import save_answer_key
from tests.conftest import make_pdf
from tests.test_grade import make_key
from tests.test_grading_modes import FakeRuntime, _backend, _grade_responses


def _ns(tmp_path, key_path, exam, **over):
    base = dict(key=str(key_path), rubric=None, resume=False, version="auto",
                exam=str(exam), grading_mode="reliability",
                rag_policy="RAG_DISABLED", course=None,
                packs_root=str(tmp_path / "packs"),
                models_config=str(tmp_path / "models.toml"), grading_policies=None)
    base.update(over)
    return argparse.Namespace(**base)


def _grade(tmp_path, monkeypatch, out_name, *, runtime, key_path, exam_name="01_50.pdf", **over):
    exam = make_pdf(tmp_path / exam_name)
    monkeypatch.setattr(orch, "setup_from_config", lambda *a, **kw: runtime)
    ns = _ns(tmp_path, key_path, exam, **over)
    return run_grade_pipeline(ns, _backend(), tmp_path / out_name, 800,
                              exam_path=exam, exam_label=out_name)


def test_packs_built_once_and_reused_across_students(tmp_path, monkeypatch):
    key_path = tmp_path / "answer_key.json"
    save_answer_key(make_key(), key_path)
    builds = []
    real_build = gp.build_all_packs
    monkeypatch.setattr(gp, "build_all_packs",
                        lambda *a, **kw: builds.append(1) or real_build(*a, **kw))

    rt = FakeRuntime(tmp_path, _grade_responses())
    _grade(tmp_path, monkeypatch, "student-1", runtime=rt, key_path=key_path)
    assert len(builds) == 1
    manifest = json.loads((tmp_path / "packs" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["packs"]                      # persisted once

    rt2 = FakeRuntime(tmp_path / "rt2", _grade_responses())
    _grade(tmp_path, monkeypatch, "student-2", runtime=rt2, key_path=key_path,
           exam_name="02_50.pdf")
    assert len(builds) == 1                       # student 2 REUSED the persisted packs


def test_rubric_change_invalidates_persisted_packs(tmp_path, monkeypatch):
    key = make_key()
    key_path = tmp_path / "answer_key.json"
    save_answer_key(key, key_path)
    rt = FakeRuntime(tmp_path, _grade_responses())
    _grade(tmp_path, monkeypatch, "s1", runtime=rt, key_path=key_path)
    fp1 = json.loads((tmp_path / "packs" / "manifest.json").read_text(encoding="utf-8"))

    key.questions[0].grading_notes = "a NEW rubric line the lecturer added"
    save_answer_key(key, key_path)                # key bytes change -> new fingerprint
    rt2 = FakeRuntime(tmp_path / "rt2", _grade_responses())
    _grade(tmp_path, monkeypatch, "s2", runtime=rt2, key_path=key_path,
           exam_name="02_50.pdf")
    fp2 = json.loads((tmp_path / "packs" / "manifest.json").read_text(encoding="utf-8"))
    assert fp1["source_fingerprint"] != fp2["source_fingerprint"]


def test_fingerprint_axes_cover_solution_index_and_rag_config():
    """Unit coverage of the invalidation axes the seam relies on: official
    solution (via key bytes), course index, retrieval config, RAG policy,
    pack schema version."""
    pol = {"1": "wrong_choice_zero"}
    base = gp.source_fingerprint(b"KEY", "idx-a", pol, 2, 1200)
    assert base != gp.source_fingerprint(b"KEY-solution-edited", "idx-a", pol, 2, 1200)
    assert base != gp.source_fingerprint(b"KEY", "idx-B", pol, 2, 1200)
    assert base != gp.source_fingerprint(b"KEY", "idx-a", pol, 3, 1200)
    assert base != gp.source_fingerprint(b"KEY", "idx-a", pol, 2, 800)
    assert base != gp.source_fingerprint(b"KEY", "idx-a", pol, 2, 1200,
                                         rag_policy="RAG_ON_UNCERTAIN")
    assert base != gp.source_fingerprint(b"KEY", "idx-a", pol, 2, 1200, pack_version="v1")


def test_prepared_rag_round_trips_through_the_store(tmp_path):
    """Lazy-policy packs persist their PREPARED chunks, so activation after a
    reload still needs no new retrieval."""
    from tests.test_gradingpack import CHUNKS, fake_retrieve_factory

    key = make_key()
    r = fake_retrieve_factory(CHUNKS)
    packs = gp.build_all_packs(key, {}, course_id="CV", retrieve=r,
                               rag_policy="RAG_ON_UNCERTAIN")
    n_retrievals = len(r.calls)
    store = gp.PackStore(tmp_path / "packs")
    fp = gp.source_fingerprint(b"K", "idx", {}, 2, 1200, rag_policy="RAG_ON_UNCERTAIN")
    store.save(packs, fp)
    loaded = store.load(fp)
    p = loaded["1"]
    assert [e.chunk_id for e in p.rag_prepared] == ["c0", "c1"]
    active = gp.activate_rag(p)
    assert [e.chunk_id for e in active.rag_evidence] == ["c0", "c1"]
    assert active.hash != p.hash                  # activation is a new identity
    assert len(r.calls) == n_retrievals           # no retrieval after reload
