"""End-to-end offline tests: the full pipeline and batch evaluation must run
with a mocked backend, no network, no Anthropic key — and must never leak the
filename-encoded grade into model input or modify the source exams."""

import argparse
import hashlib
import json
from pathlib import Path

from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend, dump_fixture
from autograder.cli import _fingerprints, main, resolve_config, run_grade_pipeline
from autograder.key_parser import save_answer_key
from autograder.schema import (
    ExamSurvey,
    ExplanationEvaluation,
    ExplanationJudgement,
    PageInfo,
    QuestionExtraction,
    SubItemExtraction,
)
from tests.conftest import make_pdf
from tests.test_grade import make_key

# Deterministic outcome of the fixture set below against make_key() version A1:
# Q1: all 8 matching items correct with valid explanations -> 32
# Q3: items 9-15, 16 (C), 17-20 correct -> 12 * 2 = 24
EXPECTED_TOTAL = 56.0

_Q1_ANSWERS = {"1": "F", "2": "G", "3": "D", "4": "H", "5": "C", "6": "I", "7": "A", "8": "E"}


def _fixture_extraction() -> QuestionExtraction:
    subs = []
    for i in range(1, 21):
        sid = str(i)
        answer = _Q1_ANSWERS.get(sid, "C" if sid == "16" else "B")
        subs.append(
            SubItemExtraction(
                sub_item_id=sid,
                status="answered",
                final_answer=answer,
                explanation_transcription=(
                    f"student explanation {sid}" if int(sid) <= 8 else None
                ),
                explanation_legibility="full" if int(sid) <= 8 else "none",
                interpretation_rationale="clean single mark",
                confidence=1.0,
            )
        )
    return QuestionExtraction(
        question_id="1",
        source_pages=[1, 2],
        authoritative_source="answer table",
        sub_items=subs,
    )


def _fixture_survey() -> ExamSurvey:
    return ExamSurvey(
        pages=[
            PageInfo(page_number=1, content_summary="p1", question_ids=["1", "3"]),
            PageInfo(page_number=2, content_summary="p2", question_ids=["1", "3"]),
        ],
        student_ink_description="blue pen",
        grader_annotations_description="red ink scores (ignored)",
    )


def _fixture_judgement() -> ExplanationJudgement:
    return ExplanationJudgement(
        evaluations=[
            ExplanationEvaluation(sub_item_id=str(i), verdict="valid", reasoning="ok")
            for i in range(1, 9)
        ]
    )


def _build_fixtures(fixtures_dir: Path) -> Path:
    dump_fixture(make_key(), fixtures_dir)
    dump_fixture(_fixture_survey(), fixtures_dir)
    dump_fixture(_fixture_extraction(), fixtures_dir)
    dump_fixture(_fixture_judgement(), fixtures_dir)
    return fixtures_dir


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_pipeline_offline_via_cli(tmp_path, monkeypatch, no_network):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GRADER_API_KEY", raising=False)
    exam = make_pdf(tmp_path / "01_50.pdf")
    key_path = tmp_path / "answer_key.json"
    save_answer_key(make_key(), key_path)
    fixtures = _build_fixtures(tmp_path / "fixtures")
    out = tmp_path / "out"

    rc = main(
        [
            "grade",
            "--backend", "mock",
            "--model", str(fixtures),
            "--key", str(key_path),
            "--exam", str(exam),
            "--out", str(out),
        ]
    )
    assert rc == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["total_awarded"] == EXPECTED_TOTAL
    assert result["detected_version"] == "A1"
    assert result["model"].startswith("mock:")
    assert result["backend_info"]["backend"] == "mock"
    assert (out / "report.md").exists()


def test_grade_and_filename_never_reach_model_input(tmp_path, no_network):
    exam = make_pdf(tmp_path / "01_50.pdf")
    key_path = tmp_path / "answer_key.json"
    save_answer_key(make_key(), key_path)

    fixtures = {
        "AnswerKey": make_key(),
        "ExamSurvey": _fixture_survey(),
        "QuestionExtraction": _fixture_extraction(),
        "ExplanationJudgement": _fixture_judgement(),
    }
    backend = MockBackend(
        config=BackendConfig(backend="mock", model="recording"),
        responder=lambda model, system, blocks: fixtures[model.__name__].model_copy(deep=True),
    )
    ns = argparse.Namespace(
        key=str(key_path), rubric=None, resume=False, version="auto", exam=str(exam)
    )
    result = run_grade_pipeline(
        ns, backend, tmp_path / "out", 800, exam_path=exam, exam_label="exam-001"
    )
    assert result.exam_file == "exam-001"
    assert backend.calls, "pipeline made no model calls?"
    for call in backend.calls:
        text = call.all_text()
        assert "01_50" not in text, f"filename leaked into {call.output_model} input"
        assert str(exam) not in text, "exam path leaked into model input"
    # The label comparison target (grade 50) is likewise absent as a filename
    # token; result artefacts use the anonymized label only.
    assert result.total_awarded == EXPECTED_TOTAL


def test_crash_during_extraction_resumes_without_redoing_survey(tmp_path, no_network):
    """A crash after the survey stage must not discard it: each stage's
    fingerprint is persisted the moment its file is written, so --resume
    re-runs only the unfinished stages."""
    exam = make_pdf(tmp_path / "01_50.pdf")
    key_path = tmp_path / "answer_key.json"
    save_answer_key(make_key(), key_path)
    out = tmp_path / "out"

    fixtures = {
        "ExamSurvey": _fixture_survey(),
        "QuestionExtraction": _fixture_extraction(),
        "ExplanationJudgement": _fixture_judgement(),
    }

    calls = {"survey": 0, "extraction": 0}

    def crashing(model, system, blocks):
        if model.__name__ == "ExamSurvey":
            calls["survey"] += 1
        if model.__name__ == "QuestionExtraction":
            raise RuntimeError("simulated crash mid-extraction")
        return fixtures[model.__name__].model_copy(deep=True)

    backend = MockBackend(
        config=BackendConfig(backend="mock", model="crashing"), responder=crashing
    )
    ns = argparse.Namespace(
        key=str(key_path), rubric=None, resume=True, version="auto", exam=str(exam)
    )
    try:
        run_grade_pipeline(ns, backend, out, 800, exam_path=exam)
        raise AssertionError("expected the simulated crash")
    except RuntimeError:
        pass
    assert (out / "survey.json").exists()
    assert calls["survey"] == 1
    stored = json.loads((out / "fingerprint.json").read_text(encoding="utf-8"))
    assert "survey" in stored, "survey fingerprint must be persisted before extraction"

    def counting(model, system, blocks):
        calls.setdefault(model.__name__, 0)
        if model.__name__ == "ExamSurvey":
            calls["survey"] += 1
        if model.__name__ == "QuestionExtraction":
            calls["extraction"] += 1
        return fixtures[model.__name__].model_copy(deep=True)

    backend2 = MockBackend(
        config=BackendConfig(backend="mock", model="crashing"), responder=counting
    )
    result = run_grade_pipeline(ns, backend2, out, 800, exam_path=exam)
    assert result.total_awarded == EXPECTED_TOTAL
    assert calls["survey"] == 1, "survey must be reused from disk, not recomputed"
    assert calls["extraction"] > 0, "extraction must run on resume"


def test_eval_batch_offline_end_to_end(tmp_path, monkeypatch, no_network):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    dataset = tmp_path / "dataset"
    exams = [make_pdf(dataset / "01_50.pdf"), make_pdf(dataset / "02_80.pdf")]
    hashes_before = {p.name: _sha(p) for p in exams}
    key_path = tmp_path / "answer_key.json"
    save_answer_key(make_key(), key_path)
    fixtures = _build_fixtures(tmp_path / "fixtures")
    manifest_dir = tmp_path / "datasets"
    eval_out = tmp_path / "eval_out"

    assert main(
        ["make-manifests", "--dataset-root", str(dataset), "--manifest-dir", str(manifest_dir)]
    ) == 0

    rc = main(
        [
            "eval-batch",
            "--backend", "mock",
            "--model", str(fixtures),
            "--key", str(key_path),
            "--manifest-dir", str(manifest_dir),
            "--split", "all",
            "--out", str(eval_out),
        ]
    )
    assert rc == 0

    combined = json.loads((eval_out / "combined_results.json").read_text(encoding="utf-8"))
    assert combined["metrics"]["processed"] == 2
    assert combined["metrics"]["failures"] == 0
    by_id = {e["anon_id"]: e for e in combined["exams"]}
    assert by_id["exam-001"]["predicted"] == EXPECTED_TOTAL
    assert by_id["exam-001"]["expected"] == 50.0
    assert by_id["exam-002"]["expected"] == 80.0
    assert (eval_out / "summary.md").exists()
    assert (eval_out / "combined_results.csv").exists()
    assert (eval_out / "failed_exams.json").exists()
    assert (eval_out / "exams" / "exam-001" / "masking.json").exists()

    # Source exams must be byte-identical after evaluation.
    for p in exams:
        assert _sha(p) == hashes_before[p.name], f"{p.name} was modified!"

    # No output path may carry the grade-bearing source filename.
    for path in eval_out.rglob("*"):
        assert "01_50" not in path.name and "02_80" not in path.name


def test_eval_batch_continues_after_individual_failure(tmp_path, monkeypatch, no_network):
    dataset = tmp_path / "dataset"
    make_pdf(dataset / "01_50.pdf")
    # 02 is a corrupt "PDF" that will fail to load — the batch must continue.
    bad = dataset / "02_80.pdf"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a pdf")
    key_path = tmp_path / "answer_key.json"
    save_answer_key(make_key(), key_path)
    fixtures = _build_fixtures(tmp_path / "fixtures")
    manifest_dir = tmp_path / "datasets"
    eval_out = tmp_path / "eval_out"

    main(["make-manifests", "--dataset-root", str(dataset), "--manifest-dir", str(manifest_dir)])
    rc = main(
        [
            "eval-batch",
            "--backend", "mock", "--model", str(fixtures),
            "--key", str(key_path),
            "--manifest-dir", str(manifest_dir),
            "--split", "all",
            "--out", str(eval_out),
        ]
    )
    assert rc == 0
    combined = json.loads((eval_out / "combined_results.json").read_text(encoding="utf-8"))
    assert combined["metrics"]["processed"] == 2
    assert combined["metrics"]["failures"] == 1
    failed = json.loads((eval_out / "failed_exams.json").read_text(encoding="utf-8"))
    assert len(failed) == 1 and failed[0]["anon_id"] == "exam-002"
    ok = [e for e in combined["exams"] if not e["failed"]]
    assert len(ok) == 1 and ok[0]["predicted"] == EXPECTED_TOTAL


def test_fingerprint_changes_with_backend_config(tmp_path):
    key_path = tmp_path / "answer_key.json"
    save_answer_key(make_key(), key_path)
    exam = make_pdf(tmp_path / "01_50.pdf")
    ns = argparse.Namespace(key=str(key_path), rubric=None, exam=str(exam))

    b1 = MockBackend(config=BackendConfig(backend="mock", model="model-a"))
    b2 = MockBackend(config=BackendConfig(backend="mock", model="model-b"))
    b3 = MockBackend(config=BackendConfig(backend="mock", model="model-a", temperature=0.7))

    fp1 = _fingerprints(ns, b1, 800, include_exam=True, exam_path=exam)
    fp2 = _fingerprints(ns, b2, 800, include_exam=True, exam_path=exam)
    fp3 = _fingerprints(ns, b3, 800, include_exam=True, exam_path=exam)
    fp1_again = _fingerprints(ns, b1, 800, include_exam=True, exam_path=exam)

    assert fp1 == fp1_again
    assert fp1["exam"] != fp2["exam"], "different model must invalidate resume"
    assert fp1["exam"] != fp3["exam"], "different generation config must invalidate resume"
    assert fp1["key"] != fp2["key"]

    other_exam = make_pdf(tmp_path / "02_80.pdf", pages=3)
    fp_other = _fingerprints(ns, b1, 800, include_exam=True, exam_path=other_exam)
    assert fp1["exam"] != fp_other["exam"], "different exam must invalidate resume"

    other_key = tmp_path / "other_key.json"
    key2 = make_key()
    key2.exam_title = "a different exam form"
    save_answer_key(key2, other_key)
    ns_other_key = argparse.Namespace(key=str(other_key), rubric=None, exam=str(exam))
    fp_other_key = _fingerprints(ns_other_key, b1, 800, include_exam=True, exam_path=exam)
    assert fp1["key"] != fp_other_key["key"], "different key must invalidate resume"
    assert fp1["exam"] != fp_other_key["exam"]


def test_resolve_config_toml_and_cli_precedence(tmp_path):
    cfg = tmp_path / "grader.toml"
    cfg.write_text(
        '[backend]\nbackend = "openai"\nmodel = "toml-model"\n'
        'base_url = "http://toml:1234/v1"\ntimeout_s = 111.0\n'
        "[grading]\nmax_image_edge = 1111\nmax_tokens = 2222\nsurvey_image_edge = 333\n",
        encoding="utf-8",
    )
    ns = argparse.Namespace(
        config=str(cfg),
        backend=None,
        model="cli-model",  # CLI wins over TOML
        base_url=None,
        api_key_env=None,
        structured_mode=None,
        max_tokens=None,
        temperature=None,
        timeout=None,
        transport_retries=None,
        validation_retries=None,
        max_image_edge=None,
        survey_image_edge=None,
    )
    backend_config, max_edge, survey_edge = resolve_config(ns)
    assert backend_config.model == "cli-model"
    assert backend_config.base_url == "http://toml:1234/v1"
    assert backend_config.timeout_s == 111.0
    assert backend_config.max_tokens == 2222
    assert max_edge == 1111
    assert survey_edge == 333

    # CLI overrides TOML; absent everywhere -> library default.
    ns.survey_image_edge = 555
    assert resolve_config(ns)[2] == 555
    cfg.write_text("[backend]\nbackend = \"openai\"\nmodel = \"m\"\n", encoding="utf-8")
    ns.survey_image_edge = None
    from autograder.config import GraderConfig

    assert resolve_config(ns)[2] == GraderConfig.survey_image_long_edge


def test_doctor_with_mock_backend(no_network):
    assert main(["doctor", "--backend", "mock", "--model", "anything"]) == 0
