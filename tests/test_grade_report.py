"""The two-layer grading report (owner directive 2026-08-28).

Layer A (grader quality) is judged ONLY against instructor-derived verdicts;
Layer B (end-to-end) compares the system's predicted final score with the
ACTUAL instructor score from the original graded test. Blind A/B/C/D audit
decisions never define, exclude or relabel a target — they surface as flags.

Runs are synthetic (a run.json + scored.jsonl.json written into tmp_path);
the dataset is the real frozen grade_primary dataset. No provider is
contacted anywhere in this file.
"""

from __future__ import annotations

import json

import pytest

from autograder.benchmark.gradereport import (
    GradeReportError,
    build_grade_report,
    render_markdown,
    verify_targets_against_instructor,
)
from autograder.benchmark.manifests import BenchCase, load_manifest

ROLE = "grade_primary"

WRONG_SELECTION_DEV = {"e002_q1_r1", "e003_q1_r3", "e003_q2_r3",
                       "e003_q2_r4", "e003_q2_r5", "e003_q2_r6"}


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(ROLE)


def _derivable(manifest, split):
    return sorted(c.case_id for c in manifest.by_split(split)
                  if c.label.get("explanation_verdict_derivable"))


def _mk_run(tmp_path, manifest, verdicts: dict[str, str | None], *, split="DEV",
            subset="dev_verdict", name="run"):
    """A synthetic executed run: verdicts maps case_id -> predicted verdict
    (None = schema failure, no parseable output)."""
    d = tmp_path / name
    d.mkdir()
    scored = []
    for cid, v in verdicts.items():
        row = {"case_id": cid, "split": split, "component": "ALL",
               "schema_failure": v is None,
               "decision": "AUTO" if v is not None else "REVIEW"}
        if v is not None:
            row["predicted_verdict"] = v
        scored.append(row)
    (d / "run.json").write_text(json.dumps({
        "run_id": name, "git_commit": "test",
        "config": {"role": ROLE, "split": split, "subset": subset, "component": None,
                   "candidate": "test-model", "backend": "ollama",
                   "base_url": "http://localhost:11434/v1",
                   "prompt_version": "grade-v4-charitable",
                   "adapter_version": "grade-bench-v2",
                   "manifest_hashes": manifest.hashes}}), encoding="utf-8")
    (d / "scored.jsonl.json").write_text(json.dumps(scored), encoding="utf-8")
    return d


# --------------------------------------------------------- target provenance ----


def test_every_frozen_target_rederives_from_the_instructor_score(manifest):
    prov = verify_targets_against_instructor(manifest.cases)
    assert prov["verified_cases"] == 67
    assert "original_instructor_grade" in prov["instructor_score_source"]


def test_a_label_that_disagrees_with_the_instructor_derivation_refuses():
    tampered = BenchCase(
        case_id="x_q1_r1", split="DEV", component="ALL",
        inputs={"transcription": "text", "pack": {"max_score": 4.0}},
        label={"score": 4.0, "ground_truth_source": "original_instructor_grade",
               # full credit implies `valid`; an audit-style relabel must refuse
               "explanation_verdict": "partially_valid",
               "explanation_verdict_derivable": True,
               "selection_correct": None, "max_score": 4.0})
    with pytest.raises(GradeReportError, match="instructor-derived"):
        verify_targets_against_instructor([tampered])


def test_a_non_instructor_ground_truth_source_refuses():
    foreign = BenchCase(
        case_id="x_q1_r2", split="DEV", component="ALL",
        inputs={"transcription": "text", "pack": {"max_score": 4.0}},
        label={"score": 4.0, "ground_truth_source": "blind_audit_majority",
               "explanation_verdict": "valid", "explanation_verdict_derivable": True,
               "selection_correct": None, "max_score": 4.0})
    with pytest.raises(GradeReportError, match="ground_truth_source"):
        verify_targets_against_instructor([foreign])


# ------------------------------------------------------------------ layer A ----


def test_fulldev_layer_a_scores_only_the_26_derivable_cases(tmp_path, manifest):
    ids = _derivable(manifest, "DEV")
    assert len(ids) == 26
    run = _mk_run(tmp_path, manifest, {cid: "valid" for cid in ids})
    rep = build_grade_report(run, manifest=manifest)
    a = rep["layer_a_local_grader_quality"]
    assert a["population_derivable"] == 26
    assert a["class_support"] == {"valid": 22, "partially_valid": 4}
    assert a["scored"] == 26
    # all-valid predictions: the 22 valid cases match, the 4 partial are upgrades
    assert a["verdict_exact_pct"] == pytest.approx(100 * 22 / 26, abs=0.01)
    assert a["harmful_verdict_upgrades"] == 4
    assert a["harmful_verdict_downgrades"] == 0


def test_wrong_selection_cases_never_enter_layer_a_even_if_graded(tmp_path, manifest):
    ids = _derivable(manifest, "DEV")
    verdicts = {cid: "valid" for cid in ids}
    verdicts["e002_q1_r1"] = "valid"          # a wrong-selection case, graded anyway
    run = _mk_run(tmp_path, manifest, verdicts, subset=None)
    rep = build_grade_report(run, manifest=manifest)
    a = rep["layer_a_local_grader_quality"]
    assert a["scored"] == 26                   # not 27
    assert a["excluded_wrong_selection"] == 6
    assert not any(c["case_id"] == "e002_q1_r1" for c in a["cases"])
    row = next(r for r in rep["cases"] if r["case_id"] == "e002_q1_r1")
    assert row["layer_b_bucket"] == "policy_deterministic_zero"
    assert row["predicted_final"] == 0.0
    assert row["model_verdict_diagnostic_only"] is True


# ------------------------------------------------------------------ layer B ----


def test_fulldev_layer_b_end_to_end_math(tmp_path, manifest):
    ids = _derivable(manifest, "DEV")
    run = _mk_run(tmp_path, manifest, {cid: "valid" for cid in ids})
    rep = build_grade_report(run, manifest=manifest)
    b = rep["layer_b_end_to_end_test_grade_agreement"]
    assert b["population"] == 32
    fs = b["full_system"]
    # 26 model-scored + 6 deterministic zeros; the 4 partial-credit cases are
    # overgraded (predicted 4 vs instructor 2)
    assert fs["cases"] == 32
    assert fs["final_exact"] == 28
    assert fs["harmful_overgrades"] == 4
    assert fs["harmful_undergrades"] == 0
    assert fs["final_score_mae"] == pytest.approx((4 * 2.0) / 32)
    ms = b["model_scored_subpopulation"]
    assert ms["cases"] == 26 and ms["final_exact"] == 22
    det = b["wrong_selection_policy_report"]
    assert set(det["cases"]) == WRONG_SELECTION_DEV
    assert det["agreement"]["cases"] == 6 and det["agreement"]["final_exact"] == 6
    conf = b["confusion_by_actual_score"]
    assert conf["0"] == {"0": 6}
    assert conf["2"] == {"4": 4}
    assert conf["4"] == {"4": 22}


def test_partially_valid_prediction_maps_to_two_points(tmp_path, manifest):
    ids = _derivable(manifest, "DEV")
    run = _mk_run(tmp_path, manifest, {cid: "partially_valid" for cid in ids})
    rep = build_grade_report(run, manifest=manifest)
    fs = rep["layer_b_end_to_end_test_grade_agreement"]["full_system"]
    # the 4 partial cases now match; the 22 valid cases are undergraded 4 -> 2
    assert fs["final_exact"] == 6 + 4
    assert fs["harmful_undergrades"] == 22
    assert fs["harmful_overgrades"] == 0


def test_schema_failure_yields_no_automated_score(tmp_path, manifest):
    ids = _derivable(manifest, "DEV")
    verdicts = {cid: "valid" for cid in ids}
    verdicts[ids[0]] = None                    # schema failure
    run = _mk_run(tmp_path, manifest, verdicts)
    rep = build_grade_report(run, manifest=manifest)
    a, b = rep["layer_a_local_grader_quality"], rep["layer_b_end_to_end_test_grade_agreement"]
    assert a["scored"] == 25 and a["not_scored_schema_failure"] == 1
    assert b["no_automated_score_schema_failure"] == [ids[0]]
    assert b["full_system"]["cases"] == 31
    assert sum(b["confusion_by_actual_score"][k].get("no_automated_score", 0)
               for k in b["confusion_by_actual_score"]) == 1


def test_calibration_population_excludes_only_unresolved_selection(tmp_path, manifest):
    ids = _derivable(manifest, "CALIBRATION")
    assert len(ids) == 12
    run = _mk_run(tmp_path, manifest, {cid: "valid" for cid in ids},
                  split="CALIBRATION", subset="calibration_verdict_v4")
    rep = build_grade_report(run, manifest=manifest)
    b = rep["layer_b_end_to_end_test_grade_agreement"]
    assert b["population"] == 14
    assert b["excluded_selection_unresolved"] == ["e004_q2_r4", "e004_q2_r5"]
    assert b["full_system"]["cases"] == 12     # no deterministic zeros here


# ------------------------------------------------------- audit decisions ----


def test_audit_decisions_are_flags_only_and_exclude_nothing(tmp_path, manifest):
    ids = _derivable(manifest, "CALIBRATION")
    run = _mk_run(tmp_path, manifest, {cid: "valid" for cid in ids},
                  split="CALIBRATION", subset="calibration_verdict_v4")
    rep = build_grade_report(run, manifest=manifest)
    aud = rep["audit_decisions"]
    flags = {f["case_id"]: f for f in aud["flags_in_population"]}
    if not flags:
        pytest.skip("no frozen experiment audit block in this checkout")
    # the C-decided case is flagged, NOT excluded: it stays in the population
    assert flags["e004_q2_r8"]["flag"] == "evidence_transcription_concern"
    assert rep["layer_a_local_grader_quality"]["population_derivable"] == 12
    assert any(c["case_id"] == "e004_q2_r8" for c in rep["cases"])
    # A-decisions confirm the label and carry no flag
    assert flags["e004_q2_r6"]["flag"] is None
    assert "never replace, modify or determine an expected label" in aud["policy"]


def test_dev_report_carries_no_audit_flags(tmp_path, manifest):
    ids = _derivable(manifest, "DEV")
    run = _mk_run(tmp_path, manifest, {cid: "valid" for cid in ids})
    rep = build_grade_report(run, manifest=manifest)
    assert rep["audit_decisions"]["flags_in_population"] == []


# ------------------------------------------------------------- refusals ----


def test_refuses_a_run_whose_dataset_has_drifted(tmp_path, manifest):
    ids = _derivable(manifest, "DEV")
    run = _mk_run(tmp_path, manifest, {cid: "valid" for cid in ids})
    r = json.loads((run / "run.json").read_text(encoding="utf-8"))
    r["config"]["manifest_hashes"] = {"inputs_sha256": "0" * 64}
    (run / "run.json").write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(GradeReportError, match="no longer matches"):
        build_grade_report(run, manifest=manifest)


def test_refuses_a_dry_run_directory(tmp_path, manifest):
    d = tmp_path / "dry"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({
        "run_id": "dry", "config": {"role": ROLE, "split": "DEV",
                                    "manifest_hashes": manifest.hashes}}), encoding="utf-8")
    with pytest.raises(GradeReportError, match="scored.jsonl.json"):
        build_grade_report(d, manifest=manifest)


# ------------------------------------------------------------- rendering ----


def test_markdown_states_the_two_layers_and_the_provenance(tmp_path, manifest):
    ids = _derivable(manifest, "DEV")
    run = _mk_run(tmp_path, manifest, {cid: "valid" for cid in ids})
    rep = build_grade_report(run, manifest=manifest)
    md = render_markdown(rep)
    assert "A. LOCAL GRADER QUALITY" in md
    assert "B. END-TO-END TEST-GRADE AGREEMENT" in md
    assert "ACTUAL instructor-assigned grades" in md
    assert "audit decisions appear only as diagnostic flags" in md
    assert "never combined" in md
