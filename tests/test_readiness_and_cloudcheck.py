"""Zero-key dry run (`autograder readiness`) and friendly cloud refusals."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autograder.cli import main
from autograder.cloudcheck import CloudNotReady, explain_cloud_error, require_cloud_task
from autograder.gateway import GatewayConfigError, ModelGateway
from autograder.readiness import format_readiness, readiness_report, role_status

REPO = Path(__file__).resolve().parents[1]


def test_readiness_report_is_complete_and_makes_no_network_calls(tmp_path, monkeypatch, no_network):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GRADER_JOBS_DIR", str(tmp_path / "jobs"))
    rep = readiness_report(models_config=REPO / "models.example.toml", state_root=tmp_path / "state")
    for section in ("course_store", "exam_packages", "benchmarks", "model_roles", "local_models", "rag_index",
                    "openrouter", "budget", "verifier_crops", "gui", "held_out", "blockers"):
        assert section in rep, section
    assert rep["network_calls"] == 0
    assert rep["openrouter"]["configured"] == "NO"
    assert rep["openrouter"]["calls_made_by_this_check"] == 0
    assert "NOT called" in rep["openrouter"]["key_metadata_endpoint"]
    assert rep["benchmarks"]["ocr_verify"]["status"] == "FROZEN"
    assert rep["benchmarks"]["ocr_primary"]["status"] == "FROZEN"
    assert rep["benchmarks"]["grade_primary"]["status"] in ("FROZEN", "NOT_BUILT")
    roles = rep["model_roles"]["tasks"]
    assert roles["grade_primary"]["status"] == "UNSELECTED"
    assert roles["mc_resolve"]["status"] == "SELECTED_LOCAL"
    assert rep["local_models"]["probed"] is False
    assert rep["budget"]["policy"] == {"warning_usd": 8.0, "hard_stop_usd": 10.0,
                                       "source": "evaluation/model_selection/candidates.toml [budget]"}
    assert rep["budget"]["status"]["state"] == "OK"
    assert rep["verifier_crops"]["status"] == "UNAVAILABLE"
    assert rep["held_out"]["untouched"] is True and rep["held_out"]["executions"] == 0
    assert rep["gui"]["screens"] == ["Dashboard", "Exam setup", "Grading progress", "Review queue",
                                     "Results / export", "Advanced / diagnostics"]
    cats = {c["category"]: c for c in rep["categories"]}
    assert cats["API KEY"]["status"] == "NOT INSTALLED" and cats["API KEY"]["ok"] is True
    assert cats["REFERENCE GROUND TRUTH"]["status"] == "OK"
    assert cats["OCR_PRIMARY DATASET"]["status"] == "READY" and "smoke frozen" in cats["OCR_PRIMARY DATASET"]["detail"]
    assert cats["OCR_VERIFY DATASET"]["status"] == "READY"
    assert cats["HELD_OUT PROTECTION"]["ok"] is True
    assert cats["PRODUCTION VERIFIER"]["status"] == "BENCHMARK READY / PRODUCTION SAFE BUT UNAVAILABLE"
    # the missing key NEVER appears as a blocker; the example config (with ${ENV} slugs) does
    assert not any("credential" in b.lower() for b in rep["blockers"])
    assert any("MODEL CONFIG" in b for b in rep["blockers"])
    assert rep["ready_for_zero_key_dry_run"] is True
    text = format_readiness(rep)
    assert "PRE-API SETUP COMPLETE:" in text and "READY FOR API KEY:" in text
    assert "OpenRouter        : configured: NO" in text and "network calls     : 0" in text
    json.dumps(rep, default=str)          # serializable


def test_readiness_cli_json(tmp_path, monkeypatch, capsys, no_network):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GRADER_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["readiness", "--json", "--models-config", str(REPO / "models.example.toml"),
               "--state-root", str(tmp_path / "state")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["openrouter"]["configured"] == "NO" and out["network_calls"] == 0


def test_role_status_marks_unset_env_slugs_unselected_without_echoing_values(monkeypatch):
    monkeypatch.delenv("OCR_PRIMARY_MODEL", raising=False)
    monkeypatch.setenv("GRADE_PRIMARY_MODEL", "vendor/some-model")
    rs = role_status(REPO / "models.example.toml")
    assert rs["tasks"]["ocr_primary"]["status"] == "UNSELECTED"
    assert rs["tasks"]["ocr_primary"]["model"] == "${OCR_PRIMARY_MODEL} (unset)"
    # the 2026-08 architecture: grading is LOCAL and unselected until the
    # local benchmark picks a winner; the env slug belongs to the research
    # config only and no longer configures production grading
    assert rs["tasks"]["grade_primary"]["status"] == "UNSELECTED"
    assert rs["tasks"]["grade_primary"]["cloud"] is False
    assert "vendor/some-model" not in json.dumps(rs)      # env values are never echoed


def test_role_status_blocks_cloud_grading_configs_for_the_gui(tmp_path, monkeypatch):
    """A models.toml that routes grading to the cloud shows the role as
    BLOCKED_IN_PRODUCTION (the GUI renders DISABLED IN PRODUCTION) — the
    boundary refuses it at call time anyway."""
    monkeypatch.setenv("GRADE_PRIMARY_MODEL", "vendor/some-model")
    rs = role_status(REPO / "models.research.example.toml")
    g = rs["tasks"]["grade_primary"]
    assert g["cloud"] is True and g["blocked_in_production"] is True
    assert g["status"] == "BLOCKED_IN_PRODUCTION"
    o = rs["tasks"]["ocr_primary"]
    assert o["blocked_in_production"] is False


def test_friendly_messages_for_unselected_and_missing_credential(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    gw = ModelGateway.from_dict({"models": {
        "grade_primary": {"backend": "openrouter", "model": "UNSELECTED"},
        "ocr_verify": {"backend": "openrouter", "model": "vendor/x"},
    }})
    with pytest.raises(CloudNotReady) as ei:
        require_cloud_task(gw, "grade_primary")
    assert str(ei.value) == "grade_primary model is not selected" and ei.value.code == "UNSELECTED"
    with pytest.raises(CloudNotReady) as ei:
        require_cloud_task(gw, "ocr_verify")
    assert str(ei.value).startswith("OpenRouter credential is not configured") and ei.value.code == "NO_CREDENTIAL"
    with pytest.raises(CloudNotReady) as ei:
        require_cloud_task(gw, "mc_resolve_cloud")
    assert "is not configured in models.toml" in str(ei.value)
    assert explain_cloud_error(GatewayConfigError("task 'ocr_primary' is UNSELECTED: no model")).task == "ocr_primary"
    assert explain_cloud_error(ValueError("something else")) is None


def test_cli_grade_in_reliability_mode_fails_with_a_sentence_not_a_trace(tmp_path, monkeypatch, capsys):
    """A cloud-dependent CLI operation with an UNSELECTED role exits 2 with
    the one-sentence explanation first (no stack trace)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import autograder.cli as cli
    from autograder.reliability import GradingModeError

    def _boom(args):
        raise GradingModeError("grading mode 'reliability' cannot start: required model role 'grade_primary' "
                               "is not usable (task 'grade_primary' is UNSELECTED: no model has been chosen)")
    monkeypatch.setattr(cli, "cmd_grade", _boom)     # build_parser binds the module global at call time
    rc = main(["grade", "--exam", str(tmp_path / "x.pdf"), "--key", str(tmp_path / "k.pdf"),
               "--backend", "mock", "--model", "m"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "ERROR: grade_primary model is not selected" in out
    assert "detail:" in out and "Traceback" not in out
