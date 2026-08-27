"""The frozen local grade_primary experiment and its strong-PC runner.

Everything here is metadata verification: ZERO inference, zero network, no
model discovery (the Ollama listing lives in a function these tests never
call). HELD_OUT content is never read.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.local_grade_freeze import (FREEZE_PATH, HELD_OUT_WRITERS,  # noqa: E402
                                        build_freeze, verify_freeze)

RUNNER = REPO / "scripts" / "run_local_grade_primary.ps1"
pytestmark = pytest.mark.skipif(not FREEZE_PATH.exists(),
                                reason="freeze record not in this checkout")


@pytest.fixture(scope="module")
def frozen():
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------ the freeze ----


def test_the_freeze_matches_the_live_repository(frozen):
    assert verify_freeze() == []


def test_frozen_populations_are_the_scientifically_valid_ones(frozen):
    dev = frozen["populations"]["dev_verdict"]
    cal = frozen["populations"]["calibration_verdict_v4"]
    assert len(dev["case_ids"]) == 26
    assert dev["class_distribution"] == {"valid": 22, "partially_valid": 4}
    assert len(cal["case_ids"]) == 12
    assert cal["class_distribution"] == {"valid": 7, "partially_valid": 5}
    smoke = frozen["populations"]["smoke"]
    assert set(smoke["case_ids"]) <= set(dev["case_ids"]) | {"e002_q1_r8", "e007_q1_r1"}
    assert "invalid" not in dev["class_distribution"]
    assert "NOT MEASURED" in frozen["limitations"]["invalid_class"]


def test_no_held_out_writer_appears_anywhere(frozen):
    for pop in frozen["populations"].values():
        for cid in pop["case_ids"]:
            assert cid.split("_")[0] not in HELD_OUT_WRITERS, cid
    blob = json.dumps(frozen["populations"], ensure_ascii=False)
    for w in HELD_OUT_WRITERS:
        assert w not in blob


def test_prompt_and_schema_hashes_are_real(frozen):
    from autograder.escalation import GradeResult, grade_system_for
    assert frozen["prompt_version"] == "grade-v4-charitable"
    assert frozen["prompt_sha256"] == hashlib.sha256(
        grade_system_for("grade-v4-charitable").encode("utf-8")).hexdigest()
    schema = json.dumps(GradeResult.model_json_schema(), sort_keys=True)
    assert frozen["schema_sha256"] == hashlib.sha256(schema.encode("utf-8")).hexdigest()
    assert frozen["rag_policy"] == "RAG_DISABLED"


def test_audit_decisions_come_from_the_saved_artifact_and_c_is_flagged(frozen):
    audit = json.loads((REPO / frozen["human_audit"]["artifact"]).read_text(encoding="utf-8"))
    saved = {c["case_id"]: c["human_decision"] for c in audit["cases"]}
    assert frozen["human_audit"]["decisions"] == saved
    # a case decided C is an evidence/transcription problem, NOT a clean hard
    # target: it must be flagged for review and excluded from strict metrics
    c_cases = sorted(cid for cid, d in saved.items() if d == "C")
    assert frozen["human_audit"]["evidence_review_required"] == c_cases
    assert "excluded from strict-accuracy" in frozen["human_audit"]["strict_metrics_policy"]
    # and every C case actually sits in a frozen population (it is not
    # silently dropped either — it runs, it just doesn't count strictly)
    all_ids = set(frozen["populations"]["dev_verdict"]["case_ids"]) | \
        set(frozen["populations"]["calibration_verdict_v4"]["case_ids"])
    for cid in c_cases:
        assert cid in all_ids


def test_a_tampered_artifact_is_refused(tmp_path, frozen, monkeypatch):
    """§9-6: any hash/case-list drift makes verify_freeze report problems —
    which preflight turns into exit 2, refusing execution."""
    import copy

    import scripts.local_grade_freeze as fz
    tampered = copy.deepcopy(frozen)
    tampered["populations"]["dev_verdict"]["case_ids"] = \
        tampered["populations"]["dev_verdict"]["case_ids"][:-1]
    p = tmp_path / "freeze.json"
    p.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    problems = fz.verify_freeze(p)
    assert any("case_ids" in x for x in problems)


def test_no_machine_specific_paths_in_portable_artifacts(frozen):
    """§9-10: the freeze, the registry, the handoff and the runner carry no
    user-specific absolute paths."""
    pat = re.compile(r"[A-Za-z]:\\Users|/home/|/Users/|C:/Users", re.I)
    for f in (FREEZE_PATH,
              REPO / "evaluation" / "model_selection" / "candidates.toml",
              REPO / "docs" / "strong-pc-local-grading-handoff.md",
              RUNNER):
        text = f.read_text(encoding="utf-8")
        assert not pat.search(text), f


def test_the_freeze_hash_is_self_consistent(frozen):
    payload = json.dumps({k: v for k, v in frozen.items() if k != "experiment_sha256"},
                         ensure_ascii=False, sort_keys=True)
    assert frozen["experiment_sha256"] == hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------ preflight (zero) ----


def test_preflight_verification_makes_zero_network_calls(monkeypatch):
    """§9-1: the freeze/boundary/config checks complete with sockets disabled.
    (Model discovery is a separate function these checks never call.)"""
    import socket

    from scripts import local_grade_preflight as pf

    def _no(*a, **kw):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", _no)
    assert verify_freeze() == []
    b = pf.boundary_report()
    assert b["cloud_grading_blocked"] and b["local_grading_allowed"]
    assert b["cloud_ocr_allowlist"] == ["ocr_primary", "ocr_verify"]
    cfg = pf.models_toml_report()
    assert cfg["grade_routes_local"] is True


def test_preflight_module_cannot_infer_or_download():
    """No inference/download primitive exists in the preflight/freeze code:
    they never construct a backend, never call a gateway, never pull."""
    for f in ("scripts/local_grade_preflight.py", "scripts/local_grade_freeze.py"):
        src = (REPO / f).read_text(encoding="utf-8")
        for banned in ("gateway.call", "create_backend", "backend_for", ".parse(",
                       "ollama pull", "ollama run", "httpx.post", "requests.post"):
            assert banned not in src, (f, banned)


# ----------------------------------------------------------- the runner -----


def test_runner_defaults_to_zero_inference_and_gates_on_execute():
    src = RUNNER.read_text(encoding="utf-8")
    assert "[switch]$Execute" in src
    assert "-not $Execute" in src, "the plan-only exit must gate on -Execute"
    # the plan-only branch exits BEFORE the execution branch
    assert src.index("-not $Execute") < src.index("EXECUTING")
    # preflight runs before any execution and aborts on failure
    assert src.index("Invoke-Preflight") < src.index("EXECUTING")
    assert "$LASTEXITCODE -ne 0" in src


def test_runner_cannot_reach_held_out_or_research_mode():
    src = RUNNER.read_text(encoding="utf-8")
    assert '"--split", "dev"' in src, "split is hardcoded to dev"
    for banned in ("held_out", "held-out", "final-eval", "confirm-held-out", "--research"):
        assert banned not in src.replace("HELD_OUT is not reachable", ""), banned
    assert '"--backend", "ollama"' in src, "backend is hardcoded local"


def test_runner_requires_candidate_and_smoke_before_fulldev():
    src = RUNNER.read_text(encoding="utf-8")
    assert "-Smoke/-FullDev require an explicit -Candidate" in src
    assert "failure-free SMOKE run" in src
    assert "machine_profile" in src, "resource provenance is recorded per execution"


def test_runs_root_convention_documented():
    readme = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary" / "README.md"
    text = readme.read_text(encoding="utf-8")
    for needed in ("run.json", "outputs.jsonl", "metrics.json", "machine_profile",
                   "never overwritten", "e004_q2_r8"):
        assert needed in text, needed
