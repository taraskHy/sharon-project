"""The SEEN-46 campaign: population, HELD_OUT seal, leakage, immutability.

Zero inference, zero network. HELD_OUT content is never read — the tests
assert ABSENCE by scanning campaign artifacts for forbidden writer ids.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.seen46_campaign import (CAMPAIGN_PATH, FORBIDDEN_WRITERS,  # noqa: E402
                                     LEAKAGE_PATH, build_campaign, verify_campaign)

pytestmark = pytest.mark.skipif(not CAMPAIGN_PATH.exists(),
                                reason="campaign not frozen in this checkout")

RUNNER = REPO / "scripts" / "run_local_grade_primary.ps1"


@pytest.fixture(scope="module")
def campaign():
    return json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))


def test_01_exactly_46_cases(campaign):
    assert len(campaign["cases"]) == 46
    assert campaign["population"]["total"] == 46


def test_02_split_counts(campaign):
    assert campaign["population"]["dev"] == 32
    assert campaign["population"]["calibration"] == 14
    assert campaign["population"]["held_out"] == 0
    assert sum(1 for c in campaign["cases"] if c["split"] == "DEV") == 32
    assert sum(1 for c in campaign["cases"] if c["split"] == "CALIBRATION") == 14
    assert not any(c["split"] == "HELD_OUT" for c in campaign["cases"])


def test_03_forbidden_writers_absent_everywhere(campaign):
    """e005/e006 must not appear ANYWHERE in campaign artifacts — not as a
    case, not in a string, not in the leakage artifact."""
    blob = CAMPAIGN_PATH.read_text(encoding="utf-8")
    for w in FORBIDDEN_WRITERS:
        assert w not in blob, f"{w} leaked into the campaign manifest"
    if LEAKAGE_PATH.exists():
        leak = LEAKAGE_PATH.read_text(encoding="utf-8")
        for w in FORBIDDEN_WRITERS:
            assert w not in leak
    for c in campaign["cases"]:
        assert c["writer"] in ("e002", "e003", "e004", "e007")


def test_04_runner_mode_runs_only_the_frozen_campaign():
    src = RUNNER.read_text(encoding="utf-8")
    assert "seen46_campaign.py" in src and "verify" in src
    assert "SEEN46_LEAKAGE_VERIFICATION" in src
    assert "ZERO-LEAKAGE VERIFIED" in src
    # the seen46 mode runs exactly dev + calibration, nothing else
    assert '@("dev", "calibration")' in src
    for banned in ("held_out", "held-out", "--research"):
        assert banned not in src.replace("HELD_OUT is not reachable", ""), banned


def test_05_zero_target_leakage_artifact_and_live_recheck(campaign):
    assert LEAKAGE_PATH.exists(), "run scripts/seen46_campaign.py leakage first"
    leak = json.loads(LEAKAGE_PATH.read_text(encoding="utf-8"))
    assert leak["verdict"] == "ZERO-LEAKAGE VERIFIED"
    assert leak["requests_verified"] == 46
    assert leak["problems"] == []
    # live spot-recheck: rebuild 3 requests and run the runner's leakage gate
    from autograder.benchmark.manifests import DEFAULT_BENCH_ROOT, load_manifest
    from autograder.benchmark.roles import GradeAdapter
    from autograder.benchmark.runner import files_root_for, leakage_check
    m = load_manifest("grade_primary")
    by_id = {c.case_id: c for c in m.cases}
    adapter = GradeAdapter("grade_primary", prompt_version=campaign["prompt_version"])
    root = files_root_for(m, DEFAULT_BENCH_ROOT)
    for cid in (campaign["cases"][0]["case_id"], campaign["cases"][31]["case_id"],
                campaign["cases"][45]["case_id"]):
        c = by_id[cid]
        req = adapter.build_request(dict(c.inputs), root)
        leakage_check(c, req, adapter.model_visible_fields)
        text = req.text_for_inspection()
        assert str(c.label.get("score")) not in ("", None)
        for banned in ("explanation_verdict", "selection_correct", "instructor",
                       "DEV", "CALIBRATION", "HELD_OUT"):
            assert banned not in text, (cid, banned)


def test_campaign_hash_is_self_consistent(campaign):
    payload = json.dumps({k: v for k, v in campaign.items() if k != "campaign_sha256"},
                         ensure_ascii=False, sort_keys=True)
    assert campaign["campaign_sha256"] == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert verify_campaign() == []


def test_targets_are_instructor_derived_and_audits_are_flags(campaign):
    t = campaign["targets"]
    assert t["verdict_derivable"] == 38
    assert t["non_derivable_diagnostic"] == 8
    assert t["strict_clean_denominator"] == 37       # derived, matches source
    assert t["evidence_issue_flagged"] == ["e004_q2_r8"]
    for c in campaign["cases"]:
        assert "actual_instructor_score" in c
        if c["audit_flag"] is not None:
            assert c["audit_flag"] in "ABCD"
            # a flag never replaces the derived verdict
            assert c["instructor_derived_verdict"] in ("valid", "partially_valid", "invalid", None)
    r8 = next(c for c in campaign["cases"] if c["case_id"] == "e004_q2_r8")
    assert r8["evidence_issue_flag"] is True
    assert r8["verdict_derivable"] is True and r8["strict_verdict_eligible"] is False
    assert r8["actual_instructor_score"] == 2.0


def test_campaign_pins_the_frozen_grader_configuration(campaign):
    import hashlib as h
    from autograder.escalation import (GRADE_VALIDATION_VERSION, GradeResult,
                                       grade_system_for)
    assert campaign["model"]["candidate"] == "qwen3-vl:8b-instruct"
    assert campaign["prompt_version"] == "grade-v4-charitable-local"
    assert campaign["prompt_sha256"] == h.sha256(
        grade_system_for("grade-v4-charitable-local").encode()).hexdigest()
    schema = json.dumps(GradeResult.model_json_schema(), sort_keys=True)
    assert campaign["schema_sha256"] == h.sha256(schema.encode()).hexdigest()
    assert campaign["validation_version"] == GRADE_VALIDATION_VERSION
    assert campaign["model"]["rag_policy"] == "RAG_DISABLED"
    assert "localhost" in campaign["model"]["base_url"]
