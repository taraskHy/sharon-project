"""Shadow-log robustness, concurrency, the SEEN-46 shadow artifacts, and the
admin-only diagnostics endpoint.

Temp files/dirs only. The live review DB and the live labeling DB are never
opened; the review46 admin tests run in-process against a temp data dir.
No model / provider / network call.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder import riskengine as re_  # noqa: E402

RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
DATE = "2026-09-02"
SHADOW_JSONL = RUNS / f"SHADOW_REPLAY_{DATE}.jsonl"
REPLAY_JSON = RUNS / f"PROSPECTIVE_POLICY_REPLAY_{DATE}.json"

NOW = "2026-09-02 00:00:00"
CLEAN = {"semantic_verdict": "valid", "schema_ok": True, "evidence_ok": True,
         "validation_ok": True, "uncertain": False,
         "transcription_complete": True, "source_integrity": "current",
         "model_output_current": True, "local_grader_available": True,
         "model_digest": "d", "prompt_version": "grade-v4-charitable-local",
         "prompt_sha256": "p", "schema_sha256": "s",
         "validation_version": "grade-validation-v2"}

needs_artifacts = pytest.mark.skipif(not SHADOW_JSONL.exists(),
                                     reason="shadow replay artifact absent")


def _event(i: int, eng, extra_offline=None):
    d = re_.ProspectiveDecisionInput.from_mapping(
        {**CLEAN, "model_digest": f"d{i}"})
    dec = eng.decide(d, now=NOW)
    return re_.build_shadow_event(f"case_{i:04d}", "run", d, dec, extra_offline)


# ------------------------------------------------------------ shadow log ----


def test_shadow_log_appends_and_skips_duplicates(tmp_path):
    eng = re_.build_engine(mode="shadow", policy_id="prospective_valid_only_v1")
    log = re_.ShadowLog(tmp_path / "log.jsonl")
    ev = _event(1, eng)
    assert log.append(ev) is True
    assert log.append(dict(ev)) is False              # idempotent duplicate
    assert len(log.events()) == 1
    # a reopened log preserves idempotency (replayed import)
    log2 = re_.ShadowLog(tmp_path / "log.jsonl")
    assert log2.append(dict(ev)) is False
    assert len(log2.events()) == 1


def test_malformed_events_and_logs_are_typed_refusals(tmp_path):
    log = re_.ShadowLog(tmp_path / "log.jsonl")
    with pytest.raises(re_.ShadowLogError, match="malformed"):
        log.append({"event_id": "x"})                 # wrong event_version
    with pytest.raises(re_.ShadowLogError, match="malformed"):
        log.append({"event_version": re_.SHADOW_EVENT_VERSION})   # no id
    # interrupted export: a truncated final line names its line number
    p = tmp_path / "trunc.jsonl"
    eng = re_.build_engine(mode="shadow", policy_id="prospective_valid_only_v1")
    good = json.dumps(_event(1, eng), ensure_ascii=False)
    p.write_text(good + "\n" + good[: len(good) // 2], encoding="utf-8")
    with pytest.raises(re_.ShadowLogError, match="line 2"):
        re_.ShadowLog(p)


def test_hebrew_rtl_content_round_trips_in_events(tmp_path):
    eng = re_.build_engine(mode="shadow", policy_id="prospective_valid_only_v1")
    log = re_.ShadowLog(tmp_path / "log.jsonl")
    hebrew = "יש טשטוש בכל התדריך — הסבר חלקי בלבד" * 40   # long RTL text
    ev = _event(7, eng, extra_offline={"offline_only": True, "note": hebrew})
    assert log.append(ev) is True
    back = re_.ShadowLog(tmp_path / "log.jsonl").events()[0]
    assert back["offline_evaluation"]["note"] == hebrew


def test_read_only_log_is_a_typed_error_with_no_partial_write(tmp_path):
    eng = re_.build_engine(mode="shadow", policy_id="prospective_valid_only_v1")
    p = tmp_path / "ro.jsonl"
    log = re_.ShadowLog(p)
    assert log.append(_event(1, eng)) is True
    before = p.read_bytes()
    os.chmod(p, stat.S_IREAD)
    try:
        with pytest.raises(re_.ShadowLogError, match="not writable"):
            log.append(_event(2, eng))
    finally:
        os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
    assert p.read_bytes() == before                    # recoverable state


def test_concurrent_writers_produce_every_event_exactly_once(tmp_path):
    eng = re_.build_engine(mode="shadow", policy_id="prospective_noninvalid_v1")
    log = re_.ShadowLog(tmp_path / "log.jsonl")
    events = [_event(i, eng) for i in range(400)]
    errors: list[Exception] = []

    def worker(chunk):
        try:
            for ev in chunk:
                log.append(dict(ev))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(events[i::8],))
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    back = re_.ShadowLog(tmp_path / "log.jsonl").events()
    assert len(back) == 400
    assert len({e["event_id"] for e in back}) == 400
    # every line parses on its own — no interleaving corruption
    for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_racing_duplicate_appends_write_exactly_one_copy(tmp_path):
    eng = re_.build_engine(mode="shadow", policy_id="prospective_valid_only_v1")
    log = re_.ShadowLog(tmp_path / "log.jsonl")
    ev = _event(99, eng)
    results: list[bool] = []
    threads = [threading.Thread(
        target=lambda: results.append(log.append(dict(ev))))
        for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1 and results.count(False) == 15
    assert len(re_.ShadowLog(tmp_path / "log.jsonl").events()) == 1


# ------------------------------------------- the SEEN-46 shadow artifacts ----


@needs_artifacts
def test_shadow_replay_has_one_event_per_case_per_prospective_policy():
    events = re_.ShadowLog(SHADOW_JSONL).events()
    assert len(events) == 138
    seen = {(e["case_id"], e["decision"]["policy_id"]) for e in events}
    assert len(seen) == 138
    policies = {e["decision"]["policy_id"] for e in events}
    assert policies == {"prospective_valid_only_v1", "prospective_noninvalid_v1",
                        "prospective_auto_all_structurally_valid_v1"}
    for e in events:
        assert e["decision"]["policy_scope"] in ("PROSPECTIVE_DEPLOYABLE",
                                                 "ANALYSIS_BASELINE_ONLY")
        assert e["decision"]["mode"] == "shadow"


@needs_artifacts
def test_decision_inputs_carry_no_post_hoc_fields():
    allowed = set(re_.ProspectiveDecisionInput.__dataclass_fields__)
    for e in re_.ShadowLog(SHADOW_JSONL).events():
        assert set(e["decision_input"]) == allowed
        assert e["offline_evaluation"]["offline_only"] is True


@needs_artifacts
def test_deleting_offline_evaluation_fields_leaves_every_decision_unchanged():
    events = re_.ShadowLog(SHADOW_JSONL).events()
    engines = {p: re_.build_engine(mode="shadow", policy_id=p)
               for p in {e["decision"]["policy_id"] for e in events}}
    for e in events:
        stripped = {k: v for k, v in e.items() if k != "offline_evaluation"}
        stripped["offline_evaluation"] = None
        redecided = re_.replay_decision_from_event(
            stripped, engines[e["decision"]["policy_id"]])
        assert redecided.to_dict() == e["decision"], e["case_id"]


@needs_artifacts
def test_prospective_replay_artifact_matches_the_shadow_events():
    doc = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    events = re_.ShadowLog(SHADOW_JSONL).events()
    for pol, m in doc["deployable_prospective"].items():
        evs = [e for e in events if e["decision"]["policy_id"] == pol]
        auto = [e for e in evs if e["decision"]["action"] == "AUTO"]
        assert len(evs) == 46 and len(auto) == m["auto"]
        assert m["auto"] + m["review"] == 46
        risk = sum(e["offline_evaluation"]["strict_weighted_loss"]
                   for e in auto)
        assert risk == m["auto_total_weighted_loss"]
        severe = sum(1 for e in auto
                     if e["offline_evaluation"]["severe_invalid_to_valid"])
        assert severe == m["invalid_to_valid_auto"] == 0


@needs_artifacts
def test_rare_event_bounds_in_the_artifact_are_exact():
    doc = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    for pol, m in doc["deployable_prospective"].items():
        ru = m["rare_event_uncertainty"]["invalid_to_valid"]
        assert ru["observed"] == 0 and ru["denominator"] == 5
        assert abs(ru["one_sided_upper_95"] - 0.450722) < 1e-4
    mn = doc["minimum_invalid_examples_for_zero_event_upper_bound"]
    assert (mn["10pct"], mn["5pct"], mn["2pct"], mn["1pct"]) == (29, 59, 149, 299)


@needs_artifacts
def test_oracle_tables_are_marked_not_deployable():
    doc = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    assert "oracle_retrospective_upper_bound_NOT_DEPLOYABLE" in doc
    for pol in doc["oracle_retrospective_upper_bound_NOT_DEPLOYABLE"]:
        assert re_.POLICY_REGISTRY[pol].scope == "RETROSPECTIVE_HUMAN_ASSISTED"
    md = (RUNS / f"PROSPECTIVE_POLICY_REPLAY_{DATE}.md").read_text(encoding="utf-8")
    assert "ORACLE-ASSISTED RETROSPECTIVE UPPER BOUND — NOT DEPLOYABLE" in md


# --------------------------------------------- admin diagnostics endpoint ---

BUNDLE = None
try:
    from review46_app import default_data_dir
    BUNDLE = default_data_dir() / "bundle"
except Exception:  # noqa: BLE001
    pass

needs_bundle = pytest.mark.skipif(
    BUNDLE is None or not (BUNDLE / "bundle46.json").exists(),
    reason="review46 bundle not built on this machine")


@pytest.fixture()
def shadow_dir(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir()
    if SHADOW_JSONL.exists():
        (d / SHADOW_JSONL.name).write_text(
            SHADOW_JSONL.read_text(encoding="utf-8"), encoding="utf-8")
        (d / REPLAY_JSON.name).write_text(
            REPLAY_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    return d


@needs_bundle
@needs_artifacts
def test_admin_shadow_endpoint_is_gated_and_serves_diagnostics(tmp_path,
                                                               shadow_dir):
    from starlette.testclient import TestClient
    from review46_app.app import create_app
    app = create_app(data_dir=tmp_path / "data", bundle_dir=BUNDLE,
                     admin_key="TESTKEY", shadow_artifacts_dir=shadow_dir)
    anon = TestClient(app)
    assert anon.get("/api/admin/shadow").status_code == 403       # reviewers: no
    admin = TestClient(app, headers={"x-admin-key": "TESTKEY"})
    r = admin.get("/api/admin/shadow")
    assert r.status_code == 200
    body = r.json()
    assert body["policy_mode"] == "shadow"
    assert body["engine_version"] == re_.RISK_ENGINE_VERSION
    assert "NOT deployable" in body["oracle_warning"] or \
        "NOT" in body["oracle_warning"]
    assert set(body["deployable_prospective"]) == {
        "prospective_valid_only_v1", "prospective_noninvalid_v1",
        "prospective_auto_all_structurally_valid_v1"}
    # per-case drill-down
    case = next(iter(body["deployable_prospective"]))
    some_case = json.loads(SHADOW_JSONL.read_text(encoding="utf-8")
                           .splitlines()[0])["case_id"]
    r2 = admin.get(f"/api/admin/shadow?case_id={some_case}")
    evs = r2.json()["case_events"]
    assert len(evs) == 3
    for ev in evs:
        assert ev["prospective_action"] in ("AUTO", "REVIEW")
        assert "offline_evaluation_admin_only" in ev


@needs_bundle
def test_admin_shadow_endpoint_fails_closed(tmp_path):
    from starlette.testclient import TestClient
    from review46_app.app import create_app
    app = create_app(data_dir=tmp_path / "data", bundle_dir=BUNDLE,
                     admin_key="TESTKEY")           # not configured
    admin = TestClient(app, headers={"x-admin-key": "TESTKEY"})
    assert admin.get("/api/admin/shadow").status_code == 404
    empty = tmp_path / "empty"
    empty.mkdir()
    app2 = create_app(data_dir=tmp_path / "data2", bundle_dir=BUNDLE,
                      admin_key="TESTKEY", shadow_artifacts_dir=empty)
    admin2 = TestClient(app2, headers={"x-admin-key": "TESTKEY"})
    assert admin2.get("/api/admin/shadow").status_code == 404
    # a tampered artifact is refused, never served
    bad = {"provenance": {}, "content_sha256": "wrong"}
    (empty / "PROSPECTIVE_POLICY_REPLAY_2026-09-02.json").write_text(
        json.dumps(bad), encoding="utf-8")
    assert admin2.get("/api/admin/shadow").status_code == 409


@needs_bundle
@needs_artifacts
def test_reviewer_payload_still_carries_no_shadow_or_model_data(tmp_path,
                                                                shadow_dir):
    from starlette.testclient import TestClient
    from review46_app.app import create_app
    app = create_app(data_dir=tmp_path / "data", bundle_dir=BUNDLE,
                     admin_key="TESTKEY", shadow_artifacts_dir=shadow_dir)
    c = TestClient(app)
    c.post("/api/session", json={"name": "Blind Tester"})
    r = c.post("/api/next")
    item = r.json().get("item") or {}
    text = json.dumps(item, ensure_ascii=False)
    for banned in ("shadow", "policy_id", "risk", "semantic_verdict",
                   "reference_verdict", "model_verdict", "AUTO_GROUNDED"):
        assert banned not in text, banned
