"""Model-selection benchmark harness (autograder/benchmark) — offline only.

Covers: frozen manifest loading + hash verification, split enforcement,
held-out confirmation + permanent log, verifier input leakage guard, resume,
raw output persistence, UNSELECTED refusal, $8 warning / $10 hard stop,
prompt parity with the historical OCR runs, no benchmark mutation, and
zero provider calls in dry-run mode.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from autograder.backends.mock import MockBackend
from autograder.benchmark.datasets import DatasetExists, write_declared_dataset
from autograder.benchmark.manifests import (BenchCase, BenchmarkIntegrityError, BenchmarkNotBuilt,
                                            all_manifest_summaries, load_manifest)
from autograder.benchmark.registry import load_registry
from autograder.benchmark.roles import (OcrVerifyAdapter, Request, _load_historical_prompts, adapter_for)
from autograder.benchmark.runner import (HeldOutRefused, LeakageError, RunSpec, UnselectedCandidate,
                                         held_out_executions, leakage_check, run_benchmark)
from autograder.gateway import ModelGateway, TaskRoute
from autograder.requestcache import RequestCache
from autograder.usage import BudgetExceeded, BudgetLimits, BudgetManager, UsageLedger

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
FROZEN_FILES = [
    BENCH / "reference_audit.json", BENCH / "reference_audit_manifest.json",
    BENCH / "items.json", BENCH / "references.json",
    BENCH / "verifier_bench" / "selected" / "cases_inputs.jsonl",
    BENCH / "verifier_bench" / "selected" / "cases_labels.jsonl",
    BENCH / "verifier_bench" / "selected" / "manifest.json",
    BENCH / "verifier_bench" / "synthetic" / "cases_inputs.jsonl",
    BENCH / "verifier_bench" / "synthetic" / "cases_labels.jsonl",
    BENCH / "verifier_bench" / "synthetic" / "manifest.json",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- manifests --

def test_frozen_manifests_load_and_verify_hashes():
    m = load_manifest("ocr_verify")
    assert m.status == "FROZEN"
    assert m.components == ["REAL", "SYNTHETIC"]
    counts = m.counts()
    assert counts["DEV"]["REAL"] + counts["CALIBRATION"]["REAL"] + counts["HELD_OUT"]["REAL"] == 303
    assert counts["DEV"]["SYNTHETIC"] + counts["CALIBRATION"]["SYNTHETIC"] + counts["HELD_OUT"]["SYNTHETIC"] == 136
    assert m.split_assignment == {"DEV": ["e002", "e003", "e007"], "CALIBRATION": ["e004"],
                                  "HELD_OUT": ["e005", "e006"]}
    # recorded hashes == on-disk bytes (the loader refused otherwise)
    assert m.hashes["real_inputs_sha256"] == _sha(BENCH / "verifier_bench" / "selected" / "cases_inputs.jsonl")
    # model-visible inputs carry exactly the three allowed fields
    for c in m.cases:
        assert set(c.inputs) == {"case_id", "crop", "candidate_transcription"}
        assert "expected_verdict" in c.label and "polarity" in c.label
    o = load_manifest("ocr_primary")
    assert o.status == "FROZEN" and len(o.cases) == 129
    assert o.extra["items_with_reference"] == 129
    # same writer split as the verifier benchmark; text-layer items are DEV
    assert all(c.split == "DEV" for c in o.cases if c.meta["tier"] == "text-layer")
    assert {c.split for c in o.cases if c.meta["writer"] == "e005"} == {"HELD_OUT"}


def test_tampered_frozen_file_is_refused(tmp_path):
    root = tmp_path / "bench"
    (root / "verifier_bench").mkdir(parents=True)
    for sub in ("selected", "synthetic"):
        shutil.copytree(BENCH / "verifier_bench" / sub, root / "verifier_bench" / sub)
    p = root / "verifier_bench" / "selected" / "cases_labels.jsonl"
    p.write_bytes(p.read_bytes() + b"\n")          # one extra byte
    with pytest.raises(BenchmarkIntegrityError):
        load_manifest("ocr_verify", bench_root=root)


def test_declared_roles_report_not_built_without_fabricating(tmp_path):
    m = load_manifest("grade_primary", datasets_root=tmp_path)
    assert m.status == "NOT_BUILT" and m.cases == []
    summ = all_manifest_summaries(datasets_root=tmp_path)
    assert summ["mc_resolve_cloud"]["status"] == "NOT_BUILT"
    assert summ["ocr_verify"]["status"] == "FROZEN"


def test_split_selection_is_explicit():
    m = load_manifest("ocr_verify")
    with pytest.raises(ValueError):
        m.by_split("train")
    assert all(c.split == "DEV" and c.component == "REAL" for c in m.by_split("dev", "REAL"))


# ----------------------------------------------------------------- registry --

def test_registry_is_data_without_prices():
    reg = load_registry()
    assert reg.experiment_total_usd == 10.0 and reg.warn_usd == 8.0
    assert set(reg.unselected_roles()) >= {"ocr_primary", "ocr_verify", "grade_primary", "grade_escalate",
                                           "mc_resolve_cloud", "variant_resolve", "align_resolve"}
    text = reg.path.read_text(encoding="utf-8")
    assert "price" not in text.lower().replace("promotional pricing", "").replace("prices you note", "") \
        or "$" not in text.split("[roles.")[1]        # no per-model dollar figures in the role sections


# ------------------------------------------------------------------ leakage --

def _align_case(content_extra: str = "", label_extra: dict | None = None, inputs_extra: dict | None = None):
    inputs = {"case_id": "c1", "question_id": "1", "canonical": [["1", "alpha"], ["2", "beta"]],
              "printed": [["a", "alpha" + content_extra], ["b", "beta"]], **(inputs_extra or {})}
    label = {"mapping": {"a": "1", "b": "2"}, **(label_extra or {})}
    return BenchCase("c1", "DEV", "ALL", inputs, label)


def test_leakage_guard_refuses_label_values_split_names_and_extra_inputs(tmp_path):
    ad = adapter_for("align_resolve")
    ok = _align_case()
    leakage_check(ok, ad.build_request(dict(ok.inputs), tmp_path), ad.model_visible_fields)
    leaky = _align_case(content_extra=" ZEBRA-ANSWER", label_extra={"secret": "ZEBRA-ANSWER"})
    with pytest.raises(LeakageError):
        leakage_check(leaky, ad.build_request(dict(leaky.inputs), tmp_path), ad.model_visible_fields)
    split = _align_case(content_extra=" HELD_OUT")
    with pytest.raises(LeakageError):
        leakage_check(split, ad.build_request(dict(split.inputs), tmp_path), ad.model_visible_fields)
    extra = _align_case(inputs_extra={"expected_verdict": "review"})
    with pytest.raises(LeakageError):
        leakage_check(extra, Request("sys", [{"type": "text", "text": "x"}], type(ad.build_request(dict(ok.inputs), tmp_path).output_model), "v"),
                      ad.model_visible_fields)


def test_verifier_request_carries_only_crop_and_candidate():
    """The production verifier contract: image + candidate, nothing else."""
    m = load_manifest("ocr_verify")
    case = m.by_split("DEV", "SYNTHETIC")[0]
    ad = OcrVerifyAdapter()
    req = ad.build_request(dict(case.inputs), BENCH)
    leakage_check(case, req, ad.model_visible_fields)
    text = req.text_for_inspection()
    for forbidden in (case.label["audited_reference"] if case.label["audited_reference"] != case.inputs["candidate_transcription"] else "\x00",
                      "expected_verdict", "polarity", "corruption_type", "DEV", "CALIBRATION", "HELD_OUT",
                      "cer_vs_audited", "rubric", "solution"):
        assert forbidden not in text
    assert case.inputs["candidate_transcription"] in text
    assert sum(1 for b in req.content_blocks if b["type"] == "image") == 1


# ------------------------------------------------------------------- runner --

def _tiny_align_dataset(root: Path) -> Path:
    d = root / "align_resolve"
    inputs, labels = [], []
    for i, split in enumerate(["DEV", "DEV", "CALIBRATION", "HELD_OUT"]):
        inputs.append({"case_id": f"case{i}", "question_id": str(i), "canonical": [["1", "alpha"], ["2", "beta"]],
                       "printed": [["a", "alpha"], ["b", "beta" + (" FAILME" if i == 1 else "")]]})
        labels.append({"case_id": f"case{i}", "split": split, "mapping": {"a": "1", "b": "2"}})
    write_declared_dataset(d, name="tiny align", cases_inputs=inputs, cases_labels=labels,
                           split_assignment={"DEV": ["q0", "q1"], "CALIBRATION": ["q2"], "HELD_OUT": ["q3"]},
                           now="2026-08-22 00:00:00")
    with pytest.raises(DatasetExists):
        write_declared_dataset(d, name="again", cases_inputs=inputs, cases_labels=labels)
    return d


def _registry(tmp_path: Path) -> Path:
    p = tmp_path / "candidates.toml"
    p.write_text('[meta]\nversion = 1\nupdated = "2026-08-22"\nrule = "x"\n[budget]\nexperiment_total_usd = 10.0\nwarn_usd = 8.0\n'
                 '[roles.align_resolve]\nstatus = "UNSELECTED"\ngateway_task = "align_resolve_cloud"\n'
                 'env_slug = "ALIGN_RESOLVE_CLOUD_MODEL"\ncandidates = ["mock/model-a"]\n', encoding="utf-8")
    return p


def _mock_gateway(tmp_path: Path, responder, budget: BudgetManager | None = None, ledger: UsageLedger | None = None):
    from autograder.alignment import PermutationProposal  # noqa: F401 — output model of the adapter
    route = TaskRoute(task="align_resolve_cloud", backend="mock", model="mock/model-a", prompt_version="align-v1",
                      max_tokens=300)
    ledger = ledger or UsageLedger(tmp_path / "state" / "gateway_ledger" / "usage.jsonl")
    gw = ModelGateway({route.task: route},
                      backend_factory=lambda cfg: MockBackend(config=cfg, responder=responder),
                      cache=RequestCache(tmp_path / "state" / "gateway_cache"), ledger=ledger, budget=budget)
    return gw


def _spec(tmp_path: Path, **kw) -> RunSpec:
    base = dict(role="align_resolve", split="dev", candidate="mock/model-a", backend="mock",
                skip_key_preflight=True,
                registry_path=_registry(tmp_path), datasets_root=tmp_path / "ds", state_root=tmp_path / "state",
                runs_root=tmp_path / "runs", held_out_log=tmp_path / "HELD_OUT.jsonl", dry_run=False)
    base.update(kw)
    return RunSpec(**base)


def _responder(model, system, blocks):
    from autograder.backends.base import BackendError
    if "FAILME" in blocks[0]["text"]:
        raise BackendError("simulated provider failure")
    return model(question_id="q", printed_to_key={"a": "1", "b": "2"}, confident=True)


def test_live_run_persists_raw_outputs_and_resumes(tmp_path, no_network):
    _tiny_align_dataset(tmp_path / "ds")
    gw = _mock_gateway(tmp_path, _responder)
    res = run_benchmark(_spec(tmp_path), gateway=gw)
    assert res.cases_selected == 2 and res.cases_done == 1 and res.cases_failed == 1
    rows = [json.loads(l) for l in (res.run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    ok_rows = [r for r in rows if r["ok"]]
    assert ok_rows[0]["output"] == {"question_id": "q", "printed_to_key": {"a": "1", "b": "2"}, "confident": True,
                                    "notes": None}
    assert ok_rows[0]["model"] == "mock/model-a" and "fingerprint" in ok_rows[0]
    failed = [r for r in rows if r["ok"] is False][0]
    assert failed["error_type"] == "BackendError" and failed["attempt"] == 1
    run = json.loads((res.run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["config"]["prompt_version"] == "align-v1" and run["config"]["validation_retries"] == 0
    assert len(run["config"]["prompt_sha256"]) == 64 and len(run["config"]["schema_sha256"]) == 64
    assert run["config"]["manifest_hashes"]["inputs_sha256"]
    metrics = json.loads((res.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cases"] == 2 and metrics["automatic_pct"] == 50.0 and metrics["unsafe_automatic"] == 0
    # resume: the ok row is skipped, the failed row is NOT retried silently
    res2 = run_benchmark(_spec(tmp_path), gateway=gw)
    assert res2.cases_skipped_resume == 2 and res2.cases_done == 0 and res2.run_dir == res.run_dir
    # explicit retry re-attempts only the failed case and records attempt 2
    res3 = run_benchmark(_spec(tmp_path, retry_failed=True), gateway=gw)
    assert res3.cases_skipped_resume == 1 and res3.cases_failed == 1
    rows = [json.loads(l) for l in (res.run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert max(r["attempt"] for r in rows if r["case_id"] == "case1") == 2
    # a changed configuration is a DIFFERENT run directory (comparability)
    res4 = run_benchmark(_spec(tmp_path, max_tokens=999), gateway=gw)
    assert res4.run_dir != res.run_dir


def test_held_out_cannot_be_dry_run_and_only_final_eval_executes_it(tmp_path, no_network):
    _tiny_align_dataset(tmp_path / "ds")
    # dry-run is refused regardless of confirmation flags
    with pytest.raises(HeldOutRefused, match="cannot be previewed/dry-run"):
        run_benchmark(_spec(tmp_path, split="held_out", dry_run=True))
    with pytest.raises(HeldOutRefused, match="cannot be previewed/dry-run"):
        run_benchmark(_spec(tmp_path, split="held_out", dry_run=True, confirm_held_out=True, final_evaluation=True))
    # a live run through the ordinary path is refused even when confirmed
    gw = _mock_gateway(tmp_path, _responder)
    with pytest.raises(HeldOutRefused, match="final-evaluation path"):
        run_benchmark(_spec(tmp_path, split="held_out", confirm_held_out=True), gateway=gw)
    with pytest.raises(HeldOutRefused):
        run_benchmark(_spec(tmp_path, split="held_out", final_evaluation=True), gateway=gw)
    assert not (tmp_path / "HELD_OUT.jsonl").exists()          # nothing was logged for refusals
    # the explicit final-evaluation path executes and logs permanently with provenance
    res = run_benchmark(_spec(tmp_path, split="held_out", confirm_held_out=True, final_evaluation=True,
                              note="final"), gateway=gw)
    assert res.cases_selected == 1 and res.cases_done == 1
    log = held_out_executions(tmp_path / "HELD_OUT.jsonl")
    assert len(log) == 1 and log[0]["mode"] == "final_evaluation_live" and log[0]["run_id"] == res.run_id
    for k in ("config_hash", "prompt_sha256", "schema_sha256", "adapter_version", "manifest_hashes", "git_commit"):
        assert k in log[0]
    assert "no longer untouched" in log[0]["consequence"]
    # calibration and dev are never logged there
    run_benchmark(_spec(tmp_path, split="calibration", dry_run=True))
    assert len(held_out_executions(tmp_path / "HELD_OUT.jsonl")) == 1


def test_held_out_log_is_not_written_when_the_run_cannot_execute(tmp_path, monkeypatch):
    """A refused credential/readiness check must not leave a HELD_OUT record."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _tiny_align_dataset(tmp_path / "ds")
    with pytest.raises(Exception):
        run_benchmark(_spec(tmp_path, split="held_out", confirm_held_out=True, final_evaluation=True,
                            backend="openrouter"))
    assert not (tmp_path / "HELD_OUT.jsonl").exists()


def test_unselected_and_unlisted_candidates_are_refused(tmp_path):
    _tiny_align_dataset(tmp_path / "ds")
    cfg = tmp_path / "models.toml"
    cfg.write_text('[models.align_resolve_cloud]\nbackend = "openrouter"\nmodel = "${ALIGN_RESOLVE_CLOUD_MODEL}"\n',
                   encoding="utf-8")
    with pytest.raises(UnselectedCandidate, match="is not selected"):
        run_benchmark(_spec(tmp_path, candidate=None, models_config=cfg, dry_run=True))
    with pytest.raises(UnselectedCandidate, match="not a registered candidate"):
        run_benchmark(_spec(tmp_path, candidate="vendor/unlisted", dry_run=True))
    res = run_benchmark(_spec(tmp_path, candidate="vendor/unlisted", allow_unlisted=True, dry_run=True))
    assert res.dry_run and res.cases_done == 2


def test_dry_run_makes_zero_calls_and_writes_a_plan(tmp_path, no_network):
    _tiny_align_dataset(tmp_path / "ds")
    calls = []
    gw = _mock_gateway(tmp_path, lambda *a: calls.append(a))
    res = run_benchmark(_spec(tmp_path, dry_run=True), gateway=gw)
    assert calls == [] and res.cases_done == 2
    plan = json.loads((res.run_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["mode"] == "dry_run" and plan["leakage_check"] == "passed" and len(plan["rows"]) == 2
    assert not (res.run_dir / "outputs.jsonl").exists()


def test_not_built_dataset_refuses_to_run(tmp_path):
    with pytest.raises(BenchmarkNotBuilt):
        run_benchmark(_spec(tmp_path, dry_run=True))


# ------------------------------------------------------------------- budget --

def _seed_ledger(path: Path, cost: float) -> UsageLedger:
    led = UsageLedger(path)
    led.record({"task": "x", "backend": "openrouter", "model": "m", "cloud": True, "cache_hit": False,
                "reported_cost": cost})
    return led


def test_eight_dollar_warning_and_ten_dollar_hard_stop(tmp_path, no_network):
    _tiny_align_dataset(tmp_path / "ds")
    warnings: list[str] = []
    # $8.50 already spent on the CAMPAIGN ledger -> the next call warns (>= 80% of $10)
    led = _seed_ledger(tmp_path / "state" / "gateway_ledger" / "usage.jsonl", 8.5)
    bm = BudgetManager(BudgetLimits(max_cost_total=10.0, soft_fraction=0.8), ledger=led,
                       warn=warnings.append, cloud_only=False)
    gw = _mock_gateway(tmp_path, _responder, budget=bm, ledger=led)
    res = run_benchmark(_spec(tmp_path), gateway=gw)
    assert res.cases_done == 1
    assert any("soft budget cost_total" in w for w in warnings)
    # $10.01 spent -> hard stop: the run stops BEFORE the call, recorded, no exception
    tmp2 = tmp_path / "two"
    _tiny_align_dataset(tmp2 / "ds")
    led2 = _seed_ledger(tmp2 / "state" / "gateway_ledger" / "usage.jsonl", 10.01)
    bm2 = BudgetManager(BudgetLimits(max_cost_total=10.0, soft_fraction=0.8), ledger=led2, cloud_only=False)
    calls = []
    gw2 = _mock_gateway(tmp2, lambda m, s, b: (calls.append(1), _responder(m, s, b))[1], budget=bm2, ledger=led2)
    res2 = run_benchmark(_spec(tmp2), gateway=gw2)
    assert calls == [] and res2.stopped_reason and "hard budget cost_total" in res2.stopped_reason
    with pytest.raises(BudgetExceeded):
        bm2.check(task="x", route=TaskRoute(task="x", backend="openrouter", model="m"), meta={})


# --------------------------------------------------------------- integrity --

def test_prompt_parity_with_historical_ocr_runs():
    """OCR_PRIMARY requests use the byte-identical hebrew_bench_v2 prompts
    (scripts/m2_bench_run.py PROMPTS) so new runs stay comparable."""
    prompts = _load_historical_prompts()
    src = (REPO / "scripts" / "m2_bench_run.py").read_text(encoding="utf-8")
    assert set(prompts) == {"handwritten_line", "handwritten_cell", "printed_rtl", "mixed_he_en",
                            "formula_printed", "option_row_association"}
    for cat, text in prompts.items():
        head = text.splitlines()[0][:40]
        assert head.split('"')[0][:20] in src          # the prompt head exists verbatim in the script
    ad = adapter_for("ocr_primary")
    m = load_manifest("ocr_primary")
    req = ad.build_request(dict(m.cases[0].inputs), BENCH)
    assert req.system == prompts[m.cases[0].inputs["category"]]
    assert req.output_model.model_json_schema()["required"] == ["transcription"]


def test_loading_and_dry_running_never_mutates_the_frozen_benchmarks(tmp_path):
    before = {p: _sha(p) for p in FROZEN_FILES}
    load_manifest("ocr_verify"); load_manifest("ocr_primary")
    run_benchmark(RunSpec(role="ocr_verify", split="dev", candidate="openai/gpt-5.6-luna", component="REAL",
                          runs_root=tmp_path / "runs", state_root=tmp_path / "state",
                          held_out_log=tmp_path / "HO.jsonl", limit=3, dry_run=True))
    run_benchmark(RunSpec(role="ocr_primary", split="dev", candidate="google/gemini-3.7-flash",
                          runs_root=tmp_path / "runs", state_root=tmp_path / "state",
                          held_out_log=tmp_path / "HO.jsonl", limit=3, dry_run=True))
    after = {p: _sha(p) for p in FROZEN_FILES}
    assert before == after
    assert not (tmp_path / "HO.jsonl").exists()


def test_verifier_metrics_report_real_and_synthetic_separately():
    ad = OcrVerifyAdapter()
    m = load_manifest("ocr_verify")
    cases = m.by_split("DEV")
    # a verifier that accepts everything: REAL FAR = 100%, SYNTHETIC FAR = 100%, FRR = 0
    scored = [ad.score(c, {"verdict": "supported", "confidence": "high", "omissions": [], "substitutions": [],
                           "additions": []}, None) for c in cases]
    agg = ad.aggregate(scored, [{"ok": True, "usage": {}, "latency_s": 1.0}] * len(cases))
    assert agg["REAL"]["false_accept_rate_pct"] == 100.0 and agg["REAL"]["false_reject_rate_pct"] == 0.0
    assert agg["SYNTHETIC"]["false_accept_rate_pct"] == 100.0
    assert agg["SYNTHETIC"]["numeric_math"]["cases"] > 0
    assert set(agg["SYNTHETIC"]["by_corruption_type"]) >= {"char_deletion", "digit_substitution"}
    assert "COMBINED_secondary" in agg and "primary_metric" in agg
    # a verifier that rejects everything: FAR 0, FRR 100, review rate 100
    scored = [ad.score(c, {"verdict": "review", "confidence": "low"}, None) for c in cases]
    agg = ad.aggregate(scored, [])
    assert agg["REAL"]["false_accept_rate_pct"] == 0.0 and agg["REAL"]["false_reject_rate_pct"] == 100.0
    assert agg["REAL"]["review_rate_pct"] == 100.0


def test_ocr_primary_scoring_uses_audited_references():
    ad = adapter_for("ocr_primary")
    m = load_manifest("ocr_primary")
    c = next(c for c in m.cases if c.meta["tier"] == "owner")
    exact = ad.score(c, {"transcription": c.label["reference"]}, None)
    assert exact["cer"] == 0.0 and exact["usable_25"] and not exact["number_sign_error"]
    wrong = ad.score(c, {"transcription": "xyz 999"}, None)
    assert wrong["cer"] > 0.5
    agg = ad.aggregate([exact, wrong], [])
    assert agg["overall"]["cases"] == 2 and "by_category" in agg and agg["schema_failures"] == 0
