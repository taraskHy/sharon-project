"""Pre-register (freeze) the production OpenRouter OCR validation campaign.

    python scripts/ocr_validation_freeze.py

PREPARATION ONLY — this script performs ZERO cloud/OCR/model calls, never
authenticates, and never reads HELD_OUT content. It freezes, before any
future execution:

- the two campaign stages (frozen bench smoke; SEEN-46 explanation crops);
- per-case crop file hashes and audited-transcription hashes (SEEN cases
  only, matched by case id against the frozen 46-case reference — HELD_OUT
  rows in the dataset files are skipped by id WITHOUT being parsed);
- OCR prompt versions + schema hashes for BOTH protocols (historical bench
  m2-strict-v1 for comparability; production ocr-v1 for the shipping route);
- the candidate OpenRouter models and the recorded local pricing snapshot;
- the exact expected request counts and a hard cost upper bound;
- the metric definitions and acceptance gates;
- the exact future dry-run and paid commands (owner-gated).

Writes evaluation/model_selection/experiments/OCR_VALIDATION_CAMPAIGN_<date>.json
(immutable, self-hashed) and a summary MD next to the risk artifacts.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
EXPERIMENTS = REPO / "evaluation" / "model_selection" / "experiments"
DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
SMOKE = REPO / "evaluation" / "model_selection" / "smoke" / "ocr_primary_smoke.json"
REF_PATH = RUNS / "FINAL_HUMAN_REFERENCE_2026-09-02.json"
EVAL_ROOT = REPO / "evaluation"

CANDIDATES = ("google/gemini-3.7-flash", "openai/gpt-5.6-luna",
              "anthropic/claude-sonnet-5")
PRIMARY_CANDIDATE = CANDIDATES[0]
MAX_OUTPUT_TOKENS = 600
FLAT_IMAGE_TOKENS = 1100          # the estimator's per-image constant
COST_UPPER_BOUND_USD = 2.00       # hard campaign bound, far under the $10 cap


def _sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    today = time.strftime("%Y-%m-%d")
    out_json = EXPERIMENTS / f"OCR_VALIDATION_CAMPAIGN_{today}.json"
    out_md = RUNS / f"OCR_VALIDATION_CAMPAIGN_{today}.md"
    if out_json.exists():
        print(f"REFUSED: {out_json.name} already frozen")
        return 3

    # ---- the SEEN-46 population (ids only; HELD_OUT rows never parsed) ----
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    payload = json.dumps({k: v for k, v in ref.items() if k != "reference_sha256"},
                         ensure_ascii=False, sort_keys=True)
    assert ref["reference_sha256"] == _sha_text(payload), "reference tampered"
    seen_ids = {c["case_id"] for c in ref["cases"]}
    assert len(seen_ids) == 46

    id_pat = re.compile(r'"case_id": "([^"]+)"')

    def seen_rows(path: Path) -> dict[str, dict]:
        """Parse ONLY lines whose case id is in the seen-46 set: HELD_OUT
        writers' rows are skipped by id before any JSON parsing."""
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            m = id_pat.search(line[:200])
            if not m or m.group(1) not in seen_ids:
                continue
            out[m.group(1)] = json.loads(line)
        assert set(out) == seen_ids, sorted(seen_ids - set(out))
        return out

    inputs = seen_rows(DATASET / "cases_inputs.jsonl")
    labels = seen_rows(DATASET / "cases_labels.jsonl")

    cases = []
    total_crops = 0
    for cid in sorted(seen_ids):
        images = labels[cid]["evidence_images"]
        crop_hashes = {}
        for rel in images:
            p = EVAL_ROOT / rel
            if not p.exists():
                p = DATASET / rel
            assert p.exists(), f"crop not found: {rel}"
            crop_hashes[rel] = _sha_file(p)
        total_crops += len(images)
        cases.append({
            "case_id": cid,
            "writer": cid.split("_")[0],
            "audited_transcription_sha256":
                _sha_text(inputs[cid]["transcription"] or ""),
            "transcription_chars": len(inputs[cid]["transcription"] or ""),
            "evidence_crops": crop_hashes,
            "transcription_complete": labels[cid]["transcription_complete"],
        })

    # ---- prompts / schemas (both protocols), boundary verification --------
    from autograder.benchmark.roles import (OcrPrimaryAdapter,
                                            _load_historical_prompts)
    from autograder.cloudboundary import (approved_cloud_ocr_systems,
                                          check_cloud_call, CloudBoundaryError)
    from autograder.prompts import EXPLANATION_OCR_SYSTEM
    from autograder.schema import ExplanationTranscription
    bench_adapter = OcrPrimaryAdapter()
    bench_prompts = _load_historical_prompts()     # m2-strict-v1, AST-recovered
    prod_schema = json.dumps(ExplanationTranscription.model_json_schema(),
                             sort_keys=True)
    assert EXPLANATION_OCR_SYSTEM in approved_cloud_ocr_systems(), \
        "production OCR prompt must be registered at the boundary"
    # the boundary admits ocr_primary to OpenRouter and refuses grading —
    # verified WITHOUT any network or credential use
    check_cloud_call(task="ocr_primary", backend="openrouter", base_url=None,
                     execution_mode="production", system=EXPLANATION_OCR_SYSTEM,
                     content_blocks=[])
    try:
        check_cloud_call(task="grade_primary", backend="openrouter",
                         base_url=None, execution_mode="production")
        raise AssertionError("grading must never pass the boundary")
    except CloudBoundaryError:
        pass

    # ---- smoke stage (already frozen 2026-08-22) --------------------------
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    assert smoke["role"] == "ocr_primary" and smoke["split"] == "DEV"
    smoke_n = len(smoke["cases"])

    # ---- pricing snapshot (machine-local estimator table; no fetch) -------
    pricing = {"source": "models.toml [pricing] (machine-local, gitignored, "
                         "verified 2026-08-23; estimator-only, never fetched)",
               "rows": {}}
    models_toml = REPO / "models.toml"
    if models_toml.exists():
        import tomllib
        cfg = tomllib.loads(models_toml.read_text(encoding="utf-8"))
        for slug in CANDIDATES:
            row = (cfg.get("pricing") or {}).get(slug)
            if row:
                pricing["rows"][slug] = row
    else:
        pricing["rows"] = {"note": "models.toml absent on this machine; "
                                   "pricing must be present before any live run "
                                   "(UnpricedCandidate refusal applies)"}

    # ---- request counts + cost upper bound --------------------------------
    per_call_tokens_in = FLAT_IMAGE_TOKENS + len(EXPLANATION_OCR_SYSTEM) // 4 + 100
    calls_per_candidate = smoke_n + total_crops
    cost_note = (
        f"per-call estimate = ({per_call_tokens_in} in + {MAX_OUTPUT_TOKENS} "
        "out max) tokens at the recorded per-1M prices; e.g. "
        "gemini-3.7-flash ~= $0.0017/call -> "
        f"{calls_per_candidate} calls ~= $0.09/candidate. The frozen hard "
        f"bound is ${COST_UPPER_BOUND_USD:.2f} for the whole campaign (all "
        "candidates, retries included), enforced on top of the global $10 "
        "experiment ceiling and the $8 warn line.")

    doc = {
        "experiment": f"ocr_validation_campaign_{today}",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PREPARED_NOT_EXECUTED",
        "git_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse",
                                      "HEAD"], capture_output=True, text=True,
                                     timeout=15).stdout.strip(),
        "purpose": "validate production OpenRouter OCR quality against the "
                   "audited SEEN transcriptions BEFORE any end-to-end "
                   "shipping; OCR remains a hard release blocker until this "
                   "campaign runs and passes",
        "prohibited_now": ["OpenRouter authentication", "any cloud call",
                           "any OCR call", "any HELD_OUT content"],
        "stages": [
            {"stage": 1, "name": "bench smoke (historical protocol)",
             "population": "frozen ocr_primary_smoke.json",
             "cases": smoke_n,
             "selection_sha256": smoke["selection_sha256"],
             "manifest_hashes": smoke["manifest_hashes"],
             "adapter_version": bench_adapter.adapter_version,
             "prompt_version": bench_adapter.prompt_version,
             "prompt_sha256_by_category":
                 {k: _sha_text(v) for k, v in sorted(bench_prompts.items())},
             "schema": "BenchTranscription {transcription: str}",
             "note": "byte-identical m2-strict-v1 prompts keep new runs "
                     "comparable with the frozen hebrew_bench_v2 history"},
            {"stage": 2, "name": "SEEN-46 explanation crops (production "
                                 "protocol)",
             "population": "the frozen 46-case grading campaign evidence",
             "cases": 46, "crops": total_crops,
             "reference_sha256": ref["reference_sha256"],
             "prompt_version": "ocr-v1",
             "prompt_sha256": _sha_text(EXPLANATION_OCR_SYSTEM),
             "schema_name": "ExplanationTranscription",
             "schema_sha256": _sha_text(prod_schema),
             "engineering_prep_required_before_execution": [
                 "register a frozen seen46-ocr subset for the bench CLI "
                 "(--subset currently has no seen-46 OCR entry)",
                 "add per-writer WER/omission aggregation to the scoring "
                 "path (word_align is loaded but unused by "
                 "OcrPrimaryAdapter); adapter_version must bump if its "
                 "scoring changes"],
             },
        ],
        "cases": cases,
        "request_content_allowlist": [
            "the necessary crop image", "the exact-transcription instruction",
            "minimal script/language hint", "the minimal structured OCR schema"],
        "request_content_forbidden": [
            "rubric", "official solution", "grade", "instructor label",
            "RAG context", "grading prompt", "model grading output"],
        "boundary": {"allowlist": ["ocr_primary", "ocr_verify"],
                     "production_prompt_registered": True,
                     "grading_refused": True},
        "candidates": {"primary": PRIMARY_CANDIDATE,
                       "shortlist": list(CANDIDATES),
                       "registry": "evaluation/model_selection/candidates.toml "
                                   "[roles.ocr_primary] (status UNSELECTED)"},
        "pricing_snapshot": pricing,
        "execution_budget": {
            "max_provider_calls_per_candidate": calls_per_candidate,
            "max_candidates": len(CANDIDATES),
            "max_provider_calls_total": calls_per_candidate * len(CANDIDATES),
            "cost_upper_bound_usd": COST_UPPER_BOUND_USD,
            "cost_estimate_note": cost_note,
            "cache_policy": "exact-request cache reuse allowed and reported; "
                            "never cleared; cache hits still write ledger rows",
            "global_ceiling": "models.toml [budget] max_cost_total = 10.00 "
                              "USD, soft warn 8.00 (BudgetManager enforced)"},
        "metrics": {
            "primary": "per-case CER against the audited transcription "
                       "(hebrew_bench_eval.normalize + lev), WRITER-GROUPED",
            "secondary": ["WER via hebrew_bench_eval.word_align",
                          "omission rate (deletions/gt_words)",
                          "digit/operator signature error "
                          "(refaudit.digit_op_signature)",
                          "line-loss rate (missing evidence lines)",
                          "unreadable-span behaviour (legibility field vs "
                          "hallucinated text)",
                          "verifier disagreement (ocr-verify-v2-independent, "
                          "local agreement >= 0.95 rule)",
                          "false-accept rate (agreement passed but CER > gate)",
                          "REVIEW rate", "cost per crop",
                          "projected cost per exam and per 100 exams"],
            "downstream_grade_impact": "verdict-flip rate: rerun the FROZEN "
                                       "local grader on OCR text vs the frozen "
                                       "transcription — LOCAL inference, "
                                       "owner-gated, NOT part of this freeze's "
                                       "cloud budget",
            "gates_proposed": "per-writer CER <= 5% AND zero harmful verdict "
                              "flips on seen data (calibrate on CALIBRATION "
                              "only, report writer-held)"},
        "future_commands": {
            "dry_run_now_allowed": [
                "python -m autograder bench dry-run --role ocr_primary "
                "--split dev --subset smoke --candidate "
                f"{PRIMARY_CANDIDATE} --backend openrouter "
                "--models-config models.toml"],
            "paid_stage_1_owner_gated": [
                "python -m autograder bench run --role ocr_primary --split dev "
                f"--subset smoke --candidate {PRIMARY_CANDIDATE} "
                "--backend openrouter --models-config models.toml "
                "--i-understand-this-spends-money"],
            "paid_stage_2_blocked_until_prep": [
                "(after the seen46-ocr subset + WER scoring land) "
                "python -m autograder bench run --role ocr_primary "
                "--split dev --subset seen46_ocr --candidate "
                f"{PRIMARY_CANDIDATE} --backend openrouter "
                "--models-config models.toml --i-understand-this-spends-money"],
            "never": ["bench final-eval (HELD_OUT) before grader+matrix+"
                      "policy+OCR freezes and owner sign-off"]},
        "held_out": {"rows_in_campaign": 0,
                     "policy": "HELD_OUT ids/content untouched; dataset lines "
                               "outside the seen-46 id set were skipped "
                               "without JSON parsing"},
    }
    payload = json.dumps({k: v for k, v in doc.items()
                          if k != "experiment_sha256"},
                         ensure_ascii=False, sort_keys=True)
    doc["experiment_sha256"] = _sha_text(payload)
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8", newline="\n")

    md = [f"# OCR validation campaign — FROZEN, NOT EXECUTED ({doc['created_at']})",
          "",
          f"`{out_json.name}` sha `{doc['experiment_sha256'][:16]}…`; "
          f"git `{doc['git_commit'][:12]}`.",
          "",
          f"- stage 1: {smoke_n} frozen bench smoke cases (m2-strict-v1, "
          "historical comparability)",
          f"- stage 2: 46 seen cases / {total_crops} crops (production "
          "ocr-v1 + ExplanationTranscription), blocked on two named prep "
          "items (seen46-ocr subset registration; per-writer WER scoring)",
          f"- candidates: {', '.join(CANDIDATES)} (primary "
          f"{PRIMARY_CANDIDATE}; all UNSELECTED)",
          f"- budget: <= {calls_per_candidate} calls/candidate, hard bound "
          f"${COST_UPPER_BOUND_USD:.2f} campaign-total inside the global $10 "
          "ceiling",
          "- request content: crop + exact-transcription instruction + "
          "minimal hint/schema ONLY; rubric/solution/grades/RAG are "
          "boundary-refused",
          "- OCR calls executed by this freeze: **0**; OpenRouter "
          "authentication: **none**"]
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"cases": len(cases), "crops": total_crops,
                      "calls_per_candidate": calls_per_candidate,
                      "experiment_sha256": doc["experiment_sha256"][:16]},
                     indent=1))
    print("written:", out_json.name, out_md.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
