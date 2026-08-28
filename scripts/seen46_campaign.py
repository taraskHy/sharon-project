"""The SEEN-46 diagnostic campaign: freeze, verify, leakage proof, gate.

    python scripts/seen46_campaign.py freeze     # write the immutable manifest
    python scripts/seen46_campaign.py verify     # exit 0 iff repo matches it
    python scripts/seen46_campaign.py leakage    # zero-leakage artifact, 46 requests
    python scripts/seen46_campaign.py gate       # model-run completion gate + summary

Population: every already-SEEN explanation case — the whole DEV split (32,
writers e002/e003/e007) plus the whole CALIBRATION split (14, writer e004).
HELD_OUT (e005/e006) is structurally excluded: its ids are never read here,
never enumerated, never counted beyond the split totals the frozen dataset
manifest already publishes.

Ground truth policy: the actual original instructor grade is the REFERENCE
(recorded per case, never modified); instructor-derived explanation verdicts
are recorded only where uniquely derivable; A/B/C/D audit decisions ride
along as diagnostic flags and are never targets. ZERO inference in every
subcommand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CAMPAIGN_ID = "seen46_2026-08-28"
CAMPAIGN_PATH = REPO / "evaluation" / "model_selection" / "experiments" / \
    "LOCAL_GRADE_PRIMARY_SEEN_46_CAMPAIGN_2026-08-28.json"
FREEZE_PATH = REPO / "evaluation" / "model_selection" / "experiments" / \
    "LOCAL_GRADE_CONTRACT_FREEZE_2026-08-28.json"
AUDIT_PATH = REPO / "evaluation" / "model_selection" / "runs" / "grade_primary" / \
    "CALIBRATION_AUDIT_2026-08-26.json"
RUNS_ROOT = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
LEAKAGE_PATH = RUNS_ROOT / "SEEN46_LEAKAGE_VERIFICATION_2026-08-28.json"
SUMMARY_JSON = RUNS_ROOT / "SEEN46_MODEL_RUN_2026-08-28.json"
SUMMARY_MD = RUNS_ROOT / "SEEN46_MODEL_RUN_2026-08-28.md"

ALLOWED_SPLITS = ("DEV", "CALIBRATION")
ALLOWED_WRITERS = ("e002", "e003", "e004", "e007")
FORBIDDEN_WRITERS = ("e005", "e006")
CANDIDATE = "qwen3-vl:8b-instruct"
PROMPT_VERSION = "grade-v4-charitable-local"


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()


def _ollama_digest(tag: str) -> str | None:
    """Metadata only (`ollama list` prints the manifest table; nothing loads)."""
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001 — absence is recorded, not fatal
        return None
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts and parts[0] == tag:
            return parts[1]
    return None


def _campaign_cases():
    """The 46 seen cases in FIXED order: DEV in frozen-manifest order, then
    CALIBRATION in frozen-manifest order. HELD_OUT is never touched."""
    from autograder.benchmark.manifests import load_manifest
    m = load_manifest("grade_primary")
    cases = []
    for split in ALLOWED_SPLITS:
        for c in m.by_split(split):
            w = c.case_id.split("_")[0]
            assert w in ALLOWED_WRITERS and w not in FORBIDDEN_WRITERS, c.case_id
            cases.append((split, c))
    return m, cases


def build_campaign() -> dict:
    from autograder.benchmark.roles import GradeAdapter
    from autograder.escalation import GRADE_VALIDATION_VERSION, GradeResult, grade_system_for

    manifest, cases = _campaign_cases()
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    audit_flags = {c["case_id"]: c.get("human_decision") for c in audit["cases"]}
    evidence_issue = sorted(cid for cid, d in audit_flags.items() if d == "C")

    ds = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
    finals = json.loads((ds / "final_labels.json").read_text(encoding="utf-8"))["labels"]

    rows = []
    for split, c in cases:
        lab = c.label
        cid = c.case_id
        writer, q, r = cid.split("_")
        assert finals[cid].get("ground_truth_source") == "original_instructor_grade", cid
        derivable = bool(lab.get("explanation_verdict_derivable"))
        flagged = cid in evidence_issue
        rows.append({
            "order": len(rows) + 1,
            "case_id": cid,
            "split": split,
            "writer": writer,
            "question_id": q[1:],
            "sub_item_id": r[1:],
            "actual_instructor_score": finals[cid]["score"],
            "selection_correct": lab.get("selection_correct"),
            "selection_correct_source": lab.get("selection_correct_source"),
            "instructor_derived_verdict": lab.get("explanation_verdict"),
            "verdict_derivable": derivable,
            "verdict_derivation_reason": lab.get("explanation_verdict_reason"),
            "evidence_issue_flag": flagged,
            "audit_flag": audit_flags.get(cid),   # diagnostic metadata ONLY, never a target
            "strict_verdict_eligible": derivable and not flagged,
        })

    n_dev = sum(1 for r in rows if r["split"] == "DEV")
    n_cal = sum(1 for r in rows if r["split"] == "CALIBRATION")
    derivable_n = sum(1 for r in rows if r["verdict_derivable"])
    strict_n = sum(1 for r in rows if r["strict_verdict_eligible"])

    system = grade_system_for(PROMPT_VERSION)
    schema = json.dumps(GradeResult.model_json_schema(), sort_keys=True)
    doc = {
        "campaign": CAMPAIGN_ID,
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": ("larger diagnostic run of the local grader over EVERY seen explanation "
                    "case, to be compared against (a) an independent blind human consensus "
                    "and (b) the original instructor grade, SEPARATELY. The instructor grade "
                    "is a reference source, not infallible truth; human reviewers are not "
                    "automatically infallible either; every source is preserved separately "
                    "and disagreements are adjudicated"),
        "population": {"splits": list(ALLOWED_SPLITS), "writers": list(ALLOWED_WRITERS),
                       "forbidden_writers": "the two held-out writers (ids deliberately "
                                            "withheld from every campaign artifact; enforced "
                                            "by scripts/seen46_campaign.py constants and tests)",
                       "dev": n_dev, "calibration": n_cal, "total": len(rows), "held_out": 0},
        "targets": {
            "ground_truth_policy": ("actual original instructor grades (final_labels.json, "
                                    "ground_truth_source=original_instructor_grade) recorded per "
                                    "case and never modified; instructor-derived verdicts only "
                                    "where uniquely derivable; A/B/C/D audit decisions are "
                                    "diagnostic flags, NEVER targets; no target reaches a model "
                                    "request"),
            "verdict_derivable": derivable_n,
            "non_derivable_diagnostic": len(rows) - derivable_n,
            "evidence_issue_flagged": evidence_issue,
            "strict_clean_denominator": strict_n,
        },
        "model": {"candidate": CANDIDATE, "digest": _ollama_digest(CANDIDATE),
                  "backend": "ollama", "base_url": "http://localhost:11434/v1",
                  "rag_policy": "RAG_DISABLED", "cloud_grading": "blocked (production boundary)"},
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "schema_name": "GradeResult",
        "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        "adapter_version": GradeAdapter.adapter_version,
        "validation_version": GRADE_VALIDATION_VERSION,
        "dataset": {
            "inputs_sha256": manifest.hashes["inputs_sha256"],
            "labels_sha256": manifest.hashes["labels_sha256"],
            "manifest_sha256": _sha_file(ds / "manifest.json"),
            "final_labels_sha256": _sha_file(ds / "final_labels.json"),
        },
        "parent_freeze": {"path": str(FREEZE_PATH.relative_to(REPO)).replace("\\", "/"),
                          "experiment_sha256": freeze["experiment_sha256"]},
        "git_commit": _git_head(),
        "execution": {
            "runs": [
                {"split": "dev", "subset": None, "cases": n_dev},
                {"split": "calibration", "subset": None, "cases": n_cal},
            ],
            "max_local_evaluations": len(rows),
            "cache_policy": "exact-request cache reuse allowed and reported; never cleared",
            "note": "sequential, local-only, one candidate; failures preserved, no replacements",
        },
        "cases": rows,
    }
    payload = json.dumps({k: v for k, v in doc.items() if k != "campaign_sha256"},
                         ensure_ascii=False, sort_keys=True)
    doc["campaign_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return doc


def verify_campaign() -> list[str]:
    problems: list[str] = []
    if not CAMPAIGN_PATH.exists():
        return [f"campaign manifest missing: {CAMPAIGN_PATH}"]
    frozen = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    payload = json.dumps({k: v for k, v in frozen.items() if k != "campaign_sha256"},
                         ensure_ascii=False, sort_keys=True)
    if frozen.get("campaign_sha256") != hashlib.sha256(payload.encode("utf-8")).hexdigest():
        problems.append("campaign_sha256 is not self-consistent (manifest edited?)")
    live = build_campaign()
    for key in ("prompt_sha256", "schema_sha256", "adapter_version", "validation_version"):
        if frozen.get(key) != live.get(key):
            problems.append(f"{key}: frozen {frozen.get(key)!r} != live {live.get(key)!r}")
    if frozen["dataset"] != live["dataset"]:
        problems.append("dataset hashes changed since the campaign freeze")
    if [c["case_id"] for c in frozen["cases"]] != [c["case_id"] for c in live["cases"]]:
        problems.append("case list/order changed")
    for c in frozen["cases"]:
        w = c["case_id"].split("_")[0]
        if w in FORBIDDEN_WRITERS or c["split"] not in ALLOWED_SPLITS:
            problems.append(f"forbidden case in campaign: {c['case_id']}")
    p = frozen["population"]
    if (p["dev"], p["calibration"], p["total"], p["held_out"]) != (32, 14, 46, 0):
        problems.append(f"population counts wrong: {p}")
    return problems


# ---------------------------------------------------------------- leakage ----

FORBIDDEN_REQUEST_TOKENS = (
    # names of evaluation-side fields (the output-schema exemption in
    # runner.leakage_check does not apply to any of these)
    "explanation_verdict", "selection_correct", "label_score", "ground_truth",
    "instructor", "audit", "human_decision", "reviewer",
    # split names + audit letters in decision form
    "DEV", "CALIBRATION", "HELD_OUT",
)


def leakage_proof() -> dict:
    """Build all 46 model requests exactly as the runner would and prove no
    target reaches any of them. ZERO inference — requests are built and
    inspected, never sent."""
    from autograder.benchmark.roles import GradeAdapter
    from autograder.benchmark.runner import files_root_for, leakage_check
    from autograder.benchmark.manifests import DEFAULT_BENCH_ROOT

    frozen = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    by_score = {c["case_id"]: c["actual_instructor_score"] for c in frozen["cases"]}
    manifest, cases = _campaign_cases()
    assert [c.case_id for _, c in cases] == [c["case_id"] for c in frozen["cases"]]

    adapter = GradeAdapter("grade_primary", prompt_version=PROMPT_VERSION)
    root = files_root_for(manifest, DEFAULT_BENCH_ROOT)
    rows, problems = [], []
    for split, c in cases:
        req = adapter.build_request(dict(c.inputs), root)
        leakage_check(c, req, adapter.model_visible_fields)      # raises on any label leak
        text = req.text_for_inspection()
        bad = [t for t in FORBIDDEN_REQUEST_TOKENS if t in text]
        # the instructor score as a bare number is indistinguishable from pack
        # numbers; the structural guarantee is that build_request never sees
        # the label at all — asserted here by construction (inputs only)
        if bad:
            problems.append({"case_id": c.case_id, "tokens": bad})
        rows.append({"case_id": c.case_id,
                     "request_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                     "request_chars": len(text),
                     "model_visible_fields": list(adapter.model_visible_fields),
                     "leakage_check": "passed", "forbidden_tokens": bad})
    doc = {
        "artifact": "seen46_zero_leakage_verification",
        "campaign": CAMPAIGN_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requests_verified": len(rows),
        "problems": problems,
        "verdict": "ZERO-LEAKAGE VERIFIED" if not problems else "LEAKAGE — DO NOT RUN",
        "method": ("every request built exactly as the runner builds it (same adapter, same "
                   "prompt version, same files root); runner.leakage_check per case plus a "
                   "forbidden-token scan; the model-visible whitelist is the adapter's "
                   "(case_id, pack, selected, transcription, version) and the label dict is "
                   "never handed to build_request by construction"),
        "requests": rows,
    }
    return doc


# ------------------------------------------------------------------- gate ----

def find_campaign_runs() -> dict[str, Path]:
    """The two run dirs of this campaign (dev + calibration, full splits,
    campaign candidate/prompt), located by their run.json config."""
    out: dict[str, Path] = {}
    root = RUNS_ROOT / "grade_primary"
    for d in sorted(root.iterdir()):
        rj = d / "run.json"
        if not rj.exists():
            continue
        cfg = json.loads(rj.read_text(encoding="utf-8")).get("config", {})
        if (cfg.get("candidate") == CANDIDATE and cfg.get("prompt_version") == PROMPT_VERSION
                and cfg.get("subset") is None and cfg.get("split") in ("DEV", "CALIBRATION")):
            out[cfg["split"]] = d
    return out


def gate() -> dict:
    frozen = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    runs = find_campaign_runs()
    problems: list[str] = []
    if set(runs) != {"DEV", "CALIBRATION"}:
        problems.append(f"expected DEV+CALIBRATION full-split runs, found {sorted(runs)}")
    rows = {}
    usage_total = {"live": 0, "cache_hits": 0, "failures": 0, "input_tokens": 0, "output_tokens": 0}
    lats = []
    for split, d in runs.items():
        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
        cfg = run["config"]
        for key, want in (("prompt_sha256", frozen["prompt_sha256"]),
                          ("schema_sha256", frozen["schema_sha256"]),
                          ("adapter_version", frozen["adapter_version"]),
                          ("backend", "ollama")):
            if cfg.get(key) != want:
                problems.append(f"{split}: {key} mismatch")
        if "localhost" not in str(cfg.get("base_url")):
            problems.append(f"{split}: not a localhost route")
        outs = [json.loads(l) for l in (d / "outputs.jsonl").open(encoding="utf-8")]
        latest = {}
        for r in outs:
            if r.get("ok") is not None:
                latest[r["case_id"]] = r
        for cid, r in latest.items():
            w = cid.split("_")[0]
            if w in FORBIDDEN_WRITERS:
                problems.append(f"FORBIDDEN writer executed: {cid}")
            rows[cid] = r
            if r.get("ok"):
                if r.get("cache_hit"):
                    usage_total["cache_hits"] += 1
                else:
                    usage_total["live"] += 1
                    if r.get("latency_s") is not None:
                        lats.append(float(r["latency_s"]))
                usage_total["input_tokens"] += int((r.get("usage") or {}).get("input_tokens") or 0)
                usage_total["output_tokens"] += int((r.get("usage") or {}).get("output_tokens") or 0)
            else:
                usage_total["failures"] += 1
    want_ids = [c["case_id"] for c in frozen["cases"]]
    missing = sorted(set(want_ids) - set(rows))
    extra = sorted(set(rows) - set(want_ids))
    if missing:
        problems.append(f"missing outputs for {len(missing)} case(s): {missing[:5]}")
    if extra:
        problems.append(f"unexpected outputs: {extra[:5]}")
    import statistics
    summary = {
        "artifact": "seen46_model_run_summary",
        "campaign": CAMPAIGN_ID,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": {s: str(d.relative_to(REPO)).replace("\\", "/") for s, d in runs.items()},
        "cases_intended": len(want_ids),
        "cases_with_output": len([c for c in rows.values() if c.get("ok")]),
        "failures_preserved": usage_total["failures"],
        "dev": sum(1 for cid in rows if cid.split("_")[0] != "e004"),
        "calibration": sum(1 for cid in rows if cid.split("_")[0] == "e004"),
        "held_out": 0,
        "usage": {**usage_total,
                  "latency_median_s": round(statistics.median(lats), 2) if lats else None,
                  "latency_max_s": round(max(lats), 2) if lats else None,
                  "latency_total_s": round(sum(lats), 1) if lats else None,
                  "cloud_calls": 0, "cloud_cost_usd": 0.0},
        "gate": "PASS" if not problems else "FAIL",
        "problems": problems,
        "policy": ("model outputs are FROZEN as of this gate: no prompt/config change, no "
                   "re-scoring from human outcomes, labels untouched"),
    }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["freeze", "verify", "leakage", "gate"])
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        if CAMPAIGN_PATH.exists():
            print(f"REFUSED: {CAMPAIGN_PATH.name} already exists — a campaign is never re-frozen")
            return 3
        doc = build_campaign()
        CAMPAIGN_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
        print(f"wrote {CAMPAIGN_PATH.name}")
        print(f"campaign_sha256 {doc['campaign_sha256']}")
        print(f"dev {doc['population']['dev']} calibration {doc['population']['calibration']} "
              f"total {doc['population']['total']} held_out {doc['population']['held_out']} | "
              f"derivable {doc['targets']['verdict_derivable']} "
              f"strict {doc['targets']['strict_clean_denominator']}")
        return 0
    if args.cmd == "verify":
        problems = verify_campaign()
        if problems:
            print("CAMPAIGN MISMATCH:")
            for p in problems:
                print(" -", p)
            return 2
        print("campaign verified: manifest matches the live repository")
        return 0
    if args.cmd == "leakage":
        doc = leakage_proof()
        LEAKAGE_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8")
        print(f"{doc['verdict']}: {doc['requests_verified']} requests -> {LEAKAGE_PATH.name}")
        return 0 if not doc["problems"] else 2
    if args.cmd == "gate":
        summary = gate()
        SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8")
        u = summary["usage"]
        md = [f"# SEEN-46 model run — {CAMPAIGN_ID}", "",
              f"Gate: **{summary['gate']}**. {summary['cases_with_output']} outputs / "
              f"{summary['cases_intended']} intended ({summary['failures_preserved']} preserved "
              f"failures); DEV {summary['dev']} + CALIBRATION {summary['calibration']}, "
              f"HELD_OUT 0. Live calls {u['live']}, cache hits {u['cache_hits']}, median "
              f"{u['latency_median_s']}s, max {u['latency_max_s']}s, total live time "
              f"{u['latency_total_s']}s, cloud calls 0 (cost $0).", "",
              summary["policy"], ""]
        if summary["problems"]:
            md.insert(2, "PROBLEMS: " + "; ".join(map(str, summary["problems"])) + "")
        SUMMARY_MD.write_text("\n".join(md), encoding="utf-8", newline="\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "runs"} | {"runs": summary["runs"]},
                         indent=1)[:1200])
        return 0 if summary["gate"] == "PASS" else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
