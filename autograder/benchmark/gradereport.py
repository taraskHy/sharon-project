"""Two-layer report for the grading benchmarks (GRADE_PRIMARY / GRADE_ESCALATE).

Owner directive (2026-08-28): the authoritative ground truth for grading
evaluation is the ACTUAL instructor-assigned grade from the original graded
test (final_labels.json, ground_truth_source=original_instructor_grade). The
owner's blind A/B/C/D audit decisions, model-majority votes and previous
cloud-model predictions are diagnostic metadata ONLY: they may flag a
rubric-practice mismatch, an evidence/transcription concern or an ambiguity,
but they never replace, modify or determine an expected label.

The report therefore has TWO layers that are never combined:

    A. LOCAL GRADER QUALITY
       Model canonical explanation verdict vs the instructor-DERIVED verdict,
       only where that verdict is mathematically identifiable from
       (instructor score, selection correctness, frozen production policy).
       For the frozen DEV population: 26 cases = 22 valid + 4 partially_valid
       + 0 invalid.

    B. END-TO-END TEST-GRADE AGREEMENT
       System predicted final score vs the actual instructor score over the
       WHOLE split population:

           wrong selection (audited)  -> 0.0 through the production selection
                                         gate, deterministically; a local
                                         grading call is normally unnecessary
           selection correct          -> max_points * factor(model verdict)
                                         (verdicts.final_score_for, the
                                         production composition verbatim)

       Wrong-selection cases live ONLY in this layer: their final score is
       zero because of the selection, so a final-score match proves nothing
       about explanation judgement, and they carry no explanation ground
       truth (Layer A excludes them by construction).

Every target is RE-DERIVED from the instructor score at report time
(``verdicts.derive_verdict``); a frozen label that disagrees refuses the
report instead of being silently reinterpreted. The report also refuses a
run whose recorded manifest hashes no longer match the dataset on disk.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .manifests import (DEFAULT_BENCH_ROOT, DEFAULT_DATASETS_ROOT, REPO_ROOT, BenchCase,
                        BenchmarkManifest, load_manifest)
from .roles import VERDICT_RANK, _pct, _verdict_metrics
from .verdicts import CANONICAL_VERDICTS, derive_verdict, final_score_for

#: the frozen local experiment; its human_audit block is the only audit source
#: this report reads, and it is rendered as FLAGS, never applied to a label.
DEFAULT_EXPERIMENT_PATH = (REPO_ROOT / "evaluation" / "model_selection" / "experiments"
                           / "LOCAL_GRADE_PRIMARY_FREEZE_2026-08-27.json")

GRADING_ROLES = ("grade_primary", "grade_escalate")

#: what an audit decision MEANS, and the only way it may surface in a report.
AUDIT_DECISIONS = {
    "A": {"meaning": "derived verdict is consistent with the rubric and instructor practice",
          "flag": None},
    "B": {"meaning": "instructor practice is more lenient than the literal encoded rubric",
          "flag": "rubric_practice_mismatch"},
    "C": {"meaning": "transcription/evidence/rubric artifact is incomplete or incorrect",
          "flag": "evidence_transcription_concern"},
    "D": {"meaning": "genuinely ambiguous",
          "flag": "ambiguity"},
}

GROUND_TRUTH_STATEMENT = (
    "All benchmark targets in this report originate from the ACTUAL instructor-assigned "
    "grades on the original graded tests (final_labels.json, "
    "ground_truth_source=original_instructor_grade), inverted through the frozen "
    "production scoring policy. The owner's blind A/B/C/D audit decisions, "
    "model-majority votes and previous cloud-model predictions are NOT used as "
    "expected labels anywhere in this report; audit decisions appear only as "
    "diagnostic flags. Every target was re-derived from the instructor score at "
    "report time and matched the frozen label.")


class GradeReportError(RuntimeError):
    """The report refuses rather than reinterpret: wrong role, drifted dataset,
    or a frozen label that disagrees with the instructor-score derivation."""


# ----------------------------------------------------------------------------
# target verification (the directive, enforced)
# ----------------------------------------------------------------------------

def _max_points(case: BenchCase) -> float:
    v = case.label.get("max_score")
    if v is None:
        v = (case.inputs.get("pack") or {}).get("max_score")
    return float(v if v is not None else 4.0)


def verify_targets_against_instructor(cases: list[BenchCase]) -> dict[str, Any]:
    """Re-derive every case's explanation-verdict target from (instructor
    score, selection correctness, frozen production policy) and compare with
    the frozen label. Any disagreement is a refusal, not a warning: a label
    that cannot be reproduced from the instructor grade is not a target this
    report is allowed to use.

    Compares the (derivable, verdict) pair only — unresolved cases carry more
    specific frozen reason codes (e.g. exam version unconfirmed) than the
    generic derivation can know about, and that extra precision is not a
    disagreement.
    """
    problems: list[str] = []
    for c in cases:
        lab = c.label
        score = lab.get("score")
        if score is None:
            problems.append(f"{c.case_id}: no authoritative instructor score on the label")
            continue
        if lab.get("ground_truth_source") != "original_instructor_grade":
            problems.append(f"{c.case_id}: ground_truth_source is "
                            f"{lab.get('ground_truth_source')!r}, not the original instructor grade")
        d = derive_verdict(case_id=c.case_id, instructor_final_score=float(score),
                           selection_correct=lab.get("selection_correct"),
                           max_points=_max_points(c),
                           transcription=str(c.inputs.get("transcription") or ""))
        if bool(lab.get("explanation_verdict_derivable")) != d.derivable:
            problems.append(f"{c.case_id}: frozen derivable={lab.get('explanation_verdict_derivable')} "
                            f"but the instructor-score derivation says {d.derivable}")
        elif (lab.get("explanation_verdict") or None) != d.derived_explanation_verdict:
            problems.append(f"{c.case_id}: frozen verdict {lab.get('explanation_verdict')!r} "
                            f"!= instructor-derived {d.derived_explanation_verdict!r}")
    if problems:
        raise GradeReportError(
            "refusing to report: frozen labels disagree with the instructor-score "
            "derivation (the owner directive makes the instructor grade the only "
            "authoritative target) — " + "; ".join(problems[:5])
            + (f" … and {len(problems) - 5} more" if len(problems) > 5 else ""))
    return {"verified_cases": len(cases),
            "derivation": "reliability._verdict_from_score + grade._verdict_factor + "
                          "grade._grade_sub_item (explanation_required, weight==0 branch)",
            "instructor_score_source": "final_labels.json (ground_truth_source=original_instructor_grade)"}


# ----------------------------------------------------------------------------
# per-case join
# ----------------------------------------------------------------------------

def _selection_state(label: dict) -> str:
    sc = label.get("selection_correct")
    if sc is False:
        return "wrong_selection_audited"
    if sc is True:
        return "correct_audited"
    if (label.get("score") or 0) > 0:
        # full or partial credit is unreachable through the production gate
        # unless the selection was correct — the state is implied, not assumed
        return "correct_implied_by_credit"
    return "unresolved"


def _case_row(case: BenchCase, scored: dict | None) -> dict[str, Any]:
    lab = case.label
    actual = float(lab["score"])
    max_points = _max_points(case)
    state = _selection_state(lab)
    model_verdict = (scored or {}).get("predicted_verdict")
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "split": case.split,
        "actual_instructor_score": actual,
        "max_points": max_points,
        "selection_state": state,
        "explanation_target": lab.get("explanation_verdict"),
        "explanation_target_derivable": bool(lab.get("explanation_verdict_derivable")),
        "transcription_complete": lab.get("transcription_complete", True) is not False,
        "model_ran": scored is not None,
        "model_schema_failure": bool((scored or {}).get("schema_failure")) if scored else None,
        "model_verdict": model_verdict,
        "model_decision": (scored or {}).get("decision"),
        "predicted_final": None,
        "predicted_final_basis": None,
        "layer_b_bucket": None,
    }
    if state == "wrong_selection_audited":
        row["predicted_final"] = 0.0
        row["predicted_final_basis"] = ("production_selection_gate: wrong selection -> "
                                        "deterministic 0.0 (a local grading call is "
                                        "normally unnecessary)")
        row["layer_b_bucket"] = "policy_deterministic_zero"
        if model_verdict is not None:
            # a raw-split run may still have graded it; the output is diagnostic
            # only and never changes the deterministic zero
            row["model_verdict_diagnostic_only"] = True
    elif state == "unresolved":
        row["layer_b_bucket"] = "excluded_selection_unresolved"
        row["predicted_final_basis"] = lab.get("explanation_verdict_reason")
    elif model_verdict is not None:
        row["predicted_final"] = final_score_for(selection_correct=True, verdict=model_verdict,
                                                 max_points=max_points)
        row["predicted_final_basis"] = "model_verdict_via_production_policy"
        row["layer_b_bucket"] = "model_scored"
    else:
        row["layer_b_bucket"] = ("no_automated_score_schema_failure" if scored is not None
                                 else "no_automated_score_not_in_run")
    if row["predicted_final"] is not None:
        delta = row["predicted_final"] - actual
        row["final_exact"] = abs(delta) < 1e-9
        row["final_abs_error"] = round(abs(delta), 4)
        row["harmful_overgrade"] = delta > 1e-9
        row["harmful_undergrade"] = delta < -1e-9
    return row


# ----------------------------------------------------------------------------
# aggregates
# ----------------------------------------------------------------------------

def _final_agreement(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"cases": 0}
    errs = [r["final_abs_error"] for r in rows]
    exact = sum(1 for r in rows if r["final_exact"])
    return {"cases": n,
            "final_exact": exact,
            "final_exact_pct": _pct(exact, n),
            "final_score_mae": round(statistics.mean(errs), 4),
            "harmful_overgrades": sum(1 for r in rows if r["harmful_overgrade"]),
            "harmful_undergrades": sum(1 for r in rows if r["harmful_undergrade"])}


def _fmt_score(v: float) -> str:
    return f"{v:g}"


def _confusion_by_actual(rows: list[dict]) -> dict[str, dict[str, int]]:
    """rows = every population case EXCEPT the excluded (selection-unresolved)
    ones; a case without an automated score lands in its own column so the
    denominator is never silently narrowed."""
    out: dict[str, dict[str, int]] = {}
    for r in sorted(rows, key=lambda r: r["actual_instructor_score"]):
        a = _fmt_score(r["actual_instructor_score"])
        col = (_fmt_score(r["predicted_final"]) if r["predicted_final"] is not None
               else "no_automated_score")
        out.setdefault(a, {})
        out[a][col] = out[a].get(col, 0) + 1
    return out


def _layer_a(rows: list[dict]) -> dict[str, Any]:
    derivable = [r for r in rows if r["explanation_target_derivable"] and r["transcription_complete"]]
    judged = []
    for r in derivable:
        mv, truth = r["model_verdict"], r["explanation_target"]
        if mv is None:
            continue
        judged.append({"case_id": r["case_id"], "label_verdict": truth, "predicted_verdict": mv,
                       "verdict_exact": mv == truth,
                       "harmful_verdict_upgrade": VERDICT_RANK.get(mv, 0) > VERDICT_RANK.get(truth, 0),
                       "harmful_verdict_downgrade": VERDICT_RANK.get(mv, 0) < VERDICT_RANK.get(truth, 0)})
    support: dict[str, int] = {}
    for r in derivable:
        support[r["explanation_target"]] = support.get(r["explanation_target"], 0) + 1
    out = {
        "target": "canonical explanation verdict, derived from the instructor grade "
                  "(NOT from audit decisions, model votes or prior model predictions)",
        "population_derivable": len(derivable),
        "class_support": support,
        "scored": len(judged),
        "not_scored_schema_failure": sum(1 for r in derivable
                                         if r["model_ran"] and r["model_verdict"] is None),
        "not_scored_not_in_run": sum(1 for r in derivable if not r["model_ran"]),
        "excluded_wrong_selection": sum(1 for r in rows
                                        if r["selection_state"] == "wrong_selection_audited"),
        "excluded_wrong_selection_note": (
            "wrong-selection cases carry no explanation ground truth (their zero was "
            "decided by the selection, the explanation was never scored) and never "
            "count toward explanation-model accuracy"),
        "invalid_class_note": ("`invalid` has zero authoritative support in this dataset; "
                               "performance on invalid explanations is NOT MEASURED"),
    }
    if judged:
        out.update(_verdict_metrics(judged, CANONICAL_VERDICTS))
        out["cases"] = [{k: j[k] for k in ("case_id", "label_verdict", "predicted_verdict",
                                           "verdict_exact")} for j in judged]
    return out


def _layer_b(rows: list[dict]) -> dict[str, Any]:
    included = [r for r in rows if r["layer_b_bucket"] != "excluded_selection_unresolved"]
    with_score = [r for r in included if r["predicted_final"] is not None]
    model_scored = [r for r in with_score if r["layer_b_bucket"] == "model_scored"]
    det = [r for r in with_score if r["layer_b_bucket"] == "policy_deterministic_zero"]
    return {
        "target": "actual instructor-assigned final score from the original graded test",
        "prediction": "model explanation verdict + actual selection correctness + frozen "
                      "production scoring policy",
        "population": len(rows),
        "excluded_selection_unresolved": sorted(r["case_id"] for r in rows
                                                if r["layer_b_bucket"] == "excluded_selection_unresolved"),
        "no_automated_score_schema_failure": sorted(
            r["case_id"] for r in included if r["layer_b_bucket"] == "no_automated_score_schema_failure"),
        "not_in_run": sorted(
            r["case_id"] for r in included if r["layer_b_bucket"] == "no_automated_score_not_in_run"),
        "full_system": _final_agreement(with_score),
        "model_scored_subpopulation": {
            **_final_agreement(model_scored),
            "note": "cases whose predicted score depends on the model's verdict "
                    "(selection correct); this is where grading quality shows"},
        "wrong_selection_policy_report": {
            "cases": sorted(r["case_id"] for r in det),
            "predicted": "0.0 for every case, deterministically, through the production "
                         "selection gate — before any explanation judgement",
            "agreement": _final_agreement(det),
            "note": "wrong selection -> deterministic zero -> a local grading call should "
                    "normally be avoided for these; a final-score match here proves "
                    "nothing about explanation judgement, so these cases are reported "
                    "separately and never enter Layer A"},
        "confusion_by_actual_score": _confusion_by_actual(included),
    }


# ----------------------------------------------------------------------------
# audit flags (report-only)
# ----------------------------------------------------------------------------

def _audit_section(experiment_path: Path | None, population_ids: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "policy": ("audit decisions are diagnostic flags only (rubric-practice mismatch / "
                   "evidence-transcription concern / ambiguity); they never replace, modify "
                   "or determine an expected label, and no case in this report was excluded "
                   "or relabelled because of one"),
        "flags_in_population": [],
        "decisions_outside_population": 0,
    }
    p = Path(experiment_path) if experiment_path else None
    if p is None or not p.exists():
        out["source"] = None
        return out
    exp = json.loads(p.read_text(encoding="utf-8"))
    decisions = ((exp.get("human_audit") or {}).get("decisions") or {})
    out["source"] = str(p)
    for cid, dec in sorted(decisions.items()):
        info = AUDIT_DECISIONS.get(dec, {"meaning": f"unknown decision {dec!r}", "flag": "unknown"})
        if cid not in population_ids:
            out["decisions_outside_population"] += 1
            continue
        out["flags_in_population"].append({"case_id": cid, "decision": dec,
                                           "meaning": info["meaning"], "flag": info["flag"]})
    return out


# ----------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------

def build_grade_report(run_dir: Path, *, datasets_root: Path = DEFAULT_DATASETS_ROOT,
                       bench_root: Path = DEFAULT_BENCH_ROOT,
                       manifest: BenchmarkManifest | None = None,
                       experiment_path: Path | None = DEFAULT_EXPERIMENT_PATH) -> dict[str, Any]:
    d = Path(run_dir)
    run_p, scored_p = d / "run.json", d / "scored.jsonl.json"
    if not run_p.exists():
        raise GradeReportError(f"{d} holds no run.json")
    run = json.loads(run_p.read_text(encoding="utf-8"))
    cfg = run.get("config") or {}
    role, split = cfg.get("role"), cfg.get("split")
    if role not in GRADING_ROLES:
        raise GradeReportError(f"two-layer grade reports apply to {GRADING_ROLES}; "
                               f"this run is {role!r}")
    if not scored_p.exists():
        raise GradeReportError(f"{d} has no scored.jsonl.json — dry runs and unfinished runs "
                               "cannot be reported; execute the run first")
    manifest = manifest or load_manifest(role, bench_root=Path(bench_root),
                                         datasets_root=Path(datasets_root))
    if cfg.get("manifest_hashes") and dict(cfg["manifest_hashes"]) != dict(manifest.hashes):
        raise GradeReportError(
            "the dataset on disk no longer matches the hashes this run was executed "
            "against; a report would compare model output with labels the model never "
            f"ran under (run: {cfg.get('manifest_hashes')}; disk: {manifest.hashes})")

    population = manifest.by_split(split, cfg.get("component"))
    if not population:
        raise GradeReportError(f"no {split} cases in the {role} manifest")
    provenance = verify_targets_against_instructor(population)

    scored_rows = json.loads(scored_p.read_text(encoding="utf-8"))
    by_id = {r["case_id"]: r for r in scored_rows}
    stray = sorted(set(by_id) - {c.case_id for c in population})
    rows = [_case_row(c, by_id.get(c.case_id)) for c in sorted(population, key=lambda c: c.case_id)]

    report = {
        "report": "two_layer_grade_report",
        "created_from": str(d),
        "run_id": run.get("run_id"),
        "candidate": cfg.get("candidate"),
        "role": role, "split": split, "subset": cfg.get("subset"),
        "backend": cfg.get("backend"), "base_url": cfg.get("base_url"),
        "prompt_version": cfg.get("prompt_version"),
        "adapter_version": cfg.get("adapter_version"),
        "git_commit": run.get("git_commit"),
        "dataset_hashes": dict(manifest.hashes),
        "ground_truth_statement": GROUND_TRUTH_STATEMENT,
        "target_provenance": provenance,
        "layer_a_local_grader_quality": _layer_a(rows),
        "layer_b_end_to_end_test_grade_agreement": _layer_b(rows),
        "audit_decisions": _audit_section(experiment_path, {c.case_id for c in population}),
        "cases": rows,
        "layers_note": "Layer A and Layer B answer different questions and are never combined.",
    }
    if stray:
        report["scored_rows_outside_population"] = stray
    return report


# ----------------------------------------------------------------------------
# markdown rendering
# ----------------------------------------------------------------------------

def _md_table(headers: list[str], body: list[list[Any]]) -> str:
    def row(cells):
        return "| " + " | ".join(str(c) for c in cells) + " |"
    return "\n".join([row(headers), row(["---"] * len(headers)), *[row(r) for r in body]])


def render_markdown(report: dict[str, Any]) -> str:
    a = report["layer_a_local_grader_quality"]
    b = report["layer_b_end_to_end_test_grade_agreement"]
    lines: list[str] = []
    add = lines.append
    add(f"# Two-layer grading report — {report['candidate']} ({report['split']}"
        + (f", subset {report['subset']}" if report.get("subset") else "") + ")")
    add("")
    add(f"> **Ground-truth provenance.** {report['ground_truth_statement']}")
    add("")
    add(f"Run `{report['run_id']}` | prompt `{report['prompt_version']}` | "
        f"backend `{report['backend']}` @ `{report['base_url'] or '-'}` | "
        f"adapter `{report['adapter_version']}` | commit `{(report['git_commit'] or '')[:10]}`")
    add("")

    add("## A. LOCAL GRADER QUALITY — explanation verdict vs instructor-derived verdict")
    add("")
    supp = ", ".join(f"{k} {v}" for k, v in sorted(a.get("class_support", {}).items(),
                                                   key=lambda kv: -VERDICT_RANK.get(kv[0], 0)))
    add(f"Population: **{a['population_derivable']} derivable cases** ({supp}; invalid 0 — "
        "NOT MEASURED). "
        f"Scored: **{a['scored']}**"
        + (f"; schema failures {a['not_scored_schema_failure']}" if a["not_scored_schema_failure"] else "")
        + (f"; not in this run {a['not_scored_not_in_run']}" if a["not_scored_not_in_run"] else "")
        + f". The {a['excluded_wrong_selection']} wrong-selection cases are excluded by "
          "construction (no explanation ground truth).")
    add("")
    if a.get("verdict_confusion"):
        add(_md_table(
            ["metric", "value"],
            [["verdict exact", f"{a['verdict_exact_pct']}% ({sum(1 for c in a['cases'] if c['verdict_exact'])}/{a['scored']})"],
             ["balanced accuracy", a["verdict_balanced_accuracy"]],
             ["macro-F1", a["verdict_macro_f1"]],
             ["harmful verdict upgrades", a["harmful_verdict_upgrades"]],
             ["harmful verdict downgrades", a["harmful_verdict_downgrades"]]]))
        add("")
        classes = [c for c in CANONICAL_VERDICTS]
        add("Confusion (instructor-derived truth rows x model columns):")
        add("")
        add(_md_table(["truth \\ model", *classes],
                      [[t, *[a["verdict_confusion"].get(t, {}).get(p, 0) for p in classes]]
                       for t in classes]))
    else:
        add("No derivable case was scored in this run.")
    add("")

    add("## B. END-TO-END TEST-GRADE AGREEMENT — predicted final score vs actual instructor score")
    add("")
    add(f"Population: **{b['population']} cases** (the whole {report['split']} split). "
        "Predicted final score = model explanation verdict + actual selection correctness + "
        "frozen production scoring policy.")
    add("")
    fs, ms = b["full_system"], b["model_scored_subpopulation"]
    det = b["wrong_selection_policy_report"]

    def agree_rows(block):
        return [["cases scored", block.get("cases", 0)],
                ["exact final-score match", f"{block.get('final_exact_pct')}% ({block.get('final_exact')}/{block.get('cases')})"],
                ["mean absolute score error", block.get("final_score_mae")],
                ["harmful overgrades (predicted > actual)", block.get("harmful_overgrades")],
                ["harmful undergrades (predicted < actual)", block.get("harmful_undergrades")]]

    add("**Full system** (model-scored cases + the deterministic selection-gate zeros):")
    add("")
    add(_md_table(["metric", "value"], agree_rows(fs)))
    add("")
    add("**Model-scored subpopulation** (selection correct; the score depends on the model):")
    add("")
    add(_md_table(["metric", "value"], agree_rows(ms)))
    add("")
    conf = b["confusion_by_actual_score"]
    pred_cols = sorted({c for row in conf.values() for c in row},
                       key=lambda c: (c == "no_automated_score", c))
    add("Confusion by actual instructor score (rows) vs predicted final score (columns):")
    add("")
    add(_md_table(["actual \\ predicted", *pred_cols],
                  [[k, *[conf[k].get(c, 0) for c in pred_cols]] for k in sorted(conf)]))
    add("")
    add(f"**Wrong-selection sub-report** ({len(det['cases'])} cases: {', '.join(det['cases'])}). "
        f"{det['note']}.")
    if b.get("excluded_selection_unresolved"):
        add("")
        add(f"Excluded (selection correctness unresolved): "
            f"{', '.join(b['excluded_selection_unresolved'])}.")
    if b.get("no_automated_score_schema_failure"):
        add("")
        add(f"No automated score — schema failure, production sends these to human review: "
            f"{', '.join(b['no_automated_score_schema_failure'])}.")
    if b.get("not_in_run"):
        add("")
        add(f"Not executed in this run ({len(b['not_in_run'])} cases outside the run's subset); "
            "they count in the population above but have no prediction here.")
    add("")

    aud = report["audit_decisions"]
    add("## Audit decisions (diagnostic flags only)")
    add("")
    add(aud["policy"] + ".")
    add("")
    if aud["flags_in_population"]:
        add(_md_table(["case", "decision", "flag", "meaning"],
                      [[f["case_id"], f["decision"], f["flag"] or "none (consistent)", f["meaning"]]
                       for f in aud["flags_in_population"]]))
    else:
        add(f"No audited case is in this population"
            + (f" ({aud['decisions_outside_population']} decision(s) exist on other splits)"
               if aud["decisions_outside_population"] else "") + ".")
    add("")
    add(f"_{report['layers_note']}_")
    add("")
    return "\n".join(lines)


def write_grade_report(run_dir: Path, **kwargs) -> dict[str, Any]:
    """Build and persist the two-layer report next to the run's own artifacts."""
    d = Path(run_dir)
    report = build_grade_report(d, **kwargs)
    (d / "two_layer_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (d / "two_layer_report.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")
    report["written"] = [str(d / "two_layer_report.json"), str(d / "two_layer_report.md")]
    return report


__all__ = ["GradeReportError", "AUDIT_DECISIONS", "GROUND_TRUTH_STATEMENT",
           "DEFAULT_EXPERIMENT_PATH", "build_grade_report", "render_markdown",
           "write_grade_report", "verify_targets_against_instructor"]
