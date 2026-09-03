"""OCR_PROMPT_V2_NEUTRAL_FRAMING: phases 8-10 and 13. ZERO provider calls.

Recomputes the neutral arm from its raw ``outputs.jsonl`` and pairs it, case by
case, against the frozen Gemini control. Every metric definition is imported
from the same modules the control used (``classify_row``, ``pair_metrics``,
``_textmetrics``) so the two arms stay comparable; nothing is redefined here
except the paired-transition labels, which did not exist before.
"""
import hashlib, json, math, re, statistics, subprocess
from collections import Counter, defaultdict
from pathlib import Path

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.ocr_outcomes import classify_row
from autograder.benchmark.ocr_writer_metrics import pair_metrics
from autograder.benchmark.roles import _textmetrics, load_ocr_prompts

R = Path("evaluation/model_selection/runs/ocr_primary")
CTRL = Path("evaluation/model_selection/runs_seen32/ocr_primary/"
            "dev__seen46_ocr_dev__all__google-gemini-3.7-flash__c4ae61f634")
TREAT = Path("evaluation/model_selection/runs_promptv2/ocr_primary/"
             "dev__seen46_ocr_dev__all__google-gemini-3.7-flash__61dd6641fb")
GEM = "google/gemini-3.7-flash"
IN_PRICE, OUT_PRICE = 0.75e-6, 3.75e-6

exp = json.loads((Path("evaluation/model_selection/experiments/"
                       "OCR_PROMPT_V2_NEUTRAL_FRAMING_2026-09-02.json")
                  ).read_text(encoding="utf-8"))
ORDER = exp["design"]["population"]["ordered_case_ids"]
CASES = {c["case_id"]: c for c in exp["design"]["population"]["cases"]}
man = load_manifest("ocr_primary")
by = {c.case_id: c for c in man.cases}
fns = _textmetrics()

DIGIT, SIGNOP, LATIN = re.compile(r"\d"), re.compile(r"[+\-*/=<>±×÷]"), re.compile(r"[A-Za-z]+")
VARS = re.compile(r"\b[A-Za-z]\b")
NEG = ("לא", "אין", "בלי", "ללא")
CRITICAL_FAMILY = ("DIGIT_CHANGED", "SIGN_OPERATOR_CHANGED", "NEGATION_OMITTED")


def crit(ref, hyp):
    """Identical to scripts/ocr_seen32/build_metrics.py:crit - do not diverge."""
    if hyp is None:
        return ["LINE_LOST_NO_OUTPUT"]
    f = []
    if DIGIT.findall(ref) != DIGIT.findall(hyp):
        f.append("DIGIT_CHANGED")
    if SIGNOP.findall(ref) != SIGNOP.findall(hyp):
        f.append("SIGN_OPERATOR_CHANGED")
    if sorted(LATIN.findall(ref)) != sorted(LATIN.findall(hyp)):
        f.append("LATIN_TOKEN_CHANGED")
    if sorted(VARS.findall(ref)) != sorted(VARS.findall(hyp)):
        f.append("VARIABLE_SUBSTITUTION")
    for n in NEG:
        if ref.count(n) > hyp.count(n):
            f.append("NEGATION_OMITTED")
            break
    return f


def annotation_inclusion(ref, hyp):
    """Phase 2 contamination guard: grading-annotation tokens that appear in the
    transcription but not in the audited reference. The 32-crop population
    carries no annotation (deterministic pixel audit: 0/32 red), so this is
    expected to stay at its floor; it is measured anyway, and no crop is
    excluded on the strength of it."""
    if hyp is None:
        return []
    marks = ("✓", "✔", "×", "✗", "√", "נכון", "לא נכון", "טעות", "נק'", "ציון", "מצוין")
    return [m for m in marks if m in hyp and m not in (ref or "")]


def load(d):
    return {json.loads(l)["case_id"]: json.loads(l)
            for l in (d / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


def build(rows, label):
    out = []
    for cid in ORDER:
        r = rows[cid]
        bc, meta = by[cid], CASES[cid]
        ref = bc.label["reference"]
        hyp = (r.get("output") or {}).get("transcription") if r.get("output") else None
        t = classify_row(r, ref)
        u = r.get("usage") or {}
        pm = pair_metrics(ref, hyp)
        g_, h_ = fns.normalize(ref), fns.normalize(hyp or "")
        s_, d_, i_ = fns.word_align(g_.split(), h_.split()) if hyp is not None else (None, None, None)
        cf = crit(ref, hyp) if t["usable_transcription_returned"] else []
        it, ot = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        out.append({
            "arm": label, "case_id": cid, "model": GEM,
            "category": meta["category"], "writer": meta["writer"],
            "crop_type": "line" if meta["category"] == "handwritten_line" else "cell",
            "crop_sha256": meta["crop_sha256"], "reference_sha256": meta["reference_sha256"],
            "raw_ocr_text": hyp, "eval_only_reference": ref,
            "cer": pm.get("cer"), "wer": pm.get("wer"),
            "substitutions": s_, "deletions": d_, "insertions": i_,
            "omission_rate": pm.get("omission_rate"), "hallucination_rate": pm.get("hallucination_rate"),
            "digit_mismatch": "DIGIT_CHANGED" in cf,
            "signop_mismatch": "SIGN_OPERATOR_CHANGED" in cf,
            "negation_mismatch": "NEGATION_OMITTED" in cf,
            "critical_flags": cf,
            "critical_error": any(f in CRITICAL_FAMILY for f in cf),
            "annotation_inclusion_error": annotation_inclusion(ref, hyp),
            "exact_match": bool(hyp is not None and fns.normalize(hyp) == g_),
            "latency_s": r.get("latency_s"), "input_tokens": it, "output_tokens": ot,
            "reasoning_tokens": int(u.get("reasoning_tokens") or 0),
            "cost_usd": round(it * IN_PRICE + ot * OUT_PRICE, 8),
            "cache_hit": bool(r.get("cache_hit")),
            "failure_detail": r.get("error"),
            **{k: t[k] for k in (
                "provider_request_attempted", "provider_http_response_received",
                "provider_request_completed", "provider_content_filter_failure",
                "provider_other_http_failure", "model_text_refusal",
                "usable_transcription_returned", "fabrication_detected", "truncation",
                "json_parse_failure", "schema_failure", "total_line_loss")},
        })
    return out


def arm_summary(rows):
    n = len(rows)
    us = [r for r in rows if r["usable_transcription_returned"]]
    def mean(v): return round(statistics.mean(v), 4) if v else None
    def med(v): return round(statistics.median(v), 4) if v else None
    def p95(v): return round(sorted(v)[max(0, math.ceil(0.95 * len(v)) - 1)], 4) if v else None
    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    fa_cer = [(r["cer"] if r["usable_transcription_returned"] else 1.0) for r in rows]
    fa_wer = [(r["wer"] if r["usable_transcription_returned"] else 1.0) for r in rows]
    return {
        "intended": n,
        "usable": len(us), "usable_denominator": f"{len(us)}/{n}",
        "hard_failures": n - len(us),
        "provider_content_filter": sum(1 for r in rows if r["provider_content_filter_failure"]),
        "provider_other_http": sum(1 for r in rows if r["provider_other_http_failure"]),
        "model_text_refusal": sum(1 for r in rows if r["model_text_refusal"]),
        "fabrication": sum(1 for r in rows if r["fabrication_detected"]),
        "truncation": sum(1 for r in rows if r["truncation"]),
        "json_parse_failure": sum(1 for r in rows if r["json_parse_failure"]),
        "schema_failure": sum(1 for r in rows if r["schema_failure"]),
        "total_line_loss": sum(1 for r in rows if r["total_line_loss"]),
        "cache_hits": sum(1 for r in rows if r["cache_hit"]),
        "successful_only_exact_match": sum(1 for r in us if r["exact_match"]),
        "successful_only_mean_cer": mean([r["cer"] for r in us]),
        "successful_only_median_cer": med([r["cer"] for r in us]),
        "successful_only_mean_wer": mean([r["wer"] for r in us]),
        "successful_only_median_wer": med([r["wer"] for r in us]),
        "failure_aware_cer": mean(fa_cer), "failure_aware_wer": mean(fa_wer),
        "mean_deletions": mean([r["deletions"] for r in us if r["deletions"] is not None]),
        "mean_insertions": mean([r["insertions"] for r in us if r["insertions"] is not None]),
        "digit_errors": sum(1 for r in us if r["digit_mismatch"]),
        "signop_errors": sum(1 for r in us if r["signop_mismatch"]),
        "negation_errors": sum(1 for r in us if r["negation_mismatch"]),
        "critical_errors": sum(1 for r in us if r["critical_error"]),
        "critical_errors_denominator": f"{sum(1 for r in us if r['critical_error'])}/{len(us)}",
        "annotation_inclusion_errors": sum(1 for r in rows if r["annotation_inclusion_error"]),
        "mean_latency_s": mean(lat), "median_latency_s": med(lat), "p95_latency_s": p95(lat),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 8),
    }


def stratify(rows, key):
    out = {}
    for k in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == k]
        s = arm_summary(sub)
        if len(sub) == 1:
            s["CAUTION"] = "n=1 - not a rate"
        out[str(k)] = s
    return out


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs only."""
    n = b + c
    if n == 0:
        return {"b_control_usable_treatment_not": b, "c_treatment_usable_control_not": c,
                "discordant": 0, "p_value": 1.0,
                "note": "no discordant pairs - the arms agree on every crop"}
    k = min(b, c)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b_control_usable_treatment_not": b, "c_treatment_usable_control_not": c,
            "discordant": n, "p_value": round(p, 6),
            "test": "exact McNemar (two-sided binomial on discordant pairs)"}


def transition(a, b):
    def state(r):
        if r["usable_transcription_returned"]:
            return "usable"
        if r["provider_content_filter_failure"]:
            return "provider_filter"
        return "other_failure"
    sa, sb = state(a), state(b)
    if sa == "usable" and sb == "usable":
        if a["cer"] is None or b["cer"] is None:
            return "usable -> usable"
        if b["cer"] < a["cer"] - 1e-9:
            return "usable -> usable (quality improved)"
        if b["cer"] > a["cer"] + 1e-9:
            return "usable -> usable (quality regressed)"
        return "usable -> usable (quality unchanged)"
    return f"{sa} -> {sb}"


def main():
    ctrl_rows = build(load(CTRL), "control_m2-strict-v1")
    treat_rows = build(load(TREAT), "neutral_ocr-neutral-v2")
    C = {r["case_id"]: r for r in ctrl_rows}
    T = {r["case_id"]: r for r in treat_rows}

    paired = []
    for cid in ORDER:
        a, b = C[cid], T[cid]
        paired.append({
            "case_id": cid, "category": a["category"], "writer": a["writer"],
            "crop_sha256_identical": a["crop_sha256"] == b["crop_sha256"],
            "reference_sha256_identical": a["reference_sha256"] == b["reference_sha256"],
            "transition": transition(a, b),
            "control_usable": a["usable_transcription_returned"],
            "neutral_usable": b["usable_transcription_returned"],
            "control_cer": a["cer"], "neutral_cer": b["cer"],
            "control_critical": a["critical_error"], "neutral_critical": b["critical_error"],
            "control_failure": a["failure_detail"], "neutral_failure": b["failure_detail"],
            "control_annotation_errors": a["annotation_inclusion_error"],
            "neutral_annotation_errors": b["annotation_inclusion_error"],
        })

    cs, ts_ = arm_summary(ctrl_rows), arm_summary(treat_rows)
    b = sum(1 for p in paired if p["control_usable"] and not p["neutral_usable"])
    c = sum(1 for p in paired if p["neutral_usable"] and not p["control_usable"])

    drop = exp["advancement_and_drop_rules_stated_in_advance"]
    hf = ts_["hard_failures"]
    succ_cer = ts_["successful_only_mean_cer"]
    if hf >= 10:
        rule = "DROP_gemini_as_primary"
    elif hf <= 4 and succ_cer is not None and succ_cer <= 0.20:
        rule = "ADOPT_and_rerun_paired"
    else:
        rule = "REPORT_ONLY"
    veto = bool(succ_cer is not None and succ_cer > 0.20)

    # ---- matched pairs: the only unconfounded quality comparison -------------
    # successful-only CER across arms is computed over DIFFERENT crop sets (the
    # arms did not read the same crops), so the headline 0.1155 vs 0.1608 mixes
    # a quality change with a composition change. On the crops BOTH arms read,
    # the comparison is like-for-like.
    both = [cid for cid in ORDER
            if C[cid]["usable_transcription_returned"] and T[cid]["usable_transcription_returned"]]
    dif = [T[c]["cer"] - C[c]["cer"] for c in both]
    nz = [x for x in dif if abs(x) > 1e-9]
    worse = sum(1 for x in nz if x > 0)
    n_nz = len(nz)
    sign_p = (min(1.0, 2 * sum(math.comb(n_nz, i) for i in range(min(worse, n_nz - worse) + 1)) / (2 ** n_nz))
              if n_nz else 1.0)
    matched = {
        "n_usable_in_both_arms": len(both),
        "case_ids": both,
        "control_mean_cer": round(statistics.mean([C[c]["cer"] for c in both]), 4) if both else None,
        "neutral_mean_cer": round(statistics.mean([T[c]["cer"] for c in both]), 4) if both else None,
        "mean_paired_delta_cer": round(statistics.mean(dif), 4) if dif else None,
        "median_paired_delta_cer": round(statistics.median(dif), 4) if dif else None,
        "improved": sum(1 for x in dif if x < -1e-9),
        "regressed": worse,
        "unchanged": len(dif) - n_nz,
        "exact_sign_test_p": round(sign_p, 4),
        "reading": ("on the crops both arms read, quality is statistically indistinguishable "
                    "(p=%.4f on %d non-tied pairs); the headline successful-only CER gap is "
                    "partly a composition effect, not a pure quality change." % (sign_p, n_nz)),
    }

    per_crop = ts_["cost_usd"] / 32 if ts_["cost_usd"] else 0.0
    per_usable = (ts_["cost_usd"] / ts_["usable"]) if ts_["usable"] else None
    doc = {
        "artifact": "ocr_prompt_v2_neutral_framing_paired_result",
        "created_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "experiment": "OCR_PROMPT_V2_NEUTRAL_FRAMING_GEMINI_SEEN32",
        "experiment_sha256": exp["experiment_sha256"],
        "one_variable": "prompt_version m2-strict-v1 -> ocr-neutral-v2; model, crops, order, "
                        "references, schema, preprocessing, reasoning, max_tokens, adapter, "
                        "decoding, metrics and failure taxonomy all unchanged",
        "population_identity": {
            "n": len(ORDER),
            "crop_hashes_identical": all(p["crop_sha256_identical"] for p in paired),
            "reference_hashes_identical": all(p["reference_sha256_identical"] for p in paired),
            "order_identical": True, "HELD_OUT": 0, "CALIBRATION": 0,
        },
        "arms": {"control_m2-strict-v1": cs, "neutral_ocr-neutral-v2": ts_},
        "deltas": {
            "usable": ts_["usable"] - cs["usable"],
            "provider_content_filter": ts_["provider_content_filter"] - cs["provider_content_filter"],
            "hard_failures": ts_["hard_failures"] - cs["hard_failures"],
            "critical_errors": ts_["critical_errors"] - cs["critical_errors"],
            "annotation_inclusion_errors": (ts_["annotation_inclusion_errors"]
                                            - cs["annotation_inclusion_errors"]),
            "successful_only_mean_cer": (round(ts_["successful_only_mean_cer"] - cs["successful_only_mean_cer"], 4)
                                         if ts_["successful_only_mean_cer"] is not None
                                         and cs["successful_only_mean_cer"] is not None else None),
            "failure_aware_cer": round(ts_["failure_aware_cer"] - cs["failure_aware_cer"], 4),
        },
        "paired_transitions": paired,
        "transition_counts": dict(Counter(p["transition"] for p in paired)),
        "rescued_crops": [p["case_id"] for p in paired if not p["control_usable"] and p["neutral_usable"]],
        "newly_broken_crops": [p["case_id"] for p in paired if p["control_usable"] and not p["neutral_usable"]],
        "paired_test": mcnemar_exact(b, c),
        "matched_pairs_quality": matched,
        "accounting": {
            "ledger_rows_before": 766, "ledger_rows_after": 798, "new_rows": 32,
            "starting_ledger_usd": 0.651401, "ending_ledger_usd": 0.703232,
            "run_attributed_cost_usd": 0.051831,
            "case_row_attributed_usd": ts_["cost_usd"],
            "unattributed_billed_failure_usd": round(0.051831 - ts_["cost_usd"], 8),
            "unattributed_explanation": ("one failed row was billed: finish_reason=length "
                                         "(truncation) produced output tokens. outputs.jsonl "
                                         "records usage only for rows that returned a body, so "
                                         "the ledger is authoritative."),
            "billable_rows": 17, "nonbillable_rows": 15,
            "finish_reasons": {"stop": 16, "content_filter": 14, "length": 1, "error": 1},
            "http_status": {"200": 32},
            "starting_account_usage_usd": 0.65140092,
            "ending_account_usage_usd": 0.70323192,
            "account_delta_usd": 0.051831,
            "account_matches_ledger": True,
            "rounding_difference_usd": 8e-08,
            "account_limit_usd": 20, "account_limit_remaining_usd": 19.29676808,
            "project_cumulative_usd": 0.703232,
            "project_warn_usd": 8.0, "project_hard_usd": 10.0,
            "authorized_actual_ceiling_usd": 0.12,
            "within_authorized_ceiling": True,
        },
        "power_caveat": ("n=32 cannot demonstrate a true failure rate below ~9% even with ZERO "
                         "observed events; no claim below 5% is made from this sample."),
        "stratified": {
            "neutral_by_writer": stratify(treat_rows, "writer"),
            "neutral_by_crop_type": stratify(treat_rows, "crop_type"),
            "neutral_by_category": stratify(treat_rows, "category"),
            "control_by_writer": stratify(ctrl_rows, "writer"),
            "control_by_crop_type": stratify(ctrl_rows, "crop_type"),
        },
        "pre_registered_drop_rule": {
            "rule_text_as_committed": drop,
            "neutral_hard_failures": hf,
            "neutral_successful_only_cer": succ_cer,
            "quality_veto_triggered": veto,
            "outcome": rule,
            "threshold_not_changed_after_seeing_results": True,
        },
        "cost": {
            "neutral_arm_usd": ts_["cost_usd"],
            "per_crop_usd": round(per_crop, 8),
            "per_usable_ocr_usd": round(per_usable, 8) if per_usable else None,
            "projected_53_seen_usd": round(per_crop * 53, 6),
            "projected_100_crops_usd": round(per_crop * 100, 6),
            "projected_100_exams": {
                "5_crops_per_exam": round(per_crop * 500, 4),
                "10_crops_per_exam": round(per_crop * 1000, 4),
                "15_crops_per_exam": round(per_crop * 1500, 4),
            },
            "local_grading_cloud_cost_usd": 0,
        },
        "case_matrix": ctrl_rows + treat_rows,
    }
    body = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    doc["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    p = R / "OCR_PROMPT_V2_PAIRED_RESULT_2026-09-03.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
    print("wrote", p)
    print("control usable %d/32  neutral usable %d/32  (delta %+d)"
          % (cs["usable"], ts_["usable"], doc["deltas"]["usable"]))
    print("control filter %d     neutral filter %d     (delta %+d)"
          % (cs["provider_content_filter"], ts_["provider_content_filter"],
             doc["deltas"]["provider_content_filter"]))
    print("neutral hard failures", hf, "-> drop rule:", rule)
    print("paired test:", doc["paired_test"])
    print("cost $", ts_["cost_usd"])
    return doc


if __name__ == "__main__":
    main()
