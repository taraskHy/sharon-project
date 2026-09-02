"""Phases 7-11 + overnight A/B/D/E/F/G: case-level matrix, stratified metrics,
reliability bounds, fallback replay, routing comparison. ZERO provider calls."""
import hashlib, json, math, os, re, statistics, subprocess, time, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.ocr_fallback import POLICY_ID, replay
from autograder.benchmark.ocr_outcomes import classify_row, summarize
from autograder.benchmark.ocr_writer_metrics import pair_metrics
from autograder.benchmark.roles import _textmetrics

R = Path("evaluation/model_selection/runs/ocr_primary")
S32 = Path("evaluation/model_selection/runs_seen32/ocr_primary")
GEM, SON = "google/gemini-3.7-flash", "anthropic/claude-sonnet-5"
DIRS = {GEM: S32 / "dev__seen46_ocr_dev__all__google-gemini-3.7-flash__c4ae61f634",
        SON: S32 / "dev__seen46_ocr_dev__all__anthropic-claude-sonnet-5__2f3a7c346c"}
base = json.loads((R / "OCR_SEEN32_BASELINE_2026-09-02.json").read_text(encoding="utf-8"))
ORDER = base["ordered_case_ids"]
man = load_manifest("ocr_primary")
by = {c.case_id: c for c in man.cases}
fns = _textmetrics()
DIGIT, SIGNOP, LATIN = re.compile(r"\d"), re.compile(r"[+\-*/=<>±×÷]"), re.compile(r"[A-Za-z]+")
VARS = re.compile(r"\b[A-Za-z]\b")
NEG = ("לא", "אין", "בלי", "ללא")

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


def crit(ref, hyp):
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


rows_by_model, tax_by_model, matrix = {}, {}, []
for slug, d in DIRS.items():
    rows = {json.loads(l)["case_id"]: json.loads(l)
            for l in (d / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    rows_by_model[slug] = rows
    tax_by_model[slug] = {cid: classify_row(rows[cid], by[cid].label["reference"]) for cid in ORDER}

for slug in (GEM, SON):
    for cid in ORDER:
        r = rows_by_model[slug][cid]
        bc = by[cid]
        ref = bc.label["reference"]
        hyp = (r.get("output") or {}).get("transcription") if r.get("output") else None
        t = tax_by_model[slug][cid]
        u = r.get("usage") or {}
        pm = pair_metrics(ref, hyp)
        g_, h_ = fns.normalize(ref), fns.normalize(hyp or "")
        s_, d_, i_ = fns.word_align(g_.split(), h_.split()) if hyp is not None else (None, None, None)
        img = man.root / bc.inputs["image"]
        dims = None
        if HAVE_PIL:
            try:
                with Image.open(img) as im:
                    dims = {"w": im.width, "h": im.height,
                            "aspect": round(im.width / im.height, 3), "format": im.format}
            except Exception:
                dims = None
        matrix.append({
            "case_id": cid, "model": slug, "writer": bc.meta.get("writer"),
            "crop_type": "line" if cid.startswith("hl_") else "cell",
            "category": bc.meta.get("category"), "hard": bc.label.get("hard"),
            "crop_bytes": img.stat().st_size, "crop_dims": dims,
            "crop_sha256": hashlib.sha256(img.read_bytes()).hexdigest(),
            "provider_finish_or_error": str(r.get("error") or "stop"),
            **{k: v for k, v in t.items() if k != "failure_detail"},
            "raw_ocr_text": hyp,
            "eval_only_reference": ref, "eval_only_reference_chars": len(ref),
            "eval_only_reference_digits": len(DIGIT.findall(ref)),
            "eval_only_reference_signops": len(SIGNOP.findall(ref)),
            "cer": pm["cer"], "wer": pm["wer"],
            "deletions": d_, "insertions": i_, "substitutions": s_,
            "omission_rate": pm["omission_rate"], "hallucination_rate": pm["hallucination_rate"],
            "digit_mismatch": hyp is not None and DIGIT.findall(ref) != DIGIT.findall(hyp),
            "signop_mismatch": hyp is not None and SIGNOP.findall(ref) != SIGNOP.findall(hyp),
            "negation_mismatch": hyp is not None and any(ref.count(n) > hyp.count(n) for n in NEG),
            "critical_flags": crit(ref, hyp),
            "latency_s": r.get("latency_s"), "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"), "reasoning_tokens": u.get("reasoning_tokens"),
            "cache_hit": bool(r.get("cache_hit")),
        })

MX = {(m["model"], m["case_id"]): m for m in matrix}


def block(slug, ids, label):
    rows = [MX[(slug, c)] for c in ids]
    ok = [r for r in rows if r["usable_transcription_returned"]]
    cers = [r["cer"] for r in ok if r["cer"] is not None]
    wers = [r["wer"] for r in ok if r["wer"] is not None]
    bound = [(r["cer"] if r["usable_transcription_returned"] and r["cer"] is not None else 1.0)
             for r in rows]
    boundw = [(r["wer"] if r["usable_transcription_returned"] and r["wer"] is not None else 1.0)
              for r in rows]
    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    return {
        "label": label, "intended": len(rows), "usable": len(ok),
        "usable_coverage": f"{len(ok)}/{len(rows)}",
        "usable_rate": round(len(ok) / len(rows), 4) if rows else None,
        "provider_content_filter": sum(1 for r in rows if r["provider_content_filter_failure"]),
        "provider_other_http": sum(1 for r in rows if r["provider_other_http_failure"]),
        "model_text_refusal": sum(1 for r in rows if r["model_text_refusal"]),
        "truncation": sum(1 for r in rows if r["truncation"]),
        "json_parse_failure": sum(1 for r in rows if r["json_parse_failure"]),
        "schema_failure": sum(1 for r in rows if r["schema_failure"]),
        "total_line_loss": sum(1 for r in rows if r["total_line_loss"]),
        "exact_match": sum(1 for r in ok if r["raw_ocr_text"] == r["eval_only_reference"]),
        "succ_mean_cer": round(statistics.mean(cers), 4) if cers else None,
        "succ_median_cer": round(statistics.median(cers), 4) if cers else None,
        "succ_mean_wer": round(statistics.mean(wers), 4) if wers else None,
        "succ_median_wer": round(statistics.median(wers), 4) if wers else None,
        "failure_aware_cer": round(statistics.mean(bound), 4) if bound else None,
        "failure_aware_wer": round(statistics.mean(boundw), 4) if boundw else None,
        "mean_omission": round(statistics.mean([r["omission_rate"] for r in ok]), 4) if ok else None,
        "mean_hallucination": round(statistics.mean([r["hallucination_rate"] for r in ok]), 4) if ok else None,
        "digit_mismatches": sum(1 for r in ok if r["digit_mismatch"]),
        "signop_mismatches": sum(1 for r in ok if r["signop_mismatch"]),
        "negation_mismatches": sum(1 for r in ok if r["negation_mismatch"]),
        "critical_error_cases": sum(1 for r in ok if r["critical_flags"]),
        "mean_latency_s": round(statistics.mean(lat), 3) if lat else None,
        "median_latency_s": round(statistics.median(lat), 3) if lat else None,
        "p95_latency_s": (round(sorted(lat)[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)], 3)
                          if lat else None),
        "input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "output_tokens": sum(r["output_tokens"] or 0 for r in rows),
        "reasoning_tokens": sum(r["reasoning_tokens"] or 0 for r in rows),
        "cache_hits": sum(1 for r in rows if r["cache_hit"]),
    }


LINE = [c for c in ORDER if c.startswith("hl_")]
CELL = [c for c in ORDER if c.startswith("hc_")]
WRITERS = defaultdict(list)
for c in ORDER:
    WRITERS[by[c].meta.get("writer")].append(c)

strat = {}
for slug in (GEM, SON):
    strat[slug] = {
        "all32": block(slug, ORDER, "all 32 (handwritten)"),
        "line": block(slug, LINE, "line crops"),
        "cell": block(slug, CELL, "cell crops"),
        "by_writer": {w: {**block(slug, ids, f"writer {w}"),
                          "too_small_to_interpret": len(ids) < 5}
                      for w, ids in sorted(WRITERS.items())},
        "by_category": {cat: block(slug, [c for c in ORDER if by[c].meta.get("category") == cat], cat)
                        for cat in sorted({by[c].meta.get("category") for c in ORDER})},
        "hard_flagged": block(slug, [c for c in ORDER if by[c].label.get("hard")], "hard=True"),
    }


# ---- E: exact one-sided upper bounds (Clopper-Pearson) ---------------------
def cp_upper(k, n, conf=0.95):
    """One-sided upper bound on a binomial rate; exact (Clopper-Pearson)."""
    if k >= n:
        return 1.0
    lo, hi = (k / n if n else 0.0), 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        # P(X <= k | p=mid)
        p = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        if p > 1 - conf:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def n_for_upper(target, conf=0.95):
    """Cases needed, assuming ZERO events, for a one-sided upper bound < target."""
    n = 1
    while n < 100000:
        if cp_upper(0, n, conf) < target:
            return n
        n += 1
    return None


gt = tax_by_model[GEM]
hard_fail_g = sum(1 for c in ORDER if not gt[c]["usable_transcription_returned"])
cf_g = sum(1 for c in ORDER if gt[c]["provider_content_filter_failure"])
reliability = {
    "gemini": {
        "intended": 32,
        "hard_provider_failures": hard_fail_g,
        "hard_failure_rate": round(hard_fail_g / 32, 4),
        "hard_failure_upper95": cp_upper(hard_fail_g, 32),
        "content_filter": cf_g, "content_filter_rate": round(cf_g / 32, 4),
        "content_filter_upper95": cp_upper(cf_g, 32),
        "usable": 32 - hard_fail_g, "usable_rate": round((32 - hard_fail_g) / 32, 4),
        "band": ("promising (0-1)" if hard_fail_g <= 1 else
                 "uncertain (2-4)" if hard_fail_g <= 4 else
                 "unsuitable as the sole OCR route under this configuration (5+)"),
    },
    "sonnet": {
        "intended": 32,
        "hard_provider_failures": sum(1 for c in ORDER
                                      if tax_by_model[SON][c]["provider_content_filter_failure"]
                                      or tax_by_model[SON][c]["provider_other_http_failure"]
                                      or tax_by_model[SON][c]["truncation"]
                                      or tax_by_model[SON][c]["json_parse_failure"]
                                      or tax_by_model[SON][c]["schema_failure"]),
        "model_text_refusals": sum(1 for c in ORDER if tax_by_model[SON][c]["model_text_refusal"]),
        "usable": sum(1 for c in ORDER if tax_by_model[SON][c]["usable_transcription_returned"]),
    },
    "sample_size_needed_assuming_zero_events": {
        f"{int(t*100)}%": n_for_upper(t) for t in (0.20, 0.15, 0.10, 0.05, 0.02, 0.01)},
    "caveat": ("n=32 cannot demonstrate a true failure rate below ~9% even with ZERO observed "
               "events; no claim below 5% is made from this sample."),
}

# ---- B: failure association, deterministic metadata only -------------------
def assoc(field_fn, name):
    out = {}
    for c in ORDER:
        k = field_fn(c)
        b = out.setdefault(str(k), {"n": 0, "gem_usable": 0, "gem_filter": 0})
        b["n"] += 1
        b["gem_usable"] += int(gt[c]["usable_transcription_returned"])
        b["gem_filter"] += int(gt[c]["provider_content_filter_failure"])
    for b in out.values():
        b["usable_rate"] = round(b["gem_usable"] / b["n"], 4)
        b["filter_rate"] = round(b["gem_filter"] / b["n"], 4)
    return {"dimension": name, "groups": out}


def bucket(v, edges):
    for e in edges:
        if v <= e:
            return f"<={e}"
    return f">{edges[-1]}"


association = {
    "writer": assoc(lambda c: by[c].meta.get("writer"), "writer"),
    "crop_type": assoc(lambda c: "line" if c.startswith("hl_") else "cell", "crop_type"),
    "category": assoc(lambda c: by[c].meta.get("category"), "category"),
    "hard_flag": assoc(lambda c: by[c].label.get("hard"), "hard flag"),
    "crop_bytes": assoc(lambda c: bucket(MX[(GEM, c)]["crop_bytes"], [8000, 16000, 32000]), "crop bytes"),
    "reference_chars": assoc(lambda c: bucket(len(by[c].label["reference"]), [40, 80, 120]),
                             "reference length (eval-only)"),
    "reference_has_digits": assoc(lambda c: bool(DIGIT.findall(by[c].label["reference"])),
                                  "reference contains digits (eval-only)"),
    "_discipline": ("observed ASSOCIATION only, never established cause. n=32 with 10 filter events "
                    "cannot separate these dimensions, several of which are collinear."),
}
if HAVE_PIL:
    association["aspect_ratio"] = assoc(
        lambda c: bucket((MX[(GEM, c)]["crop_dims"] or {}).get("aspect", 0), [4, 8, 16]), "aspect ratio")

# ---- historical cross-check: did these crops ever succeed elsewhere? -------
HIST = {"stage1c_gemini": Path("evaluation/model_selection/runs_stage1c/ocr_primary/"
                               "dev__smoke__all__google-gemini-3.7-flash__45297cdd83"),
        "stage1b_gemini": R / "dev__smoke__all__google-gemini-3.7-flash__45297cdd83",
        "stage1_sonnet": R / "dev__smoke__all__anthropic-claude-sonnet-5__0481873207",
        "stage1_luna": R / "dev__smoke__all__openai-gpt-5.6-luna-pro__c6f10f3603"}
hist = {}
for name, d in HIST.items():
    if not (d / "outputs.jsonl").exists():
        continue
    hr = {json.loads(l)["case_id"]: json.loads(l)
          for l in (d / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    hist[name] = {cid: bool((hr[cid].get("output") or {}).get("transcription"))
                  for cid in hr if cid in ORDER}
gem_failed = [c for c in ORDER if not gt[c]["usable_transcription_returned"]]
historical_cross = {
    "gemini_failed_here": gem_failed,
    "of_those_previously_usable": {name: sorted(c for c in gem_failed if m.get(c))
                                   for name, m in hist.items()},
}

# ---- fallback replay (prospective, reference-blind) ------------------------
gem_text = {c: (rows_by_model[GEM][c].get("output") or {}).get("transcription") for c in ORDER}
son_text = {c: (rows_by_model[SON][c].get("output") or {}).get("transcription") for c in ORDER}
fb = replay(ORDER, tax_by_model[GEM], tax_by_model[SON], gem_text, son_text,
            primary_model=GEM, secondary_model=SON)


def route_block(chosen_text, chosen_model, label, review_ids=()):
    cers, wers, crits, n_usable = [], [], 0, 0
    bound, boundw = [], []
    for c in ORDER:
        t = chosen_text.get(c)
        if t is not None:
            n_usable += 1
            m = pair_metrics(by[c].label["reference"], t)
            cers.append(m["cer"]); wers.append(m["wer"])
            bound.append(m["cer"]); boundw.append(m["wer"])
            if crit(by[c].label["reference"], t):
                crits += 1
        else:
            bound.append(1.0); boundw.append(1.0)
    return {"strategy": label, "intended": len(ORDER), "usable": n_usable,
            "usable_coverage": f"{n_usable}/{len(ORDER)}",
            "human_review_cases": len(review_ids),
            "human_review_rate": round(len(review_ids) / len(ORDER), 4),
            "succ_mean_cer": round(statistics.mean(cers), 4) if cers else None,
            "succ_mean_wer": round(statistics.mean(wers), 4) if wers else None,
            "failure_aware_cer": round(statistics.mean(bound), 4),
            "failure_aware_wer": round(statistics.mean(boundw), 4),
            "critical_error_cases": crits}


gem_only = {c: (gem_text[c] if tax_by_model[GEM][c]["usable_transcription_returned"] else None) for c in ORDER}
son_only = {c: (son_text[c] if tax_by_model[SON][c]["usable_transcription_returned"] else None) for c in ORDER}
fb_text = {d["case_id"]: d["chosen_text"] for d in fb["decisions"]}
fb_unres = [d["case_id"] for d in fb["decisions"] if not d["resolved"]]
gem_fail_ids = [c for c in ORDER if not tax_by_model[GEM][c]["usable_transcription_returned"]]

routing = {
    "1_gemini_only": {**route_block(gem_only, GEM, "Gemini only", gem_fail_ids),
                      "note": "hard failures go unresolved unless a human reads them"},
    "2_sonnet_only": {**route_block(son_only, SON, "Sonnet only",
                                    [c for c in ORDER if not tax_by_model[SON][c]["usable_transcription_returned"]])},
    "3_gemini_then_sonnet": {**route_block(fb_text, None, "Gemini -> Sonnet on hard failure", fb_unres),
                             "primary_used": fb["primary_used"], "fallback_used": fb["fallback_used"],
                             "unresolved": fb["unresolved"], "needs_review": fb["needs_review"]},
    "4_gemini_then_human": {**route_block(gem_only, GEM, "Gemini -> human review on hard failure",
                                          gem_fail_ids),
                            "note": "identical text coverage to strategy 1; the difference is that every "
                                    "hard failure becomes explicit review workload"},
    "5_gemini_then_sonnet_then_human": {
        **route_block(fb_text, None, "Gemini -> Sonnet -> human if fallback unusable", fb_unres),
        "note": "same as 3; every fallback row is additionally flagged needs_review"},
}
# oracle, for context only
oracle_text = {}
for c in ORDER:
    cands = []
    for slug, txt in ((GEM, gem_text[c]), (SON, son_text[c])):
        if tax_by_model[slug][c]["usable_transcription_returned"]:
            cands.append((pair_metrics(by[c].label["reference"], txt)["cer"], txt))
    oracle_text[c] = min(cands)[1] if cands else None
routing["ORACLE_best_of_two"] = {
    **route_block(oracle_text, None, "best-of-two by hidden reference", []),
    "WARNING": "NOT DEPLOYABLE - HIDDEN-REFERENCE ORACLE. Upper bound for context only; never a routing recommendation.",
}

# ---- G: is Sonnet actually a useful fallback? ------------------------------
trigger_ids = gem_fail_ids
son_on_trigger = block(SON, trigger_ids, "Sonnet on Gemini-fallback-trigger cases")
son_on_nontrigger = block(SON, [c for c in ORDER if c not in trigger_ids],
                          "Sonnet on cases Gemini handled")
rescue = {
    "gemini_trigger_cases": len(trigger_ids),
    "sonnet_rescued": son_on_trigger["usable"],
    "sonnet_failed_to_rescue": len(trigger_ids) - son_on_trigger["usable"],
    "rescue_rate": round(son_on_trigger["usable"] / len(trigger_ids), 4) if trigger_ids else None,
    "rescue_quality_succ_mean_cer": son_on_trigger["succ_mean_cer"],
    "sonnet_quality_on_cases_gemini_handled": son_on_nontrigger["succ_mean_cer"],
    "harder_subset": ("compare the two CER figures: if rescue quality is much worse, the crops Gemini "
                      "loses are also the crops Sonnet reads worst"),
    "on_trigger_detail": son_on_trigger, "on_nontrigger_detail": son_on_nontrigger,
}

# ---- accounting -----------------------------------------------------------
led = [json.loads(l) for l in Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl")
       .read_text(encoding="utf-8").splitlines() if l.strip()]
cost = lambda r: (r.get("cost_usd") or r.get("reported_cost") or 0)
new = led[702:]
per_model_cost = defaultdict(float)
for r in new:
    per_model_cost[r.get("model")] += cost(r)
delta = round(sum(cost(r) for r in new), 8)
# Account reconciliation needs the provider's own usage figure. This is a
# METADATA call (/key), never inference, and it is skipped when the credential
# is absent so the script stays fully offline-runnable. The key is read from the
# environment and never printed, logged or persisted.
acct = None
k = os.environ.get("OPENROUTER_API_KEY", "")
if k.strip():
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": "Bearer " + k})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            acct = json.loads(r.read().decode())["data"]
    except Exception:
        acct = None
if acct is None:
    acct = {"usage": base.get("starting_account_usage"), "limit": None, "limit_remaining": None}
start_usage = base["starting_account_usage"]
gem_cost = round(per_model_cost[GEM], 8)
son_cost = round(per_model_cost[SON], 8)
gem_usable = strat[GEM]["all32"]["usable"]
son_usable = strat[SON]["all32"]["usable"]
fb_cost = gem_cost + round(son_cost / 32 * fb["fallback_used"], 8)
COUNTS = {"all_53_seen": 53, "per_100_crops": 100, "100_exams_at_5": 500,
          "100_exams_at_10": 1000, "100_exams_at_15": 1500}
accounting = {
    "starting_ledger_rows": base["starting_ledger"]["rows"], "ending_ledger_rows": len(led),
    "new_rows": len(new), "attributed_cost_usd": delta,
    "gemini_cost_usd": gem_cost, "sonnet_cost_usd": son_cost,
    "billable": sum(1 for r in new if cost(r) > 0),
    "nonbillable_failures": sum(1 for r in new if cost(r) == 0),
    "cache_hits": strat[GEM]["all32"]["cache_hits"] + strat[SON]["all32"]["cache_hits"],
    "starting_account_usage": start_usage, "ending_account_usage": acct["usage"],
    "account_delta_usd": round(acct["usage"] - start_usage, 8),
    "rounding_difference": round(delta - (acct["usage"] - start_usage), 8),
    "account_limit": acct["limit"], "account_limit_remaining": acct["limit_remaining"],
    "ceiling_usd": 0.40, "within_ceiling": delta < 0.40,
    "project_cumulative_usd": round(sum(cost(r) for r in led), 8),
}
projections = {
    "assumptions": ("cost per ATTEMPTED crop (32 attempts per arm), same crop mix - all handwritten - "
                    "one pass, no retries. Gemini's rate benefits from 10 free content-filter rows, so "
                    "its cost-per-USABLE transcription is the fairer planning number and is given too."),
    "gemini": {"per_attempt": round(gem_cost / 32, 8),
               "per_usable": round(gem_cost / gem_usable, 8) if gem_usable else None,
               "projections_per_attempt": {k2: round(gem_cost / 32 * n, 6) for k2, n in COUNTS.items()},
               "projections_per_usable": ({k2: round(gem_cost / gem_usable * n, 6) for k2, n in COUNTS.items()}
                                          if gem_usable else None)},
    "sonnet": {"per_attempt": round(son_cost / 32, 8),
               "per_usable": round(son_cost / son_usable, 8) if son_usable else None,
               "projections_per_attempt": {k2: round(son_cost / 32 * n, 6) for k2, n in COUNTS.items()},
               "projections_per_usable": ({k2: round(son_cost / son_usable * n, 6) for k2, n in COUNTS.items()}
                                          if son_usable else None)},
    "paired_per_crop": round((gem_cost + son_cost) / 32, 8),
    "prospective_fallback_composite": {
        "measured_total_usd": round(fb_cost, 8), "per_crop": round(fb_cost / 32, 8),
        "note": f"gemini on all 32 plus sonnet on the {fb['fallback_used']} fallback-triggered crops",
        "projections": {k2: round(fb_cost / 32 * n, 6) for k2, n in COUNTS.items()}},
    "grading_cost": {"cloud_grading_usd": 0.0, "local_grading_cloud_usd": 0.0},
}

art = {
    "artifact": "ocr_seen32_paired_result",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "experiment": base["experiment"], "experiment_sha256": base["experiment_sha256"],
    "provider_requests": 64, "cache_hits": accounting["cache_hits"],
    "population": {"intended": 32, "composition": base["composition"],
                   "ordered_case_ids": ORDER,
                   "case_order_sha256": base["case_order_sha256"]},
    "case_matrix": matrix,
    "stratified": strat,
    "reliability": reliability,
    "failure_association": association,
    "historical_cross_check": historical_cross,
    "fallback": {"policy_id": POLICY_ID, **{k2: v for k2, v in fb.items() if k2 != "decisions"},
                 "decisions": fb["decisions"]},
    "routing_comparison": routing,
    "sonnet_as_fallback": rescue,
    "accounting": accounting, "cost_projections": projections,
}
body = json.dumps(art, ensure_ascii=False, indent=1, default=str)
art["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p = R / "OCR_SEEN32_PAIRED_RESULT_2026-09-02.json"
p.write_text(json.dumps(art, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("wrote", p)
print()
f4 = lambda x: "n/a" if x is None else f"{x:.4f}"
print(f"{'MODEL':28} {'usable':8} {'filter':7} {'refuse':7} {'trunc':6} {'parse':6} {'succCER':8} {'faCER':8}")
for slug in (GEM, SON):
    b = strat[slug]["all32"]
    print(f"{slug.split('/')[-1][:28]:28} {b['usable_coverage']:8} {b['provider_content_filter']:<7} "
          f"{b['model_text_refusal']:<7} {b['truncation']:<6} {b['json_parse_failure']:<6} "
          f"{f4(b['succ_mean_cer']):8} {f4(b['failure_aware_cer']):8}")
print()
print("reliability:", json.dumps(reliability["gemini"], ensure_ascii=False))
print("sample size needed (zero events):", json.dumps(reliability["sample_size_needed_assuming_zero_events"]))
print()
print("fallback:", fb["primary_used"], "primary /", fb["fallback_used"], "fallback /",
      fb["unresolved"], "unresolved | triggers", json.dumps(fb["triggers"]))
print("rescue:", json.dumps({k2: v for k2, v in rescue.items()
                             if k2 not in ("on_trigger_detail", "on_nontrigger_detail")}, ensure_ascii=False))
print()
for name, r in routing.items():
    print(f"{name:34} coverage {r['usable_coverage']:6} review {r['human_review_cases']:<3} "
          f"succCER {f4(r['succ_mean_cer'])} faCER {f4(r['failure_aware_cer'])} crit {r['critical_error_cases']}")
print()
print("accounting:", delta, "| account delta", accounting["account_delta_usd"],
      "| rounding", accounting["rounding_difference"])
