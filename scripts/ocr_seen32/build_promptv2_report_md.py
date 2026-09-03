import json, pathlib

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
d = json.loads((R / "OCR_PROMPT_V2_PAIRED_RESULT_2026-09-03.json").read_text(encoding="utf-8"))
c, t = d["arms"]["control_m2-strict-v1"], d["arms"]["neutral_ocr-neutral-v2"]
m, acc = d["matched_pairs_quality"], d["accounting"]
tc = d["transition_counts"]


def row(name, s, cost):
    return (f"| {name} | {s['usable_denominator']} | {s['provider_content_filter']} | "
            f"{s['hard_failures'] - s['provider_content_filter']} | {s['successful_only_mean_cer']} | "
            f"{s['failure_aware_cer']} | {s['critical_errors_denominator']} | "
            f"{s['annotation_inclusion_errors']} | {s['mean_latency_s']}s | ${cost} |")


L = []
A = L.append
A("# OCR_PROMPT_V2_NEUTRAL_FRAMING — result\n")
A("**Gemini-only, 32 frozen handwritten seen-DEV crops, one variable: the OCR prompt.**\n")
A("| Prompt | Usable / 32 | Provider Filter | Other Hard Failures | Successful CER | "
  "Failure-Aware CER | Critical Errors | Annotation Errors | Mean Latency | Cost |")
A("|---|---|---|---|---|---|---|---|---|---|")
A(row("original m2-strict-v1", c, "0.04492875 (ledger)"))
A(row("OCR_PROMPT_V2_NEUTRAL_FRAMING", t, f"{acc['run_attributed_cost_usd']} (ledger)"))
A("")
A("## Verdict\n")
A("**The hypothesis is refuted, and in the wrong direction.** Neutralising the exam/grading framing "
  "did not reduce Gemini's provider-filter outcomes — they rose from **10 to 14** of 32. The "
  "pre-registered drop rule fires on hard failures ≥ 10/32.\n")
A(f"- Pre-registered rule outcome: **{d['pre_registered_drop_rule']['outcome']}** "
  f"(neutral hard failures = {d['pre_registered_drop_rule']['neutral_hard_failures']}/32, "
  f"threshold ≥ 10, unchanged after seeing results)")
A(f"- Quality veto triggered: {d['pre_registered_drop_rule']['quality_veto_triggered']} "
  f"(successful-only CER {t['successful_only_mean_cer']} ≤ 0.20)")
A("- Classification: **DROP_PRIMARY_ROUTE**\n")
A("Usable coverage did rise (14 → 16), but not from fewer filters — it came from cleaner "
  "*formatting*: JSON parse failures fell 6 → 1 and truncations 2 → 1. The filter got worse while "
  "the plumbing got better.\n")
A("## The aggregate hides a swap\n")
A("The +2 usable is a net of two opposite movements, which is the most informative thing in this run:\n")
A("| Crop type | Control usable | Neutral usable | Control filter | Neutral filter |")
A("|---|---|---|---|---|")
for k in ("cell", "line"):
    cs_, ts_ = d["stratified"]["control_by_crop_type"][k], d["stratified"]["neutral_by_crop_type"][k]
    A(f"| {k} | {cs_['usable_denominator']} | {ts_['usable_denominator']} | "
      f"{cs_['provider_content_filter']} | {ts_['provider_content_filter']} |")
A("")
A("Cell crops improved (5 → 9 usable, filter 8 → 6); line crops degraded badly "
  "(9 → 7 usable, filter **2 → 8**). The two categories got different edits — the cell prompt "
  "also swapped its annotation clause, the line prompt changed only the framing sentence — so "
  "these are effectively two sub-experiments, and the line result is the clearer signal that "
  "removing the exam framing did not help.\n")
A("Writer and crop type are confounded here (e002 = all 16 cells, e003 = 15 of 16 lines), so the "
  "per-writer table restates the same split rather than adding evidence. e007 is n=1 and is not a rate.\n")
A("## Paired transitions (all 32)\n")
A("| Transition | Count |")
A("|---|---|")
for k, v in sorted(tc.items(), key=lambda kv: -kv[1]):
    A(f"| {k} | {v} |")
A("")
A(f"- Rescued (failure → usable): {len(d['rescued_crops'])} — {', '.join(d['rescued_crops'])}")
A(f"- Newly broken (usable → failure): {len(d['newly_broken_crops'])} — {', '.join(d['newly_broken_crops'])}")
pt = d["paired_test"]
A(f"- Exact McNemar on usable/not: b={pt['b_control_usable_treatment_not']}, "
  f"c={pt['c_treatment_usable_control_not']}, {pt['discordant']} discordant pairs, "
  f"**p = {pt['p_value']}** — not significant.\n")
A("## Quality, without the composition confound\n")
A("Successful-only CER across arms (0.1155 vs 0.1608) is computed over **different crop sets** — "
  "the arms did not read the same crops. On the 11 crops both arms read:\n")
A(f"- control mean CER {m['control_mean_cer']} → neutral {m['neutral_mean_cer']} "
  f"(mean paired delta **{m['mean_paired_delta_cer']:+}**, median {m['median_paired_delta_cer']:+})")
A(f"- {m['improved']} improved, {m['regressed']} regressed, {m['unchanged']} unchanged; "
  f"exact sign test **p = {m['exact_sign_test_p']}**")
A("- So quality is statistically indistinguishable on like-for-like crops. The headline CER gap "
  "is substantially a composition effect, and the exact-match drop (7 → 1) is driven by "
  "single-character divergences, not by systematic degradation.\n")
A("Critical errors (digit / sign-operator / negation) are unchanged at 2 in both arms. "
  "**Annotation-inclusion errors: 0 in both arms** — the Phase 2 guard held.\n")
A("## Red-annotation risk audit (Phase 2)\n")
A("The committed audit's finding reproduces: **0 of 32 crops carry red ink**, and 32/32 crop "
  "hashes match the freeze. But that audit only asked whether annotation could be pulled *in*. "
  "Re-auditing both directions:\n")
A("- The crops are RGB and do carry colour; **19/32 have two substantial ink colours**, including "
  "all 16 cell crops.")
A("- Visual inspection shows the handwriting is **blue ballpoint** and the non-blue ink is "
  "**printed form structure** — table borders, dashed rules, bleed-through. No instructor "
  "annotation of any colour is present.")
A("- Therefore v1's *\"ignore any red instructor ink\"* was a **no-op** on this population, while "
  "v2's *\"ignore any marks written in a different colour of ink\"* is **live on 19/32 crops**.")
A("- Consequence for interpretation: this arm compares one framing **package** against another. "
  "It is not \"exam framing removed, all else equal\". The measured contamination stayed at zero, "
  "so the clause did no harm — but the cell/line split above is consistent with the cell prompt "
  "having changed more than the line prompt.\n")
A("## Cost and accounting\n")
A(f"- Neutral arm actual: **${acc['run_attributed_cost_usd']}** (authorized ceiling $0.12; "
  f"predicted worst case $0.150003 against an authorized $0.16)")
A(f"- Per crop ${d['cost']['per_crop_usd']} · per usable OCR ${d['cost']['per_usable_ocr_usd']}")
A(f"- Projected: 53 seen crops ${d['cost']['projected_53_seen_usd']} · 100 crops "
  f"${d['cost']['projected_100_crops_usd']}")
A(f"- 100 exams: 5/exam ${d['cost']['projected_100_exams']['5_crops_per_exam']} · "
  f"10/exam ${d['cost']['projected_100_exams']['10_crops_per_exam']} · "
  f"15/exam ${d['cost']['projected_100_exams']['15_crops_per_exam']}")
A("- Local grading cloud cost remains **$0**\n")
A("Reconciliation — exact:\n")
A(f"- ledger {acc['starting_ledger_usd']} → {acc['ending_ledger_usd']} "
  f"({acc['new_rows']} new rows, {acc['ledger_rows_before']} → {acc['ledger_rows_after']})")
A(f"- account usage {acc['starting_account_usage_usd']} → {acc['ending_account_usage_usd']} "
  f"= delta **${acc['account_delta_usd']}**, identical to the ledger "
  f"(rounding difference ${acc['rounding_difference_usd']})")
A(f"- billable rows {acc['billable_rows']}, non-billable {acc['nonbillable_rows']}; "
  f"finish reasons {acc['finish_reasons']}, all HTTP 200")
A(f"- case rows attribute ${acc['case_row_attributed_usd']}; the ${acc['unattributed_billed_failure_usd']} "
  f"difference is one billed failure (finish_reason=length produced output tokens). "
  f"The ledger is authoritative.")
A("- **No accounting mismatch — OCR scaling is not blocked on accounting.**\n")
A("## Next action (recommended, NOT executed)\n")
A("**Option B — stop scaling Gemini on this OpenRouter route and pre-register a genuinely "
  "different OCR provider/model.**\n")
A("The pre-registered rule already names this: at ≥ 10 hard failures the prompt is not the cause "
  "and the filter behaviour is intrinsic to this model/provider path. Two prompt variants have now "
  "been tried; the second made filtering worse. Prompt engineering on this route is exhausted.\n")
A("Explicitly **not** recommended:")
A("- Option A (scale to the remaining 21 CALIBRATION crops) — the drop rule forbids advancing.")
A("- Option C (redesign masking) — annotation contamination measured **0** in both arms; there is "
  "nothing to fix.")
A("- Option D (another reliability batch) — the result is not inconclusive on the pre-registered "
  "question; 16/32 hard failures is a clear drop signal.\n")
A("Do not reintroduce the Gemini→Sonnet fallback: it remains strictly dominated and withdrawn.\n")
A("## Honest limits\n")
A(f"- n=32 cannot demonstrate a true failure rate below ~9% even with zero observed events.")
A(f"- The usable-count change (14 → 16) is **not** statistically significant (p = {pt['p_value']}).")
A("- The cell-vs-line swap is the strongest signal here, but each half is n=16 and the two halves "
  "received different prompt edits, so it is a hypothesis for a future freeze, not a settled result.")
A("- Nothing here is a production-readiness claim.")

p = R / "OCR_PROMPT_V2_PAIRED_RESULT_2026-09-03.md"
p.write_text("\n".join(L), encoding="utf-8", newline="\n")
print("wrote", p, len(L), "lines")
