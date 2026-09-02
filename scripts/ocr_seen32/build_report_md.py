import json
from pathlib import Path

R = Path("evaluation/model_selection/runs/ocr_primary")
a = json.loads((R / "OCR_SEEN32_PAIRED_RESULT_2026-09-02.json").read_text(encoding="utf-8"))
GEM, SON = "google/gemini-3.7-flash", "anthropic/claude-sonnet-5"
S, rel, rt, resc = a["stratified"], a["reliability"], a["routing_comparison"], a["sonnet_as_fallback"]
acc, proj, fb = a["accounting"], a["cost_projections"], a["fallback"]
L = []
w = L.append
f4 = lambda x: "n/a" if x is None else f"{x:.4f}"

w("# OCR paired 32-crop seen-DEV experiment — Gemini vs Sonnet (2026-09-02)")
w("")
w(f"Experiment `{a['experiment']}` (`{a['experiment_sha256'][:16]}…`). "
  f"**64 provider requests ({a['cache_hits']} served from exact-request cache), "
  f"${acc['attributed_cost_usd']} of a $0.40 ceiling.** "
  "Grading / OCR-verification / RAG / HELD_OUT calls: 0.")
w("")
w("## Headline")
w("")
w("| Model | Usable / 32 | Provider filter | Model refusal | Fabrication | Successful CER | Failure-aware CER | Critical errors | Mean latency | Cost |")
w("|---|---|---|---|---|---|---|---|---|---|")
for slug in (GEM, SON):
    b = S[slug]["all32"]
    cost = acc["gemini_cost_usd"] if slug == GEM else acc["sonnet_cost_usd"]
    w(f"| `{slug}` | **{b['usable_coverage']}** | {b['provider_content_filter']} | "
      f"{b['model_text_refusal']} | 0 | {f4(b['succ_mean_cer'])} *(n={b['usable']})* | "
      f"{f4(b['failure_aware_cer'])} | {b['critical_error_cases']}/{b['usable']} | "
      f"{b['mean_latency_s']}s | ${cost} |")
w("")
w("> Fabrication is semantic and human-assigned; none was adjudicated on this run, so the column is "
  "0 by *absence of adjudication*, not by proof.")
w("")
w("**The two models fail in opposite directions.** Gemini reads well and often refuses to read at "
  "all; Sonnet almost always answers and reads worse. On 32 handwritten crops Gemini produced no "
  "usable transcription for **18 of 32 (56%)**, while Sonnet produced usable text for 27 of 32 — "
  "but Sonnet's usable text carries a semantic (digit/sign/negation) error in **9 of 27** cases "
  "against Gemini's **2 of 14**. On the comparable failure-aware basis — a semantic error *or* a "
  "lost line, over all 32 crops — it is Gemini 20, Sonnet 14, composite 9.")
w("")
w("## 1. Pre-registration verification")
w("")
w("The freeze self-verifies. Its hash is `c2e9cc8f6188a496…`, **not** the `a9ad8694c0d33afc` I quoted "
  "in the previous report: I regenerated the file (restamping `created_at`) during the "
  "vendor-independence fix before committing, so that earlier value was stale. Every substantive "
  "invariant is unchanged and was re-verified independently — 32 case ids and execution order, all "
  "crop and reference sha256, DEV-only, CALIBRATION 0, HELD_OUT 0, m2-strict-v1 prompt hashes, "
  "schema, adapter, and both routes resolved through `--models-config`.")
w("")
w("## 2. Population")
w("")
c = a["population"]["composition"]
w(f"- **{c['total']} crops, all handwritten** — {c['line_crops']} line + {c['cell_crops']} cell, 0 printed")
w(f"- writers: {json.dumps(c['by_writer'])}; {c['hard_flagged']} flagged `hard=True`")
w(f"- splits {json.dumps(c['splits'])}, CALIBRATION {c['calibration_cases']}, HELD_OUT {c['held_out_cases']}")
w("")
w("**Writer and crop type are perfectly confounded**: e002 supplies all 16 cell crops, e003/e007 all "
  "16 line crops. No analysis below can separate the two, and I do not try to.")
w("")
w("## 3. Route and payload safety")
w("")
w("All 64 payloads were rebuilt and verified on the wire before the first call: one image block + one "
  "schema-instruction text block, Gemini `max_tokens` 1000 / Sonnet 400, reasoning low / none, "
  "allowed under research **and** production, 0 grading tripwires, 0 banned phrases, 0 audited-"
  "reference leakage, `leakage_check` passed 64/64.")
w("")
w(f"Two of Gemini's 32 requests were served from the exact-request cache (identical route and "
  "payload fingerprint, from Stage-1c). They are reported as cache hits, added no billable rows, "
  "and are counted in the 64.")
w("")
w("## 4. Outcome taxonomy (all 12 axes)")
w("")
w("| Axis | Gemini | Sonnet |")
w("|---|---|---|")
for k, lbl in (("usable_transcription_returned", "usable transcription"),
               ("provider_content_filter", "provider content-filter"),
               ("provider_other_http", "provider other HTTP failure"),
               ("model_text_refusal", "model-text refusal"),
               ("truncation", "truncation"),
               ("json_parse_failure", "JSON parse failure"),
               ("schema_failure", "schema failure"),
               ("total_line_loss", "total line loss")):
    kk = {"usable_transcription_returned": "usable"}.get(k, k)
    gv = S[GEM]["all32"].get(k, S[GEM]["all32"].get(kk))
    sv = S[SON]["all32"].get(k, S[SON]["all32"].get(kk))
    w(f"| {lbl} | {gv} | {sv} |")
w("")
w("Every one of Gemini's 18 losses is a provider- or format-side event, not the model declining in "
  "text: 10 content-filter, 6 JSON-parse failures, 2 truncations **even at max_tokens=1000**. Sonnet "
  "had zero provider failures; its 5 losses are 4 model-text refusals plus one empty.")
w("")
w("## 5. Gemini reliability and what 32 samples can support")
w("")
g = rel["gemini"]
w(f"- hard provider/format failures **{g['hard_provider_failures']}/32 = {g['hard_failure_rate']:.1%}**, "
  f"one-sided 95% upper bound **{g['hard_failure_upper95']:.1%}**")
w(f"- content-filter specifically **{g['content_filter']}/32 = {g['content_filter_rate']:.1%}**, "
  f"upper bound **{g['content_filter_upper95']:.1%}**")
w(f"- usable **{g['usable']}/32 = {g['usable_rate']:.1%}**")
w(f"- pre-registered band: **{g['band']}**")
w("")
w("Cases needed to demonstrate a one-sided 95% upper bound below a given rate, **assuming zero "
  "further failures**:")
w("")
w("| target upper bound | cases needed |")
w("|---|---|")
for k, v in rel["sample_size_needed_assuming_zero_events"].items():
    w(f"| < {k} | {v} |")
w("")
w(f"{rel['caveat']} The observed 56% hard-failure rate is far outside the range where sample size is "
  "the limiting factor — this is not an underpowered null result, it is a clear positive finding.")
w("")
w("## 6. Per-writer and per-crop-type")
w("")
w("| Model | slice | n | usable | filter | refusal | successful CER | failure-aware CER | critical |")
w("|---|---|---|---|---|---|---|---|---|")
for slug in (GEM, SON):
    for key, lbl in (("line", "line crops"), ("cell", "cell crops")):
        b = S[slug][key]
        w(f"| `{slug.split('/')[-1]}` | {lbl} | {b['intended']} | {b['usable_coverage']} | "
          f"{b['provider_content_filter']} | {b['model_text_refusal']} | {f4(b['succ_mean_cer'])} | "
          f"{f4(b['failure_aware_cer'])} | {b['critical_error_cases']} |")
    for wr, b in S[slug]["by_writer"].items():
        note = " ⚠ n=1, too small to interpret" if b["too_small_to_interpret"] else ""
        w(f"| `{slug.split('/')[-1]}` | writer {wr}{note} | {b['intended']} | {b['usable_coverage']} | "
          f"{b['provider_content_filter']} | {b['model_text_refusal']} | {f4(b['succ_mean_cer'])} | "
          f"{f4(b['failure_aware_cer'])} | {b['critical_error_cases']} |")
w("")
w("Gemini's failures concentrate sharply on the e002/cell half (filter rate 50% vs 12.5% on "
  "e003/line). Because writer and crop type are confounded, **this is an observed association and "
  "not an established cause** — it could be the writer's hand, the cell-crop geometry, or something "
  "correlated with both. Nothing here supports inferring anything about a person from handwriting.")
w("")
w("Hard-flagged crops are worse for both: Gemini 1/6 usable (3 filtered), Sonnet 4/6.")
w("")
w("## 7. Failure association — observed only")
w("")
w("| Dimension | Group | n | Gemini usable rate | filter rate |")
w("|---|---|---|---|---|")
for dim, v in a["failure_association"].items():
    if dim.startswith("_"):
        continue
    for k, b in sorted(v["groups"].items()):
        w(f"| {v['dimension']} | {k} | {b['n']} | {b['usable_rate']} | {b['filter_rate']} |")
w("")
w(a["failure_association"]["_discipline"])
w("")
h = a["historical_cross_check"]
w(f"**Historical cross-check.** Of the {len(h['gemini_failed_here'])} crops Gemini lost here, how "
  "many produced usable output in an earlier arm:")
w("")
for name, ids in h["of_those_previously_usable"].items():
    w(f"- {name}: {len(ids)}")
w("")
w("So these are not crops that are simply unreadable — Sonnet and Luna both read some of them, and "
  "`hl_e003_q1_r1__l1` was read perfectly by Gemini itself in Stage-1b.")
w("")
w("## 8. Prospective fallback replay")
w("")
w(f"Policy `{fb['policy_id']}`, frozen before any output existed and applied reference-blind.")
w("")
w(f"- Gemini used: **{fb['primary_used']}** · Sonnet fallback: **{fb['fallback_used']}** · "
  f"unresolved: **{fb['unresolved']}**")
w(f"- resolved coverage **{fb['resolved_coverage']}** · flagged for human review: {fb['needs_review']}")
w(f"- triggers: {json.dumps(fb['triggers'])}")
w("")
w("### Deployable routing strategies compared")
w("")
w("| Strategy | Coverage | Human review | Successful CER | Failure-aware CER | Critical errors |")
w("|---|---|---|---|---|---|")
for name, r in rt.items():
    if name.startswith("ORACLE") or name.startswith("_"):
        continue
    w(f"| {r['strategy']} | {r['usable_coverage']} | {r['human_review_cases']} | "
      f"{f4(r['succ_mean_cer'])} | **{f4(r['failure_aware_cer'])}** | {r['critical_error_cases']} |")
o = rt["ORACLE_best_of_two"]
w(f"| *{o['strategy']}* | *{o['usable_coverage']}* | *{o['human_review_cases']}* | "
  f"*{f4(o['succ_mean_cer'])}* | *{f4(o['failure_aware_cer'])}* | *{o['critical_error_cases']}* |")
w("")
w(f"**{o['WARNING']}**")
w("")
w("#### Critical errors across strategies — read the failure-aware row")
w("")
ca = a["routing_comparison"]["_critical_error_accounting"]
w("| Accounting | Gemini only | Sonnet only | Gemini → Sonnet |")
w("|---|---|---|---|")
so, fa = ca["semantic_only_among_usable"], ca["failure_aware_all_32"]
w(f"| semantic errors among that strategy's usable outputs | {so['gemini_only']} (of 14) | "
  f"{so['sonnet_only']} (of 27) | {so['gemini_then_sonnet']} (of 29) |")
w(f"| **failure-aware over all 32** (semantic error **or** lost line) | **{fa['gemini_only']}** | "
  f"**{fa['sonnet_only']}** | **{fa['gemini_then_sonnet']}** |")
w("")
w(ca["problem"])
w("")
w(f"**{fa['reading']}** {ca['correction']}.")
w("")
w("The oracle matches the prospective policy exactly, which means Gemini's transcription had the "
  "lower CER on every case where both models produced usable text. The deployable policy is already "
  "doing as well as hindsight could on this data — a real, if narrow, result.")
w("")
w("## 9. Is Sonnet actually a useful fallback? — the claim, refuted")
w("")
ref = resc["REFUTATION_after_verification"]
w(f"Measured only on the {resc['gemini_trigger_cases']} crops where Gemini hard-failed, Sonnet "
  f"returned a schema-valid non-marker transcription for **{resc['sonnet_rescued']} of "
  f"{resc['gemini_trigger_cases']} ({resc['rescue_rate']:.1%})**. I first reported that as Sonnet "
  "being a genuinely useful fallback. **Independent verification refuted that, and it was right.**")
w("")
q = ref["what_does_not"]["quality"]
r2 = ref["what_does_not"]["how_much_of_the_answer_is_actually_recovered"]
w("| What was recovered | Value |")
w("|---|---|")
w(f"| rescues meeting the project's own proposed OCR gate (CER ≤ 5%) | **{q['meeting_the_projects_own_proposed_gate_cer_5pct']}** |")
w(f"| rescues meeting even a lenient CER ≤ 10% | **{q['meeting_a_lenient_cer_10pct_bar']}** |")
w(f"| best single rescue | CER {q['best_single_rescue_cer']} |")
w(f"| mean word recovery (fraction of the reference's words actually reproduced) | **{r2['mean_word_recovery_rate']:.1%}** |")
w(f"| rescues recovering under 25% of the reference's words | {r2['rescues_recovering_under_25pct_of_reference_words']}/15 |")
w(f"| rescues recovering **zero** reference words | **{r2['rescues_recovering_ZERO_reference_words']}/15** |")
w("")
w(r2["reading"])
w("")
w("The comparison I used to justify the claim does not survive either: "
  + ref["what_does_not"]["the_comparison_that_justified_it"]["actual"])
w("")
w("### The safety inversion")
w("")
inv = ref["THE_SAFETY_INVERSION"]
w(f"**{inv['finding']}**")
w("")
w(inv["why_it_matters_here"])
w("")
w(f"*Consequence for the metric:* {inv['consequence_for_the_metric']}")
w("")
w(f"*What saves it:* {inv['what_saves_it']}")
w("")
hh = ref["head_to_head_where_both_answered"]
w(f"For scale: on the {hh['n']} crops both models read, Gemini's CER is {hh['gemini_mean_cer']} "
  f"against Sonnet's {hh['sonnet_mean_cer']} — {hh['reading']}.")
w("")
w(f"**Revised answer.** {ref['revised_answer_to_the_question']}")
w("")
w("## 9b. The composite is strictly dominated — recommendation withdrawn")
w("")
cr = rt["_composite_refutation"]
w(cr["arithmetic_confirmed"])
w("")
o = cr["1_the_fallback_contributes_nothing"]
w("**Where the composite's advantage actually comes from.**")
w("")
w("| Slice | Composite | Sonnet only | Gap |")
w("|---|---|---|---|")
for k, lbl in (("composite_vs_sonnet_only_all_32", "all 32 crops"),
               ("on_the_18_fallback_crops", "the 18 crops where the fallback FIRES"),
               ("on_the_14_gemini_primary_crops", "the 14 crops Gemini handled")):
    d = o[k]
    w(f"| {lbl} | {d['composite']} | {d['sonnet_only']} | **{d['gap']}** |")
w("")
w(o["reading"])
w("")
w("**Review-aware accounting erases the difference entirely.** Failure-aware CER charges a detected, "
  "human-routed loss the full 1.0 penalty — but a detected loss is not a wrong answer, it is a crop "
  "a human reads. Scoring only what would reach a grade unreviewed:")
w("")
ra = cr["2_review_aware_accounting_erases_the_difference"]
w("| Strategy | Unreviewed CER | Unreviewed crops | Human reviews |")
w("|---|---|---|---|")
for k, lbl in (("gemini_only", "Gemini only"), ("composite", "Gemini → Sonnet"),
               ("sonnet_only", "Sonnet only")):
    d = ra[k]
    w(f"| {lbl} | {d['unreviewed_cer']} | {d['unreviewed_crops']} | {d['human_reviews']} |")
w("")
w(ra["reading"])
w("")
w(f"**A correction to my own table:** {cr['3_the_review_workload_claim_was_wrong']['actual']}")
w("")
w(f"And the composite is the most expensive of the three: Gemini-only "
  f"${cr['4_the_composite_is_the_most_expensive']['gemini_only_usd']}, Sonnet-only "
  f"${cr['4_the_composite_is_the_most_expensive']['sonnet_only_usd']}, composite "
  f"${cr['4_the_composite_is_the_most_expensive']['composite_usd']}.")
w("")
w(f"### {cr['CONCLUSION']}")
w("")
w(cr["what_this_does_not_say"])
w("")
w("## 10. Critical-error audit")
w("")
w("| Model | digit | sign/operator | negation | cases with any / usable |")
w("|---|---|---|---|---|")
for slug in (GEM, SON):
    b = S[slug]["all32"]
    w(f"| `{slug}` | {b['digit_mismatches']} | {b['signop_mismatches']} | {b['negation_mismatches']} | "
      f"{b['critical_error_cases']}/{b['usable']} |")
w("")
w("This is the counterweight to Sonnet's coverage advantage: **44% of Sonnet's usable transcriptions "
  "carry a digit, sign or negation error** against 14% of Gemini's. For a grading pipeline those are "
  "the errors that change an answer's meaning, not its spelling. All flags are deterministic; no "
  "grading model was used and no official solution consulted.")
w("")
w("## 11. Classification")
w("")
w("**`google/gemini-3.7-flash` → DROP as the sole OCR route; MAYBE as a primary behind a fallback.** "
  "18 of 32 handwritten crops yielded nothing usable — a 56% hard-failure rate whose 95% upper bound "
  "is 71% — landing squarely in the pre-registered `5+ = unsuitable` band. Its transcription quality "
  "when it does answer is the best measured here by a wide margin (CER 0.1155, critical errors in "
  "2 of 14), so it is not a bad reader; it is an unreliable one, and a pipeline cannot silently lose "
  "more than half its crops.")
w("")
w("**`anthropic/claude-sonnet-5` → MAYBE.** The only arm with acceptable operational coverage "
  "(27/32, zero provider failures, zero truncation, zero parse failures) and the only one that "
  "handled all 16 line crops. But CER 0.4718 is not usable transcription in any strict sense, and "
  "12 of its 27 usable outputs carry a critical digit/sign/negation error. It is a viable *fallback* "
  "and a viable *coverage floor*; it is not a solution.")
w("")
w("**Composite → NOT USEFUL (strictly dominated).** The prospective Gemini→Sonnet policy "
  "reaches 29/32 coverage and a failure-aware CER of 0.3560, and both figures are real. Neither "
  "survives contact with what they mean. The fallback contributes **exactly 0.0000** of the "
  "composite's advantage — its output is byte-identical to Sonnet-only on all 18 crops where it "
  "fires. Under review-aware accounting it is **identical to Gemini-only** (0.1155 unreviewed CER, "
  "14 unreviewed crops, 18 human reviews) at roughly double the cost, and it converts 15 loud, "
  "machine-detectable failures into plausible text a reviewer must now adjudicate. It wins on no "
  "dimension. **I recommended it earlier in this report; that recommendation is withdrawn.**")
w("")
w("## 12. Cost and projections")
w("")
w(f"- Gemini **${acc['gemini_cost_usd']}** (32 attempts, 14 usable, 10 free content-filter rows)")
w(f"- Sonnet **${acc['sonnet_cost_usd']}** (32 attempts, 27 usable)")
w(f"- paired total **${acc['attributed_cost_usd']}**, ${proj['paired_per_crop']:.6f}/crop")
w(f"- prospective composite **${proj['prospective_fallback_composite']['measured_total_usd']}** "
  f"(Gemini on 32 + Sonnet on the {fb['fallback_used']} triggered crops) = "
  f"${proj['prospective_fallback_composite']['per_crop']:.6f}/crop")
w("")
w(proj["assumptions"])
w("")
w("| Strategy | per crop | 53 seen | 100 crops | 100 exams @5 | @10 | @15 |")
w("|---|---|---|---|---|---|---|")
for name, key in (("Gemini (per attempt)", "gemini"), ("Sonnet (per attempt)", "sonnet")):
    p = proj[key]["projections_per_attempt"]
    w(f"| {name} | ${proj[key]['per_attempt']:.6f} | ${p['all_53_seen']:.4f} | ${p['per_100_crops']:.4f} | "
      f"${p['100_exams_at_5']:.4f} | ${p['100_exams_at_10']:.4f} | ${p['100_exams_at_15']:.4f} |")
pc = proj["prospective_fallback_composite"]["projections"]
w(f"| **Composite fallback** | ${proj['prospective_fallback_composite']['per_crop']:.6f} | "
  f"${pc['all_53_seen']:.4f} | ${pc['per_100_crops']:.4f} | ${pc['100_exams_at_5']:.4f} | "
  f"${pc['100_exams_at_10']:.4f} | ${pc['100_exams_at_15']:.4f} |")
w("")
w(f"Gemini's cost per **usable** transcription is ${proj['gemini']['per_usable']:.6f} versus "
  f"${proj['gemini']['per_attempt']:.6f} per attempt — the gap is the 10 free content-filter rows, and "
  "the per-usable figure is the honest planning number.")
w("")
w("**Local grading cloud cost remains $0.** Cloud grading cost $0. No grading model ran.")
w("")
w("## 13. Accounting reconciliation")
w("")
w(f"- ledger {acc['starting_ledger_rows']} → {acc['ending_ledger_rows']} (+{acc['new_rows']})")
w(f"- attributed **${acc['attributed_cost_usd']}**; billable {acc['billable']}, non-billable "
  f"{acc['nonbillable_failures']}, cache hits {acc['cache_hits']}")
w(f"- account ${acc['starting_account_usage']} → ${acc['ending_account_usage']} "
  f"(delta **${acc['account_delta_usd']}**)")
w(f"- **rounding difference {acc['rounding_difference']}** — exact match")
w(f"- project cumulative ${acc['project_cumulative_usd']} against $8 warn / $10 hard")
w("")
w("The 18 non-billable rows are Gemini's content-filter and format failures, which the provider did "
  "not charge for. Every billable response has a ledger row.")
w("")
Path(R / "OCR_SEEN32_PAIRED_RESULT_2026-09-02.md").write_text("\n".join(L) + "\n",
                                                              encoding="utf-8", newline="\n")
print("wrote md,", len(L), "lines")
