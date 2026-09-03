# OCR_PROMPT_V2_NEUTRAL_FRAMING — result

**Gemini-only, 32 frozen handwritten seen-DEV crops, one variable: the OCR prompt.**

| Prompt | Usable / 32 | Provider Filter | Other Hard Failures | Successful CER | Failure-Aware CER | Critical Errors | Annotation Errors | Mean Latency | Cost |
|---|---|---|---|---|---|---|---|---|---|
| original m2-strict-v1 | 14/32 | 10 | 8 | 0.1155 | 0.613 | 2/14 | 0 | 8.6562s | $0.04492875 (ledger) |
| OCR_PROMPT_V2_NEUTRAL_FRAMING | 16/32 | 14 | 2 | 0.1608 | 0.5804 | 2/16 | 0 | 6.6864s | $0.051831 (ledger) |

## Verdict

**The hypothesis is refuted, and in the wrong direction.** Neutralising the exam/grading framing did not reduce Gemini's provider-filter outcomes — they rose from **10 to 14** of 32. The pre-registered drop rule fires on hard failures ≥ 10/32.

- Pre-registered rule outcome: **DROP_gemini_as_primary** (neutral hard failures = 16/32, threshold ≥ 10, unchanged after seeing results)
- Quality veto triggered: False (successful-only CER 0.1608 ≤ 0.20)
- Classification: **DROP_PRIMARY_ROUTE**

Usable coverage did rise (14 → 16), but not from fewer filters — it came from cleaner *formatting*: JSON parse failures fell 6 → 1 and truncations 2 → 1. The filter got worse while the plumbing got better.

## The aggregate hides a swap

The +2 usable is a net of two opposite movements, which is the most informative thing in this run:

| Crop type | Control usable | Neutral usable | Control filter | Neutral filter |
|---|---|---|---|---|
| cell | 5/16 | 9/16 | 8 | 6 |
| line | 9/16 | 7/16 | 2 | 8 |

Cell crops improved (5 → 9 usable, filter 8 → 6); line crops degraded badly (9 → 7 usable, filter **2 → 8**). The two categories got different edits — the cell prompt also swapped its annotation clause, the line prompt changed only the framing sentence — so these are effectively two sub-experiments, and the line result is the clearer signal that removing the exam framing did not help.

Writer and crop type are confounded here (e002 = all 16 cells, e003 = 15 of 16 lines), so the per-writer table restates the same split rather than adding evidence. e007 is n=1 and is not a rate.

## Paired transitions (all 32)

| Transition | Count |
|---|---|
| other_failure -> provider_filter | 6 |
| provider_filter -> provider_filter | 5 |
| usable -> usable (quality regressed) | 5 |
| provider_filter -> usable | 4 |
| usable -> usable (quality improved) | 3 |
| usable -> provider_filter | 3 |
| usable -> usable (quality unchanged) | 3 |
| other_failure -> usable | 1 |
| other_failure -> other_failure | 1 |
| provider_filter -> other_failure | 1 |

- Rescued (failure → usable): 5 — hl_e003_q1_r1__l1, hc_e002_q1_r2, hc_e002_q1_r5, hc_e002_q1_r6, hc_e002_q2_r3
- Newly broken (usable → failure): 3 — hl_e003_q1_r7__l1, hl_e003_q1_r8__l1, hl_e003_q2_r8__l1
- Exact McNemar on usable/not: b=3, c=5, 8 discordant pairs, **p = 0.726562** — not significant.

## Quality, without the composition confound

Successful-only CER across arms (0.1155 vs 0.1608) is computed over **different crop sets** — the arms did not read the same crops. On the 11 crops both arms read:

- control mean CER 0.147 → neutral 0.1727 (mean paired delta **+0.0257**, median +0.0)
- 3 improved, 5 regressed, 3 unchanged; exact sign test **p = 0.7266**
- So quality is statistically indistinguishable on like-for-like crops. The headline CER gap is substantially a composition effect, and the exact-match drop (7 → 1) is driven by single-character divergences, not by systematic degradation.

Critical errors (digit / sign-operator / negation) are unchanged at 2 in both arms. **Annotation-inclusion errors: 0 in both arms** — the Phase 2 guard held.

## Red-annotation risk audit (Phase 2)

The committed audit's finding reproduces: **0 of 32 crops carry red ink**, and 32/32 crop hashes match the freeze. But that audit only asked whether annotation could be pulled *in*. Re-auditing both directions:

- The crops are RGB and do carry colour; **19/32 have two substantial ink colours**, including all 16 cell crops.
- Visual inspection shows the handwriting is **blue ballpoint** and the non-blue ink is **printed form structure** — table borders, dashed rules, bleed-through. No instructor annotation of any colour is present.
- Therefore v1's *"ignore any red instructor ink"* was a **no-op** on this population, while v2's *"ignore any marks written in a different colour of ink"* is **live on 19/32 crops**.
- Consequence for interpretation: this arm compares one framing **package** against another. It is not "exam framing removed, all else equal". The measured contamination stayed at zero, so the clause did no harm — but the cell/line split above is consistent with the cell prompt having changed more than the line prompt.

## Cost and accounting

- Neutral arm actual: **$0.051831** (authorized ceiling $0.12; predicted worst case $0.150003 against an authorized $0.16)
- Per crop $0.00147204 · per usable OCR $0.00294408
- Projected: 53 seen crops $0.078018 · 100 crops $0.147204
- 100 exams: 5/exam $0.736 · 10/exam $1.472 · 15/exam $2.2081
- Local grading cloud cost remains **$0**

Reconciliation — exact:

- ledger 0.651401 → 0.703232 (32 new rows, 766 → 798)
- account usage 0.65140092 → 0.70323192 = delta **$0.051831**, identical to the ledger (rounding difference $8e-08)
- billable rows 17, non-billable 15; finish reasons {'stop': 16, 'content_filter': 14, 'length': 1, 'error': 1}, all HTTP 200
- case rows attribute $0.04710525; the $0.00472575 difference is one billed failure (finish_reason=length produced output tokens). The ledger is authoritative.
- **No accounting mismatch — OCR scaling is not blocked on accounting.**

## Next action (recommended, NOT executed)

**Option B — stop scaling Gemini on this OpenRouter route and pre-register a genuinely different OCR provider/model.**

The pre-registered rule already names this: at ≥ 10 hard failures the prompt is not the cause and the filter behaviour is intrinsic to this model/provider path. Two prompt variants have now been tried; the second made filtering worse. Prompt engineering on this route is exhausted.

Explicitly **not** recommended:
- Option A (scale to the remaining 21 CALIBRATION crops) — the drop rule forbids advancing.
- Option C (redesign masking) — annotation contamination measured **0** in both arms; there is nothing to fix.
- Option D (another reliability batch) — the result is not inconclusive on the pre-registered question; 16/32 hard failures is a clear drop signal.

Do not reintroduce the Gemini→Sonnet fallback: it remains strictly dominated and withdrawn.

## Honest limits

- n=32 cannot demonstrate a true failure rate below ~9% even with zero observed events.
- The usable-count change (14 → 16) is **not** statistically significant (p = 0.726562).
- The cell-vs-line swap is the strongest signal here, but each half is n=16 and the two halves received different prompt edits, so it is a hypothesis for a future freeze, not a settled result.
- Nothing here is a production-readiness claim.