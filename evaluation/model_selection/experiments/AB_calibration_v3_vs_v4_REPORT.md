# CALIBRATION A/B — grade-v3 vs grade-v4-charitable

Pre-registration `4fe60f29e6cb40742661dbbb2e14e5b11a576f3d2cf4b41b69bcfe6705af1093`,
written and pushed (`acf7b81`) before any provider output for these arms existed.

**Question.** Does grade-v4-charitable align better with the authoritative
instructor labels than grade-v3, while preserving evidence grounding and
avoiding grade inflation? More credit is not better on its own.

**Answer.** Yes for Sonnet, on every axis and with no cost anywhere. For Gemini
the verdicts are *unchanged* — but its grounding is transformed. The single
largest effect of the charitable prompt was not leniency at all: it was that
both models started quoting the student.

    invalid-class performance = NOT MEASURED

---

## 1. Results

n = 12 CALIBRATION cases (writer e004), valid 7 / partially_valid 5, identical
ids and order in all four arms. 48/48 evaluations, 0 provider failures, 0 schema
failures, 0 cache hits.

| arm | model | prompt | acc% | macro-F1 | bal.acc | partial R | valid R | harmful ↑ | harmful ↓ | AUTO | REVIEW | ev.fail | ev.engagement | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | gemini-3.7-flash | grade-v3 | 50.00 | 0.5238 | 0.4572 | 0.200 | 0.714 | 2 | 4 | 50.0% | 50.0% | 6 | **25.0%** | $0.010661 |
| A2 | gemini-3.7-flash | **v4** | 50.00 | 0.5238 | 0.4572 | 0.200 | 0.714 | 2 | 4 | **100.0%** | 0.0% | **0** | **100.0%** | $0.013176 |
| B1 | claude-sonnet-5 | grade-v3 | 58.33 | 0.7024 | 0.5572 | 0.400 | 0.714 | 0 | 5 | 75.0% | 25.0% | 3 | **57.1%** | $0.076836 |
| B2 | claude-sonnet-5 | **v4** | **66.67** | **0.7917** | **0.6572** | **0.600** | 0.714 | **0** | **4** | **100.0%** | 0.0% | **0** | **100.0%** | $0.100698 |

Independently recomputed from raw outputs without the adapter or the verdicts
module — all four arms match to rounding.

## 2. Criteria

| criterion | gemini | sonnet | verdict |
|---|---|---|---|
| A. harmful downgrades decrease | 4 → 4 | 5 → **4** | partially met |
| B. alignment improves or holds | 50.0 → 50.0 (stable) | 58.3 → **66.7** | **met** |
| C. harmful upgrades do not rise | 2 → 2 | 0 → 0 | **met — zero new upgrades anywhere** |
| D. evidence grounding preserved | 25% → **100%** | 57% → **100%** | **strongly exceeded** |
| E. REVIEW does not worsen | 50% → **0%** | 25% → **0%** | **met** |
| F. failures do not worsen | 0 → 0 | 0 → 0 | **met** |

**No case in either model got worse on any measured dimension.** v4 weakly
dominates v3 here.

## 3. The finding worth keeping

The charitable policy was written to reduce over-strictness. Its largest
measured effect was elsewhere: **evidence engagement went 25%→100% (gemini) and
57%→100% (sonnet)**.

Under v3 both models awarded credit while citing nothing verifiable — gemini on
6 of 8 credit-awarding cases, sonnet on 3 of 7. Under v4, every credit-awarding
case (8/8 for both) quotes a span that verifies against the frozen
transcription. Zero fabricated spans in any arm.

This is why AUTO reached 100%: not because credit was withheld, and not because
the validator was loosened, but because the models actually grounded the credit
they gave. The v4 clause responsible is explicit — *"Whenever you award ANY
credit above zero, you must quote a SHORT span copied VERBATIM… Leniency never
relaxes this."*

Read the causality carefully: the grounding gain is a **prompt** effect, but the
AUTO/REVIEW gain it produces is only visible because the fail-closed evidence
rule (added earlier) turns ungrounded credit into REVIEW. Without that rule,
v3's ungrounded credit would have shown up as 100% AUTO and looked *better*.

## 4. Paired transitions

### Gemini (A1 → A2)

| transition | n |
|---|---|
| correct preserved | 6 |
| wrong unchanged | 6 |
| corrected by v4 | 0 |
| broken by v4 | 0 |

**Verdicts identical on all 12 cases.** The charitable wording did not move a
single gemini judgement. What changed: evidence grounding on 6 cases, and REVIEW
50% → 0%.

### Sonnet (B1 → B2)

| transition | n |
|---|---|
| correct preserved | 7 |
| wrong unchanged | 4 |
| **wrong downgrade corrected by v4** | **1** (`e004_q1_r8`, invalid → partially_valid) |
| broken by v4 | 0 |

`e004_q1_r8` is exactly the failure mode the human audit predicted: a
directionally-correct but thin answer scored to zero under literal grading, and
to partial credit under charitable grading. Its truth is `partially_valid`.

With n=12 and a single corrected case, this is a **direction, not a
significance claim**. A one-case improvement on twelve items has a wide
interval; no paired test is reported because none would be meaningful.

## 5. Model comparison under grade-v4

| | gemini v4 | sonnet v4 |
|---|---|---|
| macro-F1 | 0.5238 | **0.7917** |
| balanced accuracy | 0.4572 | **0.6572** |
| partial recall | 0.200 | **0.600** |
| valid recall | 0.714 | 0.714 |
| harmful upgrades | **2** | **0** |
| harmful downgrades | 4 | 4 |
| evidence engagement | 100% | 100% |
| AUTO / REVIEW | 100% / 0% | 100% / 0% |
| latency (mean) | 3.10 s | 3.87 s |
| cost (12 cases) | $0.013176 | $0.100698 |

Gemini's two harmful upgrades (`e004_q1_r5`, `e004_q1_r6`) are the **only two
cases where the models disagree** under v4: gemini says `valid`, sonnet says
`partially_valid`, and the truth is `partially_valid`. Gemini over-credits
exactly where sonnet is right.

- **anthropic/claude-sonnet-5 — ADVANCE.** Best macro-F1, balanced accuracy and
  partial recall; zero harmful upgrades in both arms; full grounding. Costs
  7.6× gemini.
- **google/gemini-3.7-flash — MAYBE.** Cheapest by far, full grounding under v4,
  identical valid recall — but partial recall 0.200 and it is the only arm that
  awards credit the rubric did not earn. On a grading task, over-crediting is
  the harder error to defend.

No production winner is declared: n=12, one writer, two classes.

## 6. Relation to the human audit

The blinded five-case audit returned **A=3, B=2, C=0, D=0** — three cases where
the derived label was right and the models were too strict, two where instructor
practice is more lenient than the literal rubric. That motivated a *generic*
charitable policy; no case, answer or wording from those five entered the prompt,
and none of them is in CALIBRATION.

Did v4 reduce that same over-strictness? **Once, measurably** (sonnet
`e004_q1_r8`), and **not at all for gemini**. The four downgrades both models
still make on identical cases (`e004_q1_r1`, `e004_q1_r3`, `e004_q2_r6`,
`e004_q2_r8`) are unchanged by charity — worth a look, since three models
agreeing against a label was exactly the pattern the DEV audit examined. No
label was changed.

## 7. Accounting

| | |
|---|---|
| starting ledger | $0.270320 |
| predicted additional | $0.262701 |
| **actual additional** | **$0.201370** |
| final ledger | **$0.471690** |
| provider account | $0.471690 |
| reconciliation delta | **$0.000000** |
| cumulative stop | $0.70 — OK, $0.228310 remaining |

0 cache hits (all 48 calls were genuine misses — v3 and v4 fingerprint
differently by prompt_version, system-prompt hash and user-block hash). 0 billed
failures. 0 non-billable failures.

## 8. Limitations

- **invalid-class performance = NOT MEASURED** — zero ground-truth support in every split
- CALIBRATION is a **single writer** (e004); nothing here generalises across writers
- **n = 12**; report counts, not significance
- HELD_OUT untouched and unmeasured
- only two verdict classes measured
- only this grading policy / rubric family measured — no claim about other
  subjects, rubric structures or grading policies
