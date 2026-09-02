# OCR paired 32-crop seen-DEV experiment — Gemini vs Sonnet (2026-09-02)

Experiment `OCR_SEEN32_DEV_PAIRED_GEMINI_VS_SONNET` (`c2e9cc8f6188a496…`). **64 provider requests (2 served from exact-request cache), $0.12703875 of a $0.40 ceiling.** Grading / OCR-verification / RAG / HELD_OUT calls: 0.

## Headline

| Model | Usable / 32 | Provider filter | Model refusal | Fabrication | Successful CER | Failure-aware CER | Critical errors | Mean latency | Cost |
|---|---|---|---|---|---|---|---|---|---|
| `google/gemini-3.7-flash` | **14/32** | 10 | 0 | 0 | 0.1155 *(n=14)* | 0.6130 | 2/14 | 8.656s | $0.04492875 |
| `anthropic/claude-sonnet-5` | **27/32** | 0 | 4 | 0 | 0.4718 *(n=27)* | 0.5544 | 12/27 | 4.034s | $0.08211 |

> Fabrication is semantic and human-assigned; none was adjudicated on this run, so the column is 0 by *absence of adjudication*, not by proof.

**The two models fail in opposite directions.** Gemini reads well and often refuses to read at all; Sonnet almost always answers and reads worse. On 32 handwritten crops Gemini produced no usable transcription for **18 of 32 (56%)**, while Sonnet produced usable text for 27 of 32 — but Sonnet's usable text carries a semantic (digit/sign/negation) error in **9 of 27** cases against Gemini's **2 of 14**. On the comparable failure-aware basis — a semantic error *or* a lost line, over all 32 crops — it is Gemini 20, Sonnet 14, composite 9.

## 1. Pre-registration verification

The freeze self-verifies. Its hash is `c2e9cc8f6188a496…`, **not** the `a9ad8694c0d33afc` I quoted in the previous report: I regenerated the file (restamping `created_at`) during the vendor-independence fix before committing, so that earlier value was stale. Every substantive invariant is unchanged and was re-verified independently — 32 case ids and execution order, all crop and reference sha256, DEV-only, CALIBRATION 0, HELD_OUT 0, m2-strict-v1 prompt hashes, schema, adapter, and both routes resolved through `--models-config`.

## 2. Population

- **32 crops, all handwritten** — 16 line + 16 cell, 0 printed
- writers: {"e003": 15, "e007": 1, "e002": 16}; 6 flagged `hard=True`
- splits {"DEV": 32}, CALIBRATION 0, HELD_OUT 0

**Writer and crop type are perfectly confounded**: e002 supplies all 16 cell crops, e003/e007 all 16 line crops. No analysis below can separate the two, and I do not try to.

## 3. Route and payload safety

All 64 payloads were rebuilt and verified on the wire before the first call: one image block + one schema-instruction text block, Gemini `max_tokens` 1000 / Sonnet 400, reasoning low / none, allowed under research **and** production, 0 grading tripwires, 0 banned phrases, 0 audited-reference leakage, `leakage_check` passed 64/64.

Two of Gemini's 32 requests were served from the exact-request cache (identical route and payload fingerprint, from Stage-1c). They are reported as cache hits, added no billable rows, and are counted in the 64.

## 4. Outcome taxonomy (all 12 axes)

| Axis | Gemini | Sonnet |
|---|---|---|
| usable transcription | 14 | 27 |
| provider content-filter | 10 | 0 |
| provider other HTTP failure | 0 | 0 |
| model-text refusal | 0 | 4 |
| truncation | 2 | 0 |
| JSON parse failure | 6 | 0 |
| schema failure | 0 | 0 |
| total line loss | 18 | 5 |

Every one of Gemini's 18 losses is a provider- or format-side event, not the model declining in text: 10 content-filter, 6 JSON-parse failures, 2 truncations **even at max_tokens=1000**. Sonnet had zero provider failures; its 5 losses are 4 model-text refusals plus one empty.

## 5. Gemini reliability and what 32 samples can support

- hard provider/format failures **18/32 = 56.2%**, one-sided 95% upper bound **71.3%**
- content-filter specifically **10/32 = 31.2%**, upper bound **47.2%**
- usable **14/32 = 43.8%**
- pre-registered band: **unsuitable as the sole OCR route under this configuration (5+)**

Cases needed to demonstrate a one-sided 95% upper bound below a given rate, **assuming zero further failures**:

| target upper bound | cases needed |
|---|---|
| < 20% | 14 |
| < 15% | 19 |
| < 10% | 29 |
| < 5% | 59 |
| < 2% | 149 |
| < 1% | 300 |

n=32 cannot demonstrate a true failure rate below ~9% even with ZERO observed events; no claim below 5% is made from this sample. The observed 56% hard-failure rate is far outside the range where sample size is the limiting factor — this is not an underpowered null result, it is a clear positive finding.

## 6. Per-writer and per-crop-type

| Model | slice | n | usable | filter | refusal | successful CER | failure-aware CER | critical |
|---|---|---|---|---|---|---|---|---|
| `gemini-3.7-flash` | line crops | 16 | 9/16 | 2 | 0 | 0.0876 | 0.4868 | 0 |
| `gemini-3.7-flash` | cell crops | 16 | 5/16 | 8 | 0 | 0.1657 | 0.7393 | 2 |
| `gemini-3.7-flash` | writer e002 | 16 | 5/16 | 8 | 0 | 0.1657 | 0.7393 | 2 |
| `gemini-3.7-flash` | writer e003 | 15 | 9/15 | 2 | 0 | 0.0876 | 0.4526 | 0 |
| `gemini-3.7-flash` | writer e007 ⚠ n=1, too small to interpret | 1 | 0/1 | 0 | 0 | n/a | 1.0000 | 0 |
| `claude-sonnet-5` | line crops | 16 | 16/16 | 0 | 0 | 0.4456 | 0.4456 | 5 |
| `claude-sonnet-5` | cell crops | 16 | 11/16 | 0 | 4 | 0.5100 | 0.6631 | 7 |
| `claude-sonnet-5` | writer e002 | 16 | 11/16 | 0 | 4 | 0.5100 | 0.6631 | 7 |
| `claude-sonnet-5` | writer e003 | 15 | 15/15 | 0 | 0 | 0.4427 | 0.4427 | 4 |
| `claude-sonnet-5` | writer e007 ⚠ n=1, too small to interpret | 1 | 1/1 | 0 | 0 | 0.4896 | 0.4896 | 1 |

Gemini's failures concentrate sharply on the e002/cell half (filter rate 50% vs 12.5% on e003/line). Because writer and crop type are confounded, **this is an observed association and not an established cause** — it could be the writer's hand, the cell-crop geometry, or something correlated with both. Nothing here supports inferring anything about a person from handwriting.

Hard-flagged crops are worse for both: Gemini 1/6 usable (3 filtered), Sonnet 4/6.

## 7. Failure association — observed only

| Dimension | Group | n | Gemini usable rate | filter rate |
|---|---|---|---|---|
| writer | e002 | 16 | 0.3125 | 0.5 |
| writer | e003 | 15 | 0.6 | 0.1333 |
| writer | e007 | 1 | 0.0 | 0.0 |
| crop_type | cell | 16 | 0.3125 | 0.5 |
| crop_type | line | 16 | 0.5625 | 0.125 |
| category | handwritten_cell | 16 | 0.3125 | 0.5 |
| category | handwritten_line | 16 | 0.5625 | 0.125 |
| hard flag | False | 26 | 0.5 | 0.2692 |
| hard flag | True | 6 | 0.1667 | 0.5 |
| crop bytes | >32000 | 32 | 0.4375 | 0.3125 |
| reference length (eval-only) | <=120 | 6 | 0.3333 | 0.3333 |
| reference length (eval-only) | <=40 | 6 | 0.5 | 0.5 |
| reference length (eval-only) | <=80 | 20 | 0.45 | 0.25 |
| reference contains digits (eval-only) | False | 21 | 0.4286 | 0.381 |
| reference contains digits (eval-only) | True | 11 | 0.4545 | 0.1818 |
| aspect ratio | <=16 | 3 | 0.3333 | 0.0 |
| aspect ratio | <=8 | 29 | 0.4483 | 0.3448 |

observed ASSOCIATION only, never established cause. n=32 with 10 filter events cannot separate these dimensions, several of which are collinear.

**Historical cross-check.** Of the 18 crops Gemini lost here, how many produced usable output in an earlier arm:

- stage1c_gemini: 0
- stage1b_gemini: 1
- stage1_sonnet: 3
- stage1_luna: 3

So these are not crops that are simply unreadable — Sonnet and Luna both read some of them, and `hl_e003_q1_r1__l1` was read perfectly by Gemini itself in Stage-1b.

## 8. Prospective fallback replay

Policy `gemini_then_sonnet_hard_failure_fallback_v1`, frozen before any output existed and applied reference-blind.

- Gemini used: **14** · Sonnet fallback: **15** · unresolved: **3**
- resolved coverage **29/32** · flagged for human review: 18
- triggers: {"json_parse_failure": 6, "provider_content_filter_failure": 10, "truncation": 2}

### Deployable routing strategies compared

| Strategy | Coverage | Human review | Successful CER | Failure-aware CER | Critical errors |
|---|---|---|---|---|---|
| Gemini only | 14/32 | 18 | 0.1155 | **0.6130** | 2 |
| Sonnet only | 27/32 | 5 | 0.4718 | **0.5544** | 12 |
| Gemini -> Sonnet on hard failure | 29/32 | 3 | 0.2894 | **0.3560** | 9 |
| Gemini -> human review on hard failure | 14/32 | 18 | 0.1155 | **0.6130** | 2 |
| Gemini -> Sonnet -> human if fallback unusable | 29/32 | 3 | 0.2894 | **0.3560** | 9 |
| *best-of-two by hidden reference* | *29/32* | *0* | *0.2894* | *0.3560* | *9* |

**NOT DEPLOYABLE - HIDDEN-REFERENCE ORACLE. Upper bound for context only; never a routing recommendation.**

#### Critical errors across strategies — read the failure-aware row

| Accounting | Gemini only | Sonnet only | Gemini → Sonnet |
|---|---|---|---|
| semantic errors among that strategy's usable outputs | 2 (of 14) | 9 (of 27) | 6 (of 29) |
| **failure-aware over all 32** (semantic error **or** lost line) | **20** | **14** | **9** |

the per-strategy critical_error_cases counts are computed over each strategy's OWN usable set, so they have different denominators and must not be compared directly. The original column also used a broad flag set (digit, sign/operator, Latin token, single-letter variable, negation), which inflates the count for verbose outputs.

**this is the comparable view and it REVERSES the impression the raw column gave: Gemini-only is the WORST strategy (20/32 crops either lost or semantically wrong), not the safest. A silently lost line is a failure too.** the earlier table reported 2 / 12 / 9 and invited exactly the wrong reading.

The oracle matches the prospective policy exactly, which means Gemini's transcription had the lower CER on every case where both models produced usable text. The deployable policy is already doing as well as hindsight could on this data — a real, if narrow, result.

## 9. Is Sonnet actually a useful fallback?

Measured **only on the 18 crops where Gemini hard-failed**, not on Sonnet's overall average:

- rescued **15/18 = 83.3%**; failed to rescue 3
- rescue-subset successful CER **0.4516** versus 0.4971 on the crops Gemini handled

So the crops Gemini loses are **not** systematically harder for Sonnet — it reads them about as well as anything else it reads. The honest caveat is what "rescue" means: a transcription with ~45% character error is recovered *coverage*, not recovered *accuracy*. It is a candidate for human review, which is exactly how the policy marks it (`needs_review` on every fallback row).

## 10. Critical-error audit

| Model | digit | sign/operator | negation | cases with any / usable |
|---|---|---|---|---|
| `google/gemini-3.7-flash` | 1 | 0 | 1 | 2/14 |
| `anthropic/claude-sonnet-5` | 5 | 2 | 5 | 12/27 |

This is the counterweight to Sonnet's coverage advantage: **44% of Sonnet's usable transcriptions carry a digit, sign or negation error** against 14% of Gemini's. For a grading pipeline those are the errors that change an answer's meaning, not its spelling. All flags are deterministic; no grading model was used and no official solution consulted.

## 11. Classification

**`google/gemini-3.7-flash` → DROP as the sole OCR route; MAYBE as a primary behind a fallback.** 18 of 32 handwritten crops yielded nothing usable — a 56% hard-failure rate whose 95% upper bound is 71% — landing squarely in the pre-registered `5+ = unsuitable` band. Its transcription quality when it does answer is the best measured here by a wide margin (CER 0.1155, critical errors in 2 of 14), so it is not a bad reader; it is an unreliable one, and a pipeline cannot silently lose more than half its crops.

**`anthropic/claude-sonnet-5` → MAYBE.** The only arm with acceptable operational coverage (27/32, zero provider failures, zero truncation, zero parse failures) and the only one that handled all 16 line crops. But CER 0.4718 is not usable transcription in any strict sense, and 12 of its 27 usable outputs carry a critical digit/sign/negation error. It is a viable *fallback* and a viable *coverage floor*; it is not a solution.

**Composite → PROMISING.** The prospective Gemini→Sonnet policy reaches 29/32 coverage with a failure-aware CER of 0.3560 — better than either model alone (Gemini 0.6130, Sonnet 0.5544) — at a measured $0.084 for 32 crops, with 3 crops left for human review and every fallback row flagged. It is the best deployable configuration measured. It is **not** production-ready OCR: a third of its output still needs review, and 32 samples on one exam's handwriting cannot establish otherwise.

## 12. Cost and projections

- Gemini **$0.04492875** (32 attempts, 14 usable, 10 free content-filter rows)
- Sonnet **$0.08211** (32 attempts, 27 usable)
- paired total **$0.12703875**, $0.003970/crop
- prospective composite **$0.09069875** (Gemini on 32 + Sonnet on the 15 triggered crops) = $0.002834/crop

cost per ATTEMPTED crop (32 attempts per arm), same crop mix - all handwritten - one pass, no retries. Gemini's rate benefits from 10 free content-filter rows, so its cost-per-USABLE transcription is the fairer planning number and is given too.

| Strategy | per crop | 53 seen | 100 crops | 100 exams @5 | @10 | @15 |
|---|---|---|---|---|---|---|
| Gemini (per attempt) | $0.001404 | $0.0744 | $0.1404 | $0.7020 | $1.4040 | $2.1060 |
| Sonnet (per attempt) | $0.002566 | $0.1360 | $0.2566 | $1.2830 | $2.5659 | $3.8489 |
| **Composite fallback** | $0.002834 | $0.1502 | $0.2834 | $1.4172 | $2.8343 | $4.2515 |

Gemini's cost per **usable** transcription is $0.003209 versus $0.001404 per attempt — the gap is the 10 free content-filter rows, and the per-usable figure is the honest planning number.

**Local grading cloud cost remains $0.** Cloud grading cost $0. No grading model ran.

## 13. Accounting reconciliation

- ledger 702 → 766 (+64)
- attributed **$0.12703875**; billable 46, non-billable 18, cache hits 2
- account $0.52436217 → $0.65140092 (delta **$0.12703875**)
- **rounding difference -0.0** — exact match
- project cumulative $0.65140129 against $8 warn / $10 hard

The 18 non-billable rows are Gemini's content-filter and format failures, which the provider did not charge for. Every billable response has a ledger row.

