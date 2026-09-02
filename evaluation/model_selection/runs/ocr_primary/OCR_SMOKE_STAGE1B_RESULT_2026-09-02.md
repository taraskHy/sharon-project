# OCR Stage-1b — Gemini re-run and three-candidate comparison (2026-09-02)

Experiment `OCR_SMOKE_STAGE1B_GEMINI_REASONING_LOW` (`4de29894cc25b0cc…`). **8 new provider requests, $0.01488075 of a $0.04 ceiling.** Grading / OCR-verification / RAG / HELD_OUT calls: 0.

## Headline — handwritten first, because that is the job

| Model | HW mean CER | HW mean WER | Critical errors (HW) | Refusals | Fabrications | Printed CER | Failures | Mean latency | Cost |
|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 0.9487 *(n=5/5)* | 1.0000 | 5 | 4 | 1 | 0.4761 | 0/8 | 2.719s | $0.007107 ($0.00088833/crop) |
| `anthropic/claude-sonnet-5` | 0.6188 *(n=5/5)* | 0.9018 | 4 | 2 | 0 | 0.2694 | 0/8 | 4.383s | $0.019686 ($0.00246075/crop) |
| `google/gemini-3.7-flash` | 0.0294 *(n=2/5)* | 0.1666 | 4 | 0 | 0 | 0.2761 | 3/8 | 4.106s | $0.014881 ($0.00212582/crop) |

Gemini's Stage-1 arm is **not** in this table. It is reported separately as *invalid configuration / pre-inference failure*: 8 requests rejected with HTTP 400 `Reasoning is mandatory for this endpoint and cannot be disabled`, $0 billed, zero OCR evidence.

> **Read the `n=` column before the CER.** Gemini's 0.0294 is a mean over 2 of 5 handwritten crops; Luna's and Sonnet's are over all 5. Those are not the same measurement.

## The defect this arm found

**the run recorded max_tokens=1000 (the derived, pre-registered cap) but the provider was sent 600, the OcrPrimaryAdapter default: run_benchmark passed request.max_tokens instead of the resolved route.max_tokens**

- *This arm:* 3 of 8 cases produced no transcription — 2 explicit 'truncated at max_tokens=600' and 1 JSON-EOF schema failure consistent with the same truncation. The cap was derived specifically to stop this.
- *Stage-1 too:* Stage-1's recorded route max_tokens of 400 was likewise not what the provider received (also 600). Luna and Sonnet were unaffected in outcome — no Stage-1 response hit a length finish_reason — but their recorded decoding config was inaccurate.
- *And:* the dry-run cost prediction reads route.max_tokens, so it priced 1000 while 600 was sent
- *Fixed:* runner now passes route.max_tokens; regression test in tests/test_bench_max_tokens_resolution.py
- *Not done:* Stage-1b was NOT re-run: the 8 authorized requests are spent and a rerun was outside this authorization. The corrected cap is therefore UNVALIDATED against a live provider.

Truncation was **not** length-correlated: the shortest reference in the set (19 chars, `hc_e002_q1_r1`) truncated while a 69-char one succeeded. Reasoning-token variance consumed the cap, not transcription length — which is why raising the cap reflexively is not obviously the fix.

## Stage-1 reproduction

`33/33` checks recomputed from the committed artifacts and the live ledger — request counts per model, the identical ordered case set, task/split, the 24-row ledger delta of $0.0267926, Luna's and Sonnet's mean/median CER and WER, and proof that no audited reference, crop byte or prompt hash changed after execution. Stage-1 reproduces exactly.

## Research-boundary fix

**Root cause.** `check_cloud_call` returned immediately when `execution_mode == "research"`, *before any layer ran*. `--research` therefore meant "skip every cloud safety check": a research OCR run also lost its registered-prompt check, its grading tripwires and its block-shape limits. Stage-1's 24 payloads were verified clean offline against the production path, so nothing leaked — but the architecture would not have caught it.

**Fix.** Two layers, and only the first knows the mode:

| Layer | Production | Research |
|---|---|---|
| 1 — task / experiment | `CLOUD_OCR_ALLOWLIST` only | that allowlist **plus** what an explicit `ResearchAuthorization` names (exact campaign, task, model; no wildcards) |
| 2 — content / payload | registered OCR prompt, grading tripwires, secret scan, campaign block limits | **identical — not disableable** |

`--research` with no authorization object is now exactly as strict as production. Checks that stay active under `--research`: registered OCR system prompt (exact match); grading-context header and grading-system-prompt tripwires; credential scan on every task; campaign image/text block limits; and `runner.leakage_check`, which always ran. The bench runner builds a per-run authorization naming only that role and that resolved model.

Payload byte-identity was proven before any new call: all 24 Stage-1 payloads rebuilt after the refactor match the committed pre-execution contract on prompt hashes, block shapes, max_tokens, schema and adapter version. **Stage-1b is comparable.**

## Reference-order audit — `assoc_docB_p2_b1`

Source-verified four ways, not by model agreement:

1. **The crop itself** is a single right-to-left row: `(א) 0.39   (ב) 0.47   (ג) 0.51   (ד) 0.55`.
2. **The frozen reference's own provenance** says the pairs were derived mechanically — *"value = nearest numeric word left of the option letter"*.
3. **Historical qwen3-27b**, independent of Stage-1, read the same RTL order.
4. **Historical qwen3-8b produced the reversed association**, so agreement here is discriminative, not trivial.

**The frozen reference is semantically correct.** It encodes the right pairs, serialised left-to-right with line breaks between visual columns. A correct RTL reading scores CER 0.7083 against it. All three candidates that returned this case got the association **exactly right**.

The frozen bytes were **not** modified. An audited logical-order record sits beside them, marked `provisional_pending_owner_confirmation` — the auditor is the assistant, by direct inspection of the crop; you have not confirmed it by eye. `parse_option_associations()` takes an **explicit** convention because guessing between letter-first and value-first is precisely the defect under audit.

## Frozen vs audited-logical-order (both reported, never substituted)

| Model | frozen mean CER (n=8) | logical-order mean CER (n=8) | association exact |
|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 0.7715 *(scored 8)* | 0.6829 *(scored 8)* | yes |
| `anthropic/claude-sonnet-5` | 0.4878 *(scored 8)* | 0.3992 *(scored 8)* | yes |
| `google/gemini-3.7-flash` | 0.1774 *(scored 5)* | 0.0358 *(scored 5)* | yes |

## Every Gemini Stage-1b observation

| Case | HW | Reference | OCR | CER | WER | Critical | Reasoning tok |
|---|---|---|---|---|---|---|---|
| `hl_e003_q1_r1__l1` | yes | ניתן לראות ברמות התדרים הגבוהים שיש שפות בכיוון התנועה(הטשטו | ניתן לראות ברמות התדרים הגבוהים שיש שפות בכיוון התנועה (הטשט | 0.0000 | 0.0000 | — | 219 |
| `hc_e002_q1_r1` | yes | יש טשטוש בכל התדרים | **NO OUTPUT** — truncation_at_effective_cap | n/a | n/a | LINE_LOST_NO_OUTPUT | — |
| `hc_e002_q1_r7` | yes | סה"כ הפירמידה נראית תקינה. הדרגה 0 שלה בהירה משמעותית מהתמונ | סה״כ הפירמידה נראית תקינה, הדרגות ⏎ שלה בהירה משמעותית מהתמונה | 0.0588 | 0.3333 | DIGIT_CHANGED(ref=0,ocr=-) | 182 |
| `hc_e002_q2_r1` | yes | עבור גילוי שפות יהיה רוב התמונה ב0 ורק עבור שפות 255 | **NO OUTPUT** — schema_validation | n/a | n/a | LINE_LOST_NO_OUTPUT | — |
| `hc_e002_q2_r6` | yes | המעכה זהה לסכימת תמונה זהה עם תנודה קלה. מין תמונת echo כזאת | **NO OUTPUT** — truncation_at_effective_cap | n/a | n/a | LINE_LOST_NO_OUTPUT | — |
| `pr_docA_p1_b1` | no | 203.3730 ⏎  203.6730 /  ⏎   סמסטר א' מועד א' תשפ"ו  ⏎ 2025-2026 | 203.6730 / 203.3730 סמסטר א' מועד א' תשפ"ו ⏎ 2025-2026 | 0.1200 | 0.3077 | DIGIT_CHANGED(ref=2033730203673020252026,ocr=2036730203373020252026) | 0 |
| `pr_docA_p2_b3` | no | לאחר הפעולה בונים פרמידת WAVELETS של תמונת התוצאה. בעמוד הבא | לאחר הפעולה בונים פרמידת WAVELETS של תמונת התוצאה. בעמוד הבא | 0.0000 | 0.0000 | — | 144 |
| `assoc_docB_p2_b1` | no | 0.55 ⏎ ()ד0.51 ⏎ ()ג0.47 ⏎ ()ב0.39 ⏎ ()א | א: 0.39; ב: 0.47; ג: 0.51; ד: 0.55 | 0.7083 | 1.2222 | DIGIT_CHANGED(ref=055051047039,ocr=039047051055) | 90 |

## Handwritten-only comparison (the product view)

| Model | cases | scored | mean CER | median CER | mean WER | line loss | refusals | digit/sign errors |
|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 5 | 5 | 0.9487 | 1.0000 | 1.0000 | 0 | 4 | 3 |
| `anthropic/claude-sonnet-5` | 5 | 5 | 0.6188 | 0.4559 | 0.9018 | 0 | 2 | 3 |
| `google/gemini-3.7-flash` | 5 | 2 | 0.0294 | 0.0294 | 0.1666 | 3 | 0 | 1 |

## Printed / text-layer comparison

| Model | cases | scored | mean CER | mean WER | exact |
|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 3 | 3 | 0.4761 | 0.6638 | 0 |
| `anthropic/claude-sonnet-5` | 3 | 3 | 0.2694 | 0.5100 | 0 |
| `google/gemini-3.7-flash` | 3 | 3 | 0.2761 | 0.5100 | 0 |

Printed content flatters every candidate and must never be merged into a shipping conclusion: Luna reads printed mixed Hebrew/English at CER 0.00 and handwriting at 0.9487.

## Reliability, latency and tokens

| Model | schema-valid | failure classes | mean lat | median | p95 | input tok | output tok | reasoning tok |
|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 8/8 | — | 2.719s | 2.633s | 3.265s | 29593 | 990 | 0 |
| `anthropic/claude-sonnet-5` | 8/8 | — | 4.383s | 4.515s | 6.062s | 7918 | 385 | 0 |
| `google/gemini-3.7-flash` | 5/8 | {'truncation_at_effective_cap': 2, 'schema_validation': 1} | 4.106s | 3.969s | 6.64s | 6927 | 849 | 635 |

## Decision gates

These gates were written after the Stage-1b numbers existed — the arm had already run when the criteria were set down, so they are not blind. They are stated conservatively and none was chosen to let a particular candidate through; G4's denominator clause and G6 in fact block the strongest-looking candidate.

**`openai/gpt-5.6-luna-pro` → DROP** (3/7 gates)

- ✅ `G1_no_total_line_loss` — 0 lost
- ❌ `G2_no_repeated_refusal` — 4 refusals
- ❌ `G3_no_fabrication_pattern` — 1 fabrication(s)
- ❌ `G4_handwritten_quality` — mean CER 0.9487 on 5/5 handwritten
- ❌ `G5_bounded_critical_errors` — 3 handwritten digit/sign/negation cases
- ✅ `G6_reliability` — 8/8 scored
- ✅ `G7_latency_and_cost` — mean 2.719s, $0.00088833/crop

Confirmed by source verification, and the stratified view makes it worse than the Stage-1 headline: handwritten mean CER 0.9487 (the 0.7715 blend was diluted by printed text). It refused 4 of 5 handwritten crops against readable audited references and fabricated the fifth. Fails G2, G3 and G4. Nothing about its printed-text competence (CER 0.00 on mixed Hebrew/English) rescues a model that cannot read handwriting, which is the entire role.

**`anthropic/claude-sonnet-5` → MAYBE** (4/7 gates)

- ✅ `G1_no_total_line_loss` — 0 lost
- ❌ `G2_no_repeated_refusal` — 2 refusals
- ✅ `G3_no_fabrication_pattern` — 0 fabrication(s)
- ❌ `G4_handwritten_quality` — mean CER 0.6188 on 5/5 handwritten
- ❌ `G5_bounded_critical_errors` — 3 handwritten digit/sign/negation cases
- ✅ `G6_reliability` — 8/8 scored
- ✅ `G7_latency_and_cost` — mean 4.383s, $0.00246075/crop

Unchanged from Stage-1 and still the most RELIABLE arm: 8/8 schema-valid, zero line loss, no fabrication. But handwritten mean CER 0.6188 over the full 5 crops fails G4 by a wide margin, it refused 2 of 5, and 3 handwritten cases carry digit/sign errors. It is usable as a floor, not as a candidate to scale. It does NOT advance on this evidence.

**`google/gemini-3.7-flash` → MAYBE** (4/7 gates)

- ❌ `G1_no_total_line_loss` — 3 lost
- ✅ `G2_no_repeated_refusal` — 0 refusals
- ✅ `G3_no_fabrication_pattern` — 0 fabrication(s)
- ❌ `G4_handwritten_quality` — mean CER 0.0294 on 2/5 handwritten
- ✅ `G5_bounded_critical_errors` — 1 handwritten digit/sign/negation cases
- ❌ `G6_reliability` — 5/8 scored
- ✅ `G7_latency_and_cost` — mean 4.106s, $0.00212582/crop

The most interesting and the least conclusive. On the 5 cases it completed it is far ahead of both rivals — handwritten CER 0.0294 vs Sonnet's 0.6188, an exact reading of one handwritten line, zero refusals, zero fabrication, and the option row read correctly. But it returned no usable output on 3 of 8, so it fails G1, G6 and — decisively — G4's denominator clause: a 0.0294 mean over 2 of 5 handwritten crops is not evidence about the 5. Those 3 failures are attributable to a harness defect (the provider was sent max_tokens=600, not the derived 1000), NOT to the model, and truncation was uncorrelated with reference length — the SHORTEST reference (19 chars) truncated while a 69-char one succeeded, so mandatory-reasoning token variance is what consumed the cap. Whether 1000 is sufficient is UNVALIDATED. It is the clear frontrunner and it has not earned the larger stage yet.

## Does anything advance to the larger seen-only OCR stage?

**No.** No candidate advances to the 32/21/53-crop seen-only stage on this evidence. Luna is dropped. Sonnet and Gemini both remain plausible, but the comparison between them is not yet decidable: the arm that looks best was measured on 5 of 8 cases because of a harness defect this arm itself uncovered. Scaling now would spend the larger budget to answer a question a corrected 8-case rerun answers for about $0.02.

Required first: one Stage-1c arm: google/gemini-3.7-flash, the identical 8 frozen crops, reasoning=low, with the max_tokens fix in place so the provider actually receives the configured cap.

## Cost projections (measured)

| Model | $/crop | 32 DEV | 21 CALIB | 53 seen | 100 crops | 100 exams @5 | @10 | @15 |
|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | $0.00088833 | $0.0284 | $0.0187 | $0.0471 | $0.0888 | $0.4442 | $0.8883 | $1.3325 |
| `anthropic/claude-sonnet-5` | $0.00246075 | $0.0787 | $0.0517 | $0.1304 | $0.2461 | $1.2304 | $2.4607 | $3.6911 |
| `google/gemini-3.7-flash` | $0.00212582 | $0.0680 | $0.0446 | $0.1127 | $0.2126 | $1.0629 | $2.1258 | $3.1887 |

Per-exam figures are meaningless without the crop-count assumption, which is stated in each column. Gemini's rate is per **billable** crop: 7 of 8 requests billed and 3 produced nothing usable, so it prices attempts, not successes — a corrected-cap rerun will cost more per usable transcription.

**Local grading cloud cost is $0.** Cloud grading cost is $0. No grading model ran, cloud or local.

## Accounting reconciliation

- ledger rows 686 → 694 (+8)
- run-attributed cost **$0.01488075**; billable 7, non-billable 1
- account usage $0.49848267 → $0.51336342 (delta **$0.01488075**)
- **rounding difference 0.0** — exact match, no propagation lag to wait out
- project cumulative $0.51336379 against the $8 warn / $10 hard ceiling

Truncated and schema-failed responses were still billed — that is correct provider behaviour, and every one of them has a ledger row.

## Recommended next experiment (NOT executed)

**OCR_SMOKE_STAGE1C_GEMINI_CORRECTED_CAP** — RECOMMENDED, NOT EXECUTED — requires owner authorization

- candidate: `google/gemini-3.7-flash`; population: the identical 8 frozen Stage-1 cases; no replacements, no additions
- changes: none to the prompt, schema, adapter, crops or references; the runner now sends route.max_tokens, so the already-declared cap of 1000 actually reaches the provider (this is the harness fix, not a new decoding decision)
- question it answers: does gemini-3.7-flash return a schema-valid transcription on all 8 crops when mandatory low-effort reasoning has the derived 753-token headroom? If yes, its handwritten CER becomes comparable to Sonnet's on a full denominator and the choice is decidable. If it still truncates, the cap must be re-derived from the observed reasoning distribution rather than raised reflexively.
- predicted cost $0.037415, max 8 requests
- contingency: if 8/8 succeed and handwritten CER holds below ~0.10 on 5/5, Gemini earns the 32-crop seen DEV stage (projected $0.0595). If it truncates again, neither candidate should scale and the next move is a prompt-version or candidate change under a new pre-registration, not a bigger population.

## Confirmations

- new_provider_ocr_requests: 8
- grading_provider_calls: 0
- local_grading_calls: 0
- ocr_verification_calls: 0
- rag_calls: 0
- held_out_calls_or_exposure: 0
- frozen_audited_references_overwritten: 0
- active_grades_changed: 0
- additional_spend_usd: 0.01488075
- within_004_ceiling: True
- api_key_exposure: 0
- luna_rerun: False
- sonnet_rerun: False
- ocr_prompt_modified: False
- replacement_cases: 0
- retry_arm: 0
