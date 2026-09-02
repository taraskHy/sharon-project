# OCR Stage-1c — corrected-cap Gemini arm and three-candidate comparison (2026-09-02)

Experiment `OCR_SMOKE_STAGE1C_GEMINI_CORRECTED_CAP` (`2be5224f49142dab…`). **8 new provider requests, $0.01099875 of a $0.05 ceiling.** Grading / OCR-verification / RAG / HELD_OUT calls: 0. Luna and Sonnet not rerun.

## Headline

| Model | Handwritten success | HW mean CER | HW mean WER | Critical errors (HW) | Refusals | Fabrications | Printed success / CER | Mean latency | Cost |
|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | **5/5** | 0.9487 | 1.0000 | 3 | 4 | 1 | 3/3 / 0.4761 | 2.719s | $0.007107 |
| `anthropic/claude-sonnet-5` | **5/5** | 0.6188 | 0.9018 | 3 | 2 | 0 | 3/3 / 0.2694 | 4.383s | $0.019686 |
| `google/gemini-3.7-flash` | **2/5** | 0.1799 | 0.3333 | 1 | 0 | 0 | 3/3 / 0.2628 | 7.6s | $0.010999 |

> **The CER column is over successes only.** Read it together with the success column. Gemini's 0.1799 covers 2 of 5 handwritten crops; Luna's and Sonnet's cover 5 of 5. The failure-aware comparison on the intended denominator is below, and it reverses the ordering.

### The same comparison, failure-aware, on all 5 intended handwritten crops

| Model | successes | failures | mean CER over successes | **failure-aware CER bound** |
|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 5 | 0 | 0.9487 | **0.9487** |
| `anthropic/claude-sonnet-5` | 5 | 0 | 0.6188 | **0.6188** |
| `google/gemini-3.7-flash` | 2 | 3 | 0.1799 | **0.6720** |

The bound scores each failure as CER 1.0. It is an upper bound on error, not a prediction of what the model would have written. On it, **Gemini (0.6720) is worse than Sonnet (0.6188)** — the opposite of the successes-only reading. The truth for Gemini lies between 0.1799 and 0.6720, and eight cases cannot narrow it further.

## 1. Verified starting state

Branch `initial-prototype`, clean tree, HEAD `d548cc4` at start — containing the Stage-1b arm and the route-propagation fix. Stage-1c work is committed on top.

## 2. Stage-1 / Stage-1b reproduction

`40/40` checks, all recomputed from machine-readable run data (outputs.jsonl / run.json / the ledger / the frozen manifest), never from Markdown:

- Stage-1: Luna 8/8 parsed, Sonnet 8/8 parsed, Gemini 8/8 pre-inference HTTP 400 with zero tokens consumed
- Stage-1b Gemini: 8 attempted, 5 parsed, 3 lost — 2 explicit `truncated at max_tokens=600` plus 1 schema failure
- **The defect, pinned as a check:** `run.json` recorded `max_tokens: 1000` while the provider error text says `600`
- task `ocr_primary` only; DEV only (HELD_OUT 0); 32 ledger rows, all `ocr_primary`, zero grading calls
- audited references, crop hashes, manifest hashes and m2-strict-v1 prompt hashes all unchanged
- Luna's and Sonnet's headline CERs independently recomputed to 0.7715 / 0.4878

## 3. max_tokens propagation — root cause and proof

**Root cause.** `build_route` resolves the chain (adapter default → models.toml → declared candidate override → `--max-tokens`) into `route.max_tokens`. `run_benchmark` then called `gw.call(..., max_tokens=request.max_tokens)` — the adapter's own unresolved default. Every configured value was discarded on the way to the provider while still being recorded in the config hash and priced by the dry run.

**Proof of correction, on the wire:**

| Stage | Value |
|---|---|
| candidates.toml | 1000 |
| resolved route | 1000 |
| adapter Request default | 600 (now unused by the runner) |
| **serialized OpenRouter payload** | **1000** |

- *Offline:* the real OpenRouterBackend driven through httpx.MockTransport serialized max_tokens=1000 for all 8 payloads — deterministic, on the exact route
- *Live:* zero truncation errors this arm (Stage-1b had 2 explicit ones), and hc_e002_q1_r1 — truncated at 600 in Stage-1b — completed here
- *Honest limit:* the highest output_tokens observed live was 491, so the live run did NOT independently exercise the region above 600. The wire value is proven offline; the live arm is consistent with it but is not a second proof. The highest output_tokens observed live was 491.

A companion test asserts that passing 600 still serializes as 600, so the 1000 assertion is capable of failing.

## 4-6. Experiment identity, cases and serialized route

- experiment `2be5224f49142dab7641c0be4ca6455c12c203e000dc3dd89d8f6feee442ffb1`
- parent Stage-1b `4de29894cc25b0cc…`
- case order `8b9039cf3d15dc49…`, smoke selection `cc5c9f1ff9911a68…`
- route `55ab403ec6c3a1d5…`: `{"task": "ocr_primary", "backend": "openrouter", "model": "google/gemini-3.7-flash", "structured_mode": "json_schema", "max_tokens": 1000, "temperature": 0.0, "reasoning": {"effort": "low"}, "extra_generation": {}, "prompt_version": "m2-strict-v1"}`

The eight cases, in frozen order:

| # | case | type | category | writer | ref chars | crop sha256 |
|---|---|---|---|---|---|---|
| 1 | `hl_e003_q1_r1__l1` | handwritten | handwritten_line | e003 | 62 | `a8703841a8bd…` |
| 2 | `hc_e002_q1_r1` | handwritten | handwritten_cell | e002 | 19 | `f23d427da6e0…` |
| 3 | `hc_e002_q1_r7` | handwritten | handwritten_cell | e002 | 69 | `7cd8b22b872c…` |
| 4 | `hc_e002_q2_r1` | handwritten | handwritten_cell | e002 | 52 | `cc2e23b6c8ac…` |
| 5 | `hc_e002_q2_r6` | handwritten | handwritten_cell | e002 | 116 | `ebbbff9c23ba…` |
| 6 | `pr_docA_p1_b1` | printed | formula_printed | — | 57 | `5d759b357831…` |
| 7 | `pr_docA_p2_b3` | printed | mixed_he_en | — | 101 | `ed3c5ef5b4cb…` |
| 8 | `assoc_docB_p2_b1` | printed | option_row_association | — | 32 | `971600121151…` |

**Run isolation.** Stage-1c's config hash is identical to Stage-1b's — the route did not change, the runner's behaviour did. In the default runs root the resume rule (`skip = done_ok | done_failed`) would have skipped all 8 cases and overwritten Stage-1b. Stage-1c therefore ran under `evaluation/model_selection/runs_stage1c`, with `skipped_resume=0` confirmed by the dry run.

## 7. Request contract and research-boundary verification

All 8 payloads checked offline before any network access, at the wire level:

- allowed under `--research` with the runner's exact per-run authorization (`bench:ocr_primary:dev:smoke`, tasks ['ocr_primary'], models ['google/gemini-3.7-flash'])
- **and allowed under production with no authorization at all** — the research flag is not doing the work
- registered m2-strict-v1 prompt: True
- grading tripwire hits: 0; banned-substring hits: 0
- audited reference text in any wire payload: 0
- runner leakage check: ['passed']
- wire shape: [1] image block + [1] text block per request

**Correction carried forward.** The Stage-1 contract artifact said *zero text blocks*. That was true of the adapter's `Request.content_blocks` but **not** of the serialized request: `OpenAICompatBackend._build_payload` appends one structured-output block carrying the BenchTranscription JSON Schema. It is the allowed minimal schema and is built from the output model, so it cannot carry case data — but it is invisible to `check_cloud_call`, which scans adapter-level blocks. It is now asserted on the bytes that actually leave.

## 8. All eight Gemini Stage-1c outputs

**`hl_e003_q1_r1__l1`** — handwritten, handwritten_line, crop `a8703841a8bd…`, sent max_tokens 1000

- REF: `ניתן לראות ברמות התדרים הגבוהים שיש שפות בכיוון התנועה(הטשטוש)`
- OCR: **NO OUTPUT** — the backend refused this request (content_filter) (`content_filter_refusal`)
- critical: LINE_LOST_NO_OUTPUT

**`hc_e002_q1_r1`** — handwritten, handwritten_cell, crop `f23d427da6e0…`, sent max_tokens 1000

- REF: `יש טשטוש בכל התדרים`
- OCR: `אשר יש לטשטש בכל התדרים`
- CER 0.3158 · WER 0.5000 · omit 0.0000 · halluc 0.2000 · subs/del/ins 1/0/1
- tokens in/out/reasoning 1410/491/474 · latency 12.735s
- critical: none
- ⚠️ **HUMAN REVIEW NEEDED** — CER 0.3158 with a plausible-looking but different reading ('אשר יש לטשטש' vs 'יש טשטוש') — meaning may have flipped

**`hc_e002_q1_r7`** — handwritten, handwritten_cell, crop `7cd8b22b872c…`, sent max_tokens 1000

- REF: `סה"כ הפירמידה נראית תקינה. הדרגה 0 שלה בהירה משמעותית מהתמונה המקורית`
- OCR: `סה"כ הפירמידה נראית תקינה, הדרגאס
שלה בהירה משמעותית מהתמונה המקורית.`
- CER 0.0441 · WER 0.1667 · omit 0.0833 · halluc 0.0000 · subs/del/ins 1/1/0
- tokens in/out/reasoning 1386/485/444 · latency 6.672s
- critical: DIGIT_CHANGED(ref=0,ocr=-)

**`hc_e002_q2_r1`** — handwritten, handwritten_cell, crop `cc2e23b6c8ac…`, sent max_tokens 1000

- REF: `עבור גילוי שפות יהיה רוב התמונה ב0 ורק עבור שפות 255`
- OCR: **NO OUTPUT** — the backend refused this request (content_filter) (`content_filter_refusal`)
- critical: LINE_LOST_NO_OUTPUT

**`hc_e002_q2_r6`** — handwritten, handwritten_cell, crop `ebbbff9c23ba…`, sent max_tokens 1000

- REF: `המעכה זהה לסכימת תמונה זהה עם תנודה קלה. מין תמונת echo כזאת תגרום להיסטוגרמה החדשה להיות קרובה להכפלה ב-2 [לא קריא]`
- OCR: **NO OUTPUT** — the backend refused this request (content_filter) (`content_filter_refusal`)
- critical: LINE_LOST_NO_OUTPUT

**`pr_docA_p1_b1`** — printed, formula_printed, crop `5d759b357831…`, sent max_tokens 1000

- REF: `203.3730
 203.6730 / 
  סמסטר א' מועד א' תשפ"ו 
2025-2026`
- OCR: `203.3730 / 203.6730 סמסטר א' מועד א' תשפ"ו
2025-2026`
- CER 0.0800 · WER 0.1538 · omit 0.0769 · halluc 0.0769 · subs/del/ins 0/1/1
- tokens in/out/reasoning 1433/49/0 · latency 3.39s
- critical: none

**`pr_docA_p2_b3`** — printed, mixed_he_en, crop `ed3c5ef5b4cb…`, sent max_tokens 1000

- REF: `לאחר הפעולה בונים פרמידת WAVELETS של תמונת התוצאה. בעמוד הבא נתונות 
תמונות התוצאה ממוספרות  )
A-I
(.`
- OCR: `לאחר הפעולה בונים פרמידת WAVELETS של תמונת התוצאה. בעמוד הבא נתונות
תמונות התוצאה ממוספרות (A-I).`
- CER 0.0000 · WER 0.0000 · omit 0.0000 · halluc 0.0000 · subs/del/ins 0/0/0
- tokens in/out/reasoning 1365/395/342 · latency 6.062s
- critical: none

**`assoc_docB_p2_b1`** — printed, option_row_association, crop `971600121151…`, sent max_tokens 1000

- REF: `0.55
()ד0.51
()ג0.47
()ב0.39
()א`
- OCR: `א: 0.39; ב: 0.47; ג: 0.51; ד: 0.55`
- CER 0.7083 · WER 1.2222 · omit 0.0000 · halluc 0.2500 · subs/del/ins 8/0/3
- tokens in/out/reasoning 1296/135/98 · latency 9.141s
- critical: DIGIT_CHANGED(ref=055051047039,ocr=039047051055)

## 9. Handwritten-only, over all 5 intended cases

| Model | intended | successes | failures | exact | mean CER | median CER | mean WER | median WER | failure-aware bound | omissions | hallucinations | line loss | unreadable | digit/sign/neg (exact seq) | (signature) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 5 | 5 | 0 | 0 | 0.9487 | 1.0000 | 1.0000 | 1.0000 | 0.9487 | 0.7515 | 0.0000 | 0 | 4 | 3 | 3 |
| `anthropic/claude-sonnet-5` | 5 | 5 | 0 | 0 | 0.6188 | 0.4559 | 0.9018 | 1.0000 | 0.6188 | 0.3667 | 0.0000 | 0 | 3 | 3 | 3 |
| `google/gemini-3.7-flash` | 5 | 2 | 3 | 0 | 0.1799 | 0.1799 | 0.3333 | 0.3333 | 0.6720 | 0.0416 | 0.1000 | 3 | 0 | 1 | 1 |

## 10. Printed / text-layer, over all 3 intended cases

| Model | intended | successes | mean CER | mean WER | exact |
|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 3 | 3 | 0.4761 | 0.6638 | 0 |
| `anthropic/claude-sonnet-5` | 3 | 3 | 0.2694 | 0.5100 | 0 |
| `google/gemini-3.7-flash` | 3 | 3 | 0.2628 | 0.4587 | 0 |

Printed content flatters every candidate — Luna reads printed mixed Hebrew/English at CER 0.00 and handwriting at 0.9487 — which is why the two are never merged into one shipping number.

## 11. Frozen reference vs audited logical order

| Model | frozen mean CER (8) | logical-order mean CER (8) | `assoc` frozen CER | `assoc` logical CER | association exact |
|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 0.7715 | 0.6829 | 0.7083 | 0.0000 | yes |
| `anthropic/claude-sonnet-5` | 0.4878 | 0.3992 | 0.7083 | 0.0000 | yes |
| `google/gemini-3.7-flash` | 0.2296 | 0.0880 | 0.7083 | 0.0000 | yes |

The frozen sequence is preserved everywhere and never silently replaced. For `assoc_docB_p2_b1` the frozen reference serialises a right-to-left option row in PDF text-layer (left-to-right) order; the audited logical-order reading is `pdf_text_layer_order_not_visual_reading_order`, status **provisional_pending_owner_confirmation**. Gemini's output matches that reading **exactly (CER 0.0)** while scoring 0.7083 against the frozen bytes — the clearest demonstration yet that this case measures serialisation, not reading.

## 12. Critical errors, case by case (deterministic only)

| Model | case | flags |
|---|---|---|
| `gpt-5.6-luna-pro` | `hl_e003_q1_r1__l1` | UNREADABLE_OUTPUT_ON_READABLE_REFERENCE |
| `gpt-5.6-luna-pro` | `hc_e002_q1_r1` | LATIN_TOKEN_CHANGED(ref=[],ocr=['unreadable']), UNREADABLE_OUTPUT_ON_READABLE_REFERENCE |
| `gpt-5.6-luna-pro` | `hc_e002_q1_r7` | DIGIT_CHANGED(ref=0,ocr=-), LATIN_TOKEN_CHANGED(ref=[],ocr=['unreadable']), UNREADABLE_OUTPUT_ON_READABLE_REFERENCE |
| `gpt-5.6-luna-pro` | `hc_e002_q2_r1` | DIGIT_CHANGED(ref=0255,ocr=-), LATIN_TOKEN_CHANGED(ref=[],ocr=['unreadable']), UNREADABLE_OUTPUT_ON_READABLE_REFERENCE |
| `gpt-5.6-luna-pro` | `hc_e002_q2_r6` | DIGIT_CHANGED(ref=2,ocr=-), SIGN_OPERATOR_CHANGED(ref=-,ocr=-), LATIN_TOKEN_CHANGED(ref=['echo'],ocr=[]), NEGATION_OMITTED(לא) |
| `gpt-5.6-luna-pro` | `pr_docA_p1_b1` | DIGIT_CHANGED(ref=2033730203673020252026,ocr=2036730203373020252026) |
| `gpt-5.6-luna-pro` | `assoc_docB_p2_b1` | DIGIT_CHANGED(ref=055051047039,ocr=039047051055) |
| `claude-sonnet-5` | `hc_e002_q1_r1` | LATIN_TOKEN_CHANGED(ref=[],ocr=['unreadable']), UNREADABLE_OUTPUT_ON_READABLE_REFERENCE |
| `claude-sonnet-5` | `hc_e002_q1_r7` | DIGIT_CHANGED(ref=0,ocr=-), SIGN_OPERATOR_CHANGED(ref=-,ocr=-) |
| `claude-sonnet-5` | `hc_e002_q2_r1` | DIGIT_CHANGED(ref=0255,ocr=-), LATIN_TOKEN_CHANGED(ref=[],ocr=['unreadable']), UNREADABLE_OUTPUT_ON_READABLE_REFERENCE |
| `claude-sonnet-5` | `hc_e002_q2_r6` | DIGIT_CHANGED(ref=2,ocr=-), SIGN_OPERATOR_CHANGED(ref=-,ocr=-), LATIN_TOKEN_CHANGED(ref=['echo'],ocr=[]), NEGATION_OMITTED(לא) |
| `claude-sonnet-5` | `assoc_docB_p2_b1` | DIGIT_CHANGED(ref=055051047039,ocr=039047051055) |
| `gemini-3.7-flash` | `hl_e003_q1_r1__l1` | LINE_LOST_NO_OUTPUT |
| `gemini-3.7-flash` | `hc_e002_q1_r7` | DIGIT_CHANGED(ref=0,ocr=-) |
| `gemini-3.7-flash` | `hc_e002_q2_r1` | LINE_LOST_NO_OUTPUT |
| `gemini-3.7-flash` | `hc_e002_q2_r6` | LINE_LOST_NO_OUTPUT |
| `gemini-3.7-flash` | `assoc_docB_p2_b1` | DIGIT_CHANGED(ref=055051047039,ocr=039047051055) |

No grading model was used and no official solution was consulted; every flag is a deterministic comparison of digits, signs/operators, Latin tokens, single-letter variables, negation particles and unreadable markers.

## 13. Fair comparison

| | Luna (Stage-1) | Sonnet (Stage-1) | Gemini (Stage-1c) |
|---|---|---|---|
| handwritten coverage | 5/5 | 5/5 | 2/5 |
| HW mean CER (successes) | 0.9487 | 0.6188 | 0.1799 |
| HW failure-aware bound | 0.9487 | 0.6188 | 0.6720 |
| raw failures /8 | 0 | 0 | 3 |
| schema-valid /8 | 8 | 8 | 5 |
| full refusals (output is only a marker) | 4 | 2 | 0 |
| any unreadable marker present | 4 | 3 | 0 |
| fabrications | 1 | 0 | 0 |
| mean latency | 2.719s | 4.383s | 7.6s |

Gemini's earlier arms are excluded from quality comparison and reported as what they were: Stage-1 as **invalid configuration** (8 pre-inference HTTP 400s, $0, no OCR evidence) and Stage-1b as a **route-token-limit execution defect** (8 attempted, 5 usable, incomplete evidence).

## 14. Classification

These considerations were specified by the owner BEFORE this arm ran, in the task brief. They are applied here as written and none was adjusted after seeing the numbers — two of them block the candidate with the best transcription quality.

**`openai/gpt-5.6-luna-pro` → DROP** (6/8 criteria)

- ❌ `C1_no_repeated_refusal_pattern` — 4 unreadable-on-readable
- ✅ `C2_no_repeated_total_line_loss` — 0 lost lines
- ✅ `C3_no_repeated_fabrication` — 1 fabrication(s)
- ✅ `C4_full_or_near_full_coverage` — 8/8 schema-valid
- ❌ `C5_materially_lower_handwritten_error` — failure-aware HW bound 0.9487 vs Luna 0.9487
- ✅ `C6_critical_errors_visible` — enumerated per case
- ✅ `C7_stable_structured_output` — 0 schema failures
- ✅ `C8_acceptable_latency_and_cost` — mean 2.719s, $0.00088833/crop

Unchanged and confirmed. On the full 5-case handwritten denominator its mean CER is 0.9487 — it refused 4 of 5 handwritten crops with an unreadable marker against readable audited references, and fabricated the fifth as fluent Hebrew on an unrelated subject. It is perfectly reliable (8/8 schema-valid) and the cheapest arm at $0.00089/crop, and neither fact matters: a transcriber that refuses handwriting or invents it cannot carry this pipeline.

**`anthropic/claude-sonnet-5` → MAYBE** (7/8 criteria)

- ❌ `C1_no_repeated_refusal_pattern` — 2 unreadable-on-readable
- ✅ `C2_no_repeated_total_line_loss` — 0 lost lines
- ✅ `C3_no_repeated_fabrication` — 0 fabrication(s)
- ✅ `C4_full_or_near_full_coverage` — 8/8 schema-valid
- ✅ `C5_materially_lower_handwritten_error` — failure-aware HW bound 0.6188 vs Luna 0.9487
- ✅ `C6_critical_errors_visible` — enumerated per case
- ✅ `C7_stable_structured_output` — 0 schema failures
- ✅ `C8_acceptable_latency_and_cost` — mean 4.383s, $0.00246075/crop

The only arm with complete coverage on every view: 8/8 schema-valid, zero line loss, zero fabrication, and a handwritten mean CER of 0.6188 measured on all 5 handwritten crops. That number is also the problem — 62% character error is not usable transcription, and it refused 2 of 5. It is the reliability floor of this comparison rather than a candidate that has earned production. It remains plausible only because it is the one model whose handwriting number is not an artifact of which cases survived.

**`google/gemini-3.7-flash` → MAYBE** (5/8 criteria)

- ✅ `C1_no_repeated_refusal_pattern` — 0 unreadable-on-readable
- ❌ `C2_no_repeated_total_line_loss` — 3 lost lines
- ✅ `C3_no_repeated_fabrication` — 0 fabrication(s)
- ❌ `C4_full_or_near_full_coverage` — 5/8 schema-valid
- ❌ `C5_materially_lower_handwritten_error` — failure-aware HW bound 0.672 vs Luna 0.9487
- ✅ `C6_critical_errors_visible` — enumerated per case
- ✅ `C7_stable_structured_output` — 0 schema failures
- ✅ `C8_acceptable_latency_and_cost` — mean 7.6s, $0.00219975/crop

Stage-1c did what it was for — it proved the route fix and removed truncation entirely (zero truncation errors, versus two in Stage-1b) — and then uncovered a different, more serious problem. Three of eight requests, ALL handwritten, were refused outright by the provider's content filter before inference. On the two handwritten crops it did return, Gemini is far the best model here (mean CER 0.1799 vs Sonnet's 0.6188, and one printed case exact), with zero refusal markers and zero fabrication. But on the intended 5-case denominator, scoring the three refusals as total loss, its bound is 0.6720 — WORSE than Sonnet's measured 0.6188. Across Stage-1b and Stage-1c it has never produced more than 2 of 5 handwritten observations in a single arm, for two unrelated reasons. Its quality ceiling is the highest measured; its delivered coverage is the lowest.

## 15. Advancement decision and the next stage (not executed)

**B — Gemini and Sonnet both remain plausible; a bounded larger seen-only comparison is warranted**

No candidate is declared production-ready, and eight samples could not do that. The larger seen-only stage is now the RIGHT next experiment for a reason Stage-1c created rather than settled: the decisive unknown is no longer transcription quality but Gemini's content-filter refusal rate. Three refusals in eight requests is either a ~37% operational failure rate — which would disqualify the model regardless of how well it reads the crops it accepts — or small-sample noise. Eight cases cannot tell those apart; 32 can, for about eight cents. Sonnet travels as the reliability control because it is the only arm whose handwriting number rests on a complete denominator.

- *Why not A:* Gemini does not clearly survive: it fails C2 (3 lost lines) and C4 (5/8 coverage), and its failure-aware handwritten bound is worse than Sonnet's measured value.
- *Why not C:* The prompt is not implicated. Gemini read two handwritten crops at CER 0.18 and 0.04 and a printed one exactly, under the unchanged m2-strict-v1 prompt, so 'no model is acceptable, change the prompt' is not what the evidence says.

**OCR_SEEN32_DEV_GEMINI_VS_SONNET** — RECOMMENDED, NOT EXECUTED — requires explicit owner authorization

- population: the frozen 32-case seen DEV OCR subset (evaluation/model_selection/subsets/ocr_primary__seen46_ocr_dev.json, selection 98c8d117e6747adc), unchanged
- candidates: google/gemini-3.7-flash (reasoning=low, max_tokens=1000), anthropic/claude-sonnet-5 (as run in Stage-1); at most 64 requests
- predicted cost: gemini $0.088, sonnet $0.0787, upper bound $0.17
- primary question: is Gemini's content-filter refusal rate a stable property (~37%) or small-sample noise? 32 crops bound it far tighter than 8.
- secondary: a handwriting comparison on a denominator large enough that neither model's number depends on which cases survived
- **decision rule, stated in advance:** If Gemini's refusal rate stays above ~15% it is DROPPED on operational grounds however good its accepted transcriptions are, because a pipeline cannot silently lose one crop in six. If refusals fall below ~5% and its handwritten CER on the full denominator stays under ~0.15, it ADVANCES alone and Sonnet is dropped. Anything between is reported, not resolved.
- explicitly excluded: the 21-case CALIBRATION subset, all 53 seen crops, OCR verification, any grading, RAG, HELD_OUT, any prompt change, any new model

## 16. Cost projections

| Model | $/crop | 32 DEV | 21 CALIB | 53 seen | 100 crops | 100 exams @5 | @10 | @15 |
|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | $0.00088833 | $0.0284 | $0.0187 | $0.0471 | $0.0888 | $0.4442 | $0.8883 | $1.3325 |
| `anthropic/claude-sonnet-5` | $0.00246075 | $0.0787 | $0.0517 | $0.1304 | $0.2461 | $1.2304 | $2.4607 | $3.6911 |
| `google/gemini-3.7-flash` | $0.00219975 | $0.0704 | $0.0462 | $0.1166 | $0.2200 | $1.0999 | $2.1997 | $3.2996 |

**Assumptions.** Cost per *billable* crop, same crop mix, one pass, no retries, no cache hits, no verification pass. cost per BILLABLE crop. 3 of 8 requests were refused by the provider content filter and cost nothing, so a real campaign paying for 53 crops would need 53 BILLABLE responses — at this refusal rate that means more attempts than crops. Gemini's attempt-adjusted rate — total spend divided by all 8 requests including the 3 free refusals — is $0.00137484/attempt.

**Grading cost is separate and is $0:** cloud grading $0, local-grading cloud cost $0. No grading model ran, cloud or local.

## 17. Accounting reconciliation

- ledger rows 694 → 702 (+8)
- ledger cumulative $0.51336379 → $0.52436254
- Stage-1c attributed cost **$0.01099875**
- billable responses 5; billed failures 0; non-billable failures 3 (the 3 content-filter refusals)
- account usage $0.51336342 → $0.52436217 (delta **$0.01099875**)
- **rounding difference 0.0** — exact, no propagation lag to wait out
- account limit $20, remaining $19.47563783

Every billable response has a ledger row, and the count of billable rows equals the count of cases that returned a transcription. The three refusals cost nothing, which is correct: they were rejected before inference.

### Decoding actually sent, per arm

| Arm | route recorded | actually sent | equal? |
|---|---|---|---|
| `openai/gpt-5.6-luna-pro` (stage1) | 400 | 600 | **no** |
| `anthropic/claude-sonnet-5` (stage1) | 400 | 600 | **no** |
| `google/gemini-3.7-flash` (stage1c) | 1000 | 1000 | yes |

Only Stage-1c sent what it recorded. Stage-1's Luna and Sonnet arms recorded 400 and were sent 600; neither was affected in outcome (no response hit a length finish_reason), but their recorded decoding config was not what went on the wire. Stage-1b recorded 1000 and was sent 600, which is what cost it three cases.

Two further identity caveats worth stating plainly: Stage-1b and Stage-1c **share config hash `45297cdd83`** — the route is identical and only the runner's behaviour differed, so the arms are separable by runs-root and git commit, not by recorded identity. And each `run.json` records a single representative `prompt_sha256` (the first case's `handwritten_line` prompt), not a per-case map; all six frozen category prompts are hashed in the pre-registration.

## 18-19. Corrections found by independent verification

Eight independent agents recomputed every metric from the raw outputs and tried to refute each headline claim. Three real defects surfaced, all now fixed:

1. **A provider refusal was being labelled a schema failure.** `score()` has always been handed the error string and ignored it: `"schema_failure": output is None` in all three adapters. So Stage-1c's own `metrics.json` says `schema_failures: 3` when all three losses were provider content-filter refusals and Gemini's structured output was valid on every request it was allowed to answer. A reader of that file would have concluded the opposite of the truth. Fixed with a `classify_no_output()` taxonomy (`schema_failure` vs `provider_failure`) and 15 tests. `adapter_version` is deliberately NOT bumped: no scored metric changes (a test asserts CER is identical), and bumping would break config-hash comparability with the frozen Stage-1/1b runs over a naming correction. The historical run artifacts are left append-only; `OCR_STAGE1C_CORRECTED_TAXONOMY_2026-09-02.json` carries the corrected counts.

2. **Two defensible definitions of "digit/sign error" disagreed** and I had reported only one. Exact token-sequence equality gives 1 printed case; the harness's digit/operator *signature* gives 2, because in `pr_docA_p1_b1` the `/` moved relative to its operands even though every digit survived. Both are now reported side by side rather than one silently.

3. **A human-review note quoted CER 0.4 where the row says 0.3158.** Corrected.

## 20. Confirmations

- new_ocr_provider_requests: 8
- grading_provider_calls: 0
- local_grading_calls: 0
- ocr_verification_calls: 0
- rag_calls: 0
- held_out_calls_or_exposure: 0
- frozen_references_modified: 0
- active_grades_changed: 0
- additional_spend_usd: 0.01099875
- within_005_ceiling: True
- api_key_exposure: 0
- luna_rerun: False
- sonnet_rerun: False
- m2_strict_v1_modified: False
- replacement_cases: 0
- manual_retries: 0
- larger_ocr_stage_run: False
