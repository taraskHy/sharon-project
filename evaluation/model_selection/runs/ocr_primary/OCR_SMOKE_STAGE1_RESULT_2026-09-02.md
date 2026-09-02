# OCR Stage-1 smoke - EXECUTED (2026-09-02)

git `f7c2f602c53e`; campaign `8ce4f5eea7b233c5...`; smoke selection `cc5c9f1ff9911a68...`; schema `003ee19969c8c443...`; prompt `m2-strict-v1`; adapter `ocr-primary-bench-v1`.

**OCR provider requests: 24 / 24 authorized. Additional spend: $0.0267926 (ceiling $0.10). Grading calls: 0. HELD_OUT: 0.**

## Headline

| Model | Exact | Mean CER | Median CER | Mean WER | Critical errors | Line loss | Failures | Mean latency | Cost |
|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 0/8 | 0.7715 | 0.8717 | 0.8739 | 7 cases / 1 need human review | 0/8 | 0 | 2.719s | $0.0071066 |
| `google/gemini-3.7-flash` | 0/8 | n/a | n/a | n/a | 8 cases / 0 need human review | 8/8 | 8 | n/a | $0.0000000 |
| `anthropic/claude-sonnet-5` | 0/8 | 0.4878 | 0.4404 | 0.7549 | 5 cases / 3 need human review | 0/8 | 0 | 4.383s | $0.0196860 |

## Verdicts

- **`openai/gpt-5.6-luna-pro` -> DROP** - mean CER 0.7715 / mean WER 0.8739 on 8 cases. It refused 4 of 5 handwritten crops outright (unreadable markers against fully readable audited references) and on the fifth produced a fluent, wholly fabricated line. A transcriber that either refuses handwriting or invents it is unusable for this pipeline regardless of its $0.00089/crop price. Printed text was fine (mixed_he_en CER 0.0), but handwriting is the reason this role exists.
- **`google/gemini-3.7-flash` -> MAYBE** - NO OCR EVIDENCE EXISTS. All 8 requests failed HTTP 400 'Reasoning is mandatory for this endpoint and cannot be disabled' - the frozen route pins reasoning.effort='none'. This is a route/provider configuration incompatibility, NOT a measured OCR quality result. $0.00 billed. Neither advanced nor dropped on evidence; it needs a re-pre-registered arm with a reasoning-enabled decoding config, which was NOT run here because replacement calls are outside this authorization.
- **`anthropic/claude-sonnet-5` -> MAYBE** - Best measured arm: mean CER 0.4878 vs Luna 0.7715, and the only model that attempts handwriting rather than refusing (2 of 5 refused vs Luna's 4 of 5). Printed/mixed content is strong (CER 0.10 / 0.00) and per-writer e003 CER is 0.2131. But handwritten-cell mean CER is 0.7202 with 3 digit/sign errors, and the proposed production gate is per-writer CER <= 5% - writer e002 sits at 0.7202, over an order of magnitude away. It earns a larger seen-only OCR benchmark; it does NOT demonstrate production OCR readiness, and 8 samples cannot show that.

## Two findings that change how the numbers read

1. **`assoc_docB_p2_b1` (CER 0.7083) is a reference artifact, not a model error.** NOT a model error. The frozen reference stores the PDF text layer's raw line order; read bottom-up (visual RTL) the associations are alef=0.39, bet=0.47, gimel=0.51, dalet=0.55. BOTH models emitted exactly that association. The 0.7083 CER and the DIGIT_CHANGED flag are pure serialization-order artifacts of the reference. RECORDED ONLY - the audited reference was NOT modified (forbidden by the campaign freeze and this task). Both models' mean CER is inflated by this one case; option_row_association CER is not a usable Stage-1 signal.

2. **Gemini produced no OCR evidence at all.** All 8 requests returned HTTP 400 `Reasoning is mandatory for this endpoint and cannot be disabled`, because the frozen route pins `reasoning.effort='none'`. Nothing was billed. Re-running it with a changed decoding config would have been a replacement call, which this authorization forbids, so the arm stands unmeasured.

## Per-writer CER / WER (committed `ocr_writer_metrics`)

| Model | Writer | Cases | Mean CER | Mean WER | Line loss | Digit/sign errors |
|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | e002 | 4 | 0.9358 | 1.0 | 0 | 3 |
| `openai/gpt-5.6-luna-pro` | e003 | 1 | 1.0 | 1.0 | 0 | 0 |
| `openai/gpt-5.6-luna-pro` | no_writer | 3 | 0.4761 | 0.6638 | 0 | 2 |
| `google/gemini-3.7-flash` | e002 | 4 | n/a | n/a | 4 | 0 |
| `google/gemini-3.7-flash` | e003 | 1 | n/a | n/a | 1 | 0 |
| `google/gemini-3.7-flash` | no_writer | 3 | n/a | n/a | 3 | 0 |
| `anthropic/claude-sonnet-5` | e002 | 4 | 0.7202 | 0.9773 | 0 | 3 |
| `anthropic/claude-sonnet-5` | e003 | 1 | 0.2131 | 0.6 | 0 | 0 |
| `anthropic/claude-sonnet-5` | no_writer | 3 | 0.2694 | 0.51 | 0 | 2 |

## Every observation, individually

### `openai/gpt-5.6-luna-pro`

| Case | Writer | CER | WER | Omit | Halluc | Critical | Latency | In/Out tok |
|---|---|---|---|---|---|---|---|---|
| `hl_e003_q1_r1__l1` | e003 | 1.0 | 1.0 | 1.0 | 0.0 | UNREADABLE_OUTPUT_ON_READABLE_REFERENCE | 3.265 | 3827/113 |
| `hc_e002_q1_r1` | e002 | 1.0 | 1.0 | 0.75 | 0.0 | LATIN_TOKEN_CHANGED(ref=[],ocr=['unreadable']); UNREADABLE_OUTPUT_ON_READABLE_REFERENCE | 2.547 | 3949/72 |
| `hc_e002_q1_r7` | e002 | 1.0 | 1.0 | 0.9167 | 0.0 | DIGIT_CHANGED(ref=0,ocr=-); LATIN_TOKEN_CHANGED(ref=[],ocr=['unreadable']); UNREADABLE_OUTPUT_ON_READABLE_REFERENCE | 2.891 | 3961/84 |
| `hc_e002_q2_r1` | e002 | 1.0 | 1.0 | 0.9091 | 0.0 | DIGIT_CHANGED(ref=0255,ocr=-); LATIN_TOKEN_CHANGED(ref=[],ocr=['unreadable']); UNREADABLE_OUTPUT_ON_READABLE_REFERENCE | 2.594 | 3978/101 |
| `hc_e002_q2_r6` | e002 | 0.7434 | 1.0 | 0.1818 | 0.0 | DIGIT_CHANGED(ref=2,ocr=-); SIGN_OPERATOR_CHANGED(ref=-,ocr=-); LATIN_TOKEN_CHANGED(ref=['echo'],ocr=[]) | 2.672 | 4048/171 |
| `pr_docA_p1_b1` | no_writer | 0.72 | 0.7692 | 0.3846 | 0.3846 | DIGIT_CHANGED(ref=2033730203673020252026,ocr=2036730203373020252026) | 2.437 | 3401/147 |
| `pr_docA_p2_b3` | no_writer | 0.0 | 0.0 | 0.0 | 0.0 | - | 2.468 | 3509/164 |
| `assoc_docB_p2_b1` | no_writer | 0.7083 | 1.2222 | 0.0 | 0.25 | DIGIT_CHANGED(ref=055051047039,ocr=039047051055) | 2.875 | 2920/138 |

Reference vs OCR text:

- **`hl_e003_q1_r1__l1`**
  - REF: `ניתן לראות ברמות התדרים הגבוהים שיש שפות בכיוון התנועה(הטשטוש)`
  - OCR: `[?]`
- **`hc_e002_q1_r1`**
  - REF: `יש טשטוש בכל התדרים`
  - OCR: `[unreadable]`
- **`hc_e002_q1_r7`**
  - REF: `סה"כ הפירמידה נראית תקינה. הדרגה 0 שלה בהירה משמעותית מהתמונה המקורית`
  - OCR: `[unreadable]`
- **`hc_e002_q2_r1`**
  - REF: `עבור גילוי שפות יהיה רוב התמונה ב0 ורק עבור שפות 255`
  - OCR: `[unreadable]`
- **`hc_e002_q2_r6`**
  - REF: `המעכה זהה לסכימת תמונה זהה עם תנודה קלה. מין תמונת echo כזאת תגרום להיסטוגרמה החדשה להיות קרובה להכפלה ב-2 [לא קריא]`
  - OCR: `מבחינה למעשה להלכה יש להבחין בין שני סוגים של מצוות, מצוות שבין אדם לחבירו ומצוות שבין אדם למקום`
- **`pr_docA_p1_b1`**
  - REF: `203.3730
 203.6730 / 
  סמסטר א' מועד א' תשפ"ו 
2025-2026`
  - OCR: `סמסטר א' מועד א' תשפ"ו 203.6730 / 203.3730
2025-2026`
- **`pr_docA_p2_b3`**
  - REF: `לאחר הפעולה בונים פרמידת WAVELETS של תמונת התוצאה. בעמוד הבא נתונות 
תמונות התוצאה ממוספרות  )
A-I
(.`
  - OCR: `לאחר הפעולה בונים פרמידת WAVELETS של תמונת התוצאה. בעמוד הבא נתונות תמונות התוצאה ממוספרות (A-I).`
- **`assoc_docB_p2_b1`**
  - REF: `0.55
()ד0.51
()ג0.47
()ב0.39
()א`
  - OCR: `א: 0.39; ב: 0.47; ג: 0.51; ד: 0.55`

### `google/gemini-3.7-flash`

| Case | Writer | CER | WER | Omit | Halluc | Critical | Latency | In/Out tok |
|---|---|---|---|---|---|---|---|---|
| `hl_e003_q1_r1__l1` | e003 | n/a | n/a | n/a | n/a | LINE_LOST_NO_OUTPUT | n/a | n/a/n/a |
| `hc_e002_q1_r1` | e002 | n/a | n/a | n/a | n/a | LINE_LOST_NO_OUTPUT | n/a | n/a/n/a |
| `hc_e002_q1_r7` | e002 | n/a | n/a | n/a | n/a | LINE_LOST_NO_OUTPUT | n/a | n/a/n/a |
| `hc_e002_q2_r1` | e002 | n/a | n/a | n/a | n/a | LINE_LOST_NO_OUTPUT | n/a | n/a/n/a |
| `hc_e002_q2_r6` | e002 | n/a | n/a | n/a | n/a | LINE_LOST_NO_OUTPUT | n/a | n/a/n/a |
| `pr_docA_p1_b1` | no_writer | n/a | n/a | n/a | n/a | LINE_LOST_NO_OUTPUT | n/a | n/a/n/a |
| `pr_docA_p2_b3` | no_writer | n/a | n/a | n/a | n/a | LINE_LOST_NO_OUTPUT | n/a | n/a/n/a |
| `assoc_docB_p2_b1` | no_writer | n/a | n/a | n/a | n/a | LINE_LOST_NO_OUTPUT | n/a | n/a/n/a |

Reference vs OCR text:

- **`hl_e003_q1_r1__l1`**
  - REF: `ניתן לראות ברמות התדרים הגבוהים שיש שפות בכיוון התנועה(הטשטוש)`
  - OCR: *NO OUTPUT* - `HTTP 400 from backend: {"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400`
- **`hc_e002_q1_r1`**
  - REF: `יש טשטוש בכל התדרים`
  - OCR: *NO OUTPUT* - `HTTP 400 from backend: {"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400`
- **`hc_e002_q1_r7`**
  - REF: `סה"כ הפירמידה נראית תקינה. הדרגה 0 שלה בהירה משמעותית מהתמונה המקורית`
  - OCR: *NO OUTPUT* - `HTTP 400 from backend: {"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400`
- **`hc_e002_q2_r1`**
  - REF: `עבור גילוי שפות יהיה רוב התמונה ב0 ורק עבור שפות 255`
  - OCR: *NO OUTPUT* - `HTTP 400 from backend: {"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400`
- **`hc_e002_q2_r6`**
  - REF: `המעכה זהה לסכימת תמונה זהה עם תנודה קלה. מין תמונת echo כזאת תגרום להיסטוגרמה החדשה להיות קרובה להכפלה ב-2 [לא קריא]`
  - OCR: *NO OUTPUT* - `HTTP 400 from backend: {"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400`
- **`pr_docA_p1_b1`**
  - REF: `203.3730
 203.6730 / 
  סמסטר א' מועד א' תשפ"ו 
2025-2026`
  - OCR: *NO OUTPUT* - `HTTP 400 from backend: {"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400`
- **`pr_docA_p2_b3`**
  - REF: `לאחר הפעולה בונים פרמידת WAVELETS של תמונת התוצאה. בעמוד הבא נתונות 
תמונות התוצאה ממוספרות  )
A-I
(.`
  - OCR: *NO OUTPUT* - `HTTP 400 from backend: {"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400`
- **`assoc_docB_p2_b1`**
  - REF: `0.55
()ד0.51
()ג0.47
()ב0.39
()א`
  - OCR: *NO OUTPUT* - `HTTP 400 from backend: {"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.","code":400`

### `anthropic/claude-sonnet-5`

| Case | Writer | CER | WER | Omit | Halluc | Critical | Latency | In/Out tok |
|---|---|---|---|---|---|---|---|---|
| `hl_e003_q1_r1__l1` | e003 | 0.2131 | 0.6 | 0.0 | 0.0 | - | 5.016 | 1002/51 |
| `hc_e002_q1_r1` | e002 | 1.0 | 1.0 | 0.75 | 0.0 | LATIN_TOKEN_CHANGED(ref=[],ocr=['unreadable']); UNREADABLE_OUTPUT_ON_READABLE_REFERENCE | 3.094 | 1098/16 |
| `hc_e002_q1_r7` | e002 | 0.4559 | 1.0 | 0.0833 | 0.0 | DIGIT_CHANGED(ref=0,ocr=-); SIGN_OPERATOR_CHANGED(ref=-,ocr=-) | 6.062 | 1098/56 |
| `hc_e002_q2_r1` | e002 | 1.0 | 1.0 | 0.9091 | 0.0 | DIGIT_CHANGED(ref=0255,ocr=-); LATIN_TOKEN_CHANGED(ref=[],ocr=['unreadable']); UNREADABLE_OUTPUT_ON_READABLE_REFERENCE | 4.375 | 1098/16 |
| `hc_e002_q2_r6` | e002 | 0.4248 | 0.9091 | 0.0909 | 0.0 | DIGIT_CHANGED(ref=2,ocr=-); SIGN_OPERATOR_CHANGED(ref=-,ocr=-); LATIN_TOKEN_CHANGED(ref=['echo'],ocr=[]) | 4.828 | 1098/82 |
| `pr_docA_p1_b1` | no_writer | 0.1 | 0.3077 | 0.1538 | 0.0833 | - | 4.281 | 891/48 |
| `pr_docA_p2_b3` | no_writer | 0.0 | 0.0 | 0.0 | 0.0 | - | 4.656 | 939/76 |
| `assoc_docB_p2_b1` | no_writer | 0.7083 | 1.2222 | 0.0 | 0.25 | DIGIT_CHANGED(ref=055051047039,ocr=039047051055) | 2.75 | 694/40 |

Reference vs OCR text:

- **`hl_e003_q1_r1__l1`**
  - REF: `ניתן לראות ברמות התדרים הגבוהים שיש שפות בכיוון התנועה(הטשטוש)`
  - OCR: `ניתן לראות בראות התפרים הגבוהים שיש שפת גבול הנתונה (הלטוש)`
- **`hc_e002_q1_r1`**
  - REF: `יש טשטוש בכל התדרים`
  - OCR: `[unreadable]`
- **`hc_e002_q1_r7`**
  - REF: `סה"כ הפירמידה נראית תקינה. הדרגה 0 שלה בהירה משמעותית מהתמונה המקורית`
  - OCR: `סדר הנוישקה נחוית תקנה - הפרטס על ה בהיזה משאזרה מותמיכת המקומית`
- **`hc_e002_q2_r1`**
  - REF: `עבור גילוי שפות יהיה רוב התמונה ב0 ורק עבור שפות 255`
  - OCR: `[unreadable]`
- **`hc_e002_q2_r6`**
  - REF: `המעכה זהה לסכימת תמונה זהה עם תנודה קלה. מין תמונת echo כזאת תגרום להיסטוגרמה החדשה להיות קרובה להכפלה ב-2 [לא קריא]`
  - OCR: `ההסכה להה לסכית התנות זהה על תשונה קלה, אין השאך סתם כאחר, התרוה להסולתמג הישר לחות קרונה להכבדה כל מעי [?]`
- **`pr_docA_p1_b1`**
  - REF: `203.3730
 203.6730 / 
  סמסטר א' מועד א' תשפ"ו 
2025-2026`
  - OCR: `203.3730 / 203.6730 סמסטר א' מועד א' תשפ״ו
2025-2026`
- **`pr_docA_p2_b3`**
  - REF: `לאחר הפעולה בונים פרמידת WAVELETS של תמונת התוצאה. בעמוד הבא נתונות 
תמונות התוצאה ממוספרות  )
A-I
(.`
  - OCR: `לאחר הפעולה בונים פרמידת WAVELETS של תמונת התוצאה. בעמוד הבא נתונות תמונות התוצאה ממוספרות (A-I).`
- **`assoc_docB_p2_b1`**
  - REF: `0.55
()ד0.51
()ג0.47
()ב0.39
()א`
  - OCR: `א: 0.39; ב: 0.47; ג: 0.51; ד: 0.55`

## Aggregate behaviour

| Model | Unreadable-marker rate | Mean omission | Mean hallucination | Digit/sign errors | Schema failures | Provider failures | Cache hits | p95 latency | Total tokens |
|---|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | 0.5 | 0.5178 | 0.0793 | 5 | 0 | 0 | 0 | 3.265 | 30583 |
| `google/gemini-3.7-flash` | 0.0 | n/a | n/a | 0 | 0 | 8 | 0 | n/a | 0 |
| `anthropic/claude-sonnet-5` | 0.375 | 0.2484 | 0.0417 | 5 | 0 | 0 | 0 | 6.062 | 8303 |

## Measured cost projections

Measured cost per billable crop, multiplied out. Assumes the same crop mix, the same frozen prompt, one pass, no retries, no cache hits and no verification pass. Gemini has no measured rate because nothing was billed.

| Model | $/crop | 32 seen DEV | 21 seen CALIB | 53 all seen | 100 crops | 100 exams @5 | @10 | @15 |
|---|---|---|---|---|---|---|---|---|
| `openai/gpt-5.6-luna-pro` | $0.00088833 | $0.0284 | $0.0187 | $0.0471 | $0.0888 | $0.4442 | $0.8883 | $1.3325 |
| `google/gemini-3.7-flash` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `anthropic/claude-sonnet-5` | $0.00246075 | $0.0787 | $0.0517 | $0.1304 | $0.2461 | $1.2304 | $2.4607 | $3.6911 |

Grading cost stays separate and is **$0**: cloud grading $0, local-grading cloud cost $0. No grading model ran.

## Accounting reconciliation

- ledger rows 662 -> 686 (+24, one per request)
- ledger cumulative $0.47169044 -> $0.49848304
- run-attributed cost **$0.0267926**
- OpenRouter account usage $0.47169007 -> $0.49848267 (delta **$0.0267926**)
- **rounding difference ledger vs account: 0.0** - exact match
- billable responses 16, non-billable failures 8 (the 8 Gemini HTTP 400s), cache hits 0
- cache entries 541 -> 557 (never cleared)

Every billable response has a ledger row; every ledger row maps to one authorized OCR request.

## Recommended next experiment (NOT executed)

Re-pre-register a Stage-1b arm for **`google/gemini-3.7-flash` only**, identical 8 frozen cases, changing exactly one thing: a decoding config with reasoning enabled (the minimum the endpoint accepts) instead of `reasoning.effort='none'`. That arm answers the only open question the failure left. It needs its own pre-registration because the decoding config is part of the frozen run identity, and its cost must be re-predicted (reasoning tokens are billed as output). Only after that should any larger seen-only OCR benchmark be considered, and on this evidence that larger stage would be `anthropic/claude-sonnet-5` over the 32 seen DEV crops (projected $0.0787), never HELD_OUT.

## Confirmations

- ocr_provider_requests: 24
- grading_provider_calls: 0
- local_grading_calls: 0
- rag_calls: 0
- held_out_calls_or_exposure: 0
- audited_references_modified: 0
- active_grades_changed: 0
- additional_spend_usd: 0.0267926
- api_key_exposure: 0
- ocr_prompt_modified: False
- population_expanded: False
- replacement_calls: 0
