# Post-hoc grading-decision analysis (exact counts)

Baseline throughout = the REFERENCE-SIDE FIXED-GRADER DECISION (the same
fixed judge reading the owner-verified transcription). It is not ground
truth, not a human grade. These metrics measure decision preservation
under OCR corruption — actual grading accuracy is NOT established here.

## qwen8b_strict_contrast

- eligible n=52; agreements 10; disagreements 42
- unintelligible-shifts 31; valid->invalid 7; non-abstain downgrades 11; upward 0
- reference-side verdict pool: {'valid': 10, 'invalid': 30, 'partially_valid': 12}
- CER<=0.25: 0/0
- 0.25<CER<=0.50: 0/0
- CER>0.50: 10/52

| cell | CER | ref->ocr | preserved |
|---|---|---|---|
| e002_q1_r7 | 0.60 | invalid->invalid | yes |
| e004_q1_r4 | 0.65 | valid->valid | yes |
| e005_q1_r3 | 0.69 | invalid->unintelligible | no |
| e006_q1_r3 | 0.69 | invalid->unintelligible | no |
| e003_q1_r6 | 0.69 | partially_valid->unintelligible | no |
| e003_q2_r3 | 0.71 | valid->invalid | no |
| e006_q2_r8 | 0.71 | partially_valid->invalid | no |
| e002_q1_r3 | 0.71 | valid->invalid | no |
| e002_q1_r6 | 0.71 | partially_valid->unintelligible | no |
| e002_q1_r4 | 0.72 | valid->unintelligible | no |
| e003_q2_r4 | 0.74 | invalid->unintelligible | no |
| e006_q2_r6 | 0.74 | invalid->unintelligible | no |
| e002_q1_r5 | 0.75 | invalid->unintelligible | no |
| e002_q2_r3 | 0.75 | invalid->unintelligible | no |
| e003_q2_r7 | 0.75 | invalid->invalid | yes |
| e006_q1_r7 | 0.75 | invalid->unintelligible | no |
| e002_q1_r2 | 0.76 | valid->invalid | no |
| e005_q1_r8 | 0.76 | partially_valid->unintelligible | no |
| e005_q1_r6 | 0.76 | invalid->unintelligible | no |
| e006_q2_r7 | 0.76 | invalid->invalid | yes |
| e003_q2_r5 | 0.76 | invalid->unintelligible | no |
| e004_q2_r5 | 0.78 | invalid->invalid | yes |
| e003_q1_r3 | 0.78 | partially_valid->unintelligible | no |
| e003_q1_r8 | 0.78 | invalid->unintelligible | no |
| e005_q1_r5 | 0.78 | invalid->unintelligible | no |
| e003_q1_r5 | 0.79 | partially_valid->unintelligible | no |
| e003_q1_r7 | 0.79 | partially_valid->unintelligible | no |
| e002_q1_r8 | 0.79 | invalid->invalid | yes |
| e005_q2_r5 | 0.79 | invalid->unintelligible | no |
| e003_q1_r4 | 0.80 | valid->unintelligible | no |
| e004_q2_r3 | 0.80 | invalid->invalid | yes |
| e002_q2_r8 | 0.80 | valid->invalid | no |
| e005_q2_r6 | 0.81 | invalid->invalid | yes |
| e004_q1_r8 | 0.81 | invalid->unintelligible | no |
| e004_q2_r8 | 0.82 | invalid->unintelligible | no |
| e003_q1_r1 | 0.82 | valid->invalid | no |
| e006_q1_r6 | 0.83 | partially_valid->unintelligible | no |
| e003_q2_r2 | 0.83 | partially_valid->invalid | no |
| e006_q1_r5 | 0.83 | invalid->unintelligible | no |
| e003_q2_r1 | 0.85 | invalid->unintelligible | no |
| e003_q2_r6 | 0.86 | partially_valid->unintelligible | no |
| e004_q2_r7 | 0.86 | invalid->unintelligible | no |
| e005_q2_r3 | 0.88 | invalid->unintelligible | no |
| e004_q1_r5 | 0.89 | invalid->unintelligible | no |
| e004_q1_r1 | 0.90 | invalid->invalid | yes |
| e004_q2_r4 | 0.91 | invalid->invalid | yes |
| e005_q2_r8 | 0.95 | invalid->unintelligible | no |
| e002_q2_r2 | 1.00 | partially_valid->invalid | no |
| e003_q2_r8 | 1.10 | valid->invalid | no |
| e004_q1_r2 | 1.17 | valid->invalid | no |
| e006_q1_r2 | 1.25 | partially_valid->invalid | no |
| e004_q1_r6 | 1.26 | invalid->unintelligible | no |

## mlkit_ink_rtl_a1

- eligible n=11; agreements 1; disagreements 10
- unintelligible-shifts 7; valid->invalid 3; non-abstain downgrades 3; upward 0
- reference-side verdict pool: {'valid': 7, 'partially_valid': 2, 'invalid': 2}
- CER<=0.25: 1/1
- 0.25<CER<=0.50: 0/1
- CER>0.50: 0/9

| cell | CER | ref->ocr | preserved |
|---|---|---|---|
| e006_q1_r3 | 0.23 | invalid->invalid | yes |
| e002_q1_r2 | 0.48 | valid->unintelligible | no |
| e002_q1_r4 | 0.62 | valid->invalid | no |
| e004_q1_r2 | 0.66 | valid->invalid | no |
| e002_q1_r3 | 0.67 | valid->invalid | no |
| e004_q1_r1 | 0.69 | invalid->unintelligible | no |
| e003_q1_r4 | 0.70 | valid->unintelligible | no |
| e006_q1_r2 | 0.71 | partially_valid->unintelligible | no |
| e003_q1_r3 | 0.71 | partially_valid->unintelligible | no |
| e003_q1_r1 | 0.75 | valid->unintelligible | no |
| e003_q1_r2 | 0.80 | valid->unintelligible | no |

## gemini3_flash

- eligible n=18; agreements 8; disagreements 10
- unintelligible-shifts 1; valid->invalid 3; non-abstain downgrades 8; upward 1
- reference-side verdict pool: {'valid': 6, 'partially_valid': 6, 'invalid': 6}
- CER<=0.25: 2/2
- 0.25<CER<=0.50: 4/9
- CER>0.50: 2/7

| cell | CER | ref->ocr | preserved |
|---|---|---|---|
| e003_q2_r4 | 0.00 | invalid->invalid | yes |
| e004_q1_r2 | 0.06 | valid->valid | yes |
| e003_q1_r6 | 0.31 | partially_valid->invalid | no |
| e004_q1_r1 | 0.31 | invalid->invalid | yes |
| e003_q1_r1 | 0.34 | valid->partially_valid | no |
| e003_q1_r7 | 0.36 | partially_valid->invalid | no |
| e003_q2_r1 | 0.43 | invalid->invalid | yes |
| e003_q2_r7 | 0.45 | invalid->invalid | yes |
| e003_q1_r8 | 0.46 | invalid->unintelligible | no |
| e003_q1_r4 | 0.47 | valid->valid | yes |
| e003_q2_r8 | 0.47 | valid->invalid | no |
| e003_q2_r3 | 0.52 | valid->invalid | no |
| e003_q1_r3 | 0.52 | partially_valid->valid | no |
| e003_q2_r2 | 0.54 | partially_valid->invalid | no |
| e003_q1_r2 | 0.57 | valid->invalid | no |
| e003_q2_r5 | 0.61 | invalid->invalid | yes |
| e003_q2_r6 | 0.65 | partially_valid->partially_valid | yes |
| e003_q1_r5 | 1.02 | partially_valid->invalid | no |

## Qwen-vs-ML-Kit paradox resolution

On the IDENTICAL 10 common cells: qwen preserves 1/10, ml kit 1/10 — no difference.
The headline 19.2% vs 9.1% gap is a POOL-COMPOSITION artifact: qwen's
52-cell pool is 58% reference-side-invalid (30/52), and 8 of its 10
agreements are invalid->invalid coincidences (garbage text also judged
invalid); ML Kit's 11-cell pool is 64% reference-side-valid, where its
fluent-but-wrong Hebrew is judged 'irrelevant'/invalid — silent
downgrades of reference-valid answers. Qwen shows the same silent-
downgrade mode when its confabulation is fluent (e.g. e003_q2_r8's
education-system text judged 'completely irrelevant'). Qualitative,
small-n diagnostic — not causal proof.

## Deployment note

CER cannot serve as a deployment confidence signal (the reference is
unavailable at deployment). Any automatic-review gate needs a
REFERENCE-FREE uncertainty signal correlated with decision preservation;
none of the tested arms provides a validated one today (ML Kit has no
abstention at all; the VLM's self-reported confidence was measured
unreliable in Mission 1).