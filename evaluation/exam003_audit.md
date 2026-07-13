# Exam-003 (variant A2) per-item audit — before/after the alignment fix

Ground truth: the owner's manual student-answer transcription (2026-07-13),
used **only post-prediction**; never model-visible. Both runs: masked pages,
anonymized model-visible path (`exam-003`), `qwen3-vl:8b-instruct`,
temperature 0, json_schema, cached repaired key.

- **Before** = Stage A run (model-derived A2 alignment: claimed identity with
  malformed ids ".1"…, passed bijection validation — factually wrong).
  Predicted total **14/100** (expected 70).
- **After** = rerun with the **operator-verified alignment**
  (`Exam_solution.alignment.json`; recorded in the result as
  `question_alignment: operator-override`). Predicted total **0/100** — see
  the decomposition: the drop is honest gating + chance-level Q3 reading,
  not a regression of the mapping layer.

## Q1 (matching; printed order = canonical, verified)

| Item | Extracted (before/after) | Manual | Agreement | Key A2 accepts | Awarded (after) | Review flag |
|---|---|---|---|---|---|---|
| 1 | F / F | F | ✓✓ | F | 0 | gated-zero: empty transcription (new flag) |
| 2 | G / G | G | ✓✓ | G | 0 | same |
| 3 | D / D | **D** (red E = instructor) | ✓✓ — the instructor's correction was correctly NOT read as the student answer | E | 0 | same (selection genuinely incorrect: D≠E) |
| 4 | A / A | A | ✓✓ | A | 0 | same |
| 5 | I / I | I | ✓✓ | I | 0 | same |
| 6 | C / C | C | ✓✓ | C | 0 | same |
| 7 | H / H | H | ✓✓ | H | 0 | same |
| 8 | B / B | B | ✓✓ | B | 0 | same |

**Q1 sheet-reading accuracy: 8/8 in BOTH runs** (reproducible on the neat
sheet; instructor red ink correctly excluded by masking+prompt). Selections
7/8 correct under the A2 key, yet awarded 0/32: the handwritten
justifications were not transcribed → judged "missing" → the
explanation-required rubric gates all credit. As of this session every such
item is review-flagged ("correct selection gated to zero on an empty
explanation transcription — verify on the scan").

## Q2 (matching; printed order = canonical, verified)

| Item | Extracted (after) | Manual | Agreement | Awarded | Review flag |
|---|---|---|---|---|---|
| 1 | C | C | ✓ | 0 | gated-zero (empty transcription) |
| 2 | E | E | ✓ | 0 | same |
| 3 | D | D | ✓ | 0 | same |
| 4 | A | A | ✓ | 0 | same |
| 5 | H | H | ✓ | 0 | same |
| 6 | G | **B** | ✗ (misread) | 0 | same |
| 7 | F | **G** | ✗ (misread — row slippage 6→7→8) | 0 | same |
| 8 | B | **F** | ✗ (misread) | 0 | same |

Q2 reading: 5/8 both runs; rows 6–8 slip consistently (handwriting).

## Q3 (multiple choice; A2 print order ≠ canonical — the fixed layer)

| Printed | → Key item (operator-verified) | Extracted (after) | Manual | Agreement | Awarded |
|---|---|---|---|---|---|
| 1 | 5 | A | D | ✗ | 0 |
| 2 | 12 | B | C | ✗ | 0 |
| 3 | 18 | C | D | ✗ | 0 |
| 4 | 1 | D | C | ✗ | 0 |
| 5 | 9 | A | B | ✗ | 0 |
| 6 | 14 | B | A | ✗ | 0 |
| 7 | 3 | C | B | ✗ | 0 |
| 8 | 20 | D | D | ✓ | 0* |
| 9 | 7 | A | C | ✗ | 0 |
| 10 | 11 | B | A | ✗ | 0 |
| 11 | 6 | C | A (C,D crossed out) | ✗ | 0 |
| 12 | 15 | D | D | ✓ | 0* |
| 13 | 2 | A | D (A crossed out) | ✗ | 0 |
| 14 | 19 | B | A (D crossed out) | ✗ | 0 |
| 15 | 8 | C | D | ✗ | 0 |
| 16 | 10 | B | B | ✓ | 0* |
| 17 | 4 | A | B | ✗ | 0 |
| 18 | 17 | B | D | ✗ | 0 |
| 19 | 13 | D | D | ✓ | 0* |
| 20 | 16 | A | D | ✗ | 0 |

\* correct reads scored 0 because the extracted letter did not match the A2
key column for the mapped item and/or the item carries the
`versions_unverified` review flag (Q3's colour-only key columns pending the
instructor's override entries).

**Q3 reading accuracy: 4/20 (before: 5/20) — chance level for 4 options,
in both runs.** The extracted sequence is a near-perfect period-4
"A,B,C,D,…" staircase — a cycling template collapse on the dense bubble
grid. The uniform-collapse tripwire has been extended to periodic patterns
and now flags exactly this signature for review. The student's visible
corrections (11: A final, C/D crossed; 13: D final, A crossed; 14: A final,
D crossed) were all missed — cross-out handling on this grid never engaged
because the rows themselves were not truly read.

## Root-cause classification (after-run)

| Cause | Items | Class |
|---|---|---|
| Explanation transcriptions skipped → rubric gate zeroes correct selections | Q1 1–8, Q2 1–5 (13 correct selections worth ≥52 pts) | handwriting transcription failure (now review-flagged per item) |
| Bubble-grid reading collapse (period-4 staircase) | Q3 all 20 | wrong answer extraction (now tripwire-flagged) |
| Handwriting letter misreads | Q2 6–8 | handwriting |
| Alignment | — | **FIXED** (operator-verified mapping applied and recorded) |
| Variant detection, masking/instructor-ink separation, key columns, page routing | — | correct in both runs (Q1.3's red E excluded ✓) |

## Verdict for the Stage B gate

The **pipeline chain is correct**: variant → sheets → alignment →
authority → scoring all behave as designed, and every failure is loudly
review-flagged. The remaining errors are **8B model perception limits** —
(a) Hebrew handwritten explanation transcription (near-total miss), (b)
dense bubble-grid row reading (chance level). More batch exams would
re-measure these same limits, so per the owner's gate **Stage B is not
launched**. Highest-value next steps, in order:

1. Bubble-grid **selection** reading via row-band crops (marks, not prose)
   — still a candidate, but must be validated against ground truth first:
   the Q1.2 diagnostic showed cropped re-asking can FABRICATE marks.
2. ~~A dedicated explanation-transcription pass (crop + transcribe)~~ —
   **WITHDRAWN** after the controlled Q1.2 diagnostic
   (evaluation/diag_q1_2.md): at every resolution the 8B confabulates
   fluent Hebrew instead of reading this cursive; forcing transcription
   replaces safe omissions with dangerous fictions.
3. The **Qwen3-VL-32B bake-off** on the university vLLM server — the
   decisive lever for both open limits; its transcriptions must pass a
   fidelity check against the owner's ground truth before being trusted.
4. Instructor completes the Q3 override entries (unblocks trusted Q3
   scoring for all variants), and decides the policy question the gate
   raises: whether correct selections with unreadable explanations earn
   provisional credit pending review, or remain 0 until verified.
