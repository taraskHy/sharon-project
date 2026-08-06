# Probability-exam template (`prob-2026a-mc`) — variant mechanism and key evidence

Dataset: `prob_data/` (from `prob.zip`) — `sol.pdf` (3 pages, born-digital),
13 student scans (3 pages each, image-only), `grades.csv` (labels; never
model input). This exam family is configured **entirely separately** from the
image-processing exam (`sample_data/`): its own key, template, variant map,
alignment, caches, and evaluation outputs. Nothing is shared.

## Authoritative page rule (template-specific)

The first page of every scan is a psychometric-style response table (rows
1–10 × option columns א/ב/ג/ד with checkboxes). The sheet itself prints:

> **סמנו כאן בלבד!** אין לסמן תשובות בגוף הבחינה

("Mark HERE only! Do not mark answers in the exam body.") The template
therefore fixes `answer_sheet_pages: [1]` with
`booklet_answers_not_graded: true`: pages 2–3 (questions) may contain
circles, calculations and notes — all scratch work, never gradeable. This is
encoded in `prob_data/sol.answer_key.template.json` and applies **only to
this template**; the image-processing exam keeps its structural
sheet-detection survey.

## Variant mechanism (established by inspection, 2026-07-18)

**Indicator:** a card-suit symbol printed at the bottom of the answer-sheet
page, immediately next to "!בהצלחה". Collected across all 13 scans:

| Suit | Exams |
|---|---|
| ♡ heart | 02, 05, 06, 28 |
| ♠ spade | 13, 36 |
| ◇ diamond | 15, 21, 24 |
| ♣ club | 29, 30, 32, 37 |

**What varies:** each suit's question booklet prints the SAME 10 questions in
the SAME order, but with the four options of every question in a
suit-specific order. Verified directly:

- exams 02 and 06 (both ♡) print identical option orders on every compared
  question — same-suit consistency;
- exams 06 (♡), 13 (♠), 21 (◇), 29 (♣) print pairwise different option
  orders — cross-suit distinctness;
- question ORDER is identical everywhere (1–10, same content) — hence the
  identity question alignment (`sol.answer_key.alignment.json`).

**Correct letters per variant** were derived by matching each suit's printed
option values against the correct VALUES marked in red in `sol.pdf`
(Q1 = 0.55, Q2 = 0.427, Q3 = 1/2, Q4 = none-of-the-above, Q5 = 4/19,
Q6 = 271, Q7 = none-of-the-above, Q8 = 2/7, Q9 = 0.027, Q10 = 8.33):

| Q | sol value | ♡ heart | ♠ spade | ◇ diamond | ♣ club |
|---|---|---|---|---|---|
| 1 | 0.55 | א (A) | ב (B) | ד (D) | ג (C) |
| 2 | 0.427 | א (A) | א (A) | ג (C) | א (A) |
| 3 | 1/2 | ד (D) | ד (D) | ג (C) | ב (B) |
| 4 | אף אחד מהנ"ל | ד (D) | ד (D) | ד (D) | ד (D) |
| 5 | 4/19 | ב (B) | ד (D) | ב (B) | א (A) |
| 6 | 271 | א (A) | א (A) | ג (C) | א (A) |
| 7 | אף אחד מהנ"ל | ד (D) | ד (D) | ד (D) | ד (D) |
| 8 | 2/7 | ד (D) | ד (D) | ב (B) | ד (D) |
| 9 | 0.027 | א (A) | ג (C) | ד (D) | ב (B) |
| 10 | 8.33 | ג (C) | ב (B) | ב (B) | א (A) |

Reading basis: ♡ from exam 06 pages 2–3 (cross-checked against exam 02
page 2); ♠ from exam 13; ◇ from exam 21; ♣ from exam 29. Option orderings
were read from full-page renders at 1600 px; sol values from the born-digital
`sol.pdf`.

This is the **one-time manually verified template configuration** foreseen by
the mission brief: `sol.pdf` alone does not encode the per-variant columns
(it prints a single ordering of its own with red-marked answers), so the
columns were derived from the booklets and are shipped as the verified
structured key `prob_data/sol.answer_key.json`. `sol.pdf` is consequently
never re-parsed per student — the "parse once, cache" requirement is
satisfied by construction (zero parses per student; the one-time parse was
replaced by hand-verified structure with this documented evidence chain).

**The variant is never chosen by score.** Detection reads the suit symbol on
page 1 (a model call that sees the cover image and the marker catalogue
only); an unclear/missing symbol yields a deterministic provisional variant
plus a human-review flag (`autograder/variant.py` rules, unchanged).

## Known open items

- The four-suit collection covers all 13 available exams; future scans with
  a fifth marker (or none) will fall to the review path by design.
- Exam 02's expected grade (60) differs by one question from a manual
  pre-check of its table against the ♡ column (50) — resolved during
  evaluation by the full-resolution extraction audit (see
  `evaluation/prob/`): the per-row reading, not the key, was the suspect.
