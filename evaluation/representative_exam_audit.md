# Representative-exam per-item audit (sample_data/student_exam.pdf, variant A1)

Reference truth sources: instructor red ink on the answer sheets (read
manually at 1400 px on 2026-07-13): p11 carries "28/32" with per-row ✓/✗;
p12 carries "24/32" with per-row marks and the note "answers reversed but
explanations correct … accepting"; docs/evaluation.md ground truth (version
A1; student swapped the two answer tables; X-marks-final note on the bubble
sheet; instructor Q1=24/32, Q2=28/32).

**The swap, established from the printed/handwritten evidence:** page 11's
printed title "שאלה מספר 1" has the 1 crossed out with a handwritten 2 and
the note "התבלבלתי בין שאלות 1-2"; page 12's title "2" is crossed out with a
handwritten 1. So p11 holds **Q2's** answers (instructor 28/32) and p12
holds **Q1's** (instructor 24/32).

## Run 5 (2026-07-13 ~06:20, before fixes) — TOTAL 8/100

Chain results: variant detection ✓ (four_petal_clover → A1, confident,
bottom-third); alignment ✓ (exact identity for A1); answer-sheet pages ✓
(11–13); **swap ✗ (close-read at 1000 px found no title correction)**;
X-note partially ✓ (meaning right, transcription garbage); instructor ink:
score fractions leaked into marking-conventions ✗; chunked extraction ✓ (no
uniform-answer collapse; per-row reads).

### Q1 as graded (read from p11 — which REALLY holds Q2's answers)

| Item | Extracted | On-page (my read) | Key A1 (Q1) | True content owner | Awarded | Instructor (content) | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | E | E | F | Q2.1 (E ✓ per instructor) | 0 | ✓ | wrong-key comparison (swap) |
| 2 | C | C | G | Q2.2 (C ✓) | 0 | ✓ | swap |
| 3 | G | B | D | Q2.3 (B ✓) | 0 | ✓ | swap + handwriting misread (B→G) |
| 4 | I | I | H | Q2.4 (I ✓) | 0 | ✓ | swap |
| 5 | H | D | C | Q2.5 (D ✓) | 0 | ✓ | swap + row-attribution error (took row 6's H) |
| 6 | C | H | I | Q2.6 (H ✗ true A) | 0 | ✗ | swap + row-attribution error |
| 7 | F | F | A | Q2.7 (F ✓) | 0 | ✓ | swap |
| 8 | G | G | E | Q2.8 (G ✓ — instructor tick fixed the fragmented key group to G/F/F) | 0 | ✓ | swap |

Sheet-reading accuracy on p11 (messier handwriting): 5/8 letters exact
(items 3, 5, 6 misread); explanation transcriptions: fragments only ("low
pass"/"high pass") — the long Hebrew justifications were not transcribed.

### Q2 as graded (read from p12 — which REALLY holds Q1's answers)

| Item | Extracted | On-page (my read) | Key A1 (Q2) | True content owner | Awarded | Instructor (content) | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | F | F | E | Q1.1 (F ✓) | 0 | ✓ | swap |
| 2 | G | G | C | Q1.2 (G ✓) | 0 | ✓ | swap |
| 3 | D | D | B | Q1.3 (D ✓) | 0 | ✓ | swap |
| 4 | H | H | I | Q1.4 (H ✓) | 0 | ✓ | swap |
| 5 | I | I | D | Q1.5 (true C; student I) | 0 | ✗/accepted-pair | swap; reversed-pair 5↔6 |
| 6 | C | C | A | Q1.6 (true I; student C) | 0 | ✗/accepted-pair | swap; reversed-pair 5↔6 |
| 7 | E | E | F | Q1.7 (true A; student E) | 0 | accepted via explanation | swap; reversed-pair 7↔8 |
| 8 | A | A | G→F/F (frag) | Q1.8 (true E; student A) | 0 | accepted via explanation | swap; reversed-pair 7↔8 |

Sheet-reading accuracy on p12 (neat handwriting): **8/8 letters exact**.
Explanation transcriptions: **empty** — the reversed-pair acceptances the
instructor made (red note: "the answers are reversed but the explanations
… accepting") depend on exactly the transcriptions the model skipped; the
designed `explanation_matches_different_answer` flag could not fire.

### Q3 (p13 bubble sheet, genuinely Q3; X-marks-final convention)

Extraction produced per-row varied answers (chunking removed the earlier
all-"B" collapse; rows 17–20 carry mark observations incl. X+circle), row 9
reported unanswered. 4/20 matched the A1 key → 8 pts. Instructor's Q3 total
is not visible in the ground-truth notes; per-row verification against the
scan is still pending, so extraction-vs-student accuracy on Q3 is
**unmeasured** (the key's A1 column for Q3 is itself colour-derived and
review-flagged except item 16).

### Root-cause classification (run 5)

| Cause | Items affected | Class |
|---|---|---|
| Answer-sheet-to-question assignment (undetected swap; close-read recall at 1000 px) | 16 (all of Q1+Q2) | wrong answer-sheet detection |
| Handwriting letter misreads on the messier sheet | 3 (Q1 items 3,5,6 as printed on p11) | wrong answer extraction / handwriting |
| Explanation transcription skipped | 16 potential | handwriting transcription failure |
| Instructor score fractions ("28/32") transcribed as conventions | 2 notes | instructor-annotation leakage (into context, not into grades) |
| `answer_sheet_status` echoed as not_applicable | 3 questions | extraction schema misuse (cosmetic; authority unaffected because sheet routing used the survey) |
| Key fragment 2.8 mis-ordered (F/F/G) | 1 key item | key-parsing failure (fixed by instructor-evidence override G/F/F) |
| Version detection, alignment, page routing, chunking, review flags | — | worked as designed |

### Fixes applied before run 6 (each with a regression test)

1. Close-read at ≥1400 px via a masking-aware page loader + explicit
   title-strikethrough instruction (swap class).
2. Deterministic score-fraction filter on close-read conventions.
3. Deterministic `answer_sheet_status` derived from close-read page
   conditions (worst wins) — the authority pass no longer depends on the
   model echo.
4. Override 2.8 corrected to G/F/F on instructor-tick evidence.
5. Operator overrides re-applied on every key load; excluded from the parse
   cache fingerprint (no spurious 12-minute re-parses) while still
   invalidating per-exam grading fingerprints.

## Run 6 (after fixes) — results appended below when complete.
