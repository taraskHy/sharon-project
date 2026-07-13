# Failure-boundary diagnostic — exam-002 Q1.2 explanation transcription

Item: exam-002 (variant A3), answer sheet page 11, Q1 row 2. Batch output:
selection F (correct), `explanation_transcription: null` → verdict
`missing` → rubric gate → 0/4. Owner-visible fact: the row plainly contains
a handwritten explanation.

## Answers to the staged questions

1. **Page used:** 11 (provenance recorded per item: `source_page: 11`,
   `source_region: "answer table row 2"` — row attribution was CORRECT).
2. **Detected table bounding box:** none exists — the pipeline performs no
   table/row/cell detection; extraction receives the whole masked page.
3. **Row-2 crop:** none in the pipeline (created for this diagnostic:
   `diag_q12_B_row2_band.png`, 1720×256 from a 2000 px render).
4. **Explanation-cell crop sent to the model:** none in the pipeline
   (diagnostic input C: 1560×332 from a 2600 px render).
5. **Crop after masking:** masking removes red hues only; on p11 it masked
   0.52 % of pixels (3,670 px, 9 regions — instructor ticks/notes). The
   blue explanation line is fully intact in every input (verified
   visually), and the same masked image yielded 8/8-correct blue letters.
6. **Transcription prompt:** the pipeline has no separate transcription
   call — `EXTRACTION_SYSTEM` rule 6 requests faithful transcription inside
   the per-question extraction call (batch: 8 sub-items per call). The
   diagnostic reused the same system prompt with a single sub-item.
7. **Raw model responses (captured pre-parse, temperature 0, json_schema):**
   - A (full masked page @1000 px, batch-identical): answer F ✓,
     `legibility: full`, transcription **"הפעולה הזו היא הפעולה של רכישה"**
     — fluent Hebrew that does not appear on the page.
   - B (row-2 band, 2000 px source): answer F ✓, transcription
     **"מבחן שואל על מה שנקרא Excel"** — pure fabrication.
   - C (explanation cell only, 2600 px source): transcription
     **"הפעולה הזו היא פעולה של Excel"** — fabrication; and with the letter
     column cropped away the model also INVENTED `final_answer: "E"` with a
     described X mark that does not exist.
   (True text on the page: approx. "נשארו משמעותית רק התדרים הגבוהים" — a
   correct justification for F.)
8. **Parsed transcription value:** exactly what the model emitted, verbatim
   (raw vs parsed compared byte-for-byte in the diagnostic) — the parser
   discards nothing.
9. **No-text vs lost-text:** in the BATCH call the model emitted `null`
   (omission under 8-item load); in every SINGLE-item call it emitted
   confabulated text. Nothing was lost in parsing, resume, or cache.
10. **Scoring rule:** `grade.py`, explanation-required branch: a correct
    selection with verdict `missing` takes `_verdict_factor = 0` →
    `points_selection = 0` ("the rubric awards no credit without a valid
    explanation"). Since this session, that exact situation with an EMPTY
    transcription also sets `needs_review` with reason "…may exist on the
    sheet but be untranscribed; verify on the scan".

## Candidate-cause determination

| Candidate | Verdict |
|---|---|
| wrong row coordinates | ruled out (no coordinates exist; row attribution in provenance was correct) |
| explanation crop omitted / wrong cell | N/A — the pipeline sends the whole page by design |
| masking erased blue handwriting | **ruled out** (0.52 % red-only masking; blue intact, letters read from the same image) |
| crop resolution too low | **ruled out** — a 2600 px-source cell crop still yields fabricated text |
| row-to-sub-item mapping wrong | ruled out (F attributed to row 2 correctly in all runs) |
| model saw the crop but returned no transcription | TRUE for the batch call (null under multi-item load) |
| model returned text discarded by parser | **ruled out** (raw captured; parser preserves verbatim) |
| transcription lost in resume/cache merging | ruled out (extraction.json equals the model output) |

## Failure-boundary conclusion

The earliest failing stage is **the model's reading of this cursive Hebrew
handwriting itself** — not detection, not geometry, not masking, not
resolution, not parsing, not caching. The model always *detects* the
explanation (three different framings, `legibility: full` each time) and
cannot *read* it at any resolution; under batch load it degrades to `null`,
and when forced item-by-item it **confabulates fluent Hebrew** — including,
on the cell-only crop, a fabricated answer letter and a fabricated mark.

**Category 2 therefore splits into two modes with opposite risk profiles:**

- **2a — omission (`null`)**: what the batch produces. Safe-by-accident:
  the gate withholds points and (now) flags for review. No fiction enters
  the record.
- **2b — confabulation**: what per-row/cropped re-asking produces. ACTIVELY
  DANGEROUS: invented text would be semantically judged (category 3) and
  could grant or deny credit on fiction; a cell-only crop even fabricated
  the selection.

**Correction to the earlier recommendation list:** "a dedicated
explanation-transcription pass (crop + transcribe)" is WITHDRAWN for the
8B — the diagnostic shows it converts safe omissions into unsafe
confabulations. Row-band crops remain a candidate ONLY for the
bubble-grid *selection* reading (marks, not prose). For explanations the
honest options are: (i) current behavior — gate + per-item review flag
(human reads the sheet); (ii) a stronger model (Qwen3-VL-32B bake-off) that
must PASS a transcription fidelity check against owner ground truth before
its transcriptions are trusted; (iii) an instructor policy decision to
award selection credit provisionally pending explanation review.

## Quantified gate amplification (categories 2→4), audited runs

| Run | Correct selections | Zeroed on null transcription | Points lost to the gate |
|---|---|---|---|
| exam-002 | 18 | 7 | 28 |
| exam-003 (Stage A) | 16 | 9 | 36 |
| exam-003 (rerun) | 9 | 9 (all of them) | 36 |
| exam-004 | 15 | 7 | 28 |
| exam-007 | 13 | 0 | 0 |
| exam-008 | 14 | 4 | 16 |
| representative | 6 | 0 | 0 |
| **Total** | **91** | **36** | **144** |

(Lower bound: items where a fragment like "low pass" was transcribed and
judged `invalid` lose points through the same gate but are not counted
here.) "Visibly present explanations" was manually verified for exam-002
Q1 (all 8 rows, incl. the pasted photo) and the representative exam's
sheets; other exams' cells were not individually inspected.

## Root-cause categories for the report

1. **Selection extraction error** — separate, smaller (e.g. exam-003 Q2
   rows 6–8; bubble grids are the large case, tracked separately).
2. **Explanation detection/transcription error** — THE dominant cause;
   split 2a omission / 2b confabulation as above. Q1.2 is 2a.
3. **Semantic explanation-grading error** — cannot even be assessed until
   2 is solved with a stronger reader; judging confabulations would be
   worse than judging nothing.
4. **Scoring-gate amplification** — by rubric design ("no credit without a
   valid explanation"); every gated-zero-on-empty-transcription is now
   review-flagged rather than silent. 144 points across the audited runs
   flow through this gate from cause 2a.
