# Error analysis and taxonomy — 2026-07-13 session

Sources: representative exam (8 live runs; per-item audit with instructor
reference), Stage A (5 validation exams), exam-003 rerun with owner ground
truth, probe suite, key-parse ledger. Counts are of OBSERVED, confirmed
instances, not extrapolations.

## Taxonomy (ordered by measured grade impact)

| # | Failure class | Observed instances | Grade impact | Status |
|---|---|---|---|---|
| 1 | Hebrew handwritten **explanation transcription skipped** → rubric gate zeroes correct selections | 13/13 correct-selection zeros on exam-003 Q1+Q2; same signature on the representative exam and every Stage A exam (Q1+Q2 ≈ 0 across the board) | dominant: up to −64 pts/exam | Open (model capability). Every gated-zero-on-empty-transcription now review-flagged. Fix candidates: per-row transcription crops; 32B model |
| 2 | **Dense bubble-grid reading collapse** (uniform "all-B" measured on the representative exam; period-4 "A,B,C,D…" staircase on exam-003) — reads at chance level | 2 confirmed collapses; Q3 read accuracy 5/20 and 4/20 vs owner ground truth | −10…−30 pts/exam noise | Open (model capability). Both signatures now deterministically tripwire-flagged. Fix candidates: row-band crops; 32B |
| 3 | **Question-order misalignment** across variants (A2/A3 shuffle; model-derived alignment failed both as incomplete AND as complete-but-wrong-identity with malformed ids) | 2/2 derivation failures | Q3 scored against wrong key items on non-A1 exams | **Fixed**: operator-verified mappings (Exam_solution.alignment.json); derived alignments never silently trusted (unresolved_alignment → review); ids normalized; derivation chunked |
| 4 | **Answer-table swap** by the student (crossed-out titles + note) below the 8B's perception at 1000/1400 px + topic anchors | representative exam (three close-read configs missed it) | −52 pts on that exam if unflagged | Mitigated: deterministic crossed-agreement tripwire → review (fired live, 7/16 vs 1/16); human confirms |
| 5 | **Answer-key version-column decode** (colored letter groups flattened/missing across parses; fragmented text-layer group mis-ordered by the vision read) | 4 defective parses of 6; 1 fragment mis-order (caught by instructor-tick evidence) | whole-key corruption if trusted | **Fixed**: deterministic text-layer repair on every load + validation (reject, never cache); operator override file for colour-only values; Q3 columns flagged unverified until the instructor completes them |
| 6 | **Handwriting letter misreads / row slippage** on messy matching sheets | p11: 3/8 (rep exam); exam-003 Q2: 3/8 (rows 6–8) | −8…−12 pts/exam | Open (model capability); low-confidence/review machinery catches disagreement cases only partially |
| 7 | **Marker name echo** (detector perceived the flower but returned the description text, not the catalogue id) | 1 (exam-002 in the aborted first Stage A) | wrong provisional variant (flagged uncertain) | **Fixed**: deterministic description-fallback resolution (unique ≥80 % token match); live-verified 5/5 afterwards |
| 8 | **Instructor-ink artifacts**: score fractions ("28/32") transcribed as student conventions; a grader's red correction near an answer | 3 notes; Q1.3 on exam-003 correctly NOT taken (masking+prompt worked) | context noise | **Fixed**: deterministic score-fraction filter; grader-ink-only answers demoted by the authority pass |
| 9 | **Infrastructure/serving**: thinking-tag budget burn; grammar repetition loop; context overflows; cp1252 console crash; client timeout on the longest call | 6 distinct, all reproduced | run failures, no wrong grades | **Fixed** structurally (instruct tag, schema discipline, context table, UTF-8, generous timeout + cache) |

## Compensating-error check

Stage A totals under-shoot uniformly (all signed errors negative, −26…−56).
No case was found where wrong extraction produced a RIGHT total; the
representative exam's earlier 22/100 contained ~11 accidental Q3 points
from the all-"B" collapse against a B-heavy key column — that pathway is
now tripwire-flagged, and the honest re-run scores lower rather than
accidentally better. The design goal "never invent credit" holds in every
audited case; the failure direction is consistently under-scoring plus
review flags.

## What a stronger model must be tested on first (bake-off shortlist)

1. Handwritten Hebrew explanation transcription (taxonomy #1).
2. Dense bubble-grid row reading with corrections (taxonomy #2, incl. the
   crossed-out-and-rewritten cases the owner listed for exam-003 Q3
   11/13/14).
3. Crossed-out title digits / faint marginal notes (taxonomy #4).
