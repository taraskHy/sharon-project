# Conditioned grading-decision analysis (post-hoc, exact counts)

Terminology: rows/baselines are REFERENCE-SIDE FIXED-GRADER DECISIONS (the
same fixed judge on owner-verified text) — not ground truth, not human
grades, not actual grading accuracy. Buckets: A preserved / B safe
abstention (-> unintelligible, routable to review) / C-E silent decision
change (substantive category changed; D upward, E downward).

## Confusion matrices (rows=reference-side, cols=OCR-side)

### Qwen (n=52)
| ref\ocr | valid | partially_valid | invalid | unintelligible |
|---|---|---|---|---|
| valid (10) | 1 | 0 | 7 | 2 |
| partially_valid (12) | 0 | 0 | 4 | 8 |
| invalid (30) | 0 | 0 | 9 | 21 |

ref=valid: exact preservation 1/10; silent harmful downgrade 7 (all ->invalid); safe abstention 2.
ref=invalid: coincidental preservation 9; dangerous upward flips 0; abstain 21.

### ML Kit (n=11)
| ref\ocr | valid | partially_valid | invalid | unintelligible |
|---|---|---|---|---|
| valid (7) | 0 | 0 | 3 | 4 |
| partially_valid (2) | 0 | 0 | 0 | 2 |
| invalid (2) | 0 | 0 | 1 | 1 |

ref=valid: exact preservation 0/7; silent harmful downgrade 3; safe abstention 4.
ref=invalid: coincidental preservation 1; upward flips 0.

### Gemini (n=18)
| ref\ocr | valid | partially_valid | invalid | unintelligible |
|---|---|---|---|---|
| valid (6) | 2 | 1 | 3 | 0 |
| partially_valid (6) | 1 | 1 | 4 | 0 |
| invalid (6) | 0 | 0 | 5 | 1 |

ref=valid: exact preservation 2/6; silent harmful downgrade 4 (1 ->partial, 3 ->invalid); safe abstention 0.
ref=partially_valid: upward 1, preserved 1, downward 4, abstain 0.
ref=invalid: coincidental preservation 5; dangerous upward flips 0; abstain 1.

## Production buckets

| Arm | n | A preserved | B abstain | Silent change (C+D+E) | D up | E down |
|---|---|---|---|---|---|---|
| Qwen | 52 | 10 | 31 | 11 | 0 | 11 |
| ML Kit | 11 | 1 | 7 | 3 | 0 | 3 |
| Gemini | 18 | 8 | 1 | 9 | 1 | 8 |

## Identical common subsets (the only fair comparisons)

| Subset | n | Arm | Preserved | Abstain | Silent |
|---|---|---|---|---|---|
| qwen vs mlkit | 10 | qwen | 1 | 4 | 5 |
| | | mlkit | 1 | 6 | 3 |
| qwen vs gemini | 17 | qwen | 2 | 10 | 5 |
| | | gemini | 8 | 1 | 8 |
| mlkit vs gemini | 6 | mlkit | 0 | 5 | 1 |
| | | gemini | 3 | 0 | 3 |

## Gemini CER x bucket (exact denominators)

| Band | n | Preserved | Abstain | Silent |
|---|---|---|---|---|
| CER<=0.25 | 2 | 2 | 0 | 0 |
| 0.25-0.50 | 9 | 4 | 1 | 4 |
| >0.50 | 7 | 2 | 0 | 5 |

(Other arms have nearly no low-CER observations: mlkit CER<=0.50 n=2 ->
1 preserved / 1 abstain; qwen none below 0.50.)
CER is reference-dependent and CANNOT serve as a deployment-time
confidence score.

## Answers

1. **Fewest silent changes**: the abstaining arms. On identical subsets
   Gemini shows MORE silent changes than qwen (8 vs 5 on n=17) and than
   mlkit (3 vs 1 on n=6) — because Gemini virtually never abstains (1/18)
   and engages every text substantively. Qwen/ML Kit fail "safer" only in
   the sense that their failures route to review.
2. **Fails by abstaining rather than misleading**: ML Kit (7/11, 64%) and
   Qwen (31/52, 60%); Gemini almost never (1/18, 6%).
3. **Fair comparisons**: only the identical-subset tables above (n=10, 17,
   6). Raw pool rates are not comparable (established pool-composition
   artifact).
4. **Underpowered**: everything Gemini — 7/20 gate coverage; low-CER band
   n=2; its cells skew to one writer (e003); the decisive production
   question — does high-quality Gemini transcription keep silent changes
   near zero at low CER — has almost no observations yet. No dangerous
   invalid->valid upward flip has been observed in any arm (0 across 81
   transitions), but that also rests on small n.

**Net production reading (supported interpretation, not causal)**: Gemini
produces fewer unintelligible transcriptions, so the current pipeline
abstains less. This exposes more plausible-but-incorrect OCR outputs to
the downstream grader. On the currently matched subsets this yields more
silent grading-decision changes than Qwen/ML Kit, but sample sizes are
too small for a final provider-level conclusion. No confidence/
calibration data exists for any arm, so no claim is made about which
provider is more or less "confident" — the observation concerns the
pipeline's abstention behavior, not model calibration. The design
implication stands: a deployable pipeline needs a reference-free
uncertainty signal to decide between automatic grading and human review.

---

# UPDATE: complete Gemini gate (20/20 predictions; grading arm n=23)

Final paired 20-item transcription comparison (identical items, hard
reported separately): strict-15 Gemini CER 0.315/0.310 vs ML Kit
0.678/0.673, pairwise 14/0/1; usable@0.25 7/15 vs 1/15; @0.50 11/15 vs
3/15. Per-writer strict means (gemini | mlkit): e002 0.24|0.88, e003
0.47|0.74, e004 0.50|0.63, e005 0.02|0.44, e006 0.17|0.47.

## Updated Gemini decision buckets (n=23)

preserved 11 | safe abstention 1 | silent change 11 (up 2, down 9).
ref=valid n=9: exact 4/9, silent downgrade 5, abstain 0.
ref=invalid n=7: coincidental invalid->invalid 6, upward flips 0.

## Hypothesis test: low CER -> reliable preservation?

| CER band | n | preserved | abstain | silent change |
|---|---|---|---|---|
| <=0.25 | 5 | **5** | 0 | **0** |
| 0.25-0.50 | 10 | 4 | 1 | 5 |
| >0.50 | 8 | 2 | 0 | 6 |

Supportive: at CER<=0.25 preservation is 5/5 with zero silent changes,
and no silent change occurs below CER 0.31 anywhere in the arm. The
gradient is monotone. n=5 in the key band — promising, not conclusive.
Writer skew: 16/23 grading cells are writer e003.

## Updated identical-subset comparisons

qwen vs gemini n=22: qwen 2 pres/12 abstain/8 silent || gemini 11/1/10.
mlkit vs gemini n=11: mlkit 1/7/3 || gemini 6/0/5.
qwen vs mlkit n=10: qwen 1/4/5 || mlkit 1/6/3.

## Candidate REFERENCE-FREE uncertainty signals (identified, NOT implemented)

1. Fixed-grader abstention itself (the judge's unintelligible verdict) —
   already exists; catches garbage, misses fluent-wrong text.
2. Cross-recognizer transcription agreement (e.g. Gemini vs local qwen,
   both available at deployment; route disagreement to review). Caveat:
   the July oracle analysis showed agreement among WEAK experts carries
   no signal — must be re-tested with one strong arm in the pair.
3. Domain-vocabulary overlap: fraction of OCR tokens found in the exam's
   printed question text/domain lexicon (deterministic, cheap).
4. Local-LM plausibility of the transcription conditioned on the question
   context (reference-free perplexity via the local model).
5. ML Kit candidate-score margin (ink arm only; scores persisted).
6. Provider-reported confidence — currently unavailable (Gemini API) or
   measured-unreliable (local qwen, Mission 1).
None are validated; validation would require correlating each signal
with decision preservation on held-out cells.

---

# Reference-free uncertainty-signal analysis (offline; frozen outputs untouched)

## Upward-flip audits (manual)

**e003_q1_r3** (cell CER 0.519; partially_valid -> valid). Reference:
"עוצמת התדרים הגבוהים אחרי הפעולה גדלה ולכן הרמות הנמוכות מראות שפות
עבות יותר". Gemini transcribed ONLY the first clause and omitted the
second — the clause containing the misstatement the judge penalized on
the reference ("thicker edges"). Mechanism: TRUNCATION REMOVED THE
STUDENT'S ERROR -> unearned upgrade.

**e006_q1_r2** (cell CER 0.333; partially_valid -> valid). Student wrote
"pc" (owner-verified); Gemini normalized it to "DC" — the correct
technical term the judge then credited (reference: "'לא ישאר pc' …
unclear"). Mechanism: ERROR-CORRECTING NORMALIZATION repaired the
student's mistake (same tendency as WAVEELETS->WAVELETS). The frozen
read2 of this cell also says "DC" — the normalization is deterministic.

Both flips are fidelity failures (omission / normalization), not
hallucinations — exactly the class an image-grounded verifier should
target.

## Signal 1 — multi-read consistency (n=6 cells; quota-limited, rest pending)

Independent second reads (config gemini3_flash_read2, identical
model/prompt/settings, originals untouched): **5/6 cells byte-identical
across reads (inter-read CER 0.000), including 2 of 3 silent-change
cells** — the truncation flip reproduced verbatim. One silent cell showed
inter-read 0.862 (catchable). Discrete two-read judge disagreement:
caught 0/3 silent, flagged 1/3 preserved. **At temperature 0 Gemini is
read-deterministic; self-consistency cannot flag deterministic errors.**
Exact counts in signal1_multiread.json.

## Signal 3 — cross-recognizer agreement with local qwen (n=22, exploratory)

Inter-hypothesis distance spans 0.68-0.96 with heavy label overlap
(preserved cells at BOTH extremes; e004's two preserved cells have the
highest distances). Catching 10/10 silent changes requires reviewing
17/22 cells (flagging 6/11 preserved). **Agreement with a recognizer
this weak adds ~no usable information** — matching the July oracle
finding. Sweep table in signal3_crossrecognizer.json.

## Recommendation (ONE next experiment)

**Signal 2 — image-to-transcription verification** deserves the larger
frozen evaluation: give a verifier the crop + Gemini's frozen
transcription + printed question context; ask whether every
grading-relevant component (numbers, operators, negation, notation,
keywords) is visibly supported AND whether any visible content was
omitted; structured verdict. Rationale: the two audited upward flips are
omission/normalization failures — precisely what image-grounded
verification targets and what self-consistency (deterministic) and
weak-recognizer agreement (measured) cannot catch. To run next quota
window over all 23 grading cells; evaluated offline against
preserved/silent labels with exact counts.
