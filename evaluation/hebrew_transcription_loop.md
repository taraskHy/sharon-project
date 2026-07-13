# Hebrew handwriting transcription campaign — bounded loop (2026-07-13)

Baseline frozen at commit `d7cb2ec`. Budget: ≤8 iterations / ≤6 hours
(campaign clock started 2026-07-13 ~13:35). Constraints honoured: no Stage
B/C, no grading-policy changes, no held-out exams, ground truth hidden from
every prompt, no exam images to external services, no unvetted installers.

## Benchmark

19 explanation-cell crops (2200 px source, red-masked, blue intact) from
the three sanctioned exams: e003 Q1 rows 1–8 (neat), e002 Q1 rows 1–8
(messy cursive; r1 = struck-through case, r8 = annotator-hard), rep exam
Q1-sheet rows 1–3 (multi-line; r4 dropped for crop bleed). 16 STRICT cells
scored on CER/WER/omission/hallucination/usable; 3 HARD cells scored on
flagged-not-guessed behaviour. Ground truth: my manual transcription at
full resolution with per-token confidence flags
(`evaluation/hebrew_bench/ground_truth.json`, read only by
`scripts/hebrew_bench_eval.py`, strictly post-inference).

Metric definitions: normalized CER (punct-stripped, whitespace-collapsed,
Latin lowercased, niqqud stripped, Hebrew final letters kept); omission =
word-alignment deletions / GT words; hallucinated-word rate = insertions /
hypothesis words; usable cell = CER ≤ 0.25; stability = mean pairwise CER
between repeated runs. Acceptance gate: usable ≥ 90 %, omission ≤ 5 %,
hallucination ≤ 2 %, CER ≤ 10 %, 3 materially consistent runs, unreadable
flagged not guessed.

## CAMPAIGN PAUSED — annotation-authority correction (owner, 2026-07-13)

The owner correctly ruled that the AI assistant may not be the sole
ground-truth annotator: the system under evaluation cannot annotate its own
benchmark, especially after the Q1.2 diagnostic proved this model family
produces plausible fabricated Hebrew. Consequences, all implemented:

- The AI's readings were **demoted to unverified candidates**
  (`candidate_annotations.json`; `human_verified=false` on every row of the
  annotation CSV).
- A human-annotation package was produced for the owner (paths below); the
  evaluator now **refuses to compute any metric** until
  `verified_ground_truth.json` exists and every cell carries
  `human_verified=true` (`scripts/build_verified_gt.py` builds it from the
  owner-filled CSV).
- Verified labels remain hidden from all inference; only the post-inference
  evaluator reads them.
- Iteration 1's raw model outputs (19 cells × 3 runs) are retained under
  `evaluation/hebrew_bench/outputs/it1_baseline_8b/` and will be scored once
  verified labels exist. **No metrics were computed.** The Q8_0 model
  remains cached for iteration 3. The campaign resumes from Iteration 1 on
  the fixed benchmark when the owner returns the labels; the 8-iteration
  budget is unchanged and the clock pauses with this message.

## Iteration 1 — baseline measurement

**Hypothesis:** the frozen `qwen3-vl:8b-instruct` (Q4_K_M, Ollama 0.31.2,
temp 0, json_schema), given clean high-resolution cell crops and a plain
transcription prompt, cannot transcribe this handwriting faithfully — the
Q1.2 diagnostic predicts fluent confabulation. This run quantifies the
floor every later iteration must beat.

**Variable:** none (reference configuration). Prompt `baseline` (plain
"transcribe exactly", JSON output, no unreadable-escape guidance);
preprocessing none; 3 repeats.

**Result:** (appended below when the run completes)
