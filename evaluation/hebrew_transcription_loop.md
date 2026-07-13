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

## FINAL VERDICT: **STOP** (2026-07-13, ~1.5 h into the 6 h budget)

STOP condition met: **no justified experiment remains executable.** The
owner's mid-campaign correction (annotation authority: the evaluated system
may not be its own ground-truth annotator) is binding and correct; it makes
every remaining campaign step — scoring iteration 1, iterations 2–8 —
dependent on human-verified labels that only the owner can provide, and it
explicitly pauses further benchmark iterations until then. Continuing would
violate that order; the campaign therefore closes with STOP rather than
idle. It is designed to RESUME from Iteration 1 on the fixed benchmark the
moment `annotation_template.csv` returns verified
(`scripts/build_verified_gt.py` → evaluator unlocks → retained iteration-1
outputs are scored first, at zero additional GPU cost).

### Required final report

1. **Best model/pipeline:** UNDETERMINED — no candidate was scored, because
   no metric may be computed without human-verified ground truth (owner
   rule; enforced in code — the evaluator refuses). Candidates staged and
   ready: baseline 8B (raw outputs collected), strict-fidelity prompt, Q8_0
   quant (cached, 9.8 GB), contrast preprocessing, larger model, verifier
   selection, OCR route.
2. **Metrics vs the frozen 8B baseline:** UNAVAILABLE by design;
   `evaluation/hebrew_transcription_results.csv` intentionally contains no
   rows. The only fidelity evidence on record is the pre-campaign Q1.2
   controlled diagnostic (evaluation/diag_q1_2.md): the baseline 8B
   confabulates fluent Hebrew at every resolution and fabricated content on
   a cell-only crop — strong prior evidence AGAINST the baseline meeting
   any acceptance threshold.
3. **Safe for automatic explanation grading?** **NO — not currently.** No
   pipeline has demonstrated the gate's fidelity thresholds; the standing
   production behaviour remains correct: explanations gate to zero and
   every gated item is review-flagged for a human.
4. **Exact remaining failure modes:** (a) 8B transcription confabulation
   on cursive Hebrew (proven); (b) 8B omission under multi-item load
   (proven); (c) all other candidates untested pending labels.
5. **Hardware/runtime:** RTX 2000 Ada 15.4 GB; 20-cell × 3-run sweep =
   ~8.5 min (511 s), ~2.8 s per cell call at 8K context — a full 8-iteration
   campaign fits ~1.5 GPU-hours. Q8_0 (+9.8 GB disk) cached; nothing
   downloaded beyond it; no external services touched.
6. **Integration plan (NOT merged, per instructions):** once a candidate
   passes the gate on the verified benchmark — (i) add a
   transcription-stage backend option `transcription_model` so extraction
   letters stay on the current model while explanation cells route to the
   passing model; (ii) transcriptions carry a fidelity-tier tag; only
   gate-passing tiers feed the semantic judge; others keep the current
   review-flag path; (iii) re-run the exam-003 per-item audit and the
   representative exam as the acceptance regression before any batch use.

**Campaign artifacts:** benchmark crops + manifest
(`evaluation/hebrew_bench/`), human-annotation package
(`human_annotation/`), candidate annotations (unverified), iteration-1 raw
outputs (60 calls, unscored), runner/evaluator/builder scripts with the
human-verification interlock. Constraints honoured throughout: no held-out
exams, no GT in prompts, no external uploads, no grading-policy changes, no
Stage B/C, thresholds untouched.

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
