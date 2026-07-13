# Hebrew handwriting transcription campaign — bounded loop (2026-07-13)

## CAMPAIGN CLOSED — FINAL VERDICT: **STOP** (all 8 iterations completed)

STOP conditions met simultaneously: **8 iterations completed**; **no
meaningful improvement occurring** (best CER stuck at 0.786 across model
family, scale, quantization, prompt, preprocessing, dedicated HTR, and
multilingual OCR variables); **repeated hallucination makes the tested
approaches unsuitable** (85/85 hard-cell evaluations across all candidates
produced confabulated text; zero honest unreadable flags).

### Required final report

1. **Best pipeline:** `qwen3-vl:8b-instruct` (Q4_K_M) + strict-fidelity
   prompt + percentile contrast stretch (iteration 4) — mean CER 0.786,
   usable 0 %, halluc-word 8.7 %, omission 26.3 %, stability 0.071.
2. **Metrics vs the frozen baseline:** baseline (it1) CER 0.936 / halluc
   20.3 % / stability 0.147 → best (it4) improves every axis (CER −16 %
   relative, halluc −57 %, stability 2×) yet remains an order of magnitude
   from every acceptance threshold. Full per-iteration ledgers:
   `hebrew_transcription_results.csv` (it1–5) and
   `local_hebrew_htr_results.csv` (it6–7); head-to-head and failure-class
   verdict in `local_hebrew_htr_benchmark.md` (iteration 8).
3. **Safe for automatic explanation grading? NO.** No candidate approaches
   the gate (usable ≥90 % / omission ≤5 % / halluc ≤2 % / CER ≤10 %), and
   none ever flags unreadable text instead of guessing. The production
   policy stands: explanation credit gates to zero with per-item human
   review — grading policy was not modified.
4. **Exact remaining failure modes:** (a) fluent Hebrew confabulation by
   VLMs at every scale tested; (b) no self-detection of unreadability;
   (c) dedicated public HTR models misread modern cursive (wrong training
   distribution) or are unrunnable without approvals (HebHTR: WSL2/Docker +
   no license; ABBA-HTR: pickle-only; Surya 0.21: Docker-gated);
   (d) projection segmentation collapses on dense/messy cells.
5. **Hardware/runtime:** all candidates ran locally. Qwen ~3 s/cell (GPU,
   8K ctx); hdd-words ~2.3 s/cell CPU; Surya ~9.9 s/cell CPU; 30B MoE
   ~15 s/call at 41/59 CPU/GPU. Full campaign consumed ≈ 2 GPU-hours +
   ≈ 0.7 CPU-hours of inference across 8 iterations.
6. **Integration plan (NOT merged; requires owner approval):** none of the
   tested models may feed the semantic judge. The evidence-backed path is a
   **domain fine-tune**: kraken (first choice) or PyLaia on ~800–2,000
   owner-verified explanation lines harvested from the 41 development exams
   via the existing annotation workflow (≈1,000 lines available); training
   fits the local GPU in hours. HebHTR's 4.76 % self-reported CER on this
   exact domain is the feasibility proof. Alternative pre-trained lever if
   approved: HebHTR itself inside WSL2/Docker (license + security caveats
   documented), or the university-server 32B bake-off for the general
   pipeline (separate from transcription fidelity).

**Verdict: STOP.**

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

## CAMPAIGN RESUMED — owner labels received (2026-07-13, later the same day)

The STOP verdict below is **superseded**: the owner supplied human-verified
transcriptions for all 16 explanation cells of exam 002 (Q1 rows 1–8 and Q2
rows 1–8, the latter requiring 8 new crops from page 12). Mapping was
verified before attachment (Q1 by printed row numbers; Q2 by row position +
content anchors: "255" in r1, "בדומה לו" r2, "פריסה אחידה" r3,
"צורת ההיסטוגרמה זהה" r7, "x→2x" r8) and reported: 8 existing crops matched
Q1, 8 new crops created for Q2, no orphan entries; e003 (8) and rep (4)
crops remain unverified/excluded. Owner text copied exactly; `[לא קריא]`
preserved; any cell containing it is typed HARD (flag-expected) — strict set
= Q1 r1–r8 + Q2 r2, r3, r8 (11 cells), hard set = Q2 r1, r4, r5, r6, r7 (5).
Labels live only in `verified_ground_truth.json`, read exclusively by the
post-inference evaluator.

Empirical vindication of the annotation-authority ruling: the owner's
labels corrected ≥1 word in 4 of my 8 candidate readings for Q1
(בתמונה≠משמעותית, הדרגות≠הצבעים, הפירמידה/הדרגה 0≠התמונה/בגרסה,
האיזורים≠הרזולוציות).

The bounded loop resumes from Iteration 1 (retained outputs + the 8 new Q2
cells collected incrementally) on this fixed verified benchmark; budget
unchanged (~4.2 h remain of 6 h; iterations used: 1 of 8 after Iteration 1
scores).

## Decision-loop record (superseded by the resume above; retained for audit)

Per the campaign spec, each cycle must end in CONTINUE, ACCEPT, or STOP.
After iteration 1's data collection, each option was evaluated explicitly:

- **CONTINUE — rejected as impermissible and unjustified.** The next step of
  any iteration is step 3, "evaluate on the hidden-ground-truth benchmark".
  No human-verified ground truth exists; the owner's binding mid-campaign
  correction states the AI's own readings may not serve as ground truth and
  orders: "pause further benchmark iterations until the ground truth is
  human-verified". Even label-free work (pre-collecting raw outputs for
  iterations 2–8) would run further benchmark sweeps against that order,
  and could be invalidated anyway (the owner may amend the benchmark, e.g.
  the flagged rep_r4 recrop). Therefore no experiment is simultaneously
  permitted and useful: "no justified experiment remains" — the spec's own
  STOP trigger.
- **ACCEPT — rejected as impossible.** Acceptance requires measured
  thresholds (usable ≥90 %, omission ≤5 %, hallucination ≤2 %, CER ≤10 %,
  stability) on the hidden benchmark. Zero candidates have any measured
  metrics (metrics are forbidden until verified labels exist, enforced in
  code), and the only fidelity evidence on record (Q1.2 diagnostic) points
  the other way. Accepting would fabricate success.
- **STOP — selected.** The enumerated STOP condition "no justified
  experiment remains" holds. STOP here is a verdict on THIS bounded
  campaign run, not on the research question: the benchmark, the retained
  iteration-1 outputs, the cached Q8_0 model, and the runner/evaluator
  interlock make the owner-ordered resume ("resume the same bounded
  campaign from Iteration 1") mechanical once verified labels arrive.

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

**Result (scored on the owner-verified 16-cell benchmark; 11 strict + 5
hard):** mean CER **0.936**, usable rate **0.00**, WER 1.204, omission
0.129, hallucinated-word rate **0.203**, stability (pairwise CER between
temp-0 repeats) 0.147, hard cells flagged 0/15 — **confabulated 15/15**.
Runtime: ~2.8–7 s/cell on the RTX 2000 Ada (8K ctx); raw outputs under
`outputs/it1_baseline_8b/`. Model `qwen3-vl:8b-instruct` (Q4_K_M, Ollama ID
0533d74300e4, Apache-2.0); prompt `baseline`; preprocessing none.

**Vs gate:** every threshold fails by roughly an order of magnitude; the
model never once flags unreadable text. Matches the Q1.2 diagnostic's
prediction (fluent confabulation). Baseline floor established.

**Decision: CONTINUE** — justified next hypothesis exists (the baseline
prompt offered no escape hatch and no fidelity constraints).

## Iteration 2 — strict-fidelity prompt

**Hypothesis:** an explicit escape hatch ("[?]" per unreadable word,
"[unreadable]" per cell, "always better to output [?] than to guess") plus
word-fidelity rules will (a) convert confabulations into flagged
unreadables on hard cells and (b) reduce the hallucinated-word rate on
strict cells — possibly at the cost of higher omission. CER is not expected
to reach the gate; this isolates how much of the failure is PROMPT rather
than perception.

**Variable changed:** prompt only (`strict_fidelity` in
scripts/hebrew_bench_run.py). Model, crops, decoding, preprocessing
unchanged.

**Result:** CER 0.842 (was 0.936), usable **0.00** (unchanged), WER 1.047,
omission 0.294 (**up** from 0.129 — the predicted cost), hallucinated-word
rate **0.077** (down 2.6× from 0.203), stability 0.046 (3× more
consistent), hard cells **still 15/15 confabulated, 0 flagged** — the
escape hatch was never used despite explicit better-[?]-than-guess rules.

**Interpretation:** hypothesis (a) partially confirmed (hallucination and
stability improve materially); hypothesis (b) refuted — the model does not
recognise its own unreadability; failure is perception-bound. Prompt kept
as the new reference (safer hallucination/omission trade-off; CER no
worse) per the keep-rule.

**Decision: CONTINUE** — next single variable: quantization.

## Iteration 3 — Q8_0 quantization

**Hypothesis:** Q4_K_M weight quantization damages fine visual-character
discrimination; the cached `qwen3-vl:8b-instruct-q8_0` (9.8 GB, Ollama ID
eff3eb825b32, same architecture/license) should read characters better if
quantization is a limiting factor — and change nothing if the limit is
model scale/architecture.

**Variable changed:** model quantization only (prompt `strict_fidelity`,
crops, decoding, preprocessing unchanged).

**Result:** CER 0.892 (it2: 0.842 — within run noise), usable **0.00**,
WER 1.161, omission 0.204, hallucinated-word rate 0.194 (worse than it2's
0.077 under the identical prompt), stability 0.165 (worse), hard cells
15/15 confabulated, 0 flagged. Runtime 1323 s (~15.8 s/call — 75 % slower
than Q4).

**Interpretation:** hypothesis refuted — quantization is not the limiting
factor; fidelity is unchanged at 8-bit weights and safety metrics regress.
Change REJECTED (reference remains Q4_K_M + strict prompt).

**Decision: CONTINUE** — next single variable: input preprocessing.

## Iteration 4 — contrast-stretch preprocessing

**Hypothesis:** the crops' gray table shading and moderate ink contrast
impair character discrimination; a deterministic 5th–95th percentile
grayscale contrast stretch (ink→black, paper→white; no new dependencies)
improves character-level reading if input quality is a limiting factor.

**Variable changed:** preprocessing only (`--preproc contrast`), on the
reference config (Q4_K_M + strict prompt).

**Result:** CER **0.786** (best so far; it2 0.842), usable **0.00**, WER
1.035, omission 0.263, hallucinated-word rate 0.087, stability 0.071, hard
cells 15/15 confabulated, 0 flagged. Runtime 275 s (contrast crops also
encode smaller).

**Interpretation:** first genuine fidelity improvement (~7 % relative CER)
with no hallucination increase → change **KEPT** (reference = Q4_K_M +
strict prompt + contrast). Still an order of magnitude from the gate;
cumulative picture after four iterations (CER 0.94→0.84→0.89→0.79, usable
0 % throughout, hard cells 60/60 confabulated) says prompt/quant/preproc
are small levers on a perception-bound failure.

**Decision: CONTINUE** — next single variable: model scale
(qwen3-vl:30b-a3b-instruct MoE, downloading; expected partial CPU offload).

## Iteration 5 — model scale (Qwen3-VL-30B-A3B MoE)

**Hypothesis:** the 8B's visual encoder/decoder capacity is the binding
constraint; the 30B-A3B MoE (Apache-2.0, same family, ~19 GB Q4, 3B active
parameters) reads cursive Hebrew materially better. This is the only
remaining candidate with plausible order-of-magnitude headroom on local
hardware; it will partially offload to CPU on the 15.4 GB card.

**Variable changed:** model only (prompt strict, preproc contrast — the
kept reference configuration).

**Result:** CER 0.800 (reference 8B: 0.786 — statistically
indistinguishable), usable **0.00**, WER 1.012, omission 0.271,
hallucinated-word rate 0.065, stability 0.162 (worse — MoE routing adds
nondeterminism), hard cells 15/15 confabulated, 0 flagged. Runtime 1284 s
at a 41 %/59 % CPU/GPU split (22 GB model on the 15.4 GB card), ~15 s/call.

**Interpretation:** scale hypothesis REFUTED for the tested family — 3.75×
total parameters reads this cursive no better than 8B. Per the owner's
directive this shows only that the tested GENERAL-PURPOSE Qwen VLMs fail;
dedicated HTR models remain untested. Change rejected (8B + strict +
contrast stays the strongest Qwen result: CER 0.786 / usable 0 %).

**Decision: CONTINUE** — per the owner's revised plan, iterations 6–8
become the focused LOCAL Hebrew-HTR benchmark
(evaluation/local_hebrew_htr_benchmark.md): it6 dedicated Hebrew HTR, it7
multilingual TrOCR-style, it8 best-HTR vs best-Qwen with repeatability.
EasyOCR/Tesseract only as printed-OCR baselines. Separate venv, provenance
cards, deterministic segmentation, raw-output-only scoring.
