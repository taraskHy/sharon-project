# NEXT SESSION HANDOFF — 2026-07-13 end of session

## State
- Branch: `initial-prototype` @ latest commit (see `git log -1`; this file
  committed on top of e898549). Working tree clean; all work pushed to
  https://github.com/taraskHy/sharon-project (PRIVATE — contains real
  student scans; keep it private).
- Machine: Windows 11, Ryzen 5 5600G, 64 GB RAM, RTX 2000 Ada 15.4 GB.
  Ollama 0.31.2 (`%LOCALAPPDATA%\Programs\Ollama\ollama.exe`); start with
  `OLLAMA_CONTEXT_LENGTH=16384` (32768 only to reseed key/alignment
  caches). Venvs: `.venv` (pipeline), `.venv-htr` (transcription bench:
  torch 2.13 cpu, transformers 4.57.6, surya-ocr 0.17.1).

## Objective + privacy constraints (unchanged)
Grade scanned Hebrew exams with LOCAL open models only. Never: hosted
image APIs, exam uploads, grade-bearing filenames/labels in any
model-visible channel, ground truth in prompts, use of the 48 held-out
exams. Grading policy and review gates must not be weakened.

## Verified facts
- Tests: **122/122** (`.venv\Scripts\python.exe -m pytest`).
- Representative exam: full chain live-verified (flower→A1, sheets 11–13,
  operator alignment, X-convention, instructor-ink exclusion). Final
  14/100 with 35 review flags incl. the fired swap tripwire (crossed key
  agreement 7/16 vs own 1/16); instructor reference Q1=24/32, Q2=28/32.
  Per-item audits: `evaluation/representative_exam_audit.md`,
  `evaluation/exam003_audit.md`, `evaluation/diag_q1_2.md`.
- Stage A (5 validation exams, masked, anonymized): 0 failures, variant
  detection 5/5, MAE 39.6 (all under-scoring), mean 906 s/exam, GPU 98.1 %
  active, no CPU offload. Stage B/C intentionally not run (owner gate).
- Production behavior: correct selections with untranscribed explanations
  gate to ZERO and are review-flagged per item; Q3 key columns
  review-flagged until `sample_data/Exam_solution.versions-override.json`
  is completed by the instructor.

## Hebrew transcription campaign (CLOSED: STOP, 8/8 iterations)
- Benchmark: 28 crops (`evaluation/hebrew_bench/crops/` +
  `crops_manifest.json`); **16 owner-verified cells (ALL FROM EXAM 002 —
  ONE WRITER ONLY; generalization unknown)**: hidden GT =
  `evaluation/hebrew_bench/verified_ground_truth.json` (11 strict, 5 hard
  with [לא קריא]); owner CSV =
  `evaluation/hebrew_bench/human_annotation/annotation_template.csv`;
  12 cells (e003 ×8, rep ×4) remain unverified/excluded.
  GT is read ONLY by `scripts/hebrew_bench_eval.py`, post-inference.
- Results (strict cells; full ledgers `hebrew_transcription_results.csv`,
  `local_hebrew_htr_results.csv`):

| config | model | CER | WER | omit | halluc | usable | stability | runtime |
|---|---|---|---|---|---|---|---|---|
| it1_baseline_8b | qwen3-vl:8b-instruct Q4, plain prompt | .936 | 1.204 | .129 | .203 | 0 | .147 | ~3 s/cell GPU |
| it2_strict_prompt | + strict-fidelity prompt | .842 | 1.047 | .294 | .077 | 0 | .046 | ~3 s |
| it3_q8_quant | 8b-instruct-q8_0 | .892 | 1.161 | .204 | .194 | 0 | .165 | ~16 s |
| it4_contrast (**best**) | Q4 + strict + contrast stretch | **.786** | 1.035 | .263 | .087 | 0 | .071 | ~3 s |
| it5_moe30b | qwen3-vl:30b-a3b (41/59 CPU/GPU) | .800 | 1.012 | .271 | .065 | 0 | .162 | ~15 s |
| it6_hdd_words | sivan22/hdd-words-ocr@e089ce71 + word seg | .963 | 1.188 | .565 | .302 | 0 | .000 | ~2.3 s CPU |
| it7_surya | surya-ocr 0.17.1 (in-process) | .955 | 1.129 | .212 | .152 | 0 | .000 | ~9.9 s CPU |

  Hard cells: 85/85 evaluations confabulated; zero honest unreadable flags.
  it7 reads Hebrew cursive as ENGLISH words (script-level failure).
- Raw outputs RETAINED per cell per run:
  `evaluation/hebrew_bench/outputs/<config>/run<N>/<cell>.json`
  (fields: raw, transcription, error, latency_s). Word-seg crops:
  `evaluation/hebrew_bench/segments_words/it6_hdd_words/`.
- Commands: Qwen runner `scripts/hebrew_bench_run.py --config-id X --model
  M --prompt {baseline|strict_fidelity} [--preproc contrast] --runs N`
  (needs Ollama serving); HTR `scripts/hebrew_htr_run.py`; Surya
  `scripts/hebrew_surya_run.py` (both under `.venv-htr`). Evaluator:
  `.venv\Scripts\python.exe scripts/hebrew_bench_eval.py <config_id>
  [results_csv]` — refuses without verified GT.
- Failed-to-run models + exact reasons: **HebHTR** (best domain provenance,
  self-reported 4.76 % CER on student exams — repo archived, NO license,
  TF 1.12 + Linux-only compiled `TFWordBeamSearch.so`, master broken;
  needs WSL2/Docker = owner approval); **ABBA-HTR** (pytorch_model.bin
  pickle-only — security rule); **Surya 0.21.1** (inference manager spawns
  Docker — system-level install not approved); CHURRO-3B (historical hands,
  Qwen research license); PaddleOCR-VL (no Hebrew + trust_remote_code);
  TrOCR handwritten (English-only); medieval kraken models (wrong script
  tradition). Full provenance: `evaluation/local_hebrew_htr_benchmark.md`.
- Unresolved: faithful transcription of modern Hebrew exam cursive. Verdict
  taxonomy: VLM failure confirmed; segmentation secondary; runnable
  public HTR fails; **insufficient training data is the structural
  diagnosis** (HebHTR proves learnability); hardware not binding.
  Recommended pilot: kraken/PyLaia fine-tune, ~800–2,000 verified lines
  (~1,000 exist in the 41 dev exams), hours on this GPU — with
  WRITER-SEPARATED splits.

## Student-ink isolation experiment (2026-07-13, owner-directed; DONE — REJECTED)
- Hypothesis tested: printed table structure in crops drives recognizer
  failure; registered template subtraction + blue-ink isolation may fix it.
- Built `scripts/student_ink_isolation.py`: no blank answer sheet exists
  (Exam_solution.pdf = question booklet only), so blank templates were
  SYNTHESIZED as the per-pixel median of 16 ECC-registered dev-exam pages
  per sheet (page-search handles variable booklets; cc .59–.86, 16/16
  kept). Thresholds frozen from image stats (T_blue 25, per-page Otsu,
  texture 200@9px, production red rule); E2 images verified 16/16
  faithful BEFORE recognition (contact sheets:
  `evaluation/student_ink_isolation_artifacts/contact/`).
- Ablation (fixed qwen3-vl:8b-instruct, strict prompt, temp 0, 3 runs):
  original .866 / blue-only .870 / template-sub **.790** / lines .828 CER;
  usable **0 %** everywhere; best single cell .66. hdd-words: .978 / .937
  / .936. Ledger `evaluation/student_ink_isolation_results.csv`; report
  `evaluation/student_ink_isolation_experiment.md`.
- Verdict: **REJECT — "recognition despite clean input"** (registration,
  ink separation, segmentation all verified good). Real secondary effects:
  omission .318→.051, hallucination .134→.062 at equal CER; first honest
  [unreadable] flags ever (3/15, lines arm). Reinforces the fine-tune
  pilot; E2 cells + F line crops are reusable as its cleaned inputs.
  Runner scripts gained `--manifest` (alt crop sets); new
  `scripts/hebrew_ink_lines_run.py` joins per-line outputs per cell.
  `.venv` gained opencv-python-headless 5.0.0. Owner review pending.

## HTR-pilot annotation package (2026-07-14, owner-directed; READY FOR OWNER)
- Writer-separated deterministic split over 16 dev exams: train e003–e012,
  val e013–e015, internal_test e016–e018; e002 excluded from EVERY split
  (benchmark writer); rep exam + e019–e042 excluded/reserved; held-out
  never referenced. 441 line samples / 256 cells; zero build failures.
- Build: `scripts/htr_pilot_build.py` (sheet pages found per exam by ECC
  page-search, Q1-vs-Q2 verified by title-digit matchTemplate — e014's
  sheets are on pages 5–6; per-exam synthesized blank templates; cell rows
  from detected table rules with registration fallback — used by 5 Q1
  sheets, QA'd clean; per-sheet sidecars make re-runs incremental).
  Package: `evaluation/htr_pilot/` (images, splits/, annotations/ per
  split, contact/, summary.json, README.md with rules+recovery); sources
  map with grade-bearing scan names kept OUTSIDE at
  `evaluation/htr_pilot_sources.json` (never for training).
- App: `.venv\Scripts\python.exe -m streamlit run
  scripts/htr_annotation_app.py` (streamlit 1.59.2 now in .venv) — RTL,
  line+cell+original views, atomic autosave every button, resume, prev/
  next drafts, [לא קריא]/Blank/Bad-segmentation/Needs-recrop/Skip.
  Validator: `scripts/htr_annotation_validate.py` → RESULT: PASS.
  Tests 136/136; AppTest smoke PASS (dummy labels in temp only). NO
  authoritative labels exist yet; owner annotation ≈ 2.5–5 h. Report:
  `evaluation/htr_annotation_package.md`.

## Oracle-ensemble analysis (2026-07-14; DONE — REJECT, gate met exactly)
- Ran post-hoc over run1 of all 14 retained configs vs hidden GT
  (`scripts/oracle_ensemble_analysis.py` ->
  `evaluation/oracle_ensemble_analysis.md` + percell CSV). Oracle CER
  .717, oracle usable **0/11** (no expert reaches even CER 0.5 on any
  strict cell) -> pre-registered REJECT branch: no MoE/ensemble, proceed
  to fine-tuning. Medoid consensus (.864) is worse than best single
  (.786) — 10/14 experts share the base VLM; agreement carries no
  correctness signal (some pair agrees >=0.9 on every cell). Abstention
  must come from model-own confidence (CTC posteriors), not agreement.

## HTR fine-tune pilot scaffold (2026-07-14; READY — waiting on labels)
- Protocol + pre-registered gates: `evaluation/htr_pilot_gates.md`
  (trial budget 6, val-only selection, single internal_test decode,
  optical-only primary metric, CONTINUE/DIAGNOSE thresholds fixed before
  any run).
- `.venv-train`: py3.12 + torch 2.13.0+cu126 (CUDA verified) + opencv.
  PyLaia is NOT installable (needs py<3.11 / unresolvable pins) — pilot
  uses the in-repo CRNN+CTC trainer `scripts/htr_pilot_train.py`
  (train/decode subcommands, greedy decode + per-line confidence,
  trials.jsonl audit log). **cuDNN disabled in the trainer** — its RNN
  backward fail-fasts (0xC0000409) at teardown on this stack (isolated
  by minimal repro; native kernels fine at pilot scale).
- Data prep: `scripts/htr_train_prepare.py` (.venv) — ok-lines only
  (blank/unreadable/flagged/token-span excluded per reason), deterministic
  x5 augmentation, char symbol table, internal_test refused without
  --allow-internal-test. Eval: `scripts/htr_pilot_eval.py` (line+cell
  CER/WER, usable-rate, confidence-abstention curve). E2E smoke on
  synthetic data: `scripts/htr_train_smoke.py` — PASS (loss 14.6->2.0 on
  GPU). Workspace `evaluation/htr_train_workspace/` is git-ignored.
- NEXT: owner annotates train+val in the annotation app -> run the 4
  pipeline commands in htr_pilot_gates.md "Scaffold status".

## Real-data overfit test (2026-07-16; PASS — gate met, trainer proven)
- Owner annotated 91 train samples (72 ok / 18 bad_segmentation /
  1 needs_recrop; 56 ok-lines without [לא קריא] spans); validator PASS.
- Pre-registered gate (mean CER <= .05, >= 18/20 lines <= .10) on the
  first 20 eligible lines (sorted-id order, saved before training):
  **PASS — mean CER .0028, 20/20 within gate, 18/20 exact**; WER .019;
  mean confidence .956 (min .942); checkpoint reload delta 0.0000;
  RTL sanity: reversed-prediction CER .847 vs .0028; 71-char vocab fully
  covered; no CTC length violations; no empty decodes. Report:
  `evaluation/htr_overfit_test/report.md` (rig pkg/ws git-ignored,
  selection in selected_ids.json).
- The test caught and fixed 3 real trainer defects BEFORE the pilot:
  (1) RTL/CTC ordering — labels must be bidi display-order at train time
  (weak-bidi involution `to_display_order` in htr_train_prepare, inverse
  at decode; unit-tested); (2) LR scheduler stepped on val CER, which sits
  at 1.0 through the CTC blank-collapse phase and froze learning -> now
  steps on train loss (+ --min-epochs guard, default overfit 500);
  (3) width subsampling /8 left ~1.3 frames/char on narrow lines ->
  /4 (escape from collapse at ~epoch 126, memorized by ~550).
- Owner instruction honoured: pilot NOT started; owner continues
  annotating train+val splits.

## [COMPLETED 2026-07-14 — see oracle section above] original spec: oracle-ensemble analysis
Use ONLY the retained raw outputs above + hidden GT (post-hoc). Per
verified crop: pick the expert output with the LOWEST CER (oracle);
report oracle CER/WER/usable-rate vs each expert; pairwise error
correlation between experts; count crops where NO expert has meaningful
correct text; then test inference-time selection rules that DON'T use GT
(inter-expert agreement, abstention on disagreement) — never select on
Hebrew fluency. Decision gate: oracle usable-rate ≈ 0 → reject MoE over
these experts, proceed to the writer-separated fine-tuning pilot; oracle
materially better → document complementarity but do NOT train a gate on
this one-writer benchmark. Do not build/train an MoE either way.
