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

## FIRST TASK NEXT SESSION — oracle-ensemble analysis (cheap, no training)
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
