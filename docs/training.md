# Training and calibration strategies

The available supervision is weak: 41 exam-level final grades (train split:
25) plus one deeply-understood representative exam. This section compares
realistic strategies; the comparison is analytical — experiments require an
inference/training backend and are listed as the run plan at the bottom.

| Strategy | What it needs | Expected value here | Risks / verdict |
|---|---|---|---|
| **No fine-tuning; prompt-based use of a pretrained VLM** | nothing beyond this repo | The pipeline's structure (survey → blind extraction → judging → deterministic scoring) already encodes the domain knowledge; prompts were developed against the representative exam | **Baseline — do this first.** All development so far is here |
| **Prompt/config calibration on the train split** | eval-batch runs | Cheap wins: image resolution, structured mode, judge strictness, review thresholds; select on train, confirm on validation | Overfitting 25 exams is easy — change few knobs, justify each |
| **Total-score calibration** (e.g. shift/scale correction of systematic bias measured on train) | eval-batch runs | Detects systematic over/under-grading (mean signed error); a constant correction is defensible | Corrects symptoms, not causes; prefer fixing the causal stage |
| **LoRA / QLoRA on the VLM** (e.g. via LLaMA-Factory/ms-swift for Qwen3-VL or MiniCPM-V, which ship recipes) | GPU (≥24 GB for 8B QLoRA), derived per-question labels | Potentially large gains on the real bottleneck (Hebrew handwriting transcription + mark reading) | 25 exams ≈ a few thousand sub-item examples at best, labels must first be derived from instructor marks and verified; high risk of memorizing the single exam form. Feasible on university hardware, not on the CPU-only dev laptop |
| **Small specialized detectors** (classic CV or small nets for circle/X/bubble marks; red-ink classifier) | labeled mark crops | Could harden the most mechanical part and cheaply cross-check the VLM (disagreement ⇒ review flag) | Extra components to maintain; only worth it if benchmarks show mark-reading is a dominant error source |
| **Fine-tuning only the judging/calibration component** (a small text-only LLM judging transcribed explanations) | transcriptions + verdict labels | Decouples judging quality from vision quality | Needs verdict labels that don't exist yet; the judge is unlikely to be the first bottleneck |
| **Full fine-tuning of a large VLM** | multi-GPU cluster, large labeled corpus | — | **Not justified**: data is orders of magnitude too small; rejected without experiment |

## Recommended sequence

1. Prompt-based baseline on validation (no tuning) → identifies the real
   bottleneck from `extraction.json`/`result.json` inspection.
2. Config calibration on train (resolution, structured mode, thresholds);
   confirm on validation.
3. Only if handwriting transcription or mark-reading dominates the error
   budget: derive per-question labels from instructor annotations
   (docs/datasets.md), verify a sample by hand, and run QLoRA on the
   selected 8B VLM on university GPU hardware.
4. Keep the review-rate/quality trade-off explicit: raising confidence
   thresholds converts errors into human-review items, which the university
   may prefer over silent errors.

## Run plan for university hardware (not executed here)

```
# baseline
autograder eval-batch --split validation --backend openai --base-url http://SERVER:8000/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct --key sample_data/Exam_solution.pdf --out eval_baseline

# calibration sweep example (train split only)
for edge in 1200 1600 2000; do
  autograder eval-batch --split train --max-image-edge $edge ... --out eval_edge_$edge
done
```
