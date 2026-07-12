# Evaluation

## Benchmark tiers

1. **Representative-exam benchmark** (`sample_data/`): one fully-understood
   exam with human ground truth established by manual inspection —
   version A1; the student swapped the two answer tables and declared an
   X-marks-final convention on the bubble sheet; instructor scores Q1=24/32,
   Q2=28/32. A correct system must reproduce the table swap, the X
   convention, version A1, and flag (not silently grade) the
   reversed-with-correct-explanation items the human grader chose to accept.
   This is the single most information-dense test we have for capability
   dimensions that exam-level grades cannot measure: circle/X/bubble
   recognition, cross-out handling, convention interpretation,
   student/instructor ink separation, ambiguity detection, and Hebrew
   handwriting transcription quality (inspect `extraction.json`
   transcriptions manually against the scan).

2. **Split evaluation** (`test/` via manifests): `eval-batch` grades a whole
   split and compares totals against instructor grades — see metrics below.
   Train-split results are development signal only; **only validation-split
   results may be quoted as evidence of generalization**, and only the final
   held-out set supports the final claim.

## Commands

```
autograder make-manifests                       # once (deterministic, seed 42)
autograder eval-batch --split validation \
    --backend openai --base-url http://localhost:11434/v1 --model qwen3-vl:8b \
    --key sample_data/Exam_solution.pdf --out eval_out
autograder audit-leakage --limit 5 [backend flags]
```

`eval-batch` masks instructor annotations by default, anonymizes before
inference, processes exams independently, continues after individual
failures, supports safe resume (input-fingerprinted), writes one result
directory per exam plus `combined_results.{json,csv}`, `summary.md`,
`failed_exams.json`, and `review_cases.json`, and never modifies source
exams.

## Total-score metrics reported

Number processed, failures, exact-grade accuracy, accuracy within ±2/±5/±10,
MAE, median AE, RMSE, mean signed error (systematic over/under grading), max
absolute error, human-review rate, runtime per exam, and the exact backend +
model + generation configuration. Client-side memory is negligible; model
memory is a property of the inference server (documented in deployment.md) —
we do not fabricate per-exam RAM/VRAM numbers.

## Per-question / capability metrics

Where detailed labels exist (none yet — see docs/datasets.md), the same
runner's per-exam `result.json` supports: final-answer extraction accuracy,
per-question score accuracy, explanation-transcription quality,
explanation-grading agreement, ambiguous-answer precision/recall, and
human-review precision/recall. Until such labels are derived and verified,
these are measured only manually on the representative exam.

## Held-out final test procedure (48 exams — currently absent)

The 48 additional graded exams must remain completely unseen during
development: no training, validation, model selection, prompt tuning,
threshold tuning, calibration, debugging, fine-tuning, preprocessing
decisions, or architecture selection may touch them.

**Freeze before first contact.** When development concludes:

1. Freeze: model choice + exact weights/tag, prompts, preprocessing (render
   size, masking parameters), scoring logic, calibration, confidence and
   human-review thresholds, generation settings, backend. Practically:
   commit, tag the repository (`git tag final-freeze`), and record the tag,
   the backend `describe()` output, and the model digest (e.g.
   `ollama show <model>`) in `datasets/final_test_manifest.json` under
   `frozen_configuration`.
2. Place the 48 exams in a directory, run `make-manifests` against it with
   `--manifest-dir` pointing at a fresh directory, and copy the entries into
   `final_test_manifest.json` with split `final_test`.
3. Run `eval-batch` once with the frozen configuration.
4. Report the results as-is. **Do not modify the system based on them.** If
   changes are ever made afterwards, the final-test set is burned and must be
   reported as such.
