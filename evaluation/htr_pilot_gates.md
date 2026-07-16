# HTR fine-tune pilot — pre-registered protocol and gates

Written 2026-07-14, BEFORE any training run, per the project's audit-gated
workflow. Changes to this file after the first training run must be
flagged to the owner explicitly.

## Fixed decisions

- Engine: a minimal in-repo CRNN+CTC line recognizer
  (`scripts/htr_pilot_train.py`, plain torch 2.13+cu126 in `.venv-train`,
  no framework dependencies). Decided BEFORE any training run: PyLaia
  cannot install on this machine (1.1.x requires Python < 3.11, this venv
  is 3.12; 1.0.x is pip-unresolvable), and kraken has the same
  ecosystem-pin risk on Windows. The architecture follows the standard
  PyLaia/kraken recipe (conv stack + BiLSTM + CTC); switching to a
  maintained framework later (e.g. PyLaia on a Python 3.10 venv) does not
  reset the trial budget.
- Data: ONLY `evaluation/htr_pilot` train-split lines whose annotation is
  owner-verified `ok` without a partial [לא קריא] span, plus their
  deterministic augmentations (`scripts/htr_train_prepare.py`). Blank /
  unreadable / flagged / draft lines never train.
- Writer separation: model selection uses ONLY val decodes (writers
  e013–e015). internal_test (e016–e018) is decoded ONCE per pilot, after
  the configuration is frozen, and only via the `--allow-internal-test`
  flags. Exam 002 and the 48 held-out exams appear nowhere.
- Primary metric: optical-only (greedy CTC) cell-level CER and usable-rate
  (CER <= 0.25, campaign definition), computed by
  `scripts/htr_pilot_eval.py` with the campaign's normalization. Lexicon /
  LM decoding may be reported only as a clearly-labelled SECONDARY number.
- Trial budget: at most 6 val-scored training configurations for the
  pilot. Every trial is appended to `evaluation/htr_train_workspace/
  trials.jsonl` by the driver before its val score is known.
- Abstention: report the cell-confidence abstention curve (min line
  confidence per cell); the pilot's abstention headline is the usable-rate
  on accepted cells at the smallest tau with >= 50 % coverage. (Ensemble
  agreement was rejected as an abstention signal —
  `evaluation/oracle_ensemble_analysis.md`.)

## Gates (chosen before any result exists)

Reference points: best current system CER .786 / usable 0 % (isol2/it4);
oracle over all 14 existing experts .717 / usable 0.

- **CONTINUE (scale annotation to e019–e042, refine model):**
  best val cell CER <= 0.60 OR val usable-rate >= 10 %.
- **STRONG SIGNAL (also draft production-integration plan):**
  val usable-rate >= 25 % with abstention precision >= 80 % at >= 50 %
  coverage.
- **DIAGNOSE-BEFORE-MORE-SPEND:** after 6 trials best val cell CER >
  0.70 — analyse error classes (writer gap vs data volume vs segmentation)
  before any further training or annotation is proposed.
- Final report (single internal_test decode) happens regardless of which
  gate fired, once, after config freeze.

## Scaffold status (2026-07-14 — ready, waiting on owner labels)

- `.venv-train`: Python 3.12, torch 2.13.0+cu126 (CUDA verified on the
  RTX 2000 Ada), opencv-python-headless. cuDNN is DISABLED in the trainer
  (its RNN-backward teardown fail-fasts on this stack and corrupts exit
  codes; native kernels are sufficient at pilot scale).
- End-to-end smoke on synthetic scribbles + dummy labels
  (`scripts/htr_train_smoke.py`): prepare -> train (loss 14.6 -> 2.0 on
  GPU) -> decode with confidences -> internal_test refusal -> eval
  harness: PASS. No real annotation was used or created.
- Pipeline commands once train+val annotations exist:
  1. `.venv/Scripts/python.exe scripts/htr_train_prepare.py`
  2. `.venv-train/Scripts/python.exe scripts/htr_pilot_train.py train`
  3. `.venv-train/Scripts/python.exe scripts/htr_pilot_train.py decode
     --split val --out decodes/val_trial01.txt`
  4. `.venv/Scripts/python.exe scripts/htr_pilot_eval.py
     evaluation/htr_train_workspace/decodes/val_trial01.txt`
- Ensemble/abstention-by-agreement over the existing 14 experts was
  measured and REJECTED before this pilot
  (`evaluation/oracle_ensemble_analysis.md`: oracle usable 0/11).

## Honesty rules

- Raw decodes are saved before scoring; no post-hoc metric definitions.
- No transcription of student handwriting by AI enters any label file.
- The e002 hidden benchmark stays untouched; a fine-tuned model may be
  benchmarked on it later ONLY as a separate, owner-approved step.
