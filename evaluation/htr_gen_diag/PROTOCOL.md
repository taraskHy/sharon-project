# Writer-generalization diagnostic — pre-registered protocol

Written 2026-07-17 06:25, BEFORE any fold was prepared or trained.
This is a DIAGNOSTIC, not the official validation pilot: it does not
consume the pilot's 6-trial budget (no val-split scoring happens), and
it touches only the train split.

## Question

Do the current owner-verified train lines carry any cross-writer
learning signal, or does the CRNN only memorize writers?

## Data

Train-split lines with status `ok`, human_verified, and no partial
[לא קריא] span (the pilot's training-eligibility rule). Counts at
protocol time: e003=16, e004=18, e005=20, e006=10, e007=0 (total 64).

## Folds (max 3, fixed now)

Held-out writer = the three largest clean-ok writers, descending:

1. fold_e005 — train e003+e004+e006 (44), held-out e005 (20)
2. fold_e004 — train e003+e005+e006 (46), held-out e004 (18)
3. fold_e003 — train e004+e005+e006 (48), held-out e003 (16)

All samples of a writer stay in one side. Nothing from the held-out
writer influences training, early stopping, checkpoint selection,
augmentation, or the symbol table.

## Fixed configuration (from the overfit test, deviations documented)

- Architecture/seed/optimizer/scheduler: exactly
  `scripts/htr_pilot_train.py` (CRNN+CTC, batch 8, lr 3e-4, AdamW,
  ReduceLROnPlateau on train loss, seed 20260714, cuDNN off) — the
  configuration that passed the 2026-07-16 overfit gate.
- Checkpoint selection & early stopping: fold val list is EMPTY, so the
  trainer selects on TRAIN LOSS only (no held-out influence).
- Symbol table per fold: predeclared base alphabet
  (`htr_train_prepare.BASE_CHARS`) + chars observed in that fold's
  TRAINING labels only.
- Augmentation: deterministic ×3 per train line (pilot default is ×5;
  reduced for the overnight compute bound — documented deviation).
- Epoch budget: `--epochs 150 --max-train-seconds 2100 --patience 50`,
  identical for every fold (the overfit test ran 900 epochs on 20
  images; 3 folds of ~180 images cannot fit that in the 5-hour
  campaign — documented deviation, uniform across folds, chosen before
  any fold result was seen). No per-fold tuning of anything.

## Metrics (post-decode, campaign definitions)

Per fold: held-out-writer line CER/WER (normalized), exact-match and
usable-line rate (CER <= 0.25), omission (word deletions / ref words)
and insertion (word insertions / hyp words) rates, train-side CER at
the same checkpoint (memorization reference), confidence-vs-CER
buckets, 5 best / 5 worst / median representative raw predictions.

## Interpretation guide (fixed now)

- Held-out CER >= 0.9 on all folds: no cross-writer signal at this data
  size — writer memorization only.
- Held-out CER 0.6–0.9 with confidence-CER correlation: weak signal —
  more writers/lines justified before the pilot.
- Held-out CER < 0.6 or usable-rate > 0: real signal — annotate more
  and proceed to the pilot with confidence.
Reference: best VLM on this handwriting = CER .786; oracle over 14
existing experts = .717, usable 0.
