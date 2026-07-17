#!/bin/bash
# Writer-generalization diagnostic: train/decode/eval all 3 folds
# sequentially with the pre-registered uniform settings (PROTOCOL.md).
set -e
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../.."
for w in e005 e004 e003; do
  ws=evaluation/htr_gen_diag/fold_$w
  echo "=== FOLD $w train $(date +%H:%M:%S) ==="
  .venv-train/Scripts/python.exe scripts/htr_pilot_train.py --workspace $ws train \
      --epochs 150 --batch-size 8 --lr 3e-4 --patience 50 \
      --max-train-seconds 2100 > $ws/train_log.txt 2>&1
  tail -2 $ws/train_log.txt
  echo "=== FOLD $w decode $(date +%H:%M:%S) ==="
  .venv-train/Scripts/python.exe scripts/htr_pilot_train.py --workspace $ws \
      decode --split heldout --out decodes/heldout.txt >> $ws/train_log.txt 2>&1
  .venv-train/Scripts/python.exe scripts/htr_pilot_train.py --workspace $ws \
      decode --split trainbase --out decodes/trainbase.txt >> $ws/train_log.txt 2>&1
  echo "=== FOLD $w eval $(date +%H:%M:%S) ==="
  .venv/Scripts/python.exe scripts/writer_gen_diagnostic.py eval --heldout $w \
      | head -20
done
echo "ALL FOLDS DONE $(date +%H:%M:%S)"
