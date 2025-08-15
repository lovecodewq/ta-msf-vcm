#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/train_lmsfc_anchor.yaml}
DET_CKPT=${DET_CKPT:-checkpoints/detection/run_0.002000_16/best_model.pth}
TP_ROOT=${TP_ROOT:-thirparty}

python -m src.train_lmsfc_anchor \
  --config "$CONFIG" \
  --detection_checkpoint "$DET_CKPT" \
  --third_party_root "$TP_ROOT"

echo "Done training L-MSFC anchor."

