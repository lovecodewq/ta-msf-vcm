#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-data/processed/kitti}
DET_CKPT=${DET_CKPT:-checkpoints/detection/run_0.002000_16/best_model.pth}
LMSFC_CKPT=${LMSFC_CKPT:-checkpoints/lmsfc_anchor/run_20250813_142924_lambda_2.00e+01_lr_1.00e-04_bs_8/best_model.pth}
# Create timestamped subdir to avoid overwriting previous runs
OUT_DIR_BASE=${OUT_DIR_BASE:-evaluation_results/lmsfc_anchor}
TS=$(date +"%Y%m%d_%H%M%S")
OUT_DIR=${OUT_DIR:-${OUT_DIR_BASE}/run_${TS}}
TP_ROOT=${TP_ROOT:-thirparty}

python -m src.evaluation.run_lmsfc_anchor \
  --data_dir "$DATA_DIR" \
  --detection_model "$DET_CKPT" \
  --lmsfc_checkpoint "$LMSFC_CKPT" \
  --out_dir "$OUT_DIR" \
  --third_party_root "$TP_ROOT"

echo "Done. Outputs under $OUT_DIR"

