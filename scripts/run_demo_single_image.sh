#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env or flags)
IMG=${IMG:-example.png}
OUT=${OUT:-demo/result}
DET_CKPT=${DET_CKPT:-checkpoints/detection/run_0.002000_16/best_model.pth}
FEAT_CKPT=${FEAT_CKPT:-checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250816_010910_lambda_3.00e+01_w_1.00e-01_lr_1.00e-04_bs_8/best_model.pth}
QP=${QP:-32}
PIX_FMT=${PIX_FMT:-yuv420p10le}
ENC_BIN=${ENC_BIN:-/home/wenqiangli/code/VVCSoftware_VTM/bin/EncoderAppStatic}
DEC_BIN=${DEC_BIN:-/home/wenqiangli/code/VVCSoftware_VTM/bin/DecoderAppStatic}
ENC_CFG_BASE=${ENC_CFG_BASE:-/home/wenqiangli/code/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg}
ENC_CFG_OVR=${ENC_CFG_OVR:-/home/wenqiangli/code/VVCSoftware_VTM/cfg/encoder_ai_10bit_override.cfg}

usage() {
  cat <<USAGE
Usage: $0 [--image PATH] [--out DIR] [--det-ckpt PATH] [--feat-ckpt PATH] [--qp N] [--pix-fmt FMT]

Environment overrides:
  IMG, OUT, DET_CKPT, FEAT_CKPT, QP, PIX_FMT, ENC_BIN, DEC_BIN, ENC_CFG_BASE, ENC_CFG_OVR
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMG="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --det-ckpt) DET_CKPT="$2"; shift 2 ;;
    --feat-ckpt) FEAT_CKPT="$2"; shift 2 ;;
    --qp) QP="$2"; shift 2 ;;
    --pix-fmt) PIX_FMT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

python -m src.evaluation.demo_single_image \
  --image "$IMG" \
  --out_dir "$OUT" \
  --det_ckpt "$DET_CKPT" \
  --feat_ckpt "$FEAT_CKPT" \
  --qp "$QP" \
  --pix_fmt "$PIX_FMT" \
  --encoder_bin "$ENC_BIN" \
  --decoder_bin "$DEC_BIN" \
  --encoder_cfg "$ENC_CFG_BASE" "$ENC_CFG_OVR"

echo "Demo completed. Outputs in $OUT"


