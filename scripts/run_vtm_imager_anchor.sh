#!/usr/bin/env bash
set -euo pipefail

# Configuration (override via env vars)
ENC_BIN=${ENC_BIN:-/home/wenqiangli/code/VVCSoftware_VTM/bin/EncoderAppStatic}
DEC_BIN=${DEC_BIN:-/home/wenqiangli/code/VVCSoftware_VTM/bin/DecoderAppStatic}

CFG_BASE=${CFG_BASE:-/home/wenqiangli/code/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg}
CFG_OVR=${CFG_OVR:-/home/wenqiangli/code/VVCSoftware_VTM/cfg/encoder_ai_10bit_override.cfg}

QP=${QP:-32}
CONFIG=${CONFIG:-configs/train_joint_autoregress_prior_fused_feature_detect_loss.yaml}
SPLIT=${SPLIT:-test}
PIX_FMT=${PIX_FMT:-yuv420p10le}
DET_MODEL=${DET_MODEL:-checkpoints/detection/run_0.002000_16/best_model.pth}
OUT_DIR=${OUT_DIR:-evaluation_results/vtm_image_qp${QP}}
DATA_DIR=${DATA_DIR:-data/processed/kitti}

# Stage selector: encode | detect | both
STAGE=${1:-encode}

# Ensure override cfg exists
if [[ ! -f "$CFG_OVR" ]]; then
  cat > "$CFG_OVR" << 'EOF'
InputBitDepth : 10
InputChromaFormat : 420
IntraPeriod : 1
EOF
  dos2unix "$CFG_OVR" >/dev/null 2>&1 || true
fi

run_encode_decode() {
  python -m src.evaluation.vtm_image_anchor \
    --config "$CONFIG" \
    --split "$SPLIT" \
    --out_dir "$OUT_DIR" \
    --qp "$QP" \
    --encoder_cfg "$CFG_BASE" "$CFG_OVR" \
    --pix_fmt "$PIX_FMT" \
    --encoder_bin "$ENC_BIN" \
    --decoder_bin "$DEC_BIN"
}

run_detection() {
  # Avoid MKL threading incompatibility
  export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}
  python -m src.evaluation.run_detection \
    --data_dir "$DATA_DIR" \
    --detection_model "$DET_MODEL" \
    --mode reconstructed_images \
    --images_dir "$OUT_DIR/decoded_png" \
    --output_json "$OUT_DIR/preds.json"
}

case "$STAGE" in
  encode)
    run_encode_decode
    ;;
  detect)
    run_detection
    ;;
  both)
    run_encode_decode
    run_detection
    ;;
  *)
    echo "Unknown stage '$STAGE'. Use: encode | detect | both" >&2
    exit 1
    ;;
esac