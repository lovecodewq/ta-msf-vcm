#!/usr/bin/env bash
set -euo pipefail

# Configuration (override via env vars)
ENC_BIN=${ENC_BIN:-/home/wenqiangli/code/VVCSoftware_VTM/bin/EncoderAppStatic}
DEC_BIN=${DEC_BIN:-/home/wenqiangli/code/VVCSoftware_VTM/bin/DecoderAppStatic}

CFG_BASE=${CFG_BASE:-/home/wenqiangli/code/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg}
# 10-bit 4:4:4 override to match packed feature YUV layout
CFG_OVR=${CFG_OVR:-/home/wenqiangli/code/VVCSoftware_VTM/cfg/encoder_ai_10bit_444_override.cfg}

QP=${QP:-37}
DET_MODEL=${DET_MODEL:-checkpoints/detection/run_0.002000_16/best_model.pth}
OUT_DIR=${OUT_DIR:-evaluation_results/vtm_feature_qp${QP}}
DATA_DIR=${DATA_DIR:-data/processed/kitti}

# Ensure override cfg exists
if [[ ! -f "$CFG_OVR" ]]; then
  cat > "$CFG_OVR" << 'EOF'
InputBitDepth : 10
InputChromaFormat : 444
IntraPeriod : 1
EOF
  dos2unix "$CFG_OVR" >/dev/null 2>&1 || true
fi

# Avoid MKL threading incompatibility
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}

python -m src.evaluation.vtm_feature_anchor \
  --data_dir "$DATA_DIR" \
  --detection_model "$DET_MODEL" \
  --out_dir "$OUT_DIR" \
  --qp "$QP" \
  --encoder_bin "$ENC_BIN" \
  --decoder_bin "$DEC_BIN" \
  --encoder_cfg "$CFG_BASE" "$CFG_OVR"