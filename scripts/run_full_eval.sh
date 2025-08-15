#!/usr/bin/env bash
set -euo pipefail

# Arguments
DATA_DIR=data/processed/kitti
DET_CKPT=checkpoints/detection/run_0.002000_16/best_model.pth
# IMG_CKPT=checkpoints/factorized_prior/model_lambda_0.010.pth
IMG_CKPT=checkpoints/factorized_prior/model_lambda_0.100.pth
IMG_CKPT=checkpoints/factorized_prior/model_lambda_0.005.pth

# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature/run_20250808_023958_lambda_5.00e+00_lr_1.00e-04_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250808_155311_lambda_5.00e+00_lr_1.00e-04_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature/run_20250809_011931_lambda_5.00e-01_lr_1.00e-04_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature/run_20250809_135205_lambda_5.00e+01_lr_1.00e-04_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250810_003324_lambda_5.00e+00_w_1.00e-01_lr_1.00e-04_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250810_153818_lambda_5.00e-01_w_1.00e-02_lr_1.00e-04_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250810_223517_lambda_5.00e+01_w_1.00e-02_lr_1.00e-04_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250810_223517_lambda_5.00e+01_w_1.00e-02_lr_1.00e-04_bs_8/best_model_epoch_41_loss_0.3083_20250811_034735.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250812_023643_lambda_2.50e+00_w_1.00e-03_lr_1.00e-04_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250812_192016_lambda_1.00e+01_w_1.00e-03_lr_1.00e-04_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250813_133755_lambda_1.00e+01_w_1.00e-01_lr_1.00e-05_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250813_235639_lambda_2.00e+01_w_1.00e-03_lr_1.00e-05_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250814_145735_lambda_2.00e+01_w_1.00e-03_lr_1.00e-04_bs_8/best_model.pth
# FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250815_000502_lambda_3.00e+01_w_1.00e-02_lr_1.00e-04_bs_8/best_model.pth
FEAT_CKPT=checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250815_125856_lambda_3.00e+01_w_5.00e-02_lr_1.00e-04_bs_8/best_model.pth

# Timestamped output root to avoid overwriting previous runs
TS=$(date +"%Y%m%d_%H%M%S")
OUT_ROOT="evaluation_results/pipeline/run_${TS}"
# OUT_ROOT="evaluation_results/pipeline/run_20250808_013131"
# OUT_ROOT="evaluation_results/pipeline/run_20250809_004202"
# OUT_ROOT="evaluation_results/pipeline/run_20250809_105245"
# OUT_ROOT="evaluation_results/pipeline/run_20250811_124123"
# OUT_ROOT="evaluation_results/pipeline/run_20250812_125749"

# echo "Writing outputs to $OUT_ROOT"

RAW_OUT=$OUT_ROOT/raw
IMG_COMP_OUT=$OUT_ROOT/image_comp
FEAT_COMP_OUT=$OUT_ROOT/feature_comp

mkdir -p "$RAW_OUT" "$IMG_COMP_OUT" "$FEAT_COMP_OUT"

# echo "[1/4] Run detection on raw images"
# python -m src.evaluation.run_detection \
#   --data_dir "$DATA_DIR" \
#   --detection_model "$DET_CKPT" \
#   --mode raw \
#   --output_json "$RAW_OUT/preds.json"

# echo "[2/4] Compress images and reconstruct"
# python -m src.evaluation.run_image_compress \
#   --data_dir "$DATA_DIR" \
#   --checkpoint "$IMG_CKPT" \
#   --output_dir "$IMG_COMP_OUT"

# echo "[3/4] Run detection on reconstructed images"
# python -m src.evaluation.run_detection \
#   --data_dir "$DATA_DIR" \
#   --detection_model "$DET_CKPT" \
#   --mode reconstructed_images \
#   --images_dir "$IMG_COMP_OUT/images" \
#   --output_json "$IMG_COMP_OUT/preds.json"

echo "[4/4] Feature compress and detect"
python -m src.evaluation.run_feature_compress \
  --data_dir "$DATA_DIR" \
  --detection_model "$DET_CKPT" \
  --checkpoint "$FEAT_CKPT" \
  --output_dir "$FEAT_COMP_OUT" \
  --debug_shapes

echo "[4b] Decompress saved feature bitstreams (for portability)"
python -m src.evaluation.decompress_feature_bitstreams \
  --bitstreams_dir "$FEAT_COMP_OUT/bitstreams" \
  --checkpoint "$FEAT_CKPT" \
  --out_dir "$FEAT_COMP_OUT/features_from_bins"

# detection on features reconstructed from bins
echo "[4c] Run detection on reconstructed features"
python -m src.evaluation.run_detection \
  --data_dir "$DATA_DIR" \
  --detection_model "$DET_CKPT" \
  --mode reconstructed_features \
  --features_dir "$FEAT_COMP_OUT/features_from_bins" \
  --output_json "$FEAT_COMP_OUT/preds.json" \
  --manifest_out "$FEAT_COMP_OUT/manifest_detection_features.json" \
  --debug_shapes

echo "[viz] Aggregate and visualize"
MANIFEST_SUMMARY=$OUT_ROOT/summary_manifest.json
python - "$OUT_ROOT" "$DATA_DIR" <<'PY'
import json, os, sys
root=sys.argv[1]
data_dir=sys.argv[2]
raw=os.path.join(root,'raw','preds.json')
img=os.path.join(root,'image_comp','preds.json')
feat=os.path.join(root,'feature_comp','preds.json')
img_dir=os.path.join(root,'image_comp','images')
meta_img=os.path.join(root,'image_comp','metadata.json')
meta_feat=os.path.join(root,'feature_comp','metadata.json')
out=os.path.join(root,'summary_manifest.json')
m={
  'raw_json': raw,
  'img_json': img,
  'feat_json': feat,
  'raw_images_dir': os.path.join(data_dir, 'test', 'images'),
  'image_compression': {
    'images_dir': img_dir,
    'metadata': meta_img,
    'output_json': img,
  },
  'feature_compression': {
    'metadata': meta_feat,
    'output_json': feat,
  }
}
with open(out,'w') as f: json.dump(m,f)
print('Wrote manifest', out)
PY

python -m src.evaluation.visualize_results \
  --manifest "$MANIFEST_SUMMARY" \
  --out_dir "$OUT_ROOT/summary" \
  --num_samples 20

echo "Done. Outputs under $OUT_ROOT"

