#!/usr/bin/env bash
set -euo pipefail

# Arguments
DATA_DIR=data/processed/kitti

PIPELINE_ROOT="evaluation_results/pipeline/run_20250809_004202"

echo "Writing outputs to $PIPELINE_ROOT"

RAW_OUT=$PIPELINE_ROOT/raw
IMG_COMP_OUT=$PIPELINE_ROOT/image_comp
FEAT_COMP_OUT=$PIPELINE_ROOT/feature_comp

mkdir -p "$RAW_OUT" "$IMG_COMP_OUT" "$FEAT_COMP_OUT"

echo "[viz] Aggregate and visualize"
MANIFEST_SUMMARY=$PIPELINE_ROOT/summary_manifest.json
python - "$PIPELINE_ROOT" "$DATA_DIR" <<'PY'
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
  --out_dir "$PIPELINE_ROOT/summary" \
  --num_samples 20

echo "Done. Outputs under $PIPELINE_ROOT"

