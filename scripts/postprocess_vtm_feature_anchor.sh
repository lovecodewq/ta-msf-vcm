#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env or flags)
PREDS_JSON=${PREDS_JSON:-evaluation_results/vtm_feature_qp37/preds.json}
DATA_DIR=${DATA_DIR:-data/processed/kitti}
SPLIT=${SPLIT:-test}
OUT_JSON=${OUT_JSON:-}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--preds_json FILE] [--data_dir DIR] [--split SPLIT] [--out_json FILE]

Defaults:
  --preds_json  ${PREDS_JSON}
  --data_dir    ${DATA_DIR}
  --split       ${SPLIT}
  --out_json    <run_dir>/summary/bpp_vs_map50.json
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preds_json) PREDS_JSON="$2"; shift 2;;
    --data_dir) DATA_DIR="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --out_json) OUT_JSON="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1;;
  esac
done

PREDS_JSON=evaluation_results/vtm_feature_qp37/preds.json
DATA_DIR=data/processed/kitti
SPLIT=test

cmd=(python -m src.evaluation.postprocess_vtm_feature_anchor --preds_json "$PREDS_JSON" --data_dir "$DATA_DIR" --split "$SPLIT")
if [[ -n "$OUT_JSON" ]]; then
  cmd+=(--out_json "$OUT_JSON")
fi

"${cmd[@]}"

echo "Done."

