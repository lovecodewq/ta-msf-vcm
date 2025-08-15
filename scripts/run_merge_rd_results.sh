#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env or flags)
PIPELINE_DIR=${PIPELINE_DIR:-evaluation_results/pipeline}
VTM_IMAGE_ANCHOR_DIR=${VTM_IMAGE_ANCHOR_DIR:-evaluation_results/vtm_image_qp32}
VTM_FEATURE_ANCHOR_FILES=${VTM_FEATURE_ANCHOR_FILES:-evaluation_results/vtm_feature_qp37/summary/bpp_vs_map50.json}
LMSFC_FEATURE_ANCHOR_FILES=${LMSFC_FEATURE_ANCHOR_FILES:-}
LMSFC_FEATURE_ANCHOR_DIR=${LMSFC_FEATURE_ANCHOR_DIR:-evaluation_results/lmsfc_anchor}
OUT_DIR=${OUT_DIR:-evaluation_results/merged}
PATTERN=${PATTERN:-bpp_vs_map50.json}
OUT_JSON_DEFAULT=${OUT_DIR}/anchors.json
OUT_PLOT_DEFAULT=${OUT_DIR}/bpp_vs_map50_merged.png
OUT_JSON=${OUT_JSON:-$OUT_JSON_DEFAULT}
OUT_PLOT=${OUT_PLOT:-$OUT_PLOT_DEFAULT}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--pipeline_dir DIR] [--vtm_image_anchor_dir DIR] [--vtm_feature_anchor_files FILES] [--lmsfc_feature_anchor_dir DIR] [--out_dir DIR] [--pattern NAME] [--out_json FILE] [--out_plot FILE]

Defaults:
  --pipeline_dir             ${PIPELINE_DIR} (set to empty to skip)
  --vtm_image_anchor_dir     (none)
  --vtm_feature_anchor_files (space-separated list; none)
  --lmsfc_feature_anchor_dir   ${LMSFC_FEATURE_ANCHOR_DIR}
  --out_dir       ${OUT_DIR}
  --pattern       ${PATTERN}
  --out_json      ${OUT_JSON}
  --out_plot      ${OUT_PLOT}

Examples:
  $(basename "$0") --vtm_image_anchor_dir evaluation_results/vtm_image_qp32
  $(basename "$0") --pipeline_dir evaluation_results/pipeline --vtm_image_anchor_dir evaluation_results/vtm_image_qp32 --out_dir evaluation_results/merged
  $(basename "$0") --vtm_feature_anchor_files evaluation_results/vtm_feature_qp37/summary/bpp_vs_map50.json --out_dir evaluation_results/merged
  $(basename "$0") --lmsfc_feature_anchor_dir evaluation_results/lmsfc_anchor --out_dir evaluation_results/merged
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline_dir)
      PIPELINE_DIR="$2"; shift 2;;
    --vtm_image_anchor_dir)
      VTM_IMAGE_ANCHOR_DIR="$2"; shift 2;;
    --vtm_feature_anchor_files)
      # Accept space-separated list as a single argument or pass multiple times
      VTM_FEATURE_ANCHOR_FILES="$2"; shift 2;;
    --lmsfc_feature_anchor_dir)
      LMSFC_FEATURE_ANCHOR_DIR="$2"; shift 2;;
    --out_dir)
      OUT_DIR="$2"; shift 2;;
    --pattern)
      PATTERN="$2"; shift 2;;
    --out_json)
      OUT_JSON="$2"; shift 2;;
    --out_plot)
      OUT_PLOT="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2; usage; exit 1;;
  esac
done

mkdir -p "$OUT_DIR"

echo "Merging RD results"
echo "Writing JSON to: $OUT_JSON"
echo "Writing plot to: $OUT_PLOT"

cmd=(python -m src.evaluation.merge_rd_results)
if [[ -n "${PIPELINE_DIR}" ]]; then
  cmd+=(--inputs "$PIPELINE_DIR")
fi
if [[ -n "${VTM_IMAGE_ANCHOR_DIR}" ]]; then
  cmd+=(--vtm_image_anchor_dir "$VTM_IMAGE_ANCHOR_DIR")
fi
if [[ -n "${VTM_FEATURE_ANCHOR_FILES}" ]]; then
  # Split space-separated list into array
  read -r -a feat_files <<< "$VTM_FEATURE_ANCHOR_FILES"
  for f in "${feat_files[@]}"; do
    cmd+=(--vtm_feature_anchor_files "$f")
  done
fi

if [[ -n "${LMSFC_FEATURE_ANCHOR_DIR}" ]]; then
  cmd+=(--lmsfc_feature_anchor_dir "$LMSFC_FEATURE_ANCHOR_DIR")
fi
cmd+=(--pattern "$PATTERN" --out_json "$OUT_JSON" --out_plot "$OUT_PLOT")

"${cmd[@]}"

echo "Done."

