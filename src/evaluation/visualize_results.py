import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

from data.kitti_dataset import KITTIDetectionDataset
from utils.training_utils import get_lambda_from_checkpoint, get_weight_from_checkpoint


def draw_boxes(ax, img_np, preds, gt=None, threshold: float = 0.5, color_map: Dict[int, str] = None, title: str = ''):
    ax.imshow(img_np)
    ax.set_title(title)
    if preds is not None:
        boxes = np.array(preds['boxes'])
        scores = np.array(preds['scores'])
        labels = np.array(preds['labels'])
        keep = scores > threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=(color_map or {}).get(label, 'r'), facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, max(0, y1-5), f"{KITTIDetectionDataset.CLASSES[label-1]} {score:.2f}", color=(color_map or {}).get(label, 'r'), bbox=dict(facecolor='white', alpha=0.8))
    # Optional GT overlay in green
    if gt is not None and 'boxes' in gt and len(gt['boxes']):
        gt_boxes = np.array(gt['boxes'])
        gt_labels = np.array(gt.get('labels', []))
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            if gt_labels.size:
                lbl = int(gt_labels[i])
                ax.text(x1, y1, f"GT {KITTIDetectionDataset.CLASSES[lbl-1]}", color='g', bbox=dict(facecolor='white', alpha=0.6))
    ax.axis('off')


def load_raw_anchor(raw_json_path: str):
    """Load raw detection result JSON if present and return:
    - map_raw (float)
    - raw_map (dict image_id->pred) or None
    - raw_tgt_map (dict image_id->target) or None
    """
    p = Path(raw_json_path) if raw_json_path else None
    if not p or not p.exists():
        return float('nan'), None, None
    with open(p) as f:
        raw = json.load(f)
    def _to_map(d):
        return {item['image_id']: item for item in d['predictions']}
    def _to_tgt_map(d):
        return {item['image_id']: item for item in d['targets']}
    raw_map = _to_map(raw)
    raw_tgt_map = _to_tgt_map(raw)
    map_raw = float(raw.get('metrics', {}).get('map50', float('nan')))
    return map_raw, raw_map, raw_tgt_map


def load_image_anchor(img_json_path: str, meta_img_path: str):
    """Load image-compression outputs and metadata.
    Returns (map_img, img_map, img_tgt_map, bpp_img, img_ckpt).
    """
    p_img = Path(img_json_path) if img_json_path else None
    p_meta = Path(meta_img_path) if meta_img_path else None
    if not p_img or not p_img.exists() or not p_meta or not p_meta.exists():
        return float('nan'), None, None, None, None
    with open(p_img) as f:
        img = json.load(f)
    with open(p_meta) as f:
        meta_img = json.load(f)
    def _to_map(d):
        return {item['image_id']: item for item in d['predictions']}
    def _to_tgt_map(d):
        return {item['image_id']: item for item in d['targets']}
    img_map = _to_map(img)
    img_tgt_map = _to_tgt_map(img)
    bpp_img = {s['image_id']: s['bpp'] for s in meta_img.get('stats', [])}
    map_img = float(img.get('metrics', {}).get('map50', float('nan')))
    img_ckpt = meta_img.get('checkpoint')
    return map_img, img_map, img_tgt_map, bpp_img, img_ckpt


def load_feature_anchor(feat_json_path: str, meta_feat_path: str):
    """Load feature-compression outputs and metadata.
    Returns (map_feat, feat_map, feat_tgt_map, bpp_feat, feat_ckpt).
    """
    p_feat = Path(feat_json_path) if feat_json_path else None
    p_meta = Path(meta_feat_path) if meta_feat_path else None
    if not p_feat or not p_feat.exists() or not p_meta or not p_meta.exists():
        return float('nan'), None, None, None, None
    with open(p_feat) as f:
        feat = json.load(f)
    with open(p_meta) as f:
        meta_feat = json.load(f)
    def _to_map(d):
        return {item['image_id']: item for item in d['predictions']}
    def _to_tgt_map(d):
        return {item['image_id']: item for item in d['targets']}
    feat_map = _to_map(feat)
    feat_tgt_map = _to_tgt_map(feat)
    bpp_feat = {s['image_id']: s['bpp'] for s in meta_feat.get('stats', [])}
    map_feat = float(feat.get('metrics', {}).get('map50', float('nan')))
    feat_ckpt = meta_feat.get('checkpoint')
    return map_feat, feat_map, feat_tgt_map, bpp_feat, feat_ckpt


def main():
    parser = argparse.ArgumentParser(description='Aggregate and visualize detection results across modes')
    parser.add_argument('--raw_json', type=str, required=False)
    parser.add_argument('--img_json', type=str, required=False)
    parser.add_argument('--feat_json', type=str, required=False)
    parser.add_argument('--reconstructed_images_dir', type=str, required=False)
    parser.add_argument('--meta_img', type=str, required=False, help='metadata.json from image compression run')
    parser.add_argument('--meta_feat', type=str, required=False, help='metadata.json from feature compression run')
    parser.add_argument('--manifest', type=str, required=False, help='Optional manifest JSON with paths to all inputs')
    parser.add_argument('--image_model_type', type=str, required=False, help='Optional label for image compression model type (e.g., factorized_prior)')
    parser.add_argument('--feature_model_type', type=str, required=False, help='Optional label for feature compression model type (e.g., fused_feature, fused_feature_with_detection_loss)')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Resolve inputs either from explicit flags or the manifest
    if args.manifest:
        with open(args.manifest) as mf:
            m = json.load(mf)
        # Support a consolidated manifest that aggregates child manifests
        raw_json_path = m.get('raw_json')
        img_json_path = m.get('img_json') or (m.get('image_compression', {}).get('output_json'))
        feat_json_path = m.get('feat_json') or (m.get('feature_compression', {}).get('output_json'))
        reconstructed_images_dir = m.get('image_compression', {}).get('images_dir') or m.get('reconstructed_images_dir')
        raw_images_dir = m.get('raw_images_dir')
        meta_img_path = m.get('image_compression', {}).get('metadata') or m.get('meta_img')
        meta_feat_path = m.get('feature_compression', {}).get('metadata') or m.get('meta_feat')
    else:
        raw_json_path = args.raw_json
        img_json_path = args.img_json
        feat_json_path = args.feat_json
        reconstructed_images_dir = args.reconstructed_images_dir
        meta_img_path = args.meta_img
        meta_feat_path = args.meta_feat

    map_raw, raw_map, raw_tgt_map = load_raw_anchor(raw_json_path)
    map_img, img_map, img_tgt_map, bpp_img, img_ckpt = load_image_anchor(img_json_path, meta_img_path)
    map_feat, feat_map, feat_tgt_map, bpp_feat, feat_ckpt = load_feature_anchor(feat_json_path, meta_feat_path)
    # Derive model types if not provided
    image_model_type = args.image_model_type or ('factorized_prior' if img_ckpt and 'factorized_prior' in img_ckpt else 'image_model')
    if args.feature_model_type:
        feature_model_type = args.feature_model_type
    else:
        if feat_ckpt and 'with_detect_loss' in feat_ckpt:
            feature_model_type = 'fused_feature_with_detection_loss'
        else:
            feature_model_type = 'fused_feature'

    # raw_map, raw_tgt_map loaded via helper; img/feat maps loaded via helpers

    # Average BPP across all images (computed here for reuse in plot and summary export)
    avg_bpp_img = float(np.mean(list(bpp_img.values()))) if bpp_img else float('nan')
    avg_bpp_feat = float(np.mean(list(bpp_feat.values()))) if bpp_feat else float('nan')

    # Visualize a subset
    # Prefer raw image for 'Raw' panel to avoid scale mismatch
    image_root_raw = Path(raw_images_dir) if args.manifest and raw_images_dir else Path(reconstructed_images_dir) if reconstructed_images_dir else None
    image_root_rec = Path(reconstructed_images_dir) if reconstructed_images_dir else None
    shown = 0
    for image_id in raw_map.keys() if raw_map else img_map.keys() if img_map else feat_map.keys():
        if shown >= args.num_samples:
            break
        # Load raw and reconstructed for consistent display panels
        img_path_raw = image_root_raw / f'{image_id}.png' if image_root_raw else None
        img_path_rec = image_root_rec / f'{image_id}.png' if image_root_rec else None
        if img_path_rec is None or not img_path_rec.exists():
            # allow drawing raw/feature-only if reconstructed is missing
            pass
        # Fallback to reconstructed for raw panel if true raw not available
        pil_raw = Image.open(img_path_raw).convert('RGB') if img_path_raw and img_path_raw.exists() else (Image.open(img_path_rec).convert('RGB') if img_path_rec and img_path_rec.exists() else None)
        pil_rec = Image.open(img_path_rec).convert('RGB') if img_path_rec and img_path_rec.exists() else None
        if pil_raw is None and pil_rec is None:
            continue
        img_np_raw = np.array(pil_raw) if pil_raw is not None else None
        img_np_rec = np.array(pil_rec) if pil_rec is not None else None

        # Build available panels dynamically
        panels = []
        if raw_map is not None and img_np_raw is not None:
            panels.append(('Raw', img_np_raw, raw_map.get(image_id), raw_tgt_map.get(image_id) if raw_tgt_map else None))
        if img_map is not None and img_np_rec is not None:
            title = 'Image comp'
            if bpp_img and image_id in bpp_img:
                title = f'Image comp (BPP {bpp_img[image_id]:.4f})'
            panels.append((title, img_np_rec, img_map.get(image_id), img_tgt_map.get(image_id) if img_tgt_map else None))
        if feat_map is not None and img_np_raw is not None:
            title = 'Feature comp'
            if bpp_feat and image_id in bpp_feat:
                title = f'Feature comp (BPP {bpp_feat[image_id]:.4f})'
            panels.append((title, img_np_raw, feat_map.get(image_id), raw_tgt_map.get(image_id) if raw_tgt_map else None))
        if not panels:
            continue

        fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 8))
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, img_np, preds, gt) in zip(axes, panels):
            draw_boxes(ax, img_np, preds, gt=gt, title=title)
        plt.tight_layout()
        fig.savefig(vis_dir / f'{image_id}.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        shown += 1

    # Optionally compute simple mAP@0.5 if desired later; here we only plot BPP vs mAP if JSONs already include metrics
    logging.info(f'Saved {shown} comparison visualizations to {vis_dir}')

    # Plot BPP vs mAP50 scatter for image vs feature compression
    try:
        fig2, ax = plt.subplots(figsize=(8, 6))
        if not np.isnan(avg_bpp_img) and not np.isnan(map_img):
            ax.scatter([avg_bpp_img], [map_img], label='Image compression', marker='o', s=120)
        if not np.isnan(avg_bpp_feat) and not np.isnan(map_feat):
            ax.scatter([avg_bpp_feat], [map_feat], label='Feature compression', marker='s', s=120)
        # Baseline: raw image mAP@0.5 as a horizontal reference line
        if not np.isnan(map_raw):
            ax.axhline(y=map_raw, color='gray', linestyle='--', linewidth=1.5, label=f'Raw baseline (mAP@0.5={map_raw:.3f})')
        ax.set_xlabel('Bits Per Pixel (avg)')
        ax.set_ylabel('mAP@0.5')
        ax.set_title('BPP vs mAP@0.5')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        fig2.savefig(out_dir / 'bpp_vs_map50.png', bbox_inches='tight', dpi=150)
        plt.close(fig2)
        logging.info(f'Saved BPP vs mAP@0.5 plot to {out_dir / "bpp_vs_map50.png"}')
    except Exception as e:
        logging.warning(f'Failed to plot BPP vs mAP@0.5: {e}')

    # Export the plotted points for downstream analysis
    try:
        lambda_img = get_lambda_from_checkpoint(img_ckpt) if img_ckpt else None
        lambda_feat = get_lambda_from_checkpoint(feat_ckpt) if feat_ckpt else None
        weight_feat = get_weight_from_checkpoint(feat_ckpt) if feat_ckpt else None
        summary_points = {
            'image_compression': {
                'avg_bpp': avg_bpp_img,
                'map50': map_img,
                'checkpoint': img_ckpt,
                'model_type': image_model_type,
                'lambda': lambda_img,
            },
            'feature_compression': {
                'avg_bpp': avg_bpp_feat,
                'map50': map_feat,
                'checkpoint': feat_ckpt,
                'model_type': feature_model_type,
                'lambda': lambda_feat,
                'detection_loss_weight': weight_feat,
            },
            'raw': {
                'map50': map_raw
            }
        }
        with open(out_dir / 'bpp_vs_map50.json', 'w') as jf:
            json.dump(summary_points, jf, indent=2)
        logging.info(f"Saved BPP vs mAP@0.5 data points to {out_dir / 'bpp_vs_map50.json'}")
    except Exception as e:
        logging.warning(f'Failed to save BPP vs mAP@0.5 points: {e}')


if __name__ == '__main__':
    main()

