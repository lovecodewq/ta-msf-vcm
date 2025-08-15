import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from data.kitti_dataset import KITTIDetectionDataset


def _ensure_boxes2d(boxes_arr):
    arr = np.asarray(boxes_arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if arr.ndim == 1:
        if arr.size == 4:
            return arr.reshape(1, 4)
        if arr.size % 4 == 0:
            return arr.reshape(-1, 4)
        return np.zeros((0, 4), dtype=np.float32)
    if arr.shape[-1] != 4:
        return np.zeros((0, 4), dtype=np.float32)
    return arr


def _calculate_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return inter / np.clip(union, 1e-9, None)


def _calculate_ap(predictions, targets, iou_threshold=0.5):
    if not any(len(t['boxes']) > 0 for t in targets):
        return 0.0
    all_pred_boxes, all_pred_scores, all_target_boxes = [], [], []
    n_pos = 0
    for pred, tgt in zip(predictions, targets):
        pboxes = _ensure_boxes2d(pred.get('boxes', []))
        pscores = np.asarray(pred.get('scores', []), dtype=np.float32).reshape(-1)
        tboxes = _ensure_boxes2d(tgt.get('boxes', []))
        all_pred_boxes.append(pboxes)
        all_pred_scores.append(pscores)
        all_target_boxes.append(tboxes)
        n_pos += int(tboxes.shape[0])
    if n_pos == 0:
        return 0.0
    if len(all_pred_boxes) == 0:
        return 0.0
    pred_boxes = np.concatenate(all_pred_boxes, axis=0) if len(all_pred_boxes) else np.zeros((0, 4), np.float32)
    pred_scores = (
        np.concatenate([s.reshape(-1) for s in all_pred_scores], axis=0)
        if len(all_pred_scores)
        else np.zeros((0,), np.float32)
    )
    if pred_boxes.shape[0] == 0:
        return 0.0
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    for i, pbox in enumerate(pred_boxes):
        max_iou = 0.0
        max_img_idx = -1
        max_gt_idx = -1
        for j, tboxes in enumerate(all_target_boxes):
            if len(tboxes) == 0:
                continue
            ious = _calculate_iou(pbox, tboxes)
            k = int(np.argmax(ious))
            if ious[k] > max_iou:
                max_iou = float(ious[k])
                max_img_idx = j
                max_gt_idx = k
        if max_iou >= iou_threshold and max_img_idx >= 0 and max_gt_idx >= 0:
            tp[i] = 1
            all_target_boxes[max_img_idx] = np.delete(all_target_boxes[max_img_idx], max_gt_idx, axis=0)
        else:
            fp[i] = 1
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    recalls = ctp / max(n_pos, 1)
    precisions = ctp / np.clip(ctp + cfp, 1e-9, None)
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            continue
        ap += float(np.max(precisions[recalls >= t])) / 11.0
    return float(ap)


def _calculate_map(predictions, targets, num_classes: int, iou_threshold=0.5):
    class_aps = []
    for class_id in range(1, num_classes + 1):
        class_preds = []
        class_tgts = []
        for pred, tgt in zip(predictions, targets):
            pred_boxes = _ensure_boxes2d(pred.get('boxes', []))
            pred_scores = np.asarray(pred.get('scores', []), dtype=np.float32).reshape(-1)
            pred_labels = np.asarray(pred['labels'], dtype=np.int64)
            mask = pred_labels == class_id
            class_preds.append({
                'boxes': pred_boxes[mask] if pred_boxes.size else pred_boxes,
                'scores': pred_scores[mask] if pred_scores.size else pred_scores,
            })
            tgt_boxes = _ensure_boxes2d(tgt.get('boxes', []))
            tgt_labels = np.asarray(tgt['labels'], dtype=np.int64)
            tmask = tgt_labels == class_id
            class_tgts.append({'boxes': tgt_boxes[tmask] if tgt_boxes.size else tgt_boxes})
        ap = _calculate_ap(class_preds, class_tgts, iou_threshold)
        class_aps.append(ap)
    map50 = float(np.mean(class_aps)) if class_aps else 0.0
    return map50, class_aps


def main():
    parser = argparse.ArgumentParser(description='Post-process VTM feature anchor preds.json to summary/bpp_vs_map50.json')
    parser.add_argument('--preds_json', type=str, required=True, help='Path to preds.json produced by vtm_feature_anchor.py')
    parser.add_argument('--data_dir', type=str, required=True, help='KITTI root to load ground-truths')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--out_json', type=str, default=None, help='Output summary JSON (default: <run_dir>/summary/bpp_vs_map50.json)')
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    preds_fp = Path(args.preds_json)
    if not preds_fp.exists():
        raise FileNotFoundError(f'preds_json not found: {preds_fp}')
    run_dir = preds_fp.parent
    out_json = Path(args.out_json) if args.out_json else (run_dir / 'summary' / 'bpp_vs_map50.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(preds_fp, 'r') as f:
        data = json.load(f)
    predictions: List[Dict] = data.get('predictions', [])
    per_image_bits: List[Dict] = data.get('per_image_bits', [])
    if not predictions:
        raise ValueError('No predictions found in preds.json')
    if not per_image_bits:
        raise ValueError('No per_image_bits found in preds.json')

    # Compute avg_bpp
    avg_bpp = float(np.mean([float(x.get('bpp', 0.0)) for x in per_image_bits]))

    # Build image_id -> target map from dataset
    ds = KITTIDetectionDataset(args.data_dir, split=args.split)
    id_to_target: Dict[str, Dict] = {}
    for _, tgt in ds:
        image_id = tgt.get('image_id')
        if image_id is None:
            continue
        id_to_target[str(image_id)] = {
            'boxes': tgt['boxes'].cpu().numpy().tolist() if len(tgt['boxes']) else [],
            'labels': tgt['labels'].cpu().numpy().tolist() if len(tgt['labels']) else [],
        }

    # Align targets to predictions order
    targets_aligned: List[Dict] = []
    missing = 0
    for pred in predictions:
        iid = str(pred.get('image_id'))
        tgt = id_to_target.get(iid)
        if tgt is None:
            missing += 1
            targets_aligned.append({'boxes': [], 'labels': []})
        else:
            targets_aligned.append(tgt)
    if missing:
        logging.warning(f"Missing ground truths for {missing} images out of {len(predictions)}; those will contribute zeros to AP.")

    # Compute mAP@0.5
    num_classes = len(KITTIDetectionDataset.CLASSES)
    map50, _ = _calculate_map(predictions, targets_aligned, num_classes=num_classes, iou_threshold=0.5)

    summary = {
        'feature_compression': {
            'avg_bpp': avg_bpp,
            'map50': map50,
            'model_type': 'vtm_feature_anchor',
            'checkpoint': str(run_dir),
        }
    }

    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f'Wrote summary to {out_json}')


if __name__ == '__main__':
    main()

