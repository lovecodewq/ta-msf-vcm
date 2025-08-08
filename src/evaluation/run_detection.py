import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import json as _json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.kitti_dataset import KITTIDetectionDataset
from model.detection import DetectionModel


def to_tensor_only(image, target=None):
    import torchvision.transforms.functional as F
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def batch_features(feature_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = feature_list[0].keys()
    return {k: torch.cat([f[k] for f in feature_list], dim=0) for k in keys}


def load_features_for_batch(features_dir: Path, image_ids: List[str]) -> Dict[str, torch.Tensor]:
    feature_dicts = []
    for img_id in image_ids:
        feat_path = features_dir / f"{img_id}.pt"
        loaded = torch.load(feat_path, map_location='cpu')
        feature_dicts.append({k: v for k, v in loaded.items()})
    return batch_features(feature_dicts)


# --- mAP@0.5 utilities ---
import numpy as _np

def _ensure_boxes2d(boxes_arr):
    arr = _np.asarray(boxes_arr, dtype=_np.float32)
    if arr.size == 0:
        return _np.zeros((0, 4), dtype=_np.float32)
    if arr.ndim == 1:
        if arr.size == 4:
            return arr.reshape(1, 4)
        if arr.size % 4 == 0:
            return arr.reshape(-1, 4)
        return _np.zeros((0, 4), dtype=_np.float32)
    if arr.shape[-1] != 4:
        return _np.zeros((0, 4), dtype=_np.float32)
    return arr

def _calculate_iou(box, boxes):
    x1 = _np.maximum(box[0], boxes[:, 0])
    y1 = _np.maximum(box[1], boxes[:, 1])
    x2 = _np.minimum(box[2], boxes[:, 2])
    y2 = _np.minimum(box[3], boxes[:, 3])
    inter = _np.maximum(0, x2 - x1) * _np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return inter / _np.clip(union, 1e-9, None)

def _calculate_ap(predictions, targets, iou_threshold=0.5):
    if not any(len(t['boxes']) > 0 for t in targets):
        return 0.0
    all_pred_boxes, all_pred_scores, all_target_boxes = [], [], []
    n_pos = 0
    for pred, tgt in zip(predictions, targets):
        pboxes = _ensure_boxes2d(pred.get('boxes', []))
        pscores = _np.asarray(pred.get('scores', []), dtype=_np.float32).reshape(-1)
        tboxes = _ensure_boxes2d(tgt.get('boxes', []))
        all_pred_boxes.append(pboxes)
        all_pred_scores.append(pscores)
        all_target_boxes.append(tboxes)
        n_pos += int(tboxes.shape[0])
    if n_pos == 0:
        return 0.0
    if len(all_pred_boxes) == 0:
        return 0.0
    pred_boxes = _np.concatenate(all_pred_boxes, axis=0) if len(all_pred_boxes) else _np.zeros((0,4), _np.float32)
    pred_scores = _np.concatenate([s.reshape(-1) for s in all_pred_scores], axis=0) if len(all_pred_scores) else _np.zeros((0,), _np.float32)
    if pred_boxes.shape[0] == 0:
        return 0.0
    order = _np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    tp = _np.zeros(len(pred_boxes))
    fp = _np.zeros(len(pred_boxes))
    for i, pbox in enumerate(pred_boxes):
        max_iou = 0.0
        max_img_idx = -1
        max_gt_idx = -1
        for j, tboxes in enumerate(all_target_boxes):
            if len(tboxes) == 0:
                continue
            ious = _calculate_iou(pbox, tboxes)
            k = int(_np.argmax(ious))
            if ious[k] > max_iou:
                max_iou = float(ious[k])
                max_img_idx = j
                max_gt_idx = k
        if max_iou >= iou_threshold and max_img_idx >= 0 and max_gt_idx >= 0:
            tp[i] = 1
            all_target_boxes[max_img_idx] = _np.delete(all_target_boxes[max_img_idx], max_gt_idx, axis=0)
        else:
            fp[i] = 1
    ctp = _np.cumsum(tp)
    cfp = _np.cumsum(fp)
    recalls = ctp / max(n_pos, 1)
    precisions = ctp / _np.clip(ctp + cfp, 1e-9, None)
    ap = 0.0
    for t in _np.arange(0.0, 1.1, 0.1):
        if _np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = _np.max(precisions[recalls >= t])
        ap += p / 11.0
    return float(ap)

def _calculate_map(predictions, targets, num_classes: int, iou_threshold=0.5):
    class_aps = []
    for class_id in range(1, num_classes + 1):
        class_preds = []
        class_tgts = []
        for pred, tgt in zip(predictions, targets):
            pred_boxes = _ensure_boxes2d(pred.get('boxes', []))
            pred_scores = _np.asarray(pred.get('scores', []), dtype=_np.float32).reshape(-1)
            pred_labels = _np.asarray(pred['labels'], dtype=_np.int64)
            mask = pred_labels == class_id
            class_preds.append({'boxes': pred_boxes[mask] if pred_boxes.size else pred_boxes,
                                'scores': pred_scores[mask] if pred_scores.size else pred_scores})
            tgt_boxes = _ensure_boxes2d(tgt.get('boxes', []))
            tgt_labels = _np.asarray(tgt['labels'], dtype=_np.int64)
            tmask = tgt_labels == class_id
            class_tgts.append({'boxes': tgt_boxes[tmask] if tgt_boxes.size else tgt_boxes})
        ap = _calculate_ap(class_preds, class_tgts, iou_threshold)
        class_aps.append(ap)
    map50 = float(_np.mean(class_aps)) if class_aps else 0.0
    return map50, class_aps


def main():
    parser = argparse.ArgumentParser(description='Run detection on raw/reconstructed images or reconstructed features')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--detection_model', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['raw', 'reconstructed_images', 'reconstructed_features'], required=True)
    parser.add_argument('--images_dir', type=str, default=None, help='Directory with reconstructed images (.png) when mode=reconstructed_images')
    parser.add_argument('--features_dir', type=str, default=None, help='Directory with reconstructed feature tensors (.pt) when mode=reconstructed_features')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--manifest_out', type=str, default=None, help='Optional path to write a manifest.json summarizing outputs')
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load detection model
    checkpoint = torch.load(args.detection_model, map_location=device)
    num_classes = len(KITTIDetectionDataset.CLASSES) + 1
    model = DetectionModel(num_classes=num_classes, pretrained=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    override_dir = args.images_dir if args.mode == 'reconstructed_images' else None
    dataset = KITTIDetectionDataset(
        args.data_dir,
        split='test',
        transform=to_tensor_only,
        override_image_dir=override_dir
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    preds_out = []
    targs_out = []

    features_dir = Path(args.features_dir) if args.mode == 'reconstructed_features' else None

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f'Detecting ({args.mode})'):
            images = [img.to(device) for img in images]
            image_ids = [t.get('image_id') for t in targets]

            if args.mode == 'reconstructed_features':
                if features_dir is None:
                    raise ValueError('features_dir must be provided for mode=reconstructed_features')
                feats_batched = load_features_for_batch(features_dir, image_ids)
                feats_batched = {k: v.to(device) for k, v in feats_batched.items()}
                predictions = model.forward_from_features(images, feats_batched)
            else:
                predictions = model(images)

            for pred, tgt in zip(predictions, targets):
                preds_out.append({
                    'image_id': tgt.get('image_id'),
                    'boxes': pred['boxes'].cpu().numpy().tolist(),
                    'scores': pred['scores'].cpu().numpy().tolist(),
                    'labels': pred['labels'].cpu().numpy().tolist()
                })
                targs_out.append({
                    'image_id': tgt.get('image_id'),
                    'boxes': tgt['boxes'].cpu().numpy().tolist() if len(tgt['boxes']) else [],
                    'labels': tgt['labels'].cpu().numpy().tolist() if len(tgt['labels']) else []
                })

    # Compute mAP@0.5
    num_classes = len(KITTIDetectionDataset.CLASSES)
    map50, class_aps = _calculate_map(preds_out, targs_out, num_classes=num_classes, iou_threshold=0.5)

    output = {
        'mode': args.mode,
        'predictions': preds_out,
        'targets': targs_out,
        'metrics': {
            'map50': map50,
            'per_class_ap': {
                KITTIDetectionDataset.CLASSES[i]: float(class_aps[i]) for i in range(num_classes)
            }
        }
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f)
    logging.info(f'Saved detection outputs to {out_path}')

    # Optional manifest
    if args.manifest_out:
        manifest = {
            'kind': 'detection',
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'mode': args.mode,
            'data_dir': args.data_dir,
            'detection_model': args.detection_model,
            'output_json': str(out_path)
        }
        if args.mode == 'reconstructed_images' and args.images_dir:
            manifest['images_dir'] = str(Path(args.images_dir))
        if args.mode == 'reconstructed_features' and args.features_dir:
            manifest['features_dir'] = str(Path(args.features_dir))
        mpath = Path(args.manifest_out)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, 'w') as mf:
            _json.dump(manifest, mf)
        logging.info(f'Wrote detection manifest to {mpath}')


if __name__ == '__main__':
    main()

