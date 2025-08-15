import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.kitti_dataset import KITTIDetectionDataset
from model.detection import DetectionModel
from utils.lmsfc_import import load_lmsfc_feature_compressor


def to_tensor_only(image, target=None):
    import torchvision.transforms.functional as F
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    return image, target


def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    parser = argparse.ArgumentParser(description='Encode/decode FPN features using trained L-MSFC anchor and run detection to get mAP and BPP')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--detection_model', type=str, required=True)
    parser.add_argument('--lmsfc_checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--third_party_root', type=str, default='thirparty')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load detection model
    det_ckpt = torch.load(args.detection_model, map_location=device)
    num_classes = len(KITTIDetectionDataset.CLASSES) + 1
    det_model = DetectionModel(num_classes=num_classes, pretrained=True).to(device)
    det_model.load_state_dict(det_ckpt['model_state_dict'])
    det_model.eval()

    # Load L-MSFC compressor with correct channel dims from checkpoint
    FeatureCompressor = load_lmsfc_feature_compressor(Path(args.third_party_root))
    state = torch.load(args.lmsfc_checkpoint, map_location=device)
    # Derive model dims (N, M) from checkpoint config if available
    N_init, M_init = 256, 128
    if isinstance(state, dict) and 'config' in state:
        try:
            mcfg = state['config'].get('model', {})
            N_init = int(mcfg.get('N', N_init))
            M_init = int(mcfg.get('M', M_init))
        except Exception:
            pass
    comp_model = FeatureCompressor(N=N_init, M=M_init).to(device)
    # Load weights (strict), fallback to non-strict with warning
    try:
        payload = state['model_state_dict'] if isinstance(state, dict) and 'model_state_dict' in state else state
        comp_model.load_state_dict(payload, strict=True)
    except Exception as e:
        logging.warning(f"Strict load_state_dict failed ({e}); retrying with strict=False")
        comp_model.load_state_dict(payload, strict=False)
    try:
        comp_model.update()
    except Exception:
        comp_model.update(force=True)
    comp_model.eval()

    # Dataset
    dataset = KITTIDetectionDataset(args.data_dir, split='test', transform=to_tensor_only, debug_transforms=True, debug_samples=3)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    out_dir = Path(args.out_dir)
    bit_dir = out_dir / 'bitstreams'
    bit_dir.mkdir(parents=True, exist_ok=True)

    per_image = []
    preds_out: List[Dict] = []
    targs_out: List[Dict] = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='L-MSFC encode/decode + detect'):
            images = [img.to(device) for img in images]
            fpn = det_model.get_fpn_features(images)
            features_list = [fpn['p2'], fpn['p3'], fpn['p4'], fpn['p5'], fpn['p6']]

            # Encode/decode per image for bit accounting
            N = features_list[0].shape[0]
            for i in range(N):
                img_id = targets[i].get('image_id')
                p2_h, p2_w = features_list[0].shape[-2:]
                out_path = bit_dir / f"{img_id}.bin"
                enc = comp_model.encode(features=[f[i:i+1] for f in features_list], output_path=str(out_path), p2_height=p2_h, p2_width=p2_w)
                dec = comp_model.decode(str(out_path))
                rec_feats = dec['features']  # list [p2..p6]
                # Run detection on single-image batch using reconstructed features
                single_imgs = [images[i]]
                rec_feat_dict = {k: v for k, v in zip(['p2','p3','p4','p5','p6'], rec_feats)}
                preds = det_model.forward_from_features(single_imgs, {k: v for k, v in rec_feat_dict.items()})
                bits = enc['bit']
                h, w = targets[i]['orig_size'] if 'orig_size' in targets[i] else single_imgs[0].shape[-2:]
                bpp = float(bits) / float(h * w)

                per_image.append({'image_id': img_id, 'bits': int(bits), 'bpp': float(bpp)})
                preds_out.append({
                    'image_id': img_id,
                    'boxes': preds[0]['boxes'].cpu().numpy().tolist(),
                    'scores': preds[0]['scores'].cpu().numpy().tolist(),
                    'labels': preds[0]['labels'].cpu().numpy().tolist(),
                })
                targs_out.append({
                    'image_id': img_id,
                    'boxes': targets[i]['boxes'].cpu().numpy().tolist() if len(targets[i]['boxes']) else [],
                    'labels': targets[i]['labels'].cpu().numpy().tolist() if len(targets[i]['labels']) else [],
                })

    # Compute mAP@0.5 using existing utility from run_detection
    from src.evaluation.run_detection import _calculate_map  # reuse
    num_classes = len(KITTIDetectionDataset.CLASSES)
    map50, class_aps = _calculate_map(preds_out, targs_out, num_classes=num_classes, iou_threshold=0.5)

    avg_bpp = float(sum(x['bpp'] for x in per_image) / max(1, len(per_image)))

    out = {
        'mode': 'reconstructed_features_lmsfc_anchor',
        'predictions': preds_out,
        'targets': targs_out,
        'per_image_bits': per_image,
        'metrics': {
            'map50': map50,
            'per_class_ap': {KITTIDetectionDataset.CLASSES[i]: float(class_aps[i]) for i in range(num_classes)}
        }
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'preds.json', 'w') as f:
        json.dump(out, f, indent=2)
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump({'summary': {'avg_bpp': avg_bpp, 'num_images': len(per_image)}, 'per_image': per_image}, f, indent=2)

    # Also write summary point for merging
    summary_point = {
        'lmsfc_feature_compression': {
            'avg_bpp': avg_bpp,
            'map50': map50,
            'checkpoint': args.lmsfc_checkpoint,
            'model_type': 'lmsfc_anchor'
        }
    }
    sum_dir = out_dir / 'summary'
    sum_dir.mkdir(parents=True, exist_ok=True)
    with open(sum_dir / 'bpp_vs_map50.json', 'w') as f:
        json.dump(summary_point, f, indent=2)
    logging.info(f"Saved L-MSFC anchors to {sum_dir / 'bpp_vs_map50.json'}")


if __name__ == '__main__':
    main()

