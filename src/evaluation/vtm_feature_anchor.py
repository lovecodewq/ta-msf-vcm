import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from model.detection import DetectionModel
from data.kitti_dataset import KITTIDetectionDataset
from .vtm_utils import run_encoder, run_decoder


def to_tensor_only(image, target=None):
    import torchvision.transforms.functional as F
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def extract_fpn_features(model: DetectionModel, images: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        feats = model.get_fpn_features(images)
    return feats


def pack_features_10bit(feats: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    packed = {}
    for k, v in feats.items():
        # Normalize per-tensor to [-1, 1] using 3-sigma clipping, then map to [0, 1023]
        x = v.cpu().numpy()
        mean = x.mean()
        std = x.std() + 1e-6
        x = np.clip((x - mean) / (3.0 * std), -1.0, 1.0)
        x = ((x + 1.0) * 0.5) * 1023.0
        packed[k] = x.astype(np.uint16)
    return packed


def unpack_features_10bit(packed: Dict[str, np.ndarray], stats: Dict[str, Dict[str, float]]) -> Dict[str, torch.Tensor]:
    feats = {}
    for k, arr in packed.items():
        arr_f = arr.astype(np.float32) / 1023.0
        arr_f = (arr_f * 2.0) - 1.0
        mean = stats[k]['mean']
        std = stats[k]['std']
        x = arr_f * (3.0 * std) + mean
        feats[k] = torch.from_numpy(x)
    return feats


def write_yuv444_10bit(arr: np.ndarray, out_path: Path) -> None:
    # arr: H x W x C in [0, 1023], uint16, C=3
    assert arr.dtype == np.uint16 and arr.ndim == 3 and arr.shape[2] == 3
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # VTM expects little-endian 10-bit packed in 16-bit words
    with open(out_path, 'wb') as f:
        for c in range(3):
            arr[..., c].tofile(f)


def read_yuv444_10bit(path: Path, width: int, height: int) -> np.ndarray:
    n = width * height
    with open(path, 'rb') as f:
        y = np.fromfile(f, dtype=np.uint16, count=n).reshape(height, width)
        u = np.fromfile(f, dtype=np.uint16, count=n).reshape(height, width)
        v = np.fromfile(f, dtype=np.uint16, count=n).reshape(height, width)
    return np.stack([y, u, v], axis=-1)


def main():
    parser = argparse.ArgumentParser(description='VTM feature anchor: pack 10-bit, VTM encode/decode, run detection heads')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--detection_model', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--qp', type=int, required=True)
    parser.add_argument('--encoder_bin', type=str, default='EncoderApp')
    parser.add_argument('--decoder_bin', type=str, default='DecoderApp')
    parser.add_argument('--encoder_cfg', type=str, nargs='*', default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.detection_model, map_location=device)
    num_classes = len(KITTIDetectionDataset.CLASSES) + 1
    model = DetectionModel(num_classes=num_classes, pretrained=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = KITTIDetectionDataset(args.data_dir, split='test', transform=to_tensor_only)
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lambda b: tuple(zip(*b)))

    out_dir = Path(args.out_dir)
    yuv_dir = out_dir / 'yuv'
    bit_dir = out_dir / 'bitstreams'
    dec_dir = out_dir / 'decoded_yuv'
    rec_feat_dir = out_dir / 'reconstructed_features'
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ensure subdirectories exist for I/O
    yuv_dir.mkdir(parents=True, exist_ok=True)
    bit_dir.mkdir(parents=True, exist_ok=True)
    dec_dir.mkdir(parents=True, exist_ok=True)
    rec_feat_dir.mkdir(parents=True, exist_ok=True)

    per_image = []
    for images, targets in dl:
        images = [im.to(device) for im in images]
        image_ids = [t['image_id'] for t in targets]
        feats = extract_fpn_features(model, images)
        # Pack + quantize per image using concatenated channels of p2 as proxy image for VTM I-frame
        p2 = feats['p2'].cpu().numpy()  # N x C x H x W
        N, C, H, W = p2.shape
        # Compute stats per image for later dequant
        stats = {k: {'mean': float(v.cpu().mean().item()), 'std': float(v.cpu().std().item()) + 1e-6} for k, v in feats.items()}
        packed = pack_features_10bit(feats)
        for i in range(N):
            # Create pseudo-YUV444 10-bit image by stacking 3 channels from p2 (or tile if C<3)
            if C >= 3:
                pseudo = np.stack([
                    packed['p2'][i, 0],
                    packed['p2'][i, 1],
                    packed['p2'][i, 2],
                ], axis=-1)
            else:
                ch = packed['p2'][i, 0]
                pseudo = np.stack([ch, ch, ch], axis=-1)
            yuv_path = yuv_dir / f'{image_ids[i]}.yuv'
            write_yuv444_10bit(pseudo, yuv_path)
            bit_path = bit_dir / f'{image_ids[i]}.bin'
            run_encoder(yuv_path, W, H, 10, '444', args.qp, bit_path, encoder_bin=args.encoder_bin, encoder_cfg=[Path(c) for c in args.encoder_cfg] if args.encoder_cfg else None)
            dec_yuv = dec_dir / f'{image_ids[i]}_dec.yuv'
            run_decoder(bit_path, dec_yuv, decoder_bin=args.decoder_bin)
            dec = read_yuv444_10bit(dec_yuv, W, H)
            # Replace first 3 channels in p2 with decoded, keep others as-is
            rec_feats = {}
            for k, arr in packed.items():
                rec_feats[k] = arr.copy()
            if C >= 3:
                rec_feats['p2'][i, 0] = dec[..., 0]
                rec_feats['p2'][i, 1] = dec[..., 1]
                rec_feats['p2'][i, 2] = dec[..., 2]
            else:
                rec_feats['p2'][i, 0] = dec[..., 0]
            # Dequant and reconstruct tensors
            rec_tensors = unpack_features_10bit(rec_feats, stats)
            rec_tensors = {k: v.to(device) for k, v in rec_tensors.items()}
            # Run downstream detection heads
            with torch.no_grad():
                preds = model.forward_from_features([images[i]], {k: v[i:i+1] for k, v in rec_tensors.items()})
            bits = bit_path.stat().st_size * 8
            h, w = targets[i]['orig_size'] if 'orig_size' in targets[i] else images[i].shape[-2:]
            bpp = bits / float(h * w)
            per_image.append({
                'image_id': image_ids[i],
                'bits': int(bits),
                'bpp': float(bpp),
                'pred': {
                    'boxes': preds[0]['boxes'].cpu().numpy().tolist(),
                    'scores': preds[0]['scores'].cpu().numpy().tolist(),
                    'labels': preds[0]['labels'].cpu().numpy().tolist(),
                }
            })
            # Save reconstructed feature tensors for optional later reuse
            rec_path = rec_feat_dir / f'{image_ids[i]}.pt'
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v[i:i+1].cpu() for k, v in rec_tensors.items()}, rec_path)

    # Aggregate to preds.json compatible with visualize/run_detection outputs
    preds_out = []
    targs_out = []
    for item in per_image:
        preds_out.append({'image_id': item['image_id'], **item['pred']})
    # ground-truths are not saved here; optional post-step can merge
    out = {
        'mode': 'reconstructed_features_vtm_anchor',
        'predictions': preds_out,
        'per_image_bits': [{'image_id': x['image_id'], 'bits': x['bits'], 'bpp': x['bpp']} for x in per_image],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'preds.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()

