import argparse
import json
import logging
from pathlib import Path
import os
import struct

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from data.kitti_dataset import KITTIDetectionDataset
from data.transforms import create_transforms
from model.detection import DetectionModel
from model.joint_autoregress_fpn_compressor import JointAutoregressFPNCompressor


def to_tensor_only(image, target=None):
    import torchvision.transforms.functional as F
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser(description='Compress FPN features for KITTI test images and write reconstructed features to disk with BPPs')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--detection_model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--debug_shapes', action='store_true', help='Log detailed image/feature shapes for debugging')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load detection model for feature extraction geometry
    det_ckpt = torch.load(args.detection_model, map_location=device)
    num_classes = len(KITTIDetectionDataset.CLASSES) + 1
    det_model = DetectionModel(num_classes=num_classes, pretrained=True).to(device)
    det_model.load_state_dict(det_ckpt['model_state_dict'])
    det_model.eval()

    # Load feature compressor and its config (for resize/test transforms)
    comp_ckpt = torch.load(args.checkpoint, map_location=device)
    comp_model = JointAutoregressFPNCompressor(
        N=comp_ckpt['config']['model']['n_latent'],
        M=comp_ckpt['config']['model']['n_hidden']
    ).to(device)
    comp_model.load_state_dict(comp_ckpt['model_state_dict'])
    # Build entropy CDF tables for compression
    try:
        comp_model.update()
    except Exception:
        comp_model.update(force=True)
    comp_model.eval()

    # Best practice for feature path: no external resize/normalize; let model transform handle it
    dataset = KITTIDetectionDataset(
        args.data_dir,
        split='test',
        transform=to_tensor_only,
        debug_transforms=True,
        debug_samples=3
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    out_root = Path(args.output_dir)
    feat_out_dir = out_root / 'features'
    bin_out_dir = out_root / 'bitstreams'
    feat_out_dir.mkdir(parents=True, exist_ok=True)
    bin_out_dir.mkdir(parents=True, exist_ok=True)

    per_image_stats = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc='Compressing & reconstructing features')):
            images = [img.to(device) for img in images]

            # Extract FPN features for the batch
            fpn_features = det_model.get_fpn_features(images)
            # Keep p2..p6 order in a list as expected by compressor
            features_list = [
                fpn_features['p2'],
                fpn_features['p3'],
                fpn_features['p4'],
                fpn_features['p5'],
                fpn_features['p6'],
            ]

            # P2 dims for decompress API (batch spatial dims)
            p2_h, p2_w = fpn_features['p2'].shape[-2:]

            if args.debug_shapes and batch_idx < 5:
                img_sizes = [tuple(img.shape[-2:]) for img in images]
                feat_sizes = {k: tuple(v.shape[-2:]) for k, v in fpn_features.items()}
                logging.info(f"[DEBUG] batch {batch_idx} image_sizes={img_sizes} p2..p6={feat_sizes} p2_hw={(p2_h,p2_w)}")

            # Compress and decompress the whole batch
            compressed = comp_model.compress(features_list)
            decompressed = comp_model.decompress(compressed['strings'], p2_h, p2_w)
            recon_features_list = decompressed['features']  # list [p2..p6], each tensor shaped (N, C, H, W)

            # Compute per-image BPP from serialized file size for fidelity (use original per-image size)
            y_strings, z_strings = compressed['strings']

            # Save per image reconstructed features as .pt dicts
            for i, tgt in enumerate(targets):
                image_id = tgt.get('image_id')
                single_feats = {
                    'p2': recon_features_list[0][i:i+1].cpu(),
                    'p3': recon_features_list[1][i:i+1].cpu(),
                    'p4': recon_features_list[2][i:i+1].cpu(),
                    'p5': recon_features_list[3][i:i+1].cpu(),
                    'p6': recon_features_list[4][i:i+1].cpu(),
                }
                torch.save(single_feats, feat_out_dir / f'{image_id}.pt')

                # Serialize y and z streams with a tiny header for reproducible filesize
                y_bytes = y_strings[i]
                z_bytes = z_strings[i]
                bin_path = bin_out_dir / f'{image_id}.bin'
                with open(bin_path, 'wb') as bf:
                    # magic
                    bf.write(b'FPN1')
                    # header: p2_h, p2_w, len_y, len_z as uint32 LE
                    bf.write(struct.pack('<IIII', p2_h, p2_w, len(y_bytes), len(z_bytes)))
                    bf.write(y_bytes)
                    bf.write(z_bytes)
                bits_file = os.path.getsize(bin_path) * 8
                H, W = images[i].shape[-2], images[i].shape[-1]
                per_image_stats.append({
                    'image_id': image_id,
                    'bpp': float(bits_file) / float(H * W),
                    'bit': int(bits_file),
                    'bit_y': int(len(y_bytes) * 8),
                    'bit_z': int(len(z_bytes) * 8),
                    'bitstream_path': str(bin_path)
                })

            if args.debug_shapes and batch_idx < 5:
                sizes = [tuple(img.shape[-2:]) for img in images]
                logging.info(f"[DEBUG] batch {batch_idx} y_bits={[len(y)*8 for y in y_strings]} z_bits={[len(z)*8 for z in z_strings]} per_image_sizes={sizes}")

    meta = {
        'mode': 'feature_compression',
        'checkpoint': args.checkpoint,
        'stats': per_image_stats
    }
    meta_path = out_root / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    logging.info(f'Wrote reconstructed features to {feat_out_dir} and metadata to {meta_path}')

    # Manifest output
    manifest_path = out_root / 'manifest.json'
    manifest = {
        'kind': 'feature_compression',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'data_dir': args.data_dir,
        'detection_model': args.detection_model,
        'checkpoint': args.checkpoint,
        'features_dir': str(feat_out_dir),
        'bitstreams_dir': str(bin_out_dir),
        'metadata': str(meta_path)
    }
    with open(manifest_path, 'w') as mf:
        json.dump(manifest, mf)
    logging.info(f'Wrote feature compression manifest to {manifest_path}')


if __name__ == '__main__':
    main()

