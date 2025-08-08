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

    # Build transforms to match training/test of feature compressor (ensures stackable dims and consistency)
    test_tf_cfg = comp_ckpt['config']['data']['test_transforms']
    image_tf = create_transforms(test_tf_cfg, split='val')

    dataset = KITTIDetectionDataset(args.data_dir, split='test', transform=image_tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    out_root = Path(args.output_dir)
    feat_out_dir = out_root / 'features'
    bin_out_dir = out_root / 'bitstreams'
    feat_out_dir.mkdir(parents=True, exist_ok=True)
    bin_out_dir.mkdir(parents=True, exist_ok=True)

    per_image_stats = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Compressing & reconstructing features'):
            images = [img.to(device) for img in images]
            batch = torch.stack(images, dim=0)

            # Extract FPN features for the batch
            fpn_features = det_model.get_fpn_features(batch)
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

            # Compress and decompress the whole batch
            compressed = comp_model.compress(features_list)
            decompressed = comp_model.decompress(compressed['strings'], p2_h, p2_w)
            recon_features_list = decompressed['features']  # list [p2..p6], each tensor shaped (N, C, H, W)

            # Compute per-image BPP from serialized file size for fidelity
            H, W = batch.shape[-2], batch.shape[-1]
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
                per_image_stats.append({
                    'image_id': image_id,
                    'bpp': float(bits_file) / float(H * W),
                    'bit': int(bits_file),
                    'bit_y': int(len(y_bytes) * 8),
                    'bit_z': int(len(z_bytes) * 8),
                    'bitstream_path': str(bin_path)
                })

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

