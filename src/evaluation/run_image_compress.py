import argparse
import logging
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import torchvision.transforms as T

from data.kitti_dataset import KITTIDetectionDataset
from data.transforms import create_transforms
from model.factorized_prior import FactorizedPrior


def to_tensor_only(image, target=None):
    import torchvision.transforms.functional as F
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser(description='Compress KITTI test images and write reconstructions to disk, also record BPP per image')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write reconstructed images (.png) and metadata json')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--force_resize', type=str, default=None, help='Override resize with H,W (e.g., 384,1280) to ensure batchable tensors')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load compression model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_cfg = checkpoint['config']['model']
    model = FactorizedPrior(n_hidden=model_cfg['n_hidden'], n_channels=model_cfg['n_channels']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Always use ToTensor-only (no resize) to preserve native geometry
    logging.info('Using ToTensor-only (no resize) for image compression inputs (compress at native size).')
    image_tf = to_tensor_only

    # Dataset: resized tensors for stackable batches
    dataset = KITTIDetectionDataset(
        args.data_dir,
        split='test',
        transform=image_tf,
        debug_transforms=True,
        debug_samples=3
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    out_root = Path(args.output_dir)
    img_out_dir = out_root / 'images'
    bin_out_dir = out_root / 'bitstreams'
    img_out_dir.mkdir(parents=True, exist_ok=True)
    bin_out_dir.mkdir(parents=True, exist_ok=True)

    per_image_stats = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Compressing & reconstructing images'):
            images_list = [img.to(device) for img in images]
            # Compress per image to support variable sizes without forced resize
            # (FactorizedPrior.compress expects a batch; handle single images in a loop)
            per_image_out = []
            for img in images_list:
                comp_i = model.compress(img.unsqueeze(0))
                per_image_out.append(comp_i)
            # No single shape: use per-image shape and input_size

            for i, (img, tgt) in enumerate(zip(images_list, targets)):
                image_id = tgt.get('image_id')
                comp_i = per_image_out[i]
                y_bytes = comp_i['y_strings'][0]
                bin_path = bin_out_dir / f'{image_id}.bin'
                bin_path.write_bytes(y_bytes)

                num_bits = bin_path.stat().st_size * 8
                h, w = img.shape[-2], img.shape[-1]
                bpp = float(num_bits) / float(h * w)

                # Decompress single image and save PNG
                rec = model.decompress([y_bytes], comp_i['shape'], comp_i['input_size'])['x_hat'].squeeze(0).clamp(0, 1).cpu()
                from torchvision.utils import save_image
                png_path = img_out_dir / f'{image_id}.png'
                save_image(rec, str(png_path))

                per_image_stats.append({
                    'image_id': image_id,
                    'bpp': bpp,
                    'reconstruction_path': str(png_path),
                    'bitstream_path': str(bin_path)
                })

    meta = {
        'mode': 'image_compression',
        'checkpoint': args.checkpoint,
        'stats': per_image_stats
    }
    meta_path = out_root / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    logging.info(f'Wrote reconstructions to {img_out_dir} and metadata to {meta_path}')

    # Optional manifest output (align with detection script optionality)
    manifest_path = out_root / 'manifest.json'
    manifest = {
        'kind': 'image_compression',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'data_dir': args.data_dir,
        'checkpoint': args.checkpoint,
        'images_dir': str(img_out_dir),
        'bitstreams_dir': str(bin_out_dir),
        'metadata': str(meta_path)
    }
    with open(manifest_path, 'w') as mf:
        json.dump(manifest, mf)
    logging.info(f'Wrote image compression manifest to {manifest_path}')


if __name__ == '__main__':
    main()

