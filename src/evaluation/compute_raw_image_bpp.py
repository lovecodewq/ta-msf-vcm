import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

from PIL import Image
import numpy as np


def find_image_dir_from_config(cfg: Dict, split: str) -> Path:
    # Prefer explicit root_dir used by our detection/fused-feature configs
    root_dir = cfg.get('data', {}).get('root_dir')
    if root_dir:
        return Path(root_dir) / split / 'images'
    # Fallback: derive root from *_list if present
    list_key = f'{split}_list'
    lst = cfg.get('data', {}).get(list_key)
    if lst:
        # Expect path like data/processed/kitti/<split>.txt → use its parent parent
        p = Path(lst)
        try:
            root = p.parent.parent
            return root / split / 'images'
        except Exception:
            pass
    # Last resort: assume data/processed/kitti/<split>/images
    return Path('data/processed/kitti') / split / 'images'


def compute_bpp_for_dir(images_dir: Path) -> Dict:
    images = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')))
    if not images:
        raise FileNotFoundError(f'No images found in {images_dir}')
    per_image = []
    for img_path in images:
        # Use PIL to get H,W
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            w, h = im.size
        n_pixels = h * w
        bytes_size = img_path.stat().st_size
        bpp = (bytes_size * 8.0) / float(n_pixels)
        per_image.append({
            'image_id': img_path.stem,
            'path': str(img_path),
            'width': w,
            'height': h,
            'filesize_bytes': int(bytes_size),
            'bpp': float(bpp),
        })
    bpps = [x['bpp'] for x in per_image]
    summary = {
        'num_images': len(per_image),
        'avg_bpp': float(np.mean(bpps)),
        'median_bpp': float(np.median(bpps)),
        'min_bpp': float(np.min(bpps)),
        'max_bpp': float(np.max(bpps)),
    }
    return {'images_dir': str(images_dir), 'summary': summary, 'per_image': per_image}


def main():
    parser = argparse.ArgumentParser(description='Compute raw PNG/JPEG BPP for a dataset split using a training config')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train','val','test'])
    parser.add_argument('--out_json', type=str, required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = json.load(f) if cfg_path.suffix.lower() == '.json' else __import__('yaml').safe_load(f)

    images_dir = find_image_dir_from_config(cfg, args.split)
    logging.info(f'Computing raw image BPP for split={args.split} at {images_dir}')

    result = compute_bpp_for_dir(images_dir)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logging.info(f'Wrote BPP results to {out_path}')


if __name__ == '__main__':
    main()

