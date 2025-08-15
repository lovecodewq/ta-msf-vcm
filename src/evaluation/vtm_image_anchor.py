import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .vtm_utils import run_encoder, run_decoder, ffmpeg_rgb_to_yuv, ffmpeg_yuv_to_png


def _load_config_any(path: Path):
    with open(path, 'r') as f:
        if path.suffix.lower() in ('.yaml', '.yml'):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def _find_root_dir_from_config(cfg: dict) -> Path | None:
    root_dir = cfg.get('data', {}).get('root_dir')
    if root_dir:
        return Path(root_dir)
    # Try infer from train/val/test lists
    for key in ['test_list', 'val_list', 'train_list']:
        p = cfg.get('data', {}).get(key)
        if p:
            pth = Path(p)
            # Expect .../kitti/<split>.txt → root is parent of parent
            try:
                return pth.parent.parent
            except Exception:
                continue
    return None


def _find_images_dir_from_config(cfg: dict, split: str) -> Path | None:
    root = _find_root_dir_from_config(cfg)
    if root:
        candidate = root / split / 'images'
        if candidate.exists():
            return candidate
    return None


def rgb_to_vtm_friendly_yuv(img_path: Path, out_yuv: Path, pix_fmt: str) -> tuple[int, int, int, str]:
    # Convert to requested YUV format using ffmpeg
    with Image.open(img_path) as im:
        im = im.convert('RGB')
        w, h = im.size
    ffmpeg_rgb_to_yuv(img_path, out_yuv, pix_fmt=pix_fmt)
    bitdepth = 10 if '10' in pix_fmt else 8
    chroma = '420'
    return w, h, bitdepth, chroma


def yuv_to_png_with_ffmpeg(in_yuv: Path, width: int, height: int, pix_fmt: str, out_png: Path) -> None:
    ffmpeg_yuv_to_png(in_yuv, width, height, pix_fmt, out_png)


def main():
    parser = argparse.ArgumentParser(description='VTM image anchor: encode/decode images and compute BPP')
    parser.add_argument('--images_dir', type=str, required=False, help='Directory of source PNG/JPEG images (overrides --config)')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--qp', type=int, required=True)
    parser.add_argument('--encoder_bin', type=str, default='EncoderApp')
    parser.add_argument('--decoder_bin', type=str, default='DecoderApp')
    parser.add_argument('--encoder_cfg', type=str, nargs='*', default=None)
    parser.add_argument('--pix_fmt', type=str, default='yuv420p', choices=['yuv420p', 'yuv420p10le'], help='Raw YUV pixel format for ffmpeg I/O; must match VTM cfg/output')
    parser.add_argument('--run_detection', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None, help='KITTI root (overrides --config)')
    parser.add_argument('--config', type=str, default=None, help='Training/eval config to infer dataset paths')
    parser.add_argument('--split', type=str, default='test', choices=['train','val','test'])
    parser.add_argument('--detection_model', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
    # Resolve images_dir and data_dir from config if provided
    images_dir = None
    data_dir = None
    if args.config:
        cfg = _load_config_any(Path(args.config))
        images_dir = _find_images_dir_from_config(cfg, args.split)
        data_dir = _find_root_dir_from_config(cfg)
    if args.images_dir:
        images_dir = Path(args.images_dir)
    if args.data_dir:
        data_dir = Path(args.data_dir)
    if images_dir is None or not images_dir.exists():
        raise FileNotFoundError('Could not resolve images_dir. Provide --images_dir or a valid --config with data.root_dir and split images.')
    if args.run_detection and (data_dir is None or not data_dir.exists()):
        raise FileNotFoundError('run_detection requires dataset root. Provide --data_dir or a valid --config with data.root_dir')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')))
    per_image = []
    dec_images_dir = out_dir / 'decoded_png'
    bitstreams_dir = out_dir / 'bitstreams'
    yuv_dir = out_dir / 'yuv'
    # Pixel format is explicit; default yuv420p to match VTM defaults when no cfg
    use_pix_fmt = args.pix_fmt
    for img in image_paths:
        yuv_path = yuv_dir / f'{img.stem}.yuv'
        w, h, bd, cf = rgb_to_vtm_friendly_yuv(img, yuv_path, use_pix_fmt)
        # Use padded even dimensions for VTM encode/decode when using 4:2:0
        pad_w = int(np.ceil(w / 2) * 2)
        pad_h = int(np.ceil(h / 2) * 2)
        bit_path = bitstreams_dir / f'{img.stem}.bin'
        run_encoder(yuv_path, pad_w, pad_h, bd, cf, args.qp, bit_path, encoder_bin=args.encoder_bin, encoder_cfg=[Path(c) for c in args.encoder_cfg] if args.encoder_cfg else None)
        dec_yuv = yuv_dir / f'{img.stem}_dec.yuv'
        run_decoder(bit_path, dec_yuv, decoder_bin=args.decoder_bin)
        dec_png = dec_images_dir / f'{img.stem}.png'
        # Output YUV has padded even dims; decode using padded size
        yuv_to_png_with_ffmpeg(dec_yuv, pad_w, pad_h, use_pix_fmt, dec_png)
        # Sanity check: decoded YUV filesize vs expected
        try:
            size_bytes = dec_yuv.stat().st_size
            bytes_per_sample = 2 if use_pix_fmt.endswith('10le') else 1
            expected = int(pad_w * pad_h * 1.5 * bytes_per_sample)
            if abs(size_bytes - expected) > max(16, expected * 0.01):
                logging.warning(f"Decoded YUV size mismatch for {img.stem}: got {size_bytes}, expected ~{expected} for {use_pix_fmt} {pad_w}x{pad_h}")
        except Exception:
            pass
        bits = bit_path.stat().st_size * 8
        bpp = bits / float(w * h)
        per_image.append({'image_id': img.stem, 'orig_path': str(img), 'decoded_path': str(dec_png), 'width': w, 'height': h, 'bits': int(bits), 'bpp': float(bpp)})

    summary = {
        'qp': args.qp,
        'avg_bpp': float(np.mean([x['bpp'] for x in per_image])) if per_image else float('nan'),
        'num_images': len(per_image),
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump({'summary': summary, 'per_image': per_image}, f, indent=2)

    if args.run_detection:
        assert args.detection_model, 'detection_model is required for detection'
        # Reuse run_detection on decoded images
        cmd = [
            'python', '-m', 'src.evaluation.run_detection',
            '--data_dir', str(data_dir),
            '--detection_model', args.detection_model,
            '--mode', 'reconstructed_images',
            '--images_dir', str(dec_images_dir),
            '--output_json', str(out_dir / 'preds.json')
        ]
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()

