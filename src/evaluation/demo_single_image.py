import argparse
import json
import logging
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
from model.detection import DetectionModel
from data.kitti_dataset import KITTIDetectionDataset  # type: ignore

from src.evaluation.vtm_utils import run_encoder, run_decoder, ffmpeg_rgb_to_yuv, ffmpeg_yuv_to_png


def _to_tensor_only(image, target=None):
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    return image, target


def _bytes_readable(num_bytes: int) -> str:
    units = ['B', 'KB', 'MB', 'GB']
    size = float(num_bytes)
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _render_size_chart(out_root: Path,
                       raw_bytes: int,
                       bit_path: Optional[Path],
                       raw_feat_bytes: int,
                       feat_comp_bytes: int) -> Optional[Path]:
    labels = ['Raw Image', 'Image Anchor', 'Raw Feature', 'Compressed Feature']
    sizes = [raw_bytes,
                (os.path.getsize(bit_path) if bit_path else 0),
                raw_feat_bytes,
                feat_comp_bytes]
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']
    plt.figure(figsize=(6, 4))
    xs = range(len(labels))
    plt.bar(xs, sizes, color=colors)
    plt.xticks(xs, labels, rotation=15)
    plt.ylabel('Bytes')
    plt.title('Size Comparison')
    for i, v in enumerate(sizes):
        plt.text(i, v * 1.02 if v > 0 else 1, _bytes_readable(v), ha='center', va='bottom', fontsize=9)
    chart_path = out_root / 'size_comparison.png'
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logging.info("[Viz] size comparison chart saved to %s", chart_path)
    return chart_path


def _build_composite(
    out_root: Path,
    vis_dir: Path,
    image_path: Path,
    det_vis_path: Path,
    raw_bytes: int,
    raw_bpp: float,
    raw_feat_bytes: int,
    feat_comp_bytes: int,
    feat_bpp: float,
    chart_path: Optional[Path]
) -> Optional[Path]:
    try:
        images_for_row: List[Image.Image] = []
        # Raw (ensure copy exists for consistent visual size)
        raw_png = vis_dir / f"{image_path.stem}_raw.png"
        if not raw_png.exists():
            with Image.open(image_path).convert('RGB') as im_raw:
                im_raw.save(raw_png)
        images_for_row.append(Image.open(raw_png).convert('RGB'))
        # Detection visualization
        images_for_row.append(Image.open(det_vis_path).convert('RGB'))
        # Bar chart (optional)
        if chart_path and chart_path.exists():
            images_for_row.append(Image.open(chart_path).convert('RGB'))

        # Normalize heights by padding (no resize), align to max height
        max_h = max(im.size[1] for im in images_for_row)
        padded: List[Image.Image] = []
        for im in images_for_row:
            w, h = im.size
            if h < max_h:
                canvas = Image.new('RGB', (w, max_h), color=(255, 255, 255))
                canvas.paste(im, (0, 0))
                padded.append(canvas)
            else:
                padded.append(im)
        total_w = sum(im.size[0] for im in padded)
        row_h = max_h

        # Build table panel (text only)
        table_w = max(520, total_w // 3)
        table_h = row_h
        table = Image.new('RGB', (table_w, table_h), color=(255, 255, 255))
        d = ImageDraw.Draw(table)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        lines = [
            'Summary (Bytes & BPP):',
            f"- Raw Image: {_bytes_readable(raw_bytes)}, bpp={raw_bpp:.4f}",
            f"- Raw Feature: {_bytes_readable(raw_feat_bytes)}",
            f"- Compressed Feature: {_bytes_readable(feat_comp_bytes)}, bpp={feat_bpp:.4f}",
            '',
            'Notes:',
            '- BPP uses original image W×H.',
            '- Detection visualization is run on compressed features.',
        ]
        y = 10
        for ln in lines:
            d.text((10, y), ln, fill='black', font=font)
            y += 16

        # Compose: [row images] stacked with table on the right
        comp_w = total_w + table_w
        comp_h = row_h
        comp = Image.new('RGB', (comp_w, comp_h), color=(255, 255, 255))
        x = 0
        for im in padded:
            comp.paste(im, (x, 0))
            x += im.size[0]
        comp.paste(table, (x, 0))
        composite_path = vis_dir / f"{image_path.stem}_composite.png"
        comp.save(composite_path)
        logging.info("[Viz] composite saved to %s", composite_path)
        return composite_path
    except Exception as e:
        logging.warning("[Viz] failed to build composite visualization: %s", e)
        return None

def _draw_predictions(image_path: Path, prediction: Dict[str, "torch.Tensor"], out_path: Path, class_names: Optional[List[str]] = None) -> None:

    with Image.open(image_path).convert('RGB') as im:
        draw = ImageDraw.Draw(im)
        boxes = prediction.get('boxes', torch.zeros((0, 4)))
        scores = prediction.get('scores', torch.zeros((0,)))
        labels = prediction.get('labels', torch.zeros((0,), dtype=torch.long))
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for i in range(min(len(boxes), 200)):
            box = boxes[i].tolist()
            score = float(scores[i]) if i < len(scores) else 0.0
            label_id = int(labels[i]) if i < len(labels) else 0
            name = class_names[label_id - 1] if class_names and label_id > 0 and label_id - 1 < len(class_names) else str(label_id)
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            text = f"{name} {score:.2f}"
            if font:
                tw, th = draw.textsize(text, font=font)
                draw.rectangle([x1, max(0, y1 - th), x1 + tw, y1], fill='white')
                draw.text((x1, max(0, y1 - th)), text, fill='black', font=font)
            else:
                draw.text((x1, y1), text, fill='red')
        im.save(out_path)


def _vtm_compress_single_image(
    image_path: Path,
    out_dir: Path,
    qp: int,
    encoder_bin: Path,
    decoder_bin: Path,
    encoder_cfg: Optional[List[Path]],
    pix_fmt: str
) -> Tuple[Path, Path, int, float, int, int]:
    _ensure_dir(out_dir)
    yuv_dir = out_dir / 'yuv'
    bit_dir = out_dir / 'bitstreams'
    dec_dir = out_dir / 'decoded_png'
    _ensure_dir(yuv_dir)
    _ensure_dir(bit_dir)
    _ensure_dir(dec_dir)

    with Image.open(image_path) as im:
        im = im.convert('RGB')
        w, h = im.size

    yuv_path = yuv_dir / f"{image_path.stem}.yuv"
    ffmpeg_rgb_to_yuv(image_path, yuv_path, pix_fmt=pix_fmt)

    # 4:2:0 requires even dims
    pad_w = int((w + 1) // 2 * 2)
    pad_h = int((h + 1) // 2 * 2)
    bit_path = bit_dir / f"{image_path.stem}.bin"
    run_encoder(yuv_path, pad_w, pad_h, 10 if pix_fmt.endswith('10le') else 8, '420', qp, bit_path, encoder_bin=encoder_bin, encoder_cfg=encoder_cfg)

    dec_yuv = yuv_dir / f"{image_path.stem}_dec.yuv"
    run_decoder(bit_path, dec_yuv, decoder_bin=decoder_bin)
    dec_png = dec_dir / f"{image_path.stem}.png"
    ffmpeg_yuv_to_png(dec_yuv, pad_w, pad_h, pix_fmt, dec_png)

    bits = os.path.getsize(bit_path) * 8
    bpp = float(bits) / float(w * h)
    return bit_path, dec_png, int(bits), float(bpp), w, h


def _load_detection_model(det_ckpt_path: Path, device: "torch.device"):
    ckpt = torch.load(det_ckpt_path, map_location=device)
    num_classes = len(KITTIDetectionDataset.CLASSES) + 1
    model = DetectionModel(num_classes=num_classes, pretrained=True).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def _load_feature_compressor(feat_ckpt_path: Path, device: "torch.device"):
    from model.joint_autoregress_fpn_compressor import JointAutoregressFPNCompressor
    comp_ckpt = torch.load(feat_ckpt_path, map_location=device)
    comp_model = JointAutoregressFPNCompressor(
        N=comp_ckpt['config']['model']['n_latent'],
        M=comp_ckpt['config']['model']['n_hidden']
    ).to(device)
    comp_model.load_state_dict(comp_ckpt['model_state_dict'])
    try:
        comp_model.update()
    except Exception:
        comp_model.update(force=True)
    comp_model.eval()
    return comp_model


def main():
    parser = argparse.ArgumentParser(description='Single-image VCM demo: image anchor, feature compression, detection, and size comparison')
    parser.add_argument('--image', type=str, default='example.png', help='Path to a single image')
    parser.add_argument('--out_dir', type=str, default='demo/result', help='Output directory (will be created)')
    # Model paths (defaults mirror scripts/run_full_eval.sh)
    parser.add_argument('--det_ckpt', type=str, default='checkpoints/detection/run_0.002000_16/best_model.pth')
    parser.add_argument('--feat_ckpt', type=str, default='checkpoints/fpn_joint_auto_compression_fused_feature_with_detect_loss/run_20250816_010910_lambda_3.00e+01_w_1.00e-01_lr_1.00e-04_bs_8/best_model.pth')
    # VTM config
    parser.add_argument('--qp', type=int, default=32)
    parser.add_argument('--pix_fmt', type=str, default='yuv420p10le')
    parser.add_argument('--encoder_bin', type=str, default='/home/wenqiangli/code/VVCSoftware_VTM/bin/EncoderAppStatic')
    parser.add_argument('--decoder_bin', type=str, default='/home/wenqiangli/code/VVCSoftware_VTM/bin/DecoderAppStatic')
    parser.add_argument('--encoder_cfg', type=str, nargs='*', default=[
        '/home/wenqiangli/code/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg',
        '/home/wenqiangli/code/VVCSoftware_VTM/cfg/encoder_ai_10bit_override.cfg',
    ])
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_root = Path(args.out_dir)
    _ensure_dir(out_root)
    vis_dir = out_root / 'viz'
    _ensure_dir(vis_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device: %s", device)

    # Stage 0: raw image size
    raw_bytes = os.path.getsize(image_path)
    logging.info("[Raw] Image: %s size=%s", image_path.name, _bytes_readable(raw_bytes))

    # Stage 1: Image anchor compression via VTM
    try:
        enc_bin = Path(args.encoder_bin)
        dec_bin = Path(args.decoder_bin)
        cfgs = [Path(p) for p in (args.encoder_cfg or [])]
        vtmdir = out_root / 'image_anchor'
        bit_path, dec_png, img_bits, img_bpp, w, h = _vtm_compress_single_image(
            image_path=image_path,
            out_dir=vtmdir,
            qp=args.qp,
            encoder_bin=enc_bin,
            decoder_bin=dec_bin,
            encoder_cfg=cfgs,
            pix_fmt=args.pix_fmt,
        )
        logging.info("[ImageAnchor] bitstream=%s (bpp=%.4f) -> decoded=%s", _bytes_readable(os.path.getsize(bit_path)), img_bpp, dec_png)
        # Save decoded image and raw image copies under viz, and a side-by-side comparison
        try:
            raw_copy = vis_dir / f"{image_path.stem}_raw.png"
            dec_copy = vis_dir / f"{image_path.stem}_vtm_decoded.png"
            # Copy/Save raw
            with Image.open(image_path).convert('RGB') as im_raw:
                im_raw.save(raw_copy)
                w_raw, h_raw = im_raw.size
            # Copy/Save decoded
            with Image.open(dec_png).convert('RGB') as im_dec:
                im_dec.save(dec_copy)
                w_dec, h_dec = im_dec.size
            # Side-by-side comparison
            comp_path = vis_dir / f"{image_path.stem}_raw_vs_vtm.png"
            comp_w = w_raw + w_dec
            comp_h = max(h_raw, h_dec)
            comp = Image.new('RGB', (comp_w, comp_h), color=(255, 255, 255))
            comp.paste(Image.open(raw_copy).convert('RGB'), (0, 0))
            comp.paste(Image.open(dec_copy).convert('RGB'), (w_raw, 0))
            comp.save(comp_path)
            logging.info("[ImageAnchor] saved decoded copy: %s and comparison: %s", dec_copy, comp_path)
        except Exception as e:
            logging.warning("[ImageAnchor] failed to save decoded/compare visuals: %s", e)
    except Exception as e:
        logging.warning("Image anchor compression failed or skipped: %s", e)
        bit_path = None
        dec_png = None
        img_bits = None

    # Stage 2: Load detection and feature compressor
    det_model = _load_detection_model(Path(args.det_ckpt), device)
    comp_model = _load_feature_compressor(Path(args.feat_ckpt), device)

    # Stage 3: Feature extraction (no resize/normalize)
    _pil = Image.open(image_path).convert('RGB')
    img_w, img_h = _pil.size
    img_tensor, _ = _to_tensor_only(_pil)
    img_tensor = img_tensor.to(device)
    # Raw image BPP based on file bytes (PNG/JPEG) over pixel count
    raw_bpp = (raw_bytes * 8.0) / float(img_w * img_h)
    logging.info("[Raw] bpp=%.4f (W=%d, H=%d)", raw_bpp, img_w, img_h)
    fpn = det_model.get_fpn_features([img_tensor])
    p2_h, p2_w = fpn['p2'].shape[-2:]
    # Save raw features
    raw_feat_dir = out_root / 'features_raw'
    _ensure_dir(raw_feat_dir)
    raw_feat_path = raw_feat_dir / f"{image_path.stem}.pt"
    torch.save({k: v.detach().cpu() for k, v in fpn.items()}, raw_feat_path)
    raw_feat_bytes = os.path.getsize(raw_feat_path)
    logging.info("[FeatureRaw] saved=%s size=%s ratio(raw_feat/raw_img)=%.2fx", raw_feat_path, _bytes_readable(raw_feat_bytes), raw_feat_bytes / raw_bytes)

    # Stage 4: Feature compression + save reconstructed features and bitstream
    features_list = [fpn['p2'], fpn['p3'], fpn['p4'], fpn['p5'], fpn['p6']]
    compressed = comp_model.compress(features_list)
    decompressed = comp_model.decompress(compressed['strings'], p2_h, p2_w)
    recon_list = decompressed['features']

    feat_comp_dir = out_root / 'features_comp'
    _ensure_dir(feat_comp_dir)
    feat_recon_dir = feat_comp_dir / 'features_from_bins'
    feat_bins_dir = feat_comp_dir / 'bitstreams'
    _ensure_dir(feat_recon_dir)
    _ensure_dir(feat_bins_dir)

    # Save single-image recon features dict and bitstream with simple header
    recon_feats = {
        'p2': recon_list[0][0:1].cpu(),
        'p3': recon_list[1][0:1].cpu(),
        'p4': recon_list[2][0:1].cpu(),
        'p5': recon_list[3][0:1].cpu(),
        'p6': recon_list[4][0:1].cpu(),
    }
    recon_feat_path = feat_recon_dir / f"{image_path.stem}.pt"
    torch.save(recon_feats, recon_feat_path)

    y_strings, z_strings = compressed['strings']
    y_bytes = y_strings[0]
    z_bytes = z_strings[0]
    feat_bin_path = feat_bins_dir / f"{image_path.stem}.bin"
    with open(feat_bin_path, 'wb') as bf:
        bf.write(b'FPN1')
        bf.write(struct.pack('<IIII', p2_h, p2_w, len(y_bytes), len(z_bytes)))
        bf.write(y_bytes)
        bf.write(z_bytes)
    feat_comp_bytes = os.path.getsize(feat_bin_path)
    feat_bits = feat_comp_bytes * 8
    feat_bpp = float(feat_bits) / float(img_w * img_h)
    logging.info("[FeatureComp] bitstream=%s ratio(comp_feat/raw_img)=%.2fx bpp=%.4f", _bytes_readable(feat_comp_bytes), feat_comp_bytes / raw_bytes, feat_bpp)

    # Stage 5: Detection using reconstructed features
    # Build batched features dict as in run_detection
    feats_batched = {
        'p2': recon_feats['p2'].to(device),
        'p3': recon_feats['p3'].to(device),
        'p4': recon_feats['p4'].to(device),
        'p5': recon_feats['p5'].to(device),
        'p6': recon_feats['p6'].to(device),
    }
    predictions = det_model.forward_from_features([img_tensor], feats_batched)
    pred = predictions[0]

    class_names = list(KITTIDetectionDataset.CLASSES)
    det_vis_path = vis_dir / f"{image_path.stem}_det_on_comp_features.png"
    _draw_predictions(image_path, pred, det_vis_path, class_names=class_names)
    logging.info("[Detection] visualization saved to %s", det_vis_path)

    # Stage 6: Size comparison chart and manifest
    chart_path = _render_size_chart(out_root, raw_bytes, bit_path, raw_feat_bytes, feat_comp_bytes)

    # Composite canvas: build
    composite_path = _build_composite(
        out_root=out_root,
        vis_dir=vis_dir,
        image_path=image_path,
        det_vis_path=det_vis_path,
        raw_bytes=raw_bytes,
        raw_bpp=raw_bpp,
        raw_feat_bytes=raw_feat_bytes,
        feat_comp_bytes=feat_comp_bytes,
        feat_bpp=feat_bpp,
        chart_path=chart_path,
    )

    # Write manifest once at the end
    manifest = {
        'image': str(image_path),
        'raw_image_bytes': int(raw_bytes),
        'raw_image_bpp': float(raw_bpp),
        'image_anchor': {
            'bitstream': str(bit_path) if bit_path else None,
            'decoded_png': str(dec_png) if dec_png else None,
            'bytes': int(os.path.getsize(bit_path)) if bit_path else None,
            'bpp': float(img_bpp) if img_bits is not None else None,
            'qp': int(args.qp),
            'pix_fmt': args.pix_fmt,
            'decoded_copy': str((vis_dir / f"{image_path.stem}_vtm_decoded.png")) if dec_png else None,
            'raw_copy': str((vis_dir / f"{image_path.stem}_raw.png")),
            'comparison_image': str((vis_dir / f"{image_path.stem}_raw_vs_vtm.png")) if dec_png else None,
        },
        'feature_raw': {
            'path': str(raw_feat_path),
            'bytes': int(raw_feat_bytes),
            'ratio_vs_raw_image': float(raw_feat_bytes) / float(raw_bytes),
        },
        'feature_compressed': {
            'bitstream': str(feat_bin_path),
            'reconstructed_feature': str(recon_feat_path),
            'bytes': int(feat_comp_bytes),
            'ratio_vs_raw_image': float(feat_comp_bytes) / float(raw_bytes),
            'bpp': float(feat_bpp),
        },
        'detection': {
            'vis_image': str(det_vis_path),
            'det_ckpt': args.det_ckpt,
            'feat_ckpt': args.feat_ckpt,
        },
        'visualization': {
            'composite': str(composite_path) if composite_path else None,
            'bar_chart': str(chart_path) if (chart_path and chart_path.exists()) else None,
        }
    }
    manifest_path = out_root / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    logging.info("[Out] Manifest saved to %s", manifest_path)


if __name__ == '__main__':
    main()


