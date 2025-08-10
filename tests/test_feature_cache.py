import argparse
from pathlib import Path
import logging

import torch
from torchvision.transforms import functional as F
from tqdm import tqdm

from data.kitti_dataset import KITTIDetectionDataset
from model.detection import DetectionModel
from utils.training_utils import save_fpn_features_to_cache, load_fpn_features_from_cache


def to_tensor_only(image, target=None):
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    return image, target


def save_features(cache_dir: Path, image_ids, fpn_features):
    save_fpn_features_to_cache(cache_dir, image_ids, fpn_features)


def load_batched_features(cache_dir: Path, image_ids, device):
    return load_fpn_features_from_cache(cache_dir, image_ids, device)


def main():
    parser = argparse.ArgumentParser(description='Quick I/O test for feature cache')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--detection_checkpoint', type=str, required=False, help='Optional detection checkpoint')
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device('cpu')  # keep CPU for reproducibility

    # Dataset
    ds = KITTIDetectionDataset(args.data_dir, split='test', transform=to_tensor_only)
    # Collate few samples
    images = []
    image_ids = []
    for i in range(min(args.num_samples, len(ds))):
        img, tgt = ds[i]
        images.append(img)
        image_ids.append(tgt.get('image_id'))

    # Detection model (no internet weights)
    num_classes = len(KITTIDetectionDataset.CLASSES) + 1
    det = DetectionModel(num_classes=num_classes, pretrained=False).to(device)
    if args.detection_checkpoint:
        ckpt = torch.load(args.detection_checkpoint, map_location=device)
        det.load_state_dict(ckpt['model_state_dict'])
    det.eval()

    with torch.no_grad():
        fpn = det.get_fpn_features(images)

    cache_dir = Path(args.cache_dir)
    save_features(cache_dir, image_ids, fpn)
    logging.info(f"Saved {len(image_ids)} feature files to {cache_dir}")

    # Load back batched
    fpn_loaded = load_batched_features(cache_dir, image_ids, device)

    # Compare shapes and numerical closeness
    ok = True
    for k in fpn.keys():
        if fpn[k].shape != fpn_loaded[k].shape:
            logging.error(f"Shape mismatch for {k}: {fpn[k].shape} vs {fpn_loaded[k].shape}")
            ok = False
        else:
            same = torch.allclose(fpn[k].cpu(), fpn_loaded[k].cpu(), atol=1e-6)
            logging.info(f"{k}: shape {fpn[k].shape}, allclose={same}")
            ok = ok and same

    if ok:
        logging.info('Feature cache I/O test PASSED')
    else:
        logging.error('Feature cache I/O test FAILED')
        raise SystemExit(1)


if __name__ == '__main__':
    main()

