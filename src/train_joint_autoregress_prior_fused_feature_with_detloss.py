"""
Training script for FPN feature compression with additional detection loss.

Objective: Optimize a JointAutoregressiveHierarchicalPriors-based compressor on top of
Faster R-CNN features using a combined objective:
  total_loss = rate_distortion_loss + detection_loss_weight * detection_loss

Detection loss is computed by running the Faster R-CNN head on reconstructed features
via forward_from_features with ground-truth targets.
"""
import os
from pathlib import Path
import argparse
import yaml
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os

from data.kitti_dataset import KITTIDetectionDataset
from data.transforms import create_transforms, create_detection_transforms
from model.detection import DetectionModel
from model.joint_autoregress_fpn_compressor import JointAutoregressFPNCompressor
from utils.training_utils import (
    setup_training_environment,
    AverageMeter,
    EarlyStopping,
    log_epoch_summary,
    save_training_diagnostics,
    save_best_model,
    save_fpn_features_to_cache,
    load_fpn_features_from_cache,
    precompute_feature_cache,
)
from utils.metrics import compute_rate_distortion_loss



def parse_args():
    parser = argparse.ArgumentParser(description='Train FPN compressor with detection loss')
    parser.add_argument('--config', type=str, default='configs/train_joint_autoregress_prior_fused_feature_detect_loss.yaml')
    parser.add_argument('--detection_checkpoint', type=str, required=True,
                        help='Path to detection model checkpoint (used for loss only)')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Optional path to a pre-trained feature compressor checkpoint to initialize from (no detection loss)')
    return parser.parse_args()


def _to_tensor_only(image, target=None):
    import torchvision.transforms.functional as F
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    return image, target


def build_dataloaders(config):
    root_dir = config['data'].get('root_dir')
    if not root_dir:
        # fallback to derive root from lists: <root>/train.txt → <root>
        any_list = config['data'].get('train_list') or config['data'].get('val_list')
        if any_list:
            root_dir = any_list.rsplit('/', 1)[0].rsplit('/', 1)[0]
        else:
            raise ValueError('Please provide data.root_dir in config')

    use_ext = bool(config['data'].get('use_external_transforms', False))
    if use_ext:
        # Use detection-aware transforms that update targets (boxes) on flip; no external normalization
        train_tf = create_detection_transforms(config['data']['transforms'], apply_normalization=False)
        val_tf = create_detection_transforms(config['data']['test_transforms'], apply_normalization=False)
    else:
        train_tf = _to_tensor_only
        val_tf = _to_tensor_only

    train_set = KITTIDetectionDataset(root_dir, split='train', transform=train_tf)
    val_set = KITTIDetectionDataset(root_dir, split='val', transform=val_tf)

    train_loader = DataLoader(train_set,
                              batch_size=config['training']['batch_size'],
                              shuffle=True,
                              num_workers=config['training']['num_workers'],
                              pin_memory=True,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_set,
                            batch_size=config['training']['batch_size'],
                            shuffle=False,
                            num_workers=config['training']['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: tuple(zip(*x)))
    return train_loader, val_loader


def disable_parameter_grads(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _load_cached_features(cache_dir: Path, image_ids, device):
    return load_fpn_features_from_cache(cache_dir, image_ids, device)


def train_one_epoch(model, det_model, optimizer, lmbda, det_w, device, loader, log_interval, grad_clip_max_norm: float = 1.0, feature_cache_dir: Path = None):
    model.train()
    det_model.eval()

    loss_meter = AverageMeter()
    rd_meter = AverageMeter()
    det_meter = AverageMeter()
    ratio_meter = AverageMeter()
    mse_meter = AverageMeter()

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc='Training', leave=False)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Extract or load clean FPN features
        if feature_cache_dir is not None:
            image_ids = [t.get('image_id') for t in targets]
            clean_feats = _load_cached_features(feature_cache_dir, image_ids, device)
        else:
            with torch.no_grad():
                clean_feats = det_model.get_fpn_features(images)

        # Compress and reconstruct features
        out = model(clean_feats)
        recon_feats = out['features']

        # Rate-distortion loss
        rd_loss, rd_mse, rd_bpp = compute_rate_distortion_loss(out, clean_feats, lmbda, (len(images),) + images[0].shape)

        # Detection loss using reconstructed features (skip if weight is zero)
        if det_w > 0:
            # Compute detection losses; must set model to training mode for losses to be produced
            det_losses = det_model.compute_losses_from_features(images, recon_feats, targets)
            if isinstance(det_losses, dict):
                det_loss = sum(det_losses.values())
            else:
                det_loss = torch.zeros((), device=device)
        else:
            det_loss = torch.zeros((), device=device)

        total_loss = rd_loss + det_w * det_loss
        total_loss.backward()

        if grad_clip_max_norm and grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        optimizer.step()

        loss_meter.update(float(total_loss))
        rd_meter.update(float(rd_loss))
        det_meter.update(float(det_loss))
        mse_meter.update(float(rd_mse))
        ratio = float(det_w) * float(det_loss) / (float(rd_loss) + 1e-8)
        ratio_meter.update(ratio)

        if batch_idx % log_interval == 0:
            det_contrib = float(det_w) * float(det_loss)
            total_now = float(rd_loss) + det_contrib
            rd_share = float(rd_loss) / (total_now + 1e-8)
            det_share = det_contrib / (total_now + 1e-8)
            logging.info(
                f"Train [{batch_idx}/{len(loader)}] total(avg)={loss_meter.avg:.4f} "
                f"rd(avg)={rd_meter.avg:.4f} mse(avg)={mse_meter.avg:.6f} det(avg)={det_meter.avg:.4f} ratio(det/rd)(avg)={ratio_meter.avg:.3f} "
                f"curr_rd={float(rd_loss):.4f} curr_mse={float(rd_mse):.6f} curr_det={float(det_loss):.4f} det_w={det_w:.3f} "
                f"contrib(rd,det)=({rd_share:.2f},{det_share:.2f}) bpp={rd_bpp:.4f}"
            )

    return loss_meter.avg, rd_meter.avg, det_meter.avg, ratio_meter.avg, mse_meter.avg


@torch.no_grad()
def validate(model, det_model, lmbda, det_w, device, loader, feature_cache_dir: Path = None):
    model.eval()
    det_model.eval()

    loss_meter = AverageMeter()
    rd_meter = AverageMeter()
    det_meter = AverageMeter()
    ratio_meter = AverageMeter()
    mse_meter = AverageMeter()

    for images, targets in tqdm(loader, desc='Validating', leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if feature_cache_dir is not None:
            image_ids = [t.get('image_id') for t in targets]
            clean_feats = _load_cached_features(feature_cache_dir, image_ids, device)
        else:
            clean_feats = det_model.get_fpn_features(images)
        out = model(clean_feats)
        recon_feats = out['features']

        rd_loss, rd_mse, rd_bpp = compute_rate_distortion_loss(out, clean_feats, lmbda, (len(images),) + images[0].shape)
        if det_w > 0:
            det_losses = det_model.compute_losses_from_features(images, recon_feats, targets)
            det_loss = sum(det_losses.values()) if isinstance(det_losses, dict) else torch.zeros((), device=device)
        else:
            det_loss = torch.zeros((), device=device)

        total_loss = rd_loss + det_w * det_loss

        loss_meter.update(float(total_loss))
        rd_meter.update(float(rd_loss))
        det_meter.update(float(det_loss))
        mse_meter.update(float(rd_mse))
        ratio_meter.update(float(det_loss) / (float(rd_loss) + 1e-8))

    return loss_meter.avg, rd_meter.avg, det_meter.avg, ratio_meter.avg, mse_meter.avg


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = setup_training_environment(config['training']['save_dir'], config, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(config)

    # Detection model
    det_ckpt = torch.load(args.detection_checkpoint, map_location=device)
    num_classes = len(KITTIDetectionDataset.CLASSES) + 1
    det_model = DetectionModel(num_classes=num_classes, pretrained=False).to(device)
    det_model.load_state_dict(det_ckpt['model_state_dict'])
    det_model.eval()
    det_model.freeze_parameters()
    logging.info('Loaded detection model and disabled parameter grads')

    # Compression model
    comp_model = JointAutoregressFPNCompressor(
        N=config['model']['n_latent'],
        M=config['model']['n_hidden'],
        input_channels=config['model']['fpn_channels_per_level'],
        output_channels=config['model']['fpn_channels_per_level'],
    ).to(device)
    logging.info(f'Compression model params: {sum(p.numel() for p in comp_model.parameters())}')

    # Optional initialization from a pre-trained (no detection loss) checkpoint
    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        try:
            init_ckpt = torch.load(args.init_checkpoint, map_location=device)
            state = init_ckpt.get('model_state_dict', init_ckpt)
            comp_model.load_state_dict(state, strict=True)
            logging.info(f"Initialized compressor from {args.init_checkpoint}")
        except Exception as e:
            logging.warning(f"Failed to load init checkpoint {args.init_checkpoint}: {e}")

    optimizer = optim.Adam(
        comp_model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['optimizer']['betas'],
        eps=config['training']['optimizer']['eps'],
        weight_decay=config['training']['optimizer']['weight_decay'],
    )

    scheduler = None
    if config['training']['lr_schedule']['enabled']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config['training']['lr_schedule']['factor'],
            patience=config['training']['lr_schedule']['patience'], min_lr=config['training']['lr_schedule']['min_lr']
        )

    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta'],
        )

    lmbda = config['training']['lambda']
    det_w = float(config['training'].get('detection_loss_weight', 1.0))
    grad_clip = float(config['training'].get('grad_clip_max_norm', 1.0))
    no_det_epochs = int(config['training'].get('no_det_loss_epochs', 10))
    best_val = float('inf')

    # Optional feature cache
    cache_cfg = config.get('feature_cache', {})
    use_cache = bool(cache_cfg.get('enabled', False))
    cache_dir = Path(cache_cfg.get('dir', save_dir / 'fpn_cache'))
    if use_cache:
        logging.info(f"Feature cache enabled. Directory: {cache_dir}")
        precompute_feature_cache(det_model, train_loader, device, cache_dir, overwrite=bool(cache_cfg.get('overwrite', False)))
        precompute_feature_cache(det_model, val_loader, device, cache_dir, overwrite=False)

    for epoch in range(config['training']['epochs']):
        det_w_eff = det_w if epoch >= no_det_epochs else 0.0
        detection_enabled = epoch >= no_det_epochs
        tr_total, tr_rd, tr_det, tr_ratio, tr_mse = train_one_epoch(
            comp_model, det_model, optimizer, lmbda, det_w_eff, device, train_loader, config['training']['log_interval'], grad_clip,
            feature_cache_dir=(cache_dir if use_cache else None)
        )

        va_total, va_rd, va_det, va_ratio, va_mse = validate(
            comp_model, det_model, lmbda, det_w_eff, device, val_loader,
            feature_cache_dir=(cache_dir if use_cache else None)
        )

        logging.info(
            f"Epoch {epoch}: total(tr/va)=({tr_total:.4f}/{va_total:.4f}) "
            f"rd(tr/va)=({tr_rd:.4f}/{va_rd:.4f}) mse(tr/va)=({tr_mse:.6f}/{va_mse:.6f}) det(tr/va)=({tr_det:.4f}/{va_det:.4f}) "
            f"ratio(det/rd)(tr/va)=({tr_ratio:.3f}/{va_ratio:.3f}) det_w_eff={det_w_eff:.3f}"
        )

        # During warmup (no detection loss), skip scheduler, best-model saving, and early stopping
        if not detection_enabled:
            logging.info('Warmup epoch (detection loss disabled): skipping LR scheduling, best-model comparison, and early stopping')
            continue

        # First epoch with detection loss enabled: reset baselines and early stopping
        if epoch == no_det_epochs:
            best_val = va_total
            save_best_model(save_dir, epoch, comp_model, optimizer, scheduler, best_val, config)
            if early_stopping:
                early_stopping = EarlyStopping(
                    patience=config['training']['early_stopping']['patience'],
                    min_delta=config['training']['early_stopping']['min_delta'],
                )
            # Start scheduler from here
            if scheduler:
                scheduler.step(va_total)
        else:
            if scheduler:
                scheduler.step(va_total)

            if va_total < best_val:
                best_val = va_total
                save_best_model(save_dir, epoch, comp_model, optimizer, scheduler, best_val, config)

            if early_stopping:
                early_stopping(va_total)
                if early_stopping.early_stop:
                    logging.info('Early stopping triggered')
                    break

    logging.info('Training completed')


if __name__ == '__main__':
    main()

