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

from data.kitti_dataset import KITTIDetectionDataset
from data.transforms import create_transforms
from model.detection import DetectionModel
from model.joint_autoregress_fpn_compressor import JointAutoregressFPNCompressor
from utils.training_utils import (
    setup_training_environment,
    AverageMeter,
    EarlyStopping,
    log_epoch_summary,
    save_training_diagnostics,
    save_best_model,
)
from utils.metrics import compute_rate_distortion_loss



def parse_args():
    parser = argparse.ArgumentParser(description='Train FPN compressor with detection loss')
    parser.add_argument('--config', type=str, default='configs/train_joint_autoregress_prior_fused_feature_detect_loss.yaml')
    parser.add_argument('--detection_checkpoint', type=str, required=True,
                        help='Path to detection model checkpoint (used for loss only)')
    return parser.parse_args()


def build_dataloaders(config):
    train_tf = create_transforms(config['data']['transforms'], split='train')
    val_tf = create_transforms(config['data']['test_transforms'], split='val')

    train_set = KITTIDetectionDataset(config['data']['train_list'].rsplit('/', 1)[0].rsplit('/', 1)[0],
                                      split='train', transform=train_tf)
    val_set = KITTIDetectionDataset(config['data']['val_list'].rsplit('/', 1)[0].rsplit('/', 1)[0],
                                    split='val', transform=val_tf)

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


def train_one_epoch(model, det_model, optimizer, lmbda, det_w, device, loader, log_interval):
    model.train()
    det_model.eval()

    loss_meter = AverageMeter()
    rd_meter = AverageMeter()
    det_meter = AverageMeter()
    ratio_meter = AverageMeter()

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Extract clean FPN features
        with torch.no_grad():
            clean_feats = det_model.get_fpn_features(images)

        # Compress and reconstruct features
        out = model(clean_feats)
        recon_feats = out['features']

        # Rate-distortion loss
        rd_loss, rd_mse, rd_bpp = compute_rate_distortion_loss(out, clean_feats, lmbda, (len(images),) + images[0].shape)

        # Detection loss using reconstructed features
        det_losses = det_model.forward_from_features(images, recon_feats, targets)
        if isinstance(det_losses, dict):
            det_loss = sum(det_losses.values())
        else:
            det_loss = torch.zeros((), device=device)

        total_loss = rd_loss + det_w * det_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(float(total_loss))
        rd_meter.update(float(rd_loss))
        det_meter.update(float(det_loss))
        ratio = float(det_loss) / (float(rd_loss) + 1e-8)
        ratio_meter.update(ratio)

        if batch_idx % log_interval == 0:
            det_contrib = float(det_w) * float(det_loss)
            total_now = float(rd_loss) + det_contrib
            rd_share = float(rd_loss) / (total_now + 1e-8)
            det_share = det_contrib / (total_now + 1e-8)
            logging.info(
                f"Train [{batch_idx}/{len(loader)}] total(avg)={loss_meter.avg:.4f} "
                f"rd(avg)={rd_meter.avg:.4f} det(avg)={det_meter.avg:.4f} ratio(det/rd)(avg)={ratio_meter.avg:.3f} "
                f"curr_rd={float(rd_loss):.4f} curr_det={float(det_loss):.4f} det_w={det_w:.3f} "
                f"contrib(rd,det)=({rd_share:.2f},{det_share:.2f}) bpp={rd_bpp:.4f}"
            )

    return loss_meter.avg, rd_meter.avg, det_meter.avg, ratio_meter.avg


@torch.no_grad()
def validate(model, det_model, lmbda, det_w, device, loader):
    model.eval()
    det_model.eval()

    loss_meter = AverageMeter()
    rd_meter = AverageMeter()
    det_meter = AverageMeter()
    ratio_meter = AverageMeter()

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        clean_feats = det_model.get_fpn_features(images)
        out = model(clean_feats)
        recon_feats = out['features']

        rd_loss, rd_mse, rd_bpp = compute_rate_distortion_loss(out, clean_feats, lmbda, (len(images),) + images[0].shape)
        det_losses = det_model.forward_from_features(images, recon_feats, targets)
        det_loss = sum(det_losses.values()) if isinstance(det_losses, dict) else torch.zeros((), device=device)

        total_loss = rd_loss + det_w * det_loss

        loss_meter.update(float(total_loss))
        rd_meter.update(float(rd_loss))
        det_meter.update(float(det_loss))
        ratio_meter.update(float(det_loss) / (float(rd_loss) + 1e-8))

    return loss_meter.avg, rd_meter.avg, det_meter.avg, ratio_meter.avg


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
    disable_parameter_grads(det_model)
    logging.info('Loaded detection model and disabled parameter grads')

    # Compression model
    comp_model = JointAutoregressFPNCompressor(
        N=config['model']['n_latent'],
        M=config['model']['n_hidden'],
        input_channels=config['model']['fpn_channels_per_level'],
        output_channels=config['model']['fpn_channels_per_level'],
    ).to(device)
    logging.info(f'Compression model params: {sum(p.numel() for p in comp_model.parameters())}')

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
    best_val = float('inf')

    for epoch in range(config['training']['epochs']):
        tr_total, tr_rd, tr_det, tr_ratio = train_one_epoch(
            comp_model, det_model, optimizer, lmbda, det_w, device, train_loader, config['training']['log_interval']
        )

        va_total, va_rd, va_det, va_ratio = validate(
            comp_model, det_model, lmbda, det_w, device, val_loader
        )

        logging.info(
            f"Epoch {epoch}: total(tr/va)=({tr_total:.4f}/{va_total:.4f}) "
            f"rd(tr/va)=({tr_rd:.4f}/{va_rd:.4f}) det(tr/va)=({tr_det:.4f}/{va_det:.4f}) "
            f"ratio(det/rd)(tr/va)=({tr_ratio:.3f}/{va_ratio:.3f}) det_w={det_w:.3f}"
        )
        log_epoch_summary(epoch, (tr_total, tr_rd, tr_det), (va_total, va_rd, va_det), optimizer, config)

        if scheduler:
            scheduler.step(va_total)

        save_training_diagnostics(save_dir, epoch, (tr_total, tr_rd, tr_det), (va_total, va_rd, va_det), optimizer, comp_model, config)

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

