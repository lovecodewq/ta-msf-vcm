"""
Train L-MSFC feature-compression anchor on KITTI using detection model features.

This mirrors the structure of train_joint_autoregress_prior_fused_feature.py but
uses the third-party L-MSFC FeatureCompressor.
"""
import os
import sys
import json
import yaml
import math
import time
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import optim

from data.kitti_dataset import KITTIDetectionDataset
from data.transforms import create_detection_transforms
from model.detection import DetectionModel
from utils.training_utils import (
    setup_training_environment,
    AverageMeter,
    EarlyStopping,
    save_training_diagnostics,
    log_epoch_summary,
    save_best_model,
)
from utils.metrics import compute_rate_distortion_loss
from utils.lmsfc_import import load_lmsfc_feature_compressor


def collate_fn(batch):
    return tuple(zip(*batch))


def to_tensor_only(image, target=None):
    import torchvision.transforms.functional as F
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    return image, target


def train_one_epoch(model, det_model, data_loader, optimizer, lmbda, device, log_interval: int = 50):
    model.train()
    det_model.eval()

    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    grad_norm_meter = AverageMeter()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            fpn = det_model.get_fpn_features(images)
        features_list = [fpn['p2'], fpn['p3'], fpn['p4'], fpn['p5'], fpn['p6']]

        optimizer.zero_grad()
        out = model(features_list)

        # Compute total pixels across the batch (use original sizes when available)
        total_pixels = 0
        for i, img in enumerate(images):
            if isinstance(targets[i], dict) and 'orig_size' in targets[i]:
                h, w = targets[i]['orig_size']
            else:
                h, w = img.shape[-2], img.shape[-1]
            total_pixels += int(h) * int(w)
        # Pass a synthetic image_shape carrying the correct total pixel count
        image_shape = (total_pixels, 1, 1, 1)
        # Adapt L-MSFC output (list [p2..p6]) to dict expected by loss util
        out_feats = out.get('features')
        if isinstance(out_feats, list) and len(out_feats) == 5:
            out_for_loss = {
                'features': {
                    'p2': out_feats[0],
                    'p3': out_feats[1],
                    'p4': out_feats[2],
                    'p5': out_feats[3],
                    'p6': out_feats[4],
                },
                'likelihoods': out.get('likelihoods', {})
            }
        else:
            out_for_loss = out
        try:
            batch_loss, batch_mse, batch_bpp = compute_rate_distortion_loss(out_for_loss, fpn, lmbda, image_shape)
        except Exception as e:
            logging.error(f"[LMSFC DEBUG] compute_rate_distortion_loss failed: {e}")
            try:
                logging.error(f"[LMSFC DEBUG] image_shape={image_shape} total_pixels={total_pixels}")
                logging.error(f"[LMSFC DEBUG] fpn keys={list(fpn.keys())}")
                feats = out.get('features')
                ftype = type(feats).__name__
                flen = len(feats) if isinstance(feats, list) else 'n/a'
                logging.error(f"[LMSFC DEBUG] out.features type={ftype} len={flen}")
            except Exception:
                pass
            raise

        batch_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_meter.update(float(grad_norm))
        optimizer.step()

        loss_meter.update(float(batch_loss))
        bpp_meter.update(float(batch_bpp))
        mse_meter.update(float(batch_mse))

        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f"Train Batch [{batch_idx}/{len(data_loader)}] -> "
                f"Loss: {loss_meter.avg:.4f} | MSE: {mse_meter.avg:.6f} | "
                f"BPP: {bpp_meter.avg:.4f} | GradNorm: {grad_norm_meter.avg:.3f} | LR: {current_lr:.2e}"
            )

    return loss_meter.avg, mse_meter.avg, bpp_meter.avg, grad_norm_meter.avg


@torch.no_grad()
def validate(model, det_model, data_loader, lmbda, device):
    model.eval()
    det_model.eval()

    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        fpn = det_model.get_fpn_features(images)
        features_list = [fpn['p2'], fpn['p3'], fpn['p4'], fpn['p5'], fpn['p6']]
        out = model(features_list)
        total_pixels = 0
        for i, img in enumerate(images):
            if isinstance(targets[i], dict) and 'orig_size' in targets[i]:
                h, w = targets[i]['orig_size']
            else:
                h, w = img.shape[-2], img.shape[-1]
            total_pixels += int(h) * int(w)
        image_shape = (total_pixels, 1, 1, 1)
        # Adapt output to dict
        out_feats = out.get('features')
        if isinstance(out_feats, list) and len(out_feats) == 5:
            out_for_loss = {
                'features': {
                    'p2': out_feats[0],
                    'p3': out_feats[1],
                    'p4': out_feats[2],
                    'p5': out_feats[3],
                    'p6': out_feats[4],
                },
                'likelihoods': out.get('likelihoods', {})
            }
        else:
            out_for_loss = out
        batch_loss, batch_mse, batch_bpp = compute_rate_distortion_loss(out_for_loss, fpn, lmbda, image_shape)
        loss_meter.update(float(batch_loss))
        bpp_meter.update(float(batch_bpp))
        mse_meter.update(float(batch_mse))

    return loss_meter.avg, mse_meter.avg, bpp_meter.avg


def main():
    parser = argparse.ArgumentParser(description='Train L-MSFC feature-compression anchor on KITTI')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config for training')
    parser.add_argument('--detection_checkpoint', type=str, required=True, help='Path to detection model checkpoint')
    parser.add_argument('--third_party_root', type=str, default='thirparty', help='Root directory containing L-MSFC third-party code')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    save_dir = setup_training_environment(config['training']['save_dir'], config, args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Load detection model
    det_ckpt = torch.load(args.detection_checkpoint, map_location=device)
    num_classes = len(KITTIDetectionDataset.CLASSES) + 1
    det_model = DetectionModel(num_classes=num_classes, pretrained=False).to(device)
    det_model.load_state_dict(det_ckpt['model_state_dict'])
    det_model.eval()

    # Import and instantiate L-MSFC compressor
    FeatureCompressor = load_lmsfc_feature_compressor(Path(args.third_party_root))
    N = int(config['model'].get('N', 256))
    M = int(config['model'].get('M', 128))
    comp_model = FeatureCompressor(N=N, M=M).to(device)
    # Ensure entropy modules are ready
    try:
        comp_model.update()
    except Exception:
        comp_model.update(force=True)

    # Data
    train_tf = create_detection_transforms(config['data']['transforms'], apply_normalization=False)
    val_tf = create_detection_transforms(config['data']['test_transforms'], apply_normalization=False)

    train_dataset = KITTIDetectionDataset(config['data']['root_dir'], split='train', transform=train_tf)
    val_dataset = KITTIDetectionDataset(config['data']['root_dir'], split='val', transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], collate_fn=collate_fn, pin_memory=True)

    # Optimizer and schedulers
    optimizer = optim.Adam(comp_model.parameters(), lr=config['training']['learning_rate'])
    scheduler = None
    if config['training'].get('lr_schedule', {}).get('enabled', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['lr_schedule'].get('factor', 0.5),
            patience=config['training']['lr_schedule'].get('patience', 5),
            min_lr=config['training']['lr_schedule'].get('min_lr', 1e-6),
        )

    early_stopping = None
    if config['training'].get('early_stopping', {}).get('enabled', False):
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping'].get('patience', 10),
            min_delta=config['training']['early_stopping'].get('min_delta', 0.0),
        )

    lmbda = float(config['training']['lambda'])
    best_loss = float('inf')

    epochs = int(config['training']['epochs'])
    log_interval = int(config['training'].get('log_interval', 50))

    for epoch in range(epochs):
        train_loss, train_mse, train_bpp, train_grad_norm = train_one_epoch(
            comp_model, det_model, train_loader, optimizer, lmbda, device, log_interval
        )
        val_loss, val_mse, val_bpp = validate(comp_model, det_model, val_loader, lmbda, device)

        log_epoch_summary(epoch, (train_loss, train_mse, train_bpp, train_grad_norm), (val_loss, val_mse, val_bpp), optimizer, config)

        if scheduler:
            scheduler.step(val_loss)

        save_training_diagnostics(
            save_dir, epoch,
            (train_loss, train_mse, train_bpp, train_grad_norm),
            (val_loss, val_mse, val_bpp),
            optimizer, comp_model, config
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_best_model(save_dir, epoch, comp_model, optimizer, scheduler, best_loss, config)

        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info('Early stopping triggered')
                break

    logging.info(f"Training completed. Best validation loss: {best_loss:.4f}")
    logging.info(f"Model checkpoint saved to: {save_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()

