"""
Training script for per-level FPN feature compression with feature normalization.

This script extends train_factorized_prior_random_sample_fpn.py by adding
feature normalization before MSE computation to handle large value differences
across FPN levels.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
import random
from collections import OrderedDict
from datetime import datetime

# Local imports
from model.factorized_prior import FactorizedPrior
from model.detection import DetectionModel
from data.kitti_dataset import KITTIDetectionDataset
from data import ImageDataset
from data.transforms import create_transforms
from utils import get_project_path
from utils.training_utils import (
    setup_training_environment, AverageMeter, EarlyStopping,
    log_entropy_bottleneck_stats, save_training_diagnostics,
    log_epoch_summary, save_best_model
)


def normalize_features(features):
    """Normalize features to have zero mean and unit variance."""
    # Compute statistics across spatial dimensions while preserving batch and channel dims
    mean = features.mean(dim=(-2, -1), keepdim=True)
    std = features.std(dim=(-2, -1), keepdim=True) + 1e-6  # Add epsilon for numerical stability
    return (features - mean) / std


def compute_normalized_mse_loss(x, x_hat):
    """Compute MSE loss between normalized features."""
    x_norm = normalize_features(x)
    x_hat_norm = normalize_features(x_hat)
    return F.mse_loss(x_norm, x_hat_norm)


# Reuse the dataset class from original file
# Import FPNRandomLevelDataset from utils
from utils.fpn_dataset import FPNRandomLevelDataset


def extract_fpn_features_batch(images, detection_model, device):
    """Extract FPN features from a batch of images."""
    with torch.no_grad():
        images = images.to(device)
        image_list = [img for img in images]
        fpn_features = detection_model.get_fpn_features(image_list)
    return fpn_features


def compute_level_bpp(level_features, likelihoods):
    """Compute bits per pixel for a single FPN level."""
    # Get number of pixels for this level
    level_pixels = level_features.size(-2) * level_features.size(-1)  # H * W
    batch_size = level_features.size(0)
    
    # Compute BPP for this level
    bpp = -torch.log2(likelihoods).sum() / (level_pixels * batch_size)
    return bpp


def train_one_epoch(model, train_loader, optimizer, lmbda, device, log_interval, detection_model):
    """Train for one epoch with random FPN level selection and normalized MSE."""
    model.train()
    detection_model.eval()  # Keep detection model in eval mode
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    grad_norm_meter = AverageMeter()
    
    # Track feature statistics for monitoring
    feature_stats = {
        'mean_abs': AverageMeter(),
        'max_abs': AverageMeter(),
        'std': AverageMeter(),
        'norm_diff': AverageMeter()  # Track difference in normalized space
    }
    
    # Track level distribution for monitoring
    level_counts = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}

    for batch_idx, (images, level_keys) in enumerate(train_loader):
        # Extract FPN features from images
        fpn_features = extract_fpn_features_batch(images, detection_model, device)
        
        # Process each sample individually since they may have different spatial sizes
        batch_loss = 0.0
        batch_bpp = 0.0
        batch_mse = 0.0
        batch_samples = len(level_keys)
        
        optimizer.zero_grad()
        
        for i, level_key in enumerate(level_keys):
            level_feature = fpn_features[level_key][i:i+1]  # [1, C, H, W]
            level_counts[level_key] += 1
            
            # Forward pass for this sample
            out = model(level_feature)
            level_feature_hat = out['x_hat']
            sample_likelihoods = out['likelihoods']

            # Compute metrics for this sample
            sample_pixels = level_feature.size(-2) * level_feature.size(-1)
            sample_bpp = -torch.log2(sample_likelihoods).sum() / sample_pixels
            
            # Compute normalized MSE loss
            sample_mse = compute_normalized_mse_loss(level_feature, level_feature_hat)
            sample_loss = sample_mse + lmbda * sample_bpp

            # Accumulate loss (will be normalized by batch size)
            batch_loss += sample_loss
            batch_bpp += sample_bpp.item()
            batch_mse += sample_mse.item()
            
            # Track feature statistics (on original features)
            feature_stats['mean_abs'].update(level_feature.abs().mean().item())
            feature_stats['max_abs'].update(level_feature.abs().max().item())
            feature_stats['std'].update(level_feature.std().item())
            
            # Track normalized feature differences
            norm_diff = (normalize_features(level_feature) - normalize_features(level_feature_hat)).abs().mean().item()
            feature_stats['norm_diff'].update(norm_diff)

        # Average loss across batch
        batch_loss = batch_loss / batch_samples
        
        # Backward pass
        batch_loss.backward()
        
        # Gradient clipping and monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_meter.update(grad_norm.item())
        
        optimizer.step()

        # Update meters with averaged values
        loss_meter.update(batch_loss.item())
        bpp_meter.update(batch_bpp / batch_samples)
        mse_meter.update(batch_mse / batch_samples)

        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            # Show level distribution in this batch
            batch_level_dist = [f"L{k}:{level_keys.count(k)}" for k in ['0','1','2','3','4'] if level_keys.count(k) > 0]
            
            logging.info(f'Train Batch [{batch_idx}/{len(train_loader)}] Levels: [{",".join(batch_level_dist)}] -> '
                        f'Loss: {loss_meter.avg:.4f} | '
                        f'MSE(norm): {mse_meter.avg:.6f} | '
                        f'BPP: {bpp_meter.avg:.4f} | '
                        f'GradNorm: {grad_norm_meter.avg:.3f} | '
                        f'LR: {current_lr:.2e} | '
                        f'FeatStats: mean_abs={feature_stats["mean_abs"].avg:.3f}, '
                        f'max_abs={feature_stats["max_abs"].avg:.3f}, '
                        f'norm_diff={feature_stats["norm_diff"].avg:.3f}')

    # Log level distribution for this epoch
    total_samples = sum(level_counts.values())
    level_dist_str = " | ".join([f"L{k}:{v}/{total_samples}({v/total_samples*100:.1f}%)" 
                                for k, v in level_counts.items()])
    logging.info(f'Epoch Level Distribution: {level_dist_str}')

    return loss_meter.avg, mse_meter.avg, bpp_meter.avg, grad_norm_meter.avg


def validate(model, val_loader, lmbda, device, detection_model):
    """Validate the model on mixed FPN levels with normalized MSE."""
    model.eval()
    detection_model.eval()
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    
    # Track per-level validation performance
    level_metrics = {str(i): {'loss': AverageMeter(), 'bpp': AverageMeter(), 'mse': AverageMeter()} 
                    for i in range(5)}

    with torch.no_grad():
        for images, level_keys in val_loader:
            # Extract FPN features from images
            fpn_features = extract_fpn_features_batch(images, detection_model, device)
            
            # Process each sample individually
            batch_loss = 0.0
            batch_bpp = 0.0
            batch_mse = 0.0
            batch_samples = len(level_keys)
            
            for i, level_key in enumerate(level_keys):
                level_feature = fpn_features[level_key][i:i+1]  # [1, C, H, W]
                
                # Forward pass for this sample
                out = model(level_feature)
                level_feature_hat = out['x_hat']
                sample_likelihoods = out['likelihoods']

                # Compute metrics for this sample
                sample_pixels = level_feature.size(-2) * level_feature.size(-1)
                sample_bpp = -torch.log2(sample_likelihoods).sum() / sample_pixels
                sample_mse = compute_normalized_mse_loss(level_feature, level_feature_hat)
                sample_loss = sample_mse + lmbda * sample_bpp

                # Accumulate batch metrics
                batch_loss += sample_loss.item()
                batch_bpp += sample_bpp.item()
                batch_mse += sample_mse.item()
                
                # Track per-level metrics
                level_metrics[level_key]['loss'].update(sample_loss.item())
                level_metrics[level_key]['bpp'].update(sample_bpp.item())
                level_metrics[level_key]['mse'].update(sample_mse.item())

            # Update overall meters with averaged values
            loss_meter.update(batch_loss / batch_samples)
            bpp_meter.update(batch_bpp / batch_samples)
            mse_meter.update(batch_mse / batch_samples)

    # Log per-level validation results
    level_results = []
    for level_key in ['0', '1', '2', '3', '4']:
        if level_metrics[level_key]['loss'].count > 0:
            level_results.append(
                f"L{level_key}: Loss={level_metrics[level_key]['loss'].avg:.4f}, "
                f"MSE(norm)={level_metrics[level_key]['mse'].avg:.6f}"
            )
    
    if level_results:
        logging.info(f"Validation per-level results: {' | '.join(level_results)}")

    return loss_meter.avg, mse_meter.avg, bpp_meter.avg


def save_mixed_level_analysis(model, val_loader, device, save_dir, epoch, lambda_val, 
                             detection_model, num_samples=4):
    """Save analysis of compression performance across all FPN levels."""
    model.eval()
    detection_model.eval()
    
    save_dir = Path(save_dir) / 'mixed_level_analysis' / f'lambda_{lambda_val:.3f}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_file = save_dir / f'mixed_levels_lambda_{lambda_val:.3f}_epoch_{epoch}_analysis.txt'
    
    with torch.no_grad(), open(analysis_file, 'w') as f:
        f.write(f"Mixed FPN Levels Compression Analysis - Epoch {epoch}, Lambda {lambda_val:.3f}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, (images, level_keys) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            # Extract FPN features
            fpn_features = extract_fpn_features_batch(images, detection_model, device)
            
            f.write(f"Batch {i}:\n")
            f.write(f"  Batch Size: {len(level_keys)}\n")
            f.write(f"  Levels in Batch: {level_keys}\n")
            
            # Process each sample individually
            batch_mse = 0.0
            batch_bpp = 0.0
            
            for j, level_key in enumerate(level_keys):
                level_feature = fpn_features[level_key][j:j+1]  # [1, C, H, W]
                
                # Compress and reconstruct this sample
                out = model(level_feature)
                level_feature_hat = out['x_hat']
                sample_likelihoods = out['likelihoods']
                
                # Compute metrics for this sample
                sample_pixels = level_feature.size(-2) * level_feature.size(-1)
                sample_bpp = -torch.log2(sample_likelihoods).sum() / sample_pixels
                sample_mse = compute_normalized_mse_loss(level_feature, level_feature_hat)
                
                batch_mse += sample_mse.item()
                batch_bpp += sample_bpp.item()
                
                # Compute statistics in both original and normalized space
                orig_norm = torch.norm(level_feature)
                recon_norm = torch.norm(level_feature_hat)
                
                norm_orig = normalize_features(level_feature)
                norm_recon = normalize_features(level_feature_hat)
                norm_diff = (norm_orig - norm_recon).abs().mean().item()
                
                f.write(f"    Sample {j} (Level {level_key}):\n")
                f.write(f"      Shape: {list(level_feature.shape)}\n")
                f.write(f"      Original Stats - Mean: {level_feature.mean().item():.3f}, "
                       f"Std: {level_feature.std().item():.3f}\n")
                f.write(f"      MSE (normalized): {sample_mse.item():.6f}\n")
                f.write(f"      BPP: {sample_bpp.item():.4f}\n")
                f.write(f"      Norm Ratio (original): {(recon_norm/orig_norm).item():.3f}\n")
                f.write(f"      Mean Normalized Diff: {norm_diff:.3f}\n")
            
            # Write overall batch metrics
            f.write(f"  Overall MSE (normalized): {batch_mse / len(level_keys):.6f}\n")
            f.write(f"  Overall BPP: {batch_bpp / len(level_keys):.4f}\n")
            
            f.write("\n")
        
        logging.info(f"Saved mixed-level analysis to {analysis_file}")


# Use the same main() function as in train_factorized_prior_random_sample_fpn.py
if __name__ == '__main__':
    from train_factorized_prior_random_sample_fpn import main
    main()