"""
Training script for specific-level FPN feature compression.

This script trains a FactorizedPrior model for a single, specified FPN level (0-4).
Unlike random level sampling, this approach focuses on one level to analyze
level-specific compression characteristics.
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


def parse_args():
    parser = argparse.ArgumentParser(description='Train specific-level FPN feature compression model')
    parser.add_argument('--config', type=str, 
                       default=get_project_path('configs/train_factorized_prior_random_sample_fpn.yaml'),
                       help='Path to config file')
    parser.add_argument('--detection_checkpoint', type=str, 
                       default=get_project_path('checkpoints/detection/run_0.002000_16/best_model.pth'),
                       help='Path to detection model checkpoint')
    parser.add_argument('--fpn_level', type=int, required=True, choices=[0, 1, 2, 3, 4],
                       help='Specific FPN level to train on (0-4)')
    return parser.parse_args()


class FPNSpecificLevelDataset(torch.utils.data.Dataset):
    """Dataset that uses only a specific FPN level for training."""
    
    def __init__(self, txt_file, transform, fpn_level, training=True):
        """
        Args:
            txt_file: Path to image list file
            transform: Image transforms (applied to input images)
            fpn_level: Specific FPN level to use (0-4)
            training: Training mode flag (for consistency with random version)
        """
        self.image_dataset = ImageDataset(txt_file, transform)
        self.fpn_level = str(fpn_level)  # Convert to string for consistency
        self.training = training
        
        logging.info(f"Dataset initialized for FPN level {fpn_level}")
    
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        # Get preprocessed image
        image = self.image_dataset[idx]  # [C, H, W] - already preprocessed
        
        # Always use the specific level
        selected_level = self.fpn_level
        
        return image, selected_level


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


def train_one_epoch(model, train_loader, optimizer, lmbda, device, log_interval, detection_model, fpn_level):
    """Train for one epoch with specific FPN level."""
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
        'std': AverageMeter()
    }

    for batch_idx, (images, level_keys) in enumerate(train_loader):
        # Extract FPN features from images
        fpn_features = extract_fpn_features_batch(images, detection_model, device)
        
        # Process each sample - all should have the same level
        batch_loss = 0.0
        batch_bpp = 0.0
        batch_mse = 0.0
        batch_samples = len(level_keys)
        
        # Verify all samples use the same level
        assert all(level_key == str(fpn_level) for level_key in level_keys), \
            f"All samples should use level {fpn_level}, got {level_keys}"
        
        optimizer.zero_grad()
        
        for i, level_key in enumerate(level_keys):
            level_feature = fpn_features[level_key][i:i+1]  # [1, C, H, W]
            
            # Forward pass for this sample
            out = model(level_feature)
            level_feature_hat = out['x_hat']
            sample_likelihoods = out['likelihoods']

            # Compute metrics for this sample
            sample_pixels = level_feature.size(-2) * level_feature.size(-1)
            sample_bpp = -torch.log2(sample_likelihoods).sum() / sample_pixels
            sample_mse = F.mse_loss(level_feature, level_feature_hat)
            sample_loss = sample_mse + lmbda * sample_bpp

            # Accumulate loss (will be normalized by batch size)
            batch_loss += sample_loss
            batch_bpp += sample_bpp.item()
            batch_mse += sample_mse.item()
            
            # Track feature statistics
            feature_stats['mean_abs'].update(level_feature.abs().mean().item())
            feature_stats['max_abs'].update(level_feature.abs().max().item())
            feature_stats['std'].update(level_feature.std().item())

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
            
            logging.info(f'Train Batch [{batch_idx}/{len(train_loader)}] Level {fpn_level} -> '
                        f'Loss: {loss_meter.avg:.4f} | '
                        f'MSE: {mse_meter.avg:.6f} | '
                        f'BPP: {bpp_meter.avg:.4f} | '
                        f'GradNorm: {grad_norm_meter.avg:.3f} | '
                        f'LR: {current_lr:.2e} | '
                        f'FeatStats: mean_abs={feature_stats["mean_abs"].avg:.3f}, '
                        f'max_abs={feature_stats["max_abs"].avg:.3f}')
            
            # Feature magnitude warnings
            if feature_stats['max_abs'].avg > 100.0:
                logging.warning(f'⚠️  Large features detected: max_abs={feature_stats["max_abs"].avg:.2f}')

    logging.info(f'Epoch completed for FPN Level {fpn_level}')
    return loss_meter.avg, mse_meter.avg, bpp_meter.avg, grad_norm_meter.avg


def validate(model, val_loader, lmbda, device, detection_model, fpn_level):
    """Validate the model on the specific FPN level."""
    model.eval()
    detection_model.eval()
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()

    with torch.no_grad():
        for images, level_keys in val_loader:
            # Extract FPN features from images
            fpn_features = extract_fpn_features_batch(images, detection_model, device)
            
            # Process each sample
            batch_loss = 0.0
            batch_bpp = 0.0
            batch_mse = 0.0
            batch_samples = len(level_keys)
            
            # Verify all samples use the same level
            assert all(level_key == str(fpn_level) for level_key in level_keys), \
                f"All samples should use level {fpn_level}, got {level_keys}"
            
            for i, level_key in enumerate(level_keys):
                level_feature = fpn_features[level_key][i:i+1]  # [1, C, H, W]
                
                # Forward pass for this sample
                out = model(level_feature)
                level_feature_hat = out['x_hat']
                sample_likelihoods = out['likelihoods']

                # Compute metrics for this sample
                sample_pixels = level_feature.size(-2) * level_feature.size(-1)
                sample_bpp = -torch.log2(sample_likelihoods).sum() / sample_pixels
                sample_mse = F.mse_loss(level_feature, level_feature_hat)
                sample_loss = sample_mse + lmbda * sample_bpp

                # Accumulate batch metrics
                batch_loss += sample_loss.item()
                batch_bpp += sample_bpp.item()
                batch_mse += sample_mse.item()

            # Update overall meters with averaged values
            loss_meter.update(batch_loss / batch_samples)
            bpp_meter.update(batch_bpp / batch_samples)
            mse_meter.update(batch_mse / batch_samples)

    logging.info(f"Validation Level {fpn_level}: Loss={loss_meter.avg:.4f}, "
                f"MSE={mse_meter.avg:.6f}, BPP={bpp_meter.avg:.4f}")

    return loss_meter.avg, mse_meter.avg, bpp_meter.avg


def save_level_analysis(model, val_loader, device, save_dir, epoch, lambda_val, 
                       detection_model, fpn_level, num_samples=4):
    """Save analysis of compression performance for the specific FPN level."""
    model.eval()
    detection_model.eval()
    
    save_dir = Path(save_dir) / 'level_analysis' / f'level_{fpn_level}' / f'lambda_{lambda_val:.3f}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_file = save_dir / f'level_{fpn_level}_lambda_{lambda_val:.3f}_epoch_{epoch}_analysis.txt'
    
    with torch.no_grad(), open(analysis_file, 'w') as f:
        f.write(f"FPN Level {fpn_level} Compression Analysis - Epoch {epoch}, Lambda {lambda_val:.3f}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, (images, level_keys) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            # Extract FPN features
            fpn_features = extract_fpn_features_batch(images, detection_model, device)
            
            f.write(f"Batch {i}:\n")
            f.write(f"  Batch Size: {len(level_keys)}\n")
            f.write(f"  Level: {fpn_level} (all samples)\n")
            
            # Process each sample
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
                sample_mse = F.mse_loss(level_feature, level_feature_hat)
                
                batch_mse += sample_mse.item()
                batch_bpp += sample_bpp.item()
                
                orig_norm = torch.norm(level_feature)
                recon_norm = torch.norm(level_feature_hat)
                
                f.write(f"    Sample {j}:\n")
                f.write(f"      Shape: {list(level_feature.shape)}\n")
                f.write(f"      MSE: {sample_mse.item():.6f}\n")
                f.write(f"      BPP: {sample_bpp.item():.4f}\n")
                f.write(f"      Norm Ratio: {(recon_norm/orig_norm).item():.3f}\n")
            
            # Write overall batch metrics
            f.write(f"  Overall MSE: {batch_mse / len(level_keys):.6f}\n")
            f.write(f"  Overall BPP: {batch_bpp / len(level_keys):.4f}\n")
            
            f.write("\n")
        
        logging.info(f"Saved level {fpn_level} analysis to {analysis_file}")


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are correct type
    config['model']['n_hidden'] = int(config['model']['n_hidden'])
    config['model']['n_channels'] = int(config['model']['n_channels'])
    config['model']['fpn_channels_per_level'] = int(config['model'].get('fpn_channels_per_level', 256))
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['epochs'] = int(config['training']['epochs'])
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['lambda'] = float(config['training']['lambda'])
    config['training']['num_workers'] = int(config['training']['num_workers'])
    config['training']['log_interval'] = int(config['training']['log_interval'])
    
    # Ensure optimizer parameters are correct types
    config['training']['optimizer']['eps'] = float(config['training']['optimizer']['eps'])
    config['training']['optimizer']['weight_decay'] = float(config['training']['optimizer']['weight_decay'])
    
    # Ensure learning rate schedule parameters are correct types
    config['training']['lr_schedule']['factor'] = float(config['training']['lr_schedule']['factor'])
    config['training']['lr_schedule']['patience'] = int(config['training']['lr_schedule']['patience'])
    config['training']['lr_schedule']['min_lr'] = float(config['training']['lr_schedule']['min_lr'])
    
    # Ensure early stopping parameters are correct types
    config['training']['early_stopping']['patience'] = int(config['training']['early_stopping']['patience'])
    config['training']['early_stopping']['min_delta'] = float(config['training']['early_stopping']['min_delta'])
    
    # Ensure validation parameters are correct types
    config['validation']['interval'] = int(config['validation']['interval'])
    config['validation']['num_samples'] = int(config['validation']['num_samples'])
    
    # Modify save directory to include FPN level for distinction
    base_save_dir = config['training']['save_dir']
    config['training']['save_dir'] = f"{base_save_dir}_level_{args.fpn_level}"
    
    # Setup training environment
    save_dir = setup_training_environment(config['training']['save_dir'], config, args)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load detection model for FPN feature extraction
    logging.info("Loading detection model for FPN feature extraction...")
    if os.path.exists(args.detection_checkpoint):
        checkpoint = torch.load(args.detection_checkpoint, map_location=device)
        num_classes = len(KITTIDetectionDataset.CLASSES) + 1
        detection_model = DetectionModel(num_classes=num_classes, pretrained=False)
        detection_model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded detection model from {args.detection_checkpoint}")
        logging.info(f"Detection model has {num_classes} classes")
    else:
        logging.warning(f"Detection checkpoint not found: {args.detection_checkpoint}")
        detection_model = DetectionModel(num_classes=2)
    
    detection_model = detection_model.to(device)
    detection_model.eval()
    
    logging.info(f"\n{'='*60}")
    logging.info(f"TRAINING MODEL FOR SPECIFIC FPN LEVEL {args.fpn_level}")
    logging.info(f"{'='*60}")
    
    # Create model for the specific FPN level (256 channels input/output)
    model = FactorizedPrior(
        n_hidden=config['model']['n_hidden'],
        n_channels=config['model']['n_channels'],
        input_channels=config['model']['fpn_channels_per_level'],  # 256
        output_channels=config['model']['fpn_channels_per_level']  # 256
    ).to(device)
    
    logging.info(f"Created FPN Level {args.fpn_level} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup data loading with specific level selection
    train_transform = create_transforms(config['data']['transforms'], split='train')
    val_transform = create_transforms(config['data']['test_transforms'], split='val')
    
    train_dataset = FPNSpecificLevelDataset(
        txt_file=config['data']['train_list'],
        transform=train_transform,
        fpn_level=args.fpn_level,
        training=True
    )
    val_dataset = FPNSpecificLevelDataset(
        txt_file=config['data']['val_list'],
        transform=val_transform,
        fpn_level=args.fpn_level,
        training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    logging.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Setup optimization
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['optimizer']['betas'],
        eps=config['training']['optimizer']['eps'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Setup learning rate scheduler
    scheduler = None
    if config['training']['lr_schedule']['enabled']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['lr_schedule']['factor'],
            patience=config['training']['lr_schedule']['patience'],
            min_lr=config['training']['lr_schedule']['min_lr']
        )
    
    # Setup early stopping
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
    
    # Training parameters
    lmbda = config['training']['lambda']
    best_loss = float('inf')
    
    # Training loop
    logging.info(f"Starting training for FPN level {args.fpn_level}...")
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss, train_mse, train_bpp, train_grad_norm = train_one_epoch(
            model, train_loader, optimizer, lmbda, device,
            config['training']['log_interval'], detection_model, args.fpn_level
        )
        
        # Validate
        val_loss, val_mse, val_bpp = validate(
            model, val_loader, lmbda, device, detection_model, args.fpn_level
        )
        
        # Log epoch summary
        train_metrics = (train_loss, train_mse, train_bpp, train_grad_norm)
        val_metrics = (val_loss, val_mse, val_bpp)
        log_epoch_summary(epoch, train_metrics, val_metrics, optimizer, config)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Learning rate updated: {current_lr:.6f}')
        
        # Log entropy bottleneck statistics
        if epoch % 5 == 0:
            log_entropy_bottleneck_stats(model, device)
        
        # Save level-specific analysis
        if config['validation']['save_reconstructions'] and epoch % config['validation']['interval'] == 0:
            save_level_analysis(
                model, val_loader, device, save_dir, epoch,
                config['training']['lambda'], detection_model, args.fpn_level,
                config['validation']['num_samples']
            )
        
        # Save training diagnostics
        save_training_diagnostics(save_dir, epoch, train_metrics, val_metrics, optimizer, model, config)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_best_model(save_dir, epoch, model, optimizer, scheduler, best_loss, config)
        
        # Early stopping
        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info('Early stopping triggered')
                break
    
    # Save final training summary
    summary_file = save_dir / 'training_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"FPN Level {args.fpn_level} Compression Training Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Approach: Single model trained on FPN level {args.fpn_level} only\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Detection Checkpoint: {args.detection_checkpoint}\n")
        f.write(f"FPN Level: {args.fpn_level}\n")
        f.write(f"Lambda: {config['training']['lambda']:.3f}\n")
        f.write(f"Learning Rate: {config['training']['learning_rate']:.2e}\n")
        f.write(f"Batch Size: {config['training']['batch_size']}\n")
        f.write(f"Total Epochs: {epoch + 1}\n")
        f.write(f"Best Validation Loss: {best_loss:.4f}\n\n")
        
        f.write("Model Architecture:\n")
        f.write(f"  Input Channels: {config['model']['fpn_channels_per_level']} (FPN features)\n")
        f.write(f"  Output Channels: {config['model']['fpn_channels_per_level']} (FPN features)\n")
        f.write(f"  Hidden Channels: {config['model']['n_hidden']}\n")
        f.write(f"  Latent Channels: {config['model']['n_channels']}\n")
        f.write(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")
        
        f.write("Training Strategy:\n")
        f.write(f"  - Model specialized for FPN level {args.fpn_level}\n")
        f.write(f"  - All training samples use level {args.fpn_level} features\n")
        f.write(f"  - Level {args.fpn_level} has 256 channels\n")
        f.write(f"  - Optimized for level-specific characteristics\n")
    
    logging.info(f"Training completed successfully for FPN level {args.fpn_level}!")
    logging.info(f"Best validation loss: {best_loss:.4f}")
    logging.info(f"Summary saved to: {summary_file}")
    logging.info(f"Model checkpoint saved to: {save_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()