"""
Training script for per-level FPN feature compression using JointAutoregressiveHierarchicalPriors.

This script extends train_factorized_prior_random_sample_fpn.py by replacing the FactorizedPrior
model with JointAutoregressiveHierarchicalPriors from CompressAI, which typically achieves
better compression performance through a more sophisticated entropy model.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import yaml
import logging
from pathlib import Path

# Local imports
from model.detection import DetectionModel
from data.kitti_dataset import KITTIDetectionDataset
from data.transforms import create_transforms
from utils import get_project_path
from utils.training_utils import (
    setup_training_environment, AverageMeter, EarlyStopping,
    log_entropy_bottleneck_stats, save_training_diagnostics,
    log_epoch_summary, save_best_model
)

from utils.fpn_dataset import FPNDataset
from model.joint_autoregress_fpn_compressor import JointAutoregressFPNCompressor
from model.criterion import compute_rate_distortion_loss




def train_one_epoch(model, train_loader, optimizer, lmbda, device, log_interval, detection_model):
    """Train for one epoch with random FPN level selection."""
    model.train()
    detection_model.eval()
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    grad_norm_meter = AverageMeter()
    

    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        # Extract FPN features from images
        fpn_features = detection_model.get_fpn_features(images)
        
        optimizer.zero_grad()
            
        # Forward pass
        out = model(fpn_features)

        # Compute loss
        batch_loss, batch_mse, batch_bpp = compute_rate_distortion_loss(out, fpn_features, lmbda, images.shape)
                
        # Backward pass
        batch_loss.backward()

        
        # Gradient clipping and monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_meter.update(grad_norm.item())
        
        optimizer.step()

        # Update meters with averaged values
        loss_meter.update(batch_loss.item())
        bpp_meter.update(batch_bpp.item())
        mse_meter.update(batch_mse.item())

        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            logging.info(f'Train Batch [{batch_idx}/{len(train_loader)}] -> '
                        f'Loss: {loss_meter.avg:.4f} | '
                        f'MSE: {mse_meter.avg:.6f} | '
                        f'BPP: {bpp_meter.avg:.4f} | '
                        f'GradNorm: {grad_norm_meter.avg:.3f} | '
                        f'LR: {current_lr:.2e}')


    return loss_meter.avg, mse_meter.avg, bpp_meter.avg, grad_norm_meter.avg


def validate(model, val_loader, lmbda, device, detection_model):
    """Validate the model on mixed FPN levels."""
    model.eval()
    detection_model.eval()
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    
    # Track per-level validation performance
    metrics = {'loss': AverageMeter(), 'bpp': AverageMeter(), 'mse': AverageMeter()} 

    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            fpn_features = detection_model.get_fpn_features(images)
            # Forward pass
            out = model(fpn_features)
            
            # Compute loss
            batch_loss, batch_mse, batch_bpp = compute_rate_distortion_loss(out, fpn_features, lmbda, images.shape)
            
            # Accumulate batch metrics
            loss_meter.update(batch_loss.item())
            bpp_meter.update(batch_bpp.item())
            mse_meter.update(batch_mse.item())
            
            # Track per-level metrics
            metrics['loss'].update(batch_loss.item())
            metrics['bpp'].update(batch_bpp.item())
            metrics['mse'].update(batch_mse.item())


    # Log per-level validation results
    level_results = []
    if metrics['loss'].count > 0:
        level_results.append(
            f"Loss={metrics['loss'].avg:.4f}, "
                f"MSE={metrics['mse'].avg:.6f}, "
                f"BPP={metrics['bpp'].avg:.4f}"
            )
    
    if level_results:
        logging.info(f"Validation per-level results: {' | '.join(level_results)}")

    return loss_meter.avg, mse_meter.avg, bpp_meter.avg


def main():
    parser = argparse.ArgumentParser(description='Train per-level FPN feature compression models')
    parser.add_argument('--config', type=str, 
                       default=get_project_path('configs/train_joint_autoregress_prior_fused_feature.yaml'),
                       help='Path to config file')
    parser.add_argument('--detection_checkpoint', type=str, 
                       default=get_project_path('checkpoints/detection/run_0.002000_16/best_model.pth'),
                       help='Path to detection model checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup training environment
    save_dir = setup_training_environment(config['training']['save_dir'], config, args)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load detection model
    if os.path.exists(args.detection_checkpoint):
        checkpoint = torch.load(args.detection_checkpoint, map_location=device)
        num_classes = len(KITTIDetectionDataset.CLASSES) + 1
        detection_model = DetectionModel(num_classes=num_classes, pretrained=False)
        detection_model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded detection model from {args.detection_checkpoint}")
    else:
        logging.warning(f"Detection checkpoint not found: {args.detection_checkpoint}")
        detection_model = DetectionModel(num_classes=2)
    
    detection_model = detection_model.to(device)
    detection_model.eval()
    
    # Create compression model
    model = JointAutoregressFPNCompressor(
        N=config['model']['n_latent'],
        M=config['model']['n_hidden'],
        input_channels=config['model']['fpn_channels_per_level'],  # 256
        output_channels=config['model']['fpn_channels_per_level']  # 256
    ).to(device)
    
    logging.info(f"Created JointAutoregressFPNCompressor model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup data loading
    train_transform = create_transforms(config['data']['transforms'], split='train')
    val_transform = create_transforms(config['data']['test_transforms'], split='val')
    
    train_dataset = FPNDataset(
        txt_file=config['data']['train_list'],
        transform=train_transform
    )
    val_dataset = FPNDataset(
        txt_file=config['data']['val_list'],
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
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
    logging.info("Starting training with JointAutoregressiveHierarchicalPriors model...")
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss, train_mse, train_bpp, train_grad_norm = train_one_epoch(
            model, train_loader, optimizer, lmbda, device,
            config['training']['log_interval'], detection_model
        )
        
        # Validate
        val_loss, val_mse, val_bpp = validate(
            model, val_loader, lmbda, device, detection_model
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
    
    logging.info(f"Training completed successfully!")
    logging.info(f"Best validation loss: {best_loss:.4f}")
    logging.info(f"Model checkpoint saved to: {save_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()