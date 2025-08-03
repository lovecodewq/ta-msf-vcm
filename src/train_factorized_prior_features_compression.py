import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from model.factorized_prior_features_compression import FactorizedPriorFPNCompression
from model.detection import DetectionModel
from data.kitti_dataset import KITTIDetectionDataset
import yaml
import logging
from pathlib import Path
import numpy as np
from utils import get_project_path, AverageMeter, setup_training_environment
from data import ImageDataset
from data.transforms import create_transforms
from collections import OrderedDict
import torch.nn.functional as F
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=get_project_path('configs/train_factorized_prior_fpn.yaml'))
    parser.add_argument('--detection_checkpoint', type=str, 
                       default=get_project_path('checkpoints/detection/run_0.002000_16/best_model.pth'),
                       help='Path to detection model checkpoint')
    return parser.parse_args()

class FPNFeatureDataset(torch.utils.data.Dataset):
    """Dataset that extracts FPN features from images using a detection model.
    
    Flow:
    1. Load raw image from dataset
    2. Apply image preprocessing (resize, normalize) - THIS IS FOR THE INPUT IMAGE
    3. Feed preprocessed image to detection model
    4. Extract FPN features from detection model
    5. Return FPN features (no additional transforms applied to features)
    """
    
    def __init__(self, txt_file, transform):
        self.image_dataset = ImageDataset(txt_file, transform)  # transform applied to images, not FPN features
    
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        # Get preprocessed image from the base dataset (transforms already applied)
        image = self.image_dataset[idx]  # [C, H, W] - already resized and normalized
        
        # Keep image on CPU - we'll move to GPU and extract features in collate_fn
        return image

def compute_fpn_mse(fpn_features_orig, fpn_features_recon):
    """Compute MSE loss between original and reconstructed FPN features."""
    total_mse = 0.0
    total_elements = 0
    
    for key in fpn_features_orig.keys():
        if key in fpn_features_recon:
            orig = fpn_features_orig[key]
            recon = fpn_features_recon[key]
            
            # Compute MSE for this level
            level_mse = F.mse_loss(orig, recon, reduction='sum')
            total_mse += level_mse
            total_elements += orig.numel()
    
    # Return average MSE across all elements
    return total_mse / total_elements if total_elements > 0 else 0.0

def compute_fpn_bpp(fpn_features, likelihoods):
    """Compute bits per pixel for FPN features."""
    # Get total number of pixels across all FPN levels
    total_pixels = 0
    for feat in fpn_features.values():
        total_pixels += feat.size(-2) * feat.size(-1)  # H * W for each level
    
    # Compute rate
    bpp = -torch.log2(likelihoods).sum() / (total_pixels * fpn_features['0'].size(0))  # Normalize by batch size
    return bpp

def train_one_epoch(model, train_loader, optimizer, lmbda, device, log_interval, detection_model):
    """Train for one epoch."""
    model.train()
    detection_model.eval()  # Keep detection model in eval mode
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    grad_norm_meter = AverageMeter()
    
    # Individual FPN level tracking
    fpn_level_mse = {str(i): AverageMeter() for i in range(5)}

    for batch_idx, images in enumerate(train_loader):
        # Move images to device and extract FPN features in main process
        images = images.to(device)  # [B, C, H, W]
        image_list = [img for img in images]  # Convert to list format for detection model
        
        with torch.no_grad():
            fpn_features_device = detection_model.get_fpn_features(image_list)

        optimizer.zero_grad()

        # Forward pass
        out = model(fpn_features_device)
        fpn_features_hat = out['fpn_features_hat']
        likelihoods = out['likelihoods']

        # Compute rate (bits per pixel equivalent for FPN features)
        bpp = compute_fpn_bpp(fpn_features_device, likelihoods)

        # Compute distortion (mean squared error between FPN features)
        mse = compute_fpn_mse(fpn_features_device, fpn_features_hat)
        
        # Compute individual FPN level MSEs for detailed analysis
        fpn_feature_stats = {}
        for level_key in fpn_features_device.keys():
            level_mse = F.mse_loss(fpn_features_device[level_key], fpn_features_hat[level_key])
            fpn_level_mse[level_key].update(level_mse.item())
            
            # Track feature magnitudes for stability monitoring
            orig_feat = fpn_features_device[level_key]
            fpn_feature_stats[level_key] = {
                'mean_abs': orig_feat.abs().mean().item(),
                'max_abs': orig_feat.abs().max().item(),
                'std': orig_feat.std().item()
            }

        # Total loss = MSE + lambda * Rate
        loss = mse + lmbda * bpp

        # Backward pass
        loss.backward()
        
        # Compute gradient norm BEFORE clipping (for monitoring)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_meter.update(grad_norm.item())
        
        optimizer.step()

        # Update meters
        loss_meter.update(loss.item())
        bpp_meter.update(bpp.item())
        mse_meter.update(mse.item())

        if batch_idx % log_interval == 0:
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Individual FPN level MSEs for detailed analysis
            fpn_mse_str = " ".join([f"L{k}:{v.avg:.5f}" for k, v in fpn_level_mse.items()])
            
            logging.info(f'Train Batch: [{batch_idx}/{len(train_loader)}] '
                        f'Loss: {loss_meter.avg:.4f} '
                        f'MSE: {mse_meter.avg:.6f} '
                        f'BPP: {bpp_meter.avg:.4f} '
                        f'GradNorm: {grad_norm_meter.avg:.3f} '
                        f'LR: {current_lr:.2e} '
                        f'FPN_MSE: [{fpn_mse_str}]')
            
            # Log feature magnitude warnings for stability monitoring
            for level_key, stats in fpn_feature_stats.items():
                if stats['max_abs'] > 100.0:  # Threshold for concerning feature magnitudes
                    logging.warning(f'⚠️  Large FPN features detected L{level_key}: max_abs={stats["max_abs"]:.2f}, mean_abs={stats["mean_abs"]:.2f}')
                    
            # Log detailed stats every 50 batches for debugging
            if batch_idx % (log_interval * 5) == 0 and batch_idx > 0:
                feature_stats_str = " ".join([f"L{k}:max={v['max_abs']:.2f}" for k, v in fpn_feature_stats.items()])
                logging.info(f'Feature Magnitudes: [{feature_stats_str}]')

    return loss_meter.avg, mse_meter.avg, bpp_meter.avg, grad_norm_meter.avg

def validate(model, val_loader, lmbda, device, detection_model):
    """Validate the model."""
    model.eval()
    detection_model.eval()  # Keep detection model in eval mode
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()

    with torch.no_grad():
        for images in val_loader:
            # Move images to device and extract FPN features in main process
            images = images.to(device)  # [B, C, H, W]
            image_list = [img for img in images]  # Convert to list format for detection model
            
            fpn_features_device = detection_model.get_fpn_features(image_list)

            out = model(fpn_features_device)
            fpn_features_hat = out['fpn_features_hat']
            likelihoods = out['likelihoods']

            bpp = compute_fpn_bpp(fpn_features_device, likelihoods)
            mse = compute_fpn_mse(fpn_features_device, fpn_features_hat)
            loss = mse + lmbda * bpp

            loss_meter.update(loss.item())
            bpp_meter.update(bpp.item())
            mse_meter.update(mse.item())

    return loss_meter.avg, mse_meter.avg, bpp_meter.avg

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=20, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def log_entropy_bottleneck_stats(model, device):
    """Log entropy bottleneck statistics for debugging."""
    try:
        entropy_bottleneck = model.entropy_bottleneck
        if hasattr(entropy_bottleneck, '_medians') and entropy_bottleneck._medians is not None:
            medians = entropy_bottleneck._medians.flatten()
            logging.info(f'Entropy Bottleneck Medians -> Mean: {medians.mean():.4f} | Std: {medians.std():.4f} | Range: [{medians.min():.4f}, {medians.max():.4f}]')
        
        if hasattr(entropy_bottleneck, '_bias') and entropy_bottleneck._bias is not None:
            bias = entropy_bottleneck._bias.flatten()
            logging.info(f'Entropy Bottleneck Bias -> Mean: {bias.mean():.4f} | Std: {bias.std():.4f}')
    except Exception as e:
        logging.debug(f'Could not log entropy bottleneck stats: {e}')

def save_fpn_feature_analysis(model, val_loader, device, save_dir, epoch, lambda_val, detection_model, num_samples=4):
    """Save FPN feature compression analysis.
    
    Args:
        model: The compression model
        val_loader: Validation data loader
        device: Device to run on
        save_dir: Base directory for saving results
        epoch: Current epoch number
        lambda_val: Current lambda value for rate-distortion trade-off
        detection_model: Detection model for FPN feature extraction
        num_samples: Number of samples to analyze
    """
    model.eval()
    detection_model.eval()  # Keep detection model in eval mode
    # Create lambda-specific directory
    save_dir = Path(save_dir) / 'fpn_analysis' / f'lambda_{lambda_val:.3f}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_file = save_dir / f'lambda_{lambda_val:.3f}_epoch_{epoch}_analysis.txt'
    
    with torch.no_grad(), open(analysis_file, 'w') as f:
        f.write(f"FPN Feature Compression Analysis - Epoch {epoch}, Lambda {lambda_val:.3f}\n")
        f.write("=" * 60 + "\n\n")
        
        for i, images in enumerate(val_loader):
            if i >= num_samples:
                break
                
            # Move images to device and extract FPN features in main process
            images = images.to(device)  # [B, C, H, W]
            image_list = [img for img in images]  # Convert to list format for detection model
            
            fpn_features_device = detection_model.get_fpn_features(image_list)
            
            out = model(fpn_features_device)
            fpn_features_hat = out['fpn_features_hat']
            likelihoods = out['likelihoods']
            
            # Compute metrics
            bpp = compute_fpn_bpp(fpn_features_device, likelihoods)
            mse = compute_fpn_mse(fpn_features_device, fpn_features_hat)
            
            f.write(f"Sample {i}:\n")
            f.write(f"  Overall MSE: {mse.item():.6f}\n")
            f.write(f"  Overall BPP: {bpp.item():.4f}\n")
            f.write(f"  FPN Level Analysis:\n")
            
            for level_key in fpn_features_device.keys():
                orig_feat = fpn_features_device[level_key]
                recon_feat = fpn_features_hat[level_key]
                
                level_mse = F.mse_loss(orig_feat, recon_feat)
                orig_norm = torch.norm(orig_feat)
                recon_norm = torch.norm(recon_feat)
                
                f.write(f"    Level {level_key}: Shape {list(orig_feat.shape)}, "
                       f"MSE: {level_mse.item():.6f}, "
                       f"Orig Norm: {orig_norm.item():.3f}, "
                       f"Recon Norm: {recon_norm.item():.3f}\n")
            f.write("\n")
        
        logging.info(f"Saved FPN analysis to {analysis_file}")

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are of correct type
    config['model']['n_hidden'] = int(config['model']['n_hidden'])
    config['model']['n_channels'] = int(config['model']['n_channels'])
    config['model']['fpn_channels_per_level'] = int(config['model'].get('fpn_channels_per_level', 256))
    config['model']['num_fpn_levels'] = int(config['model'].get('num_fpn_levels', 5))
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
    
    # Setup training environment (logging, directories, metadata)
    save_dir = setup_training_environment(config['training']['save_dir'], config, args)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load detection model for FPN feature extraction
    logging.info("Loading detection model for FPN feature extraction...")
    
    if os.path.exists(args.detection_checkpoint):
        # Load checkpoint to determine correct number of classes
        checkpoint = torch.load(args.detection_checkpoint, map_location=device)
        # Create model with same number of classes as KITTI dataset (+1 for background)
        num_classes = len(KITTIDetectionDataset.CLASSES) + 1
        detection_model = DetectionModel(num_classes=num_classes, pretrained=False)
        detection_model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded detection model from {args.detection_checkpoint}")
        logging.info(f"Model has {num_classes} classes")
    else:
        logging.warning(f"Detection checkpoint not found: {args.detection_checkpoint}")
        logging.info("Using pretrained detection model weights")
        # Use default pretrained model (fallback with 2 classes)
        detection_model = DetectionModel(num_classes=2)
    
    detection_model = detection_model.to(device)
    detection_model.eval()
    
    # Create FPN compression model
    model = FactorizedPriorFPNCompression(
        n_hidden=config['model']['n_hidden'],
        n_channels=config['model']['n_channels'],
        fpn_channels_per_level=config['model']['fpn_channels_per_level'],
        num_fpn_levels=config['model']['num_fpn_levels']
    ).to(device)
    
    logging.info(f"Created FPN compression model with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup data loading with FPN features
    # Note: These transforms are applied to INPUT IMAGES, not FPN features
    train_transform = create_transforms(config['data']['transforms'], split='train')
    val_transform = create_transforms(config['data']['test_transforms'], split='val')
    
    train_dataset = FPNFeatureDataset(
        txt_file=config['data']['train_list'],
        transform=train_transform
    )
    val_dataset = FPNFeatureDataset(
        txt_file=config['data']['val_list'],
        transform=val_transform
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
    if config['training']['lr_schedule']['enabled']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['lr_schedule']['factor'],
            patience=config['training']['lr_schedule']['patience'],
            min_lr=config['training']['lr_schedule']['min_lr']
        )
    
    # Setup early stopping
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
    
    # Rate-distortion trade-off parameter (lambda)
    lmbda = config['training']['lambda']
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss, train_mse, train_bpp, train_grad_norm = train_one_epoch(
            model, train_loader, optimizer, lmbda, device,
            config['training']['log_interval'], detection_model
        )
        
        # Validate
        val_loss, val_mse, val_bpp = validate(model, val_loader, lmbda, device, detection_model)
        
        # Enhanced epoch logging with diagnostics
        current_lr = optimizer.param_groups[0]['lr']
        rate_weight = lmbda * val_bpp
        distortion_weight = val_mse
        
        logging.info(f'=== EPOCH {epoch} SUMMARY ===')
        logging.info(f'Training   -> Loss: {train_loss:.4f} | MSE: {train_mse:.6f} | BPP: {train_bpp:.4f} | GradNorm: {train_grad_norm:.3f}')
        logging.info(f'Validation -> Loss: {val_loss:.4f} | MSE: {val_mse:.6f} | BPP: {val_bpp:.4f}')
        logging.info(f'Rate-Distortion Balance -> λ×BPP: {rate_weight:.6f} | MSE: {distortion_weight:.6f} | Ratio: {rate_weight/distortion_weight:.3f}')
        logging.info(f'Optimization -> LR: {current_lr:.2e} | Lambda: {lmbda:.2e}')
        
        # Gradient explosion warning
        if train_grad_norm > 10.0:
            logging.warning(f'⚠️  HIGH GRADIENT NORM: {train_grad_norm:.3f} - Potential instability!')
        
        # Rate-distortion balance warning  
        rd_ratio = rate_weight / distortion_weight
        if rd_ratio > 10.0:
            logging.warning(f'⚠️  RATE DOMINATES: Ratio={rd_ratio:.1f} - Consider reducing lambda')
        elif rd_ratio < 0.1:
            logging.warning(f'⚠️  DISTORTION DOMINATES: Ratio={rd_ratio:.3f} - Consider increasing lambda')
        
        # Learning rate scheduling
        if config['training']['lr_schedule']['enabled']:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Learning rate updated: {current_lr:.6f}')
        
        # Log entropy bottleneck statistics every 5 epochs
        if epoch % 5 == 0:
            log_entropy_bottleneck_stats(model, device)
        
        # Save FPN feature analysis
        if config['validation']['save_reconstructions'] and epoch % config['validation']['interval'] == 0:
            save_fpn_feature_analysis(
                model, val_loader, device, save_dir, epoch,
                config['training']['lambda'], detection_model,
                config['validation']['num_samples']
            )
        
        # Save diagnostic summary every epoch
        diagnostic_file = save_dir / f'training_diagnostics_epoch_{epoch}.txt'
        with open(diagnostic_file, 'w') as f:
            f.write(f"Training Diagnostics - Epoch {epoch}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Loss Metrics:\n")
            f.write(f"  Training Loss: {train_loss:.6f}\n")
            f.write(f"  Validation Loss: {val_loss:.6f}\n")
            f.write(f"  Training MSE: {train_mse:.6f}\n")
            f.write(f"  Validation MSE: {val_mse:.6f}\n")
            f.write(f"  Training BPP: {train_bpp:.6f}\n")
            f.write(f"  Validation BPP: {val_bpp:.6f}\n\n")
            
            f.write(f"Optimization Health:\n")
            f.write(f"  Gradient Norm: {train_grad_norm:.6f}\n")
            f.write(f"  Learning Rate: {current_lr:.2e}\n")
            f.write(f"  Lambda: {lmbda:.2e}\n")
            f.write(f"  Rate-Distortion Ratio: {rd_ratio:.6f}\n\n")
            
            f.write(f"Model Parameters:\n")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"  Total Parameters: {total_params:,}\n")
            f.write(f"  Trainable Parameters: {trainable_params:,}\n\n")
            
            if train_grad_norm > 5.0:
                f.write("⚠️  WARNING: High gradient norm detected - potential instability\n")
            if rd_ratio > 5.0:
                f.write("⚠️  WARNING: Rate term dominates - consider reducing lambda\n")
            if rd_ratio < 0.2:
                f.write("⚠️  WARNING: Distortion term dominates - consider increasing lambda\n")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = save_dir / f'best_model_epoch_{epoch}_loss_{best_loss:.4f}_{timestamp}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if config['training']['lr_schedule']['enabled'] else None,
                'best_loss': best_loss,
                'config': config,
                'timestamp': timestamp,
                'run_directory': str(save_dir)
            }, checkpoint_path)
            logging.info(f'Saved best model with loss: {best_loss:.4f} to {checkpoint_path}')
            
            # Also save a simple "best_model.pth" for easy access
            best_model_link = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if config['training']['lr_schedule']['enabled'] else None,
                'best_loss': best_loss,
                'config': config,
                'timestamp': timestamp,
                'run_directory': str(save_dir)
            }, best_model_link)
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info('Early stopping triggered')
                break

if __name__ == '__main__':
    main() 