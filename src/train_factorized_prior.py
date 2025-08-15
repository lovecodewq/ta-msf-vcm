import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from model.factorized_prior import FactorizedPrior
import yaml
import logging
from pathlib import Path
import numpy as np
from utils import get_project_path, AverageMeter
from data import ImageDataset
from data.transforms import create_transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=get_project_path('configs/train_factorized_prior.yaml'))
    return parser.parse_args()

def setup_logging(save_dir):
    log_file = save_dir / 'train.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_one_epoch(model, train_loader, optimizer, lmbda, device, log_interval):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()

    for batch_idx, x in enumerate(train_loader):
        x = x.to(device)

        optimizer.zero_grad()

        # Forward pass
        out = model(x)
        x_hat = out['x_hat']
        likelihoods = out['likelihoods']

        # Compute rate (bits per pixel), normalized over batch and spatial dims
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp = -torch.log2(likelihoods).sum() / num_pixels

        # Compute distortion (mean squared error)
        mse = torch.mean((x - x_hat).pow(2))

        # Total loss = MSE + lambda * Rate
        loss = mse + lmbda * bpp

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update meters
        loss_meter.update(loss.item())
        bpp_meter.update(bpp.item())
        mse_meter.update(mse.item())

        if batch_idx % log_interval == 0:
            logging.info(f'Train Batch: [{batch_idx}/{len(train_loader)}] '
                        f'Loss: {loss_meter.avg:.4f} '
                        f'MSE: {mse_meter.avg:.4f} '
                        f'BPP: {bpp_meter.avg:.4f}')

    return loss_meter.avg, mse_meter.avg, bpp_meter.avg

def validate(model, val_loader, lmbda, device):
    """Validate the model."""
    model.eval()
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            out = model(x)
            x_hat = out['x_hat']
            likelihoods = out['likelihoods']

            num_pixels = x.size(0) * x.size(2) * x.size(3)
            bpp = -torch.log2(likelihoods).sum() / num_pixels
            mse = torch.mean((x - x_hat).pow(2))
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

def save_reconstructions(model, val_loader, device, save_dir, epoch, lambda_val, num_samples=4):
    """Save sample reconstructions.
    
    Args:
        model: The compression model
        val_loader: Validation data loader
        device: Device to run on
        save_dir: Base directory for saving results
        epoch: Current epoch number
        lambda_val: Current lambda value for rate-distortion trade-off
        num_samples: Number of samples to save
    """
    model.eval()
    # Create lambda-specific directory
    save_dir = Path(save_dir) / 'reconstructions' / f'lambda_{lambda_val:.3f}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    to_pil = transforms.ToPILImage()
    
    with torch.no_grad():
        for i, x in enumerate(val_loader):
            if i >= num_samples:
                break
                
            x = x.to(device)
            out = model(x)
            x_hat = out['x_hat']
            
            # Save original and reconstruction with lambda in filename
            original = to_pil(x[0].cpu())
            reconstruction = to_pil(x_hat[0].cpu().clamp(0, 1))
            
            original.save(save_dir / f'lambda_{lambda_val:.3f}_epoch_{epoch}_sample_{i}_original.png')
            reconstruction.save(save_dir / f'lambda_{lambda_val:.3f}_epoch_{epoch}_sample_{i}_reconstruction.png')

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are of correct type
    config['model']['n_hidden'] = int(config['model']['n_hidden'])
    config['model']['n_channels'] = int(config['model']['n_channels'])
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['epochs'] = int(config['training']['epochs'])
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['lambda'] = float(config['training']['lambda'])
    config['training']['num_workers'] = int(config['training']['num_workers'])
    config['training']['log_interval'] = int(config['training']['log_interval'])
    config['training']['optimizer']['eps'] = float(config['training']['optimizer']['eps'])
    config['training']['optimizer']['weight_decay'] = float(config['training']['optimizer']['weight_decay'])
    config['training']['lr_schedule']['factor'] = float(config['training']['lr_schedule']['factor'])
    config['training']['lr_schedule']['patience'] = int(config['training']['lr_schedule']['patience'])
    config['training']['lr_schedule']['min_lr'] = float(config['training']['lr_schedule']['min_lr'])
    config['training']['early_stopping']['patience'] = int(config['training']['early_stopping']['patience'])
    config['training']['early_stopping']['min_delta'] = float(config['training']['early_stopping']['min_delta'])
    config['validation']['interval'] = int(config['validation']['interval'])
    config['validation']['num_samples'] = int(config['validation']['num_samples'])
    
    # Create save directory
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Create model
    model = FactorizedPrior(
        n_hidden=config['model']['n_hidden'],
        n_channels=config['model']['n_channels']
    ).to(device)

    # Setup data loading with configurable transforms
    train_transform = create_transforms(config['data']['transforms'], split='train')
    val_transform = create_transforms(config['data']['test_transforms'], split='val')  # Use test transforms for validation
    
    train_dataset = ImageDataset(
        txt_file=config['data']['train_list'],
        transform=train_transform
    )
    val_dataset = ImageDataset(
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
        train_loss, train_mse, train_bpp = train_one_epoch(
            model, train_loader, optimizer, lmbda, device,
            config['training']['log_interval']
        )
        
        # Validate
        val_loss, val_mse, val_bpp = validate(model, val_loader, lmbda, device)
        
        # Log metrics
        logging.info(f'Epoch: {epoch} '
                    f'Train Loss: {train_loss:.4f} '
                    f'Train MSE: {train_mse:.4f} '
                    f'Train BPP: {train_bpp:.4f} '
                    f'Val Loss: {val_loss:.4f} '
                    f'Val MSE: {val_mse:.4f} '
                    f'Val BPP: {val_bpp:.4f}')
        
        # Learning rate scheduling
        if config['training']['lr_schedule']['enabled']:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Current learning rate: {current_lr:.6f}')
        
        # Save reconstructions
        if config['validation']['save_reconstructions'] and epoch % config['validation']['interval'] == 0:
            save_reconstructions(
                model, val_loader, device, save_dir, epoch,
                config['training']['lambda'],  # Pass lambda value
                config['validation']['num_samples']
            )

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = save_dir / f'model_lambda_{config["training"]["lambda"]:.3f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if config['training']['lr_schedule']['enabled'] else None,
                'best_loss': best_loss,
                'config': config
            }, checkpoint_path)
            logging.info(f'Saved best model with loss: {best_loss:.4f} to {checkpoint_path}')
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info('Early stopping triggered')
                break

if __name__ == '__main__':
    main()
