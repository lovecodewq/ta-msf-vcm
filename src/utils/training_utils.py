"""
Training utilities for compression models.
Extracted common functionality for better maintainability.
Extends the existing utils with compression-specific functionality.
"""
import logging
import torch
from pathlib import Path
from datetime import datetime

# Import existing utilities
from .logging_utils import setup_training_environment
from .metrics import AverageMeter


class EarlyStopping:
    """Early stopping to prevent overfitting."""
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


def save_training_diagnostics(save_dir, epoch, train_metrics, val_metrics, optimizer, model, config):
    """Save detailed training diagnostics to file."""
    diagnostic_file = save_dir / f'training_diagnostics_epoch_{epoch}.txt'
    
    train_loss, train_mse, train_bpp, train_grad_norm = train_metrics
    val_loss, val_mse, val_bpp = val_metrics
    current_lr = optimizer.param_groups[0]['lr']
    lmbda = config['training']['lambda']
    
    rate_weight = lmbda * val_bpp
    distortion_weight = val_mse
    rd_ratio = rate_weight / distortion_weight if distortion_weight > 0 else float('inf')
    
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
        
        # Warnings
        if train_grad_norm > 5.0:
            f.write("⚠️  WARNING: High gradient norm detected - potential instability\n")
        if rd_ratio > 5.0:
            f.write("⚠️  WARNING: Rate term dominates - consider reducing lambda\n")
        if rd_ratio < 0.2:
            f.write("⚠️  WARNING: Distortion term dominates - consider increasing lambda\n")


def log_epoch_summary(epoch, train_metrics, val_metrics, optimizer, config):
    """Log comprehensive epoch summary with diagnostics."""
    train_loss, train_mse, train_bpp, train_grad_norm = train_metrics
    val_loss, val_mse, val_bpp = val_metrics
    current_lr = optimizer.param_groups[0]['lr']
    lmbda = config['training']['lambda']
    
    rate_weight = lmbda * val_bpp
    distortion_weight = val_mse
    rd_ratio = rate_weight / distortion_weight if distortion_weight > 0 else float('inf')
    
    logging.info(f'=== EPOCH {epoch} SUMMARY ===')
    logging.info(f'Training   -> Loss: {train_loss:.4f} | MSE: {train_mse:.6f} | BPP: {train_bpp:.4f} | GradNorm: {train_grad_norm:.3f}')
    logging.info(f'Validation -> Loss: {val_loss:.4f} | MSE: {val_mse:.6f} | BPP: {val_bpp:.4f}')
    logging.info(f'Rate-Distortion Balance -> λ×BPP: {rate_weight:.6f} | MSE: {distortion_weight:.6f} | Ratio: {rd_ratio:.3f}')
    logging.info(f'Optimization -> LR: {current_lr:.2e} | Lambda: {lmbda:.2e}')
    
    # Warnings
    if train_grad_norm > 10.0:
        logging.warning(f'⚠️  HIGH GRADIENT NORM: {train_grad_norm:.3f} - Potential instability!')
    
    if rd_ratio > 10.0:
        logging.warning(f'⚠️  RATE DOMINATES: Ratio={rd_ratio:.1f} - Consider reducing lambda')
    elif rd_ratio < 0.1:
        logging.warning(f'⚠️  DISTORTION DOMINATES: Ratio={rd_ratio:.3f} - Consider increasing lambda')


def save_best_model(save_dir, epoch, model, optimizer, scheduler, best_loss, config):
    """Save best model checkpoint with timestamp and metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = save_dir / f'best_model_epoch_{epoch}_loss_{best_loss:.4f}_{timestamp}.pth'
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_loss': best_loss,
        'config': config,
        'timestamp': timestamp,
        'run_directory': str(save_dir)
    }
    
    # Save timestamped checkpoint
    torch.save(checkpoint, checkpoint_path)
    logging.info(f'Saved best model with loss: {best_loss:.4f} to {checkpoint_path}')
    
    # Also save a simple "best_model.pth" for easy access
    best_model_link = save_dir / 'best_model.pth'
    torch.save(checkpoint, best_model_link)
    
    return checkpoint_path


def get_lambda_from_checkpoint(checkpoint_path):
    """Extract lambda value from compression model checkpoint path.
    
    Handles two formats:
    1. Image compression: model_lambda_0.010.pth
    2. Feature compression: run_20250807_000250_lambda_5.00e-01_lr_1.00e-04_bs_8/best_model_epoch_11_loss_0.0229_20250807_003803.pth
    """
    path = Path(checkpoint_path)
    try:
        # Check if this is a feature compression checkpoint (has run_ prefix and contains lr)
        if 'run_' in str(path) and 'lr' in str(path):
            # Split the path into parts and find the lambda value
            parts = str(path).split('_')
            for i, part in enumerate(parts):
                if part == 'lambda' and i + 1 < len(parts):
                    # Convert scientific notation (e.g., 5.00e-01) to float
                    lambda_str = parts[i + 1]
                    logging.info(f'Found feature compression lambda: {lambda_str}')
                    return float(lambda_str)
        
        # Image compression format (model_lambda_X.XXX.pth)
        if path.stem.startswith('model_lambda'):
            lambda_str = path.stem.split('_')[-1]
            logging.info(f'Found image compression lambda: {lambda_str}')
            return float(lambda_str)
            
        logging.warning(f'Unrecognized checkpoint format: {checkpoint_path}')
        logging.warning(f'Path parts: {str(path).split("_")}')
        return None
    except (IndexError, ValueError) as e:
        logging.warning(f'Failed to extract lambda from {checkpoint_path}: {str(e)}')
        logging.warning(f'Path parts: {str(path).split("_")}')
        return None