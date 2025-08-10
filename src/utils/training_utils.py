"""
Training utilities for compression models.
Extracted common functionality for better maintainability.
Extends the existing utils with compression-specific functionality.
"""
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

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


def save_fpn_features_to_cache(cache_dir: Path, image_ids: List[str], fpn_features: Dict[str, torch.Tensor]) -> None:
    """Save batched FPN features to per-image .pt files under cache_dir.
    Each saved file contains a dict with keys p2..p6 and shape (1, C, H, W).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(fpn_features.values())).shape[0]
    for i, img_id in enumerate(image_ids):
        single = {k: v[i:i+1].cpu() for k, v in fpn_features.items()}
        torch.save(single, cache_dir / f"{img_id}.pt")


def load_fpn_features_from_cache(cache_dir: Path, image_ids: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
    """Load per-image cached features and batch them along dim 0 on the given device."""
    loaded: List[Dict[str, torch.Tensor]] = []
    for img_id in image_ids:
        path = cache_dir / f"{img_id}.pt"
        feats = torch.load(path, map_location=device)
        loaded.append(feats)
    keys = loaded[0].keys()
    batched = {k: torch.cat([d[k].to(device) for d in loaded], dim=0) for k in keys}
    return batched


@torch.no_grad()
def precompute_feature_cache(det_model, loader, device: torch.device, cache_dir: Path, overwrite: bool = False, limit: int | None = None) -> None:
    """Precompute and save FPN features for a loader into cache_dir.
    Skips existing files unless overwrite=True.
    """
    from tqdm import tqdm as _tqdm
    det_model.eval()
    cache_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for images, targets in _tqdm(loader, desc='Precomputing FPN cache', leave=False):
        image_ids = [t.get('image_id') for t in targets]
        if not overwrite and all((cache_dir / f"{iid}.pt").exists() for iid in image_ids):
            continue
        images = [img.to(device) for img in images]
        fpn = det_model.get_fpn_features(images)
        save_fpn_features_to_cache(cache_dir, image_ids, fpn)
        written += len(image_ids)
        if limit is not None and written >= limit:
            break

def get_lambda_from_checkpoint(checkpoint_path):
    """Extract lambda value from compression model checkpoint path.
    
    Handles two formats:
    1. Image compression: model_lambda_0.010.pth
    2. Feature compression: run_20250807_000250_lambda_5.00e-01_lr_1.00e-04_bs_8/best_model_epoch_11_loss_0.0229_20250807_003803.pth
    """
    path = Path(checkpoint_path)
    # Check if this is a feature compression checkpoint (has run_ prefix and contains lr)
    if 'run_' in str(path) and 'lr' in str(path):
        # Split the path into parts and find the lambda value
        parts = str(path).split('_')
        for i, part in enumerate(parts):
            if part == 'lambda' and i + 1 < len(parts):
                # Convert scientific notation (e.g., 5.00e-01) to float
                lambda_str = parts[i + 1]
                logging.info(f'Found feature compression lambda: {lambda_str}')
                try:
                    return float(lambda_str)
                except Exception:
                    pass
    
    # Image compression format (model_lambda_X.XXX.pth)
    if path.stem.startswith('model_lambda'):
        lambda_str = path.stem.split('_')[-1]
        logging.info(f'Found image compression lambda: {lambda_str}')
        try:
            return float(lambda_str)
        except Exception:
            pass
    
    logging.warning(f'Unrecognized checkpoint format: {checkpoint_path}')
    logging.warning(f'Path parts: {str(path).split("_")}')
    return None

def get_weight_from_checkpoint(checkpoint_path):
    """Extract detection loss weight (w) from feature compression checkpoint path if present.

    Expected pattern in path: ... run_..._lambda_<val>_w_<weight>_lr_.../best_model_...
    Returns float or None if not found.
    """
    path = Path(checkpoint_path)
    s = str(path)
    try:
        parts = s.split('_')
        for i, part in enumerate(parts):
            if part == 'w' and i + 1 < len(parts):
                return float(parts[i + 1])
            if part.startswith('w') and len(part) > 1 and part[1] in '0123456789.':
                return float(part[1:])
        if '_w_' in s:
            after = s.split('_w_')[1]
            num = after.split('_')[0]
            return float(num)
    except Exception:
        pass
    return None