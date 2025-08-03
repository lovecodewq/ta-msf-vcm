"""
Training script for fine-tuning Faster R-CNN on KITTI dataset.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path
import logging
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data.kitti_dataset import KITTIDetectionDataset
from utils import AverageMeter, get_project_path
from data.transforms import create_detection_transforms
from model.detection import DetectionModel
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=get_project_path('configs/train_detection.yaml'))
    return parser.parse_args()

def setup_logging(save_dir):
    """Setup logging configuration."""
    log_file = save_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_one_epoch(model, optimizer, data_loader, device, log_interval, scaler=None, grad_accum_steps=1):
    """Train model for one epoch."""
    model.train()  # Set model to training mode
    loss_meter = AverageMeter()
    
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc='Training')):
        images = [img.to(device) for img in images]
        
        # Move target tensors to device, preserving non-tensor items
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass and compute loss
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # In training mode with targets, model returns dict of losses
            loss_dict = model(images, targets)
            if isinstance(loss_dict, list):
                logging.error("Model returned predictions instead of losses. Check model mode.")
                continue
            total_loss = sum(loss for loss in loss_dict.values())
        
        # Scale loss by gradient accumulation steps
        total_loss = total_loss / grad_accum_steps
        
        # Backward pass
        if scaler:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Update meter with unscaled loss
        loss_meter.update(float(total_loss) * grad_accum_steps)
        
        if batch_idx % log_interval == 0:
            # Log individual losses
            loss_str = ' '.join(f'{k}: {v:.4f}' for k, v in loss_dict.items())
            logging.info(f'Train Batch: [{batch_idx}/{len(data_loader)}] '
                        f'Total Loss: {loss_meter.avg:.4f} ({loss_str})')
    
    return loss_meter.avg

def validate(model, data_loader, device):
    """Run validation."""
    was_training = model.training
    model.train()  # Set to train mode to compute losses
    loss_meter = AverageMeter()
    
    try:
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc='Validating'):
                images = [img.to(device) for img in images]
                # Move target tensors to device, preserving non-tensor items
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in t.items()} for t in targets]
                
                # Forward pass with targets to compute validation loss
                loss_dict = model(images, targets)
                if isinstance(loss_dict, list):
                    logging.error("Model returned predictions instead of losses. Check model mode.")
                    continue
                total_loss = sum(loss for loss in loss_dict.values())
                loss_meter.update(float(total_loss))
    finally:
        model.train(was_training)  # Restore original training mode
    
    return loss_meter.avg

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=1e-4):
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

def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a normalized image tensor back to [0,1] range."""
    mean = torch.tensor(mean, device=image.device).view(3, 1, 1)
    std = torch.tensor(std, device=image.device).view(3, 1, 1)
    return image * std + mean

def save_predictions(model, val_loader, device, save_dir, epoch, config):
    """Save sample predictions with bounding boxes."""
    model.eval()
    # Create epoch-specific directory
    pred_dir = save_dir / f'epoch_{epoch:03d}'
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Get parameters from config
    confidence_threshold = config['validation']['metrics']['confidence_threshold']
    mean = config['data']['val_transforms']['normalize']['mean']
    std = config['data']['val_transforms']['normalize']['std']
    
    # Define colors for different classes
    colors = ['r', 'g', 'b', 'c', 'm']
    
    # Use non-interactive backend
    plt.switch_backend('agg')
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if i >= 10:  # Limit to 10 samples
                break
                
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for j, (image, target, prediction) in enumerate(zip(images, targets, predictions)):
                # Denormalize and prepare image
                img_np = denormalize_image(image, mean=mean, std=std)
                img_np = img_np.cpu().permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                
                # Create figure with ground truth and predictions side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Plot ground truth
                ax1.imshow(img_np)
                ax1.set_title('Ground Truth')
                for box, label in zip(target['boxes'], target['labels']):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    class_idx = label.item()
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2,
                        edgecolor=colors[class_idx % len(colors)],
                        facecolor='none'
                    )
                    ax1.add_patch(rect)
                    ax1.text(
                        x1, y1-5,
                        f'{KITTIDetectionDataset.CLASSES[class_idx-1]}',
                        color=colors[class_idx % len(colors)],
                        bbox=dict(facecolor='white', alpha=0.8)
                    )
                
                # Plot predictions
                ax2.imshow(img_np)
                ax2.set_title(f'Predictions (conf > {confidence_threshold:.2f})')
                keep = prediction['scores'] > confidence_threshold
                boxes = prediction['boxes'][keep]
                labels = prediction['labels'][keep]
                scores = prediction['scores'][keep]
                
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    class_idx = label.item()
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2,
                        edgecolor=colors[class_idx % len(colors)],
                        facecolor='none'
                    )
                    ax2.add_patch(rect)
                    ax2.text(
                        x1, y1-5,
                        f'{KITTIDetectionDataset.CLASSES[class_idx-1]} {score:.2f}',
                        color=colors[class_idx % len(colors)],
                        bbox=dict(facecolor='white', alpha=0.8)
                    )
                
                # Remove axes
                ax1.axis('off')
                ax2.axis('off')
                
                # Save visualization
                plt.tight_layout()
                plt.savefig(pred_dir / f'pred_{i:02d}.png', bbox_inches='tight', dpi=150)
                plt.close(fig)
    
    logging.info(f'Saved {min(10, i+1)} prediction samples to {pred_dir}')

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create save directory with structured path
    save_dir = Path('checkpoints') / 'detection' / f"run_{config['training']['learning_rate']:.6f}_{config['training']['batch_size']}"
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir)

    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Setup transforms
    train_transform = create_detection_transforms(config['data']['transforms'])
    val_transform = create_detection_transforms(config['data']['val_transforms'])
    
    # Create datasets
    train_dataset = KITTIDetectionDataset(
        config['data']['root_dir'],
        split='train',
        transform=train_transform
    )
    val_dataset = KITTIDetectionDataset(
        config['data']['root_dir'],
        split='val',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Create model
    num_classes = len(KITTIDetectionDataset.CLASSES) + 1  # +1 for background
    model = DetectionModel(
        num_classes=num_classes,
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Setup optimizer
    if config['training']['optimizer']['type'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['optimizer']['momentum'],
            weight_decay=config['training']['optimizer']['weight_decay'],
            nesterov=config['training']['optimizer'].get('nesterov', False)
        )
    
    # Setup learning rate scheduler
    if config['training']['lr_schedule']['enabled']:
        scheduler_config = config['training']['lr_schedule']
        if scheduler_config['type'].lower() == 'cosine':
            # Ensure min_lr is float
            min_lr = float(scheduler_config['min_lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs'],
                eta_min=min_lr
            )
            
            # If warmup is enabled, wrap the scheduler
            if scheduler_config.get('warmup_epochs', 0) > 0:
                warmup_epochs = int(scheduler_config['warmup_epochs'])
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=0.1,
                            total_iters=warmup_epochs
                        ),
                        scheduler
                    ],
                    milestones=[warmup_epochs]
                )
        elif scheduler_config['type'].lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_config['step_size']),
                gamma=float(scheduler_config['gamma'])
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
        
        logging.info(f"Using {scheduler_config['type']} learning rate scheduler")
        if scheduler_config.get('warmup_epochs', 0) > 0:
            logging.info(f"With {scheduler_config['warmup_epochs']} epochs of warmup")
    
    # Setup early stopping if enabled
    early_stopping = None
    if config['training'].get('early_stopping', {}).get('enabled', False):
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
    
    # Setup gradient accumulation
    grad_accum_steps = config['training'].get('gradient_accumulation', {}).get('steps', 1)
    effective_batch_size = config['training']['batch_size'] * grad_accum_steps
    logging.info(f"Training with batch size {config['training']['batch_size']} "
                f"and {grad_accum_steps} gradient accumulation steps "
                f"(effective batch size: {effective_batch_size})")
    
    # Setup mixed precision training if enabled
    if config['mixed_precision']['enabled']:
        scaler = torch.cuda.amp.GradScaler(
            init_scale=config['mixed_precision'].get('scale_factor', 128.0)
        )
        logging.info("Mixed precision training enabled")
    else:
        scaler = None
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device,
            config['training']['log_interval'],
            scaler=scaler,
            grad_accum_steps=grad_accum_steps
        )
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Log metrics
        logging.info(f'Epoch: {epoch} '
                    f'Train Loss: {train_loss:.4f} '
                    f'Val Loss: {val_loss:.4f}')
        
        # Learning rate scheduling
        if config['training']['lr_schedule']['enabled']:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Current learning rate: {current_lr:.6f}')
        
        # Save predictions
        if config['validation']['save_predictions'] and epoch % config['validation']['interval'] == 0:
            predictions_dir = save_dir / 'predictions'
            save_predictions(
                model, val_loader, device, predictions_dir, epoch,
                config=config
            )

        # Save latest checkpoint
        checkpoint_path = save_dir / 'latest_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if config['training']['lr_schedule']['enabled'] else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }, checkpoint_path)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if config['training']['lr_schedule']['enabled'] else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': best_loss,
                'config': config
            }, best_checkpoint_path)
            logging.info(f'Saved new best model with validation loss: {val_loss:.4f} to {best_checkpoint_path}')
        
        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info('Early stopping triggered')
                break

    logging.info('Training completed!')

if __name__ == '__main__':
    main() 