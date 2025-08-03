"""
Tests for data loading and visualization.
"""
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import argparse
import yaml
from pathlib import Path
from data.datasets import ImageDataset
from data.transforms import create_transforms
from utils.paths import get_project_path

def save_batch(images, save_path, title='Batch of Images'):
    """Save a batch of images in a grid.
    
    Args:
        images: Tensor of shape (B, C, H, W)
        save_path: Path to save the visualization
        title: Title for the plot
    """
    # Create directory if it doesn't exist
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Make a grid of images
    grid = vutils.make_grid(images, nrow=4, padding=2, normalize=True)
    
    # Save using matplotlib
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory

def main():
    parser = argparse.ArgumentParser(description='Test dataset loading and visualization')
    parser.add_argument('--config', type=str, default=get_project_path('configs/train_factorized_prior.yaml'),
                      help='Path to config file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                      help='Which split to visualize (default: train)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for visualization (default: 16)')
    parser.add_argument('--save_dir', type=str, default='test_outputs',
                      help='Directory to save visualizations (default: test_outputs)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create transforms from config
    transform = create_transforms(config['data']['transforms'], split=args.split)

    # Create dataset
    txt_file = f"data/processed/kitti/{args.split}.txt"
    dataset = ImageDataset(txt_file=txt_file, transform=transform)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Get one batch
    images = next(iter(dataloader))
    
    # Save batch
    save_path = Path(args.save_dir) / f'batch_{args.split}_split.png'
    title = f'Batch of {args.batch_size} Images from {args.split} split'
    save_batch(images, save_path, title=title)
    print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    main() 