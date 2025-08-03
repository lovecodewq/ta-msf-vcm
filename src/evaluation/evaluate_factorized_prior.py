"""
Evaluation script for neural image compression models.
Calculates PSNR and MS-SSIM metrics and generates rate-distortion curves.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
import logging
import yaml
import argparse
from tqdm import tqdm
from model.factorized_prior import FactorizedPrior
from data.datasets import ImageDataset
from data.transforms import create_transforms
from utils.paths import get_project_path
import torchvision.transforms as transforms

def setup_logging(save_dir):
    log_file = save_dir / 'evaluation.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def calculate_psnr(x, x_hat):
    """Calculate PSNR."""
    mse = torch.mean((x - x_hat) ** 2)
    if mse == 0:
        return float('inf')
    max_val = 1.0
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def evaluate_model(model, data_loader, device, lambda_val):
    """Evaluate model on test set."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for x in tqdm(data_loader, desc=f'Evaluating (lambda={lambda_val})'):
            x = x.to(device)
            
            # Compress and decompress
            compressed = model.compress(x)
            strings = compressed['y_strings']
            shape = compressed['shape']
            input_size = compressed['input_size']
            decompressed = model.decompress(strings, shape, input_size=input_size)
            x_hat = decompressed['x_hat']
            
            # Calculate metrics for each image in batch
            for i in range(x.size(0)):
                # Get original and reconstructed images
                orig = x[i:i+1]
                recon = x_hat[i:i+1]
                
                # Calculate PSNR
                psnr = calculate_psnr(orig, recon)
                
                # Calculate MS-SSIM
                ms_ssim_val = ms_ssim(orig, recon, data_range=1.0)
                
                # Calculate bpp
                num_pixels = orig.size(2) * orig.size(3)
                # CompressAI's entropy bottleneck returns a list of strings where each string
                # represents the compressed data for one channel
                total_bits = sum(len(s) * 8 for s in strings)  # Each byte is 8 bits
                bpp = total_bits / num_pixels
                
                results.append({
                    'psnr': float(psnr),
                    'ms_ssim': float(ms_ssim_val),
                    'bpp': bpp
                })
    
    return results

def plot_rd_curves(results_dict, save_dir):
    """Plot rate-distortion curves."""
    plt.figure(figsize=(10, 8))
    
    # Plot PSNR curve
    plt.subplot(2, 1, 1)
    for lambda_val, results in results_dict.items():
        bpp = [r['bpp'] for r in results]
        psnr = [r['psnr'] for r in results]
        plt.scatter(bpp, psnr, label=f'λ={lambda_val}')
    
    plt.xlabel('Bits per pixel (bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Curve - PSNR')
    plt.grid(True)
    plt.legend()
    
    # Plot MS-SSIM curve
    plt.subplot(2, 1, 2)
    for lambda_val, results in results_dict.items():
        bpp = [r['bpp'] for r in results]
        ms_ssim = [r['ms_ssim'] for r in results]
        plt.scatter(bpp, ms_ssim, label=f'λ={lambda_val}')
    
    plt.xlabel('Bits per pixel (bpp)')
    plt.ylabel('MS-SSIM')
    plt.title('Rate-Distortion Curve - MS-SSIM')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'rd_curves.png')
    plt.close()

def save_results(results_dict, save_dir):
    """Save numerical results to file."""
    results_file = save_dir / 'evaluation_results.txt'
    
    with open(results_file, 'w') as f:
        for lambda_val, results in results_dict.items():
            bpp = np.mean([r['bpp'] for r in results])
            psnr = np.mean([r['psnr'] for r in results])
            ms_ssim = np.mean([r['ms_ssim'] for r in results])
            
            f.write(f'Lambda: {lambda_val}\n')
            f.write(f'Average BPP: {bpp:.4f}\n')
            f.write(f'Average PSNR: {psnr:.4f} dB\n')
            f.write(f'Average MS-SSIM: {ms_ssim:.4f}\n')
            f.write('-' * 50 + '\n')

def get_lambda_from_filename(filename):
    """Extract lambda value from checkpoint filename."""
    # Expected format: model_lambda_0.010.pth
    try:
        lambda_str = filename.stem.split('_')[-1]  # Get the last part after splitting by '_'
        return float(lambda_str)
    except (IndexError, ValueError):
        return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate compression model')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing model checkpoints')
    parser.add_argument('--test_data', type=str, default='data/processed/kitti/test.txt', required=True, help='Path to test dataset list file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/factorized_prior')
    args = parser.parse_args()
    
    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Find all checkpoint files and extract lambda values
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob('model_lambda_*.pth'))
    if not checkpoint_files:
        logging.error('No checkpoint files found matching pattern "model_lambda_*.pth"')
        return
        
    # Extract and sort lambda values
    lambda_values = []
    for checkpoint_file in checkpoint_files:
        lambda_val = get_lambda_from_filename(checkpoint_file)
        if lambda_val is not None:
            lambda_values.append(lambda_val)
    
    lambda_values.sort()  # Sort for consistent ordering in plots
    logging.info(f'Found checkpoints with lambda values: {lambda_values}')
    
    # Create test dataset with no transforms (we want to evaluate on full images)
    test_dataset = ImageDataset(
        txt_file=args.test_data,
        transform=transforms.ToTensor()  # Only convert to tensor, no other transforms
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for accurate metrics
        shuffle=False,
        num_workers=4
    )
    
    results_dict = {}
    
    # Evaluate each model checkpoint
    for lambda_val in lambda_values:
        checkpoint_path = checkpoint_dir / f'model_lambda_{lambda_val:.3f}.pth'
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        model = FactorizedPrior(
            n_hidden=config['model']['n_hidden'],
            n_channels=config['model']['n_channels']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        logging.info(f'Evaluating model with lambda={lambda_val}')
        results = evaluate_model(model, test_loader, device, lambda_val)
        results_dict[lambda_val] = results
    
    # Generate plots and save results
    if results_dict:
        plot_rd_curves(results_dict, save_dir)
        save_results(results_dict, save_dir)
        logging.info(f'Results saved to {save_dir}')
    else:
        logging.error('No models were evaluated. Check checkpoint paths.')

if __name__ == '__main__':
    main() 