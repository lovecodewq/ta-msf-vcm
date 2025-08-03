"""
Plot rate-distortion curves from evaluation results.
Creates two plots:
1. PSNR vs BPP
2. MS-SSIM vs BPP
Each data point corresponds to a different lambda value.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot rate-distortion curves from evaluation results')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to evaluation results text file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Output directory for plots (default: evaluation_results)')
    parser.add_argument('--output_name', type=str, default='rate_distortion_curves.png',
                      help='Output filename for the plot (default: rate_distortion_curves.png)')
    return parser.parse_args()

def parse_results_file(results_file):
    """Parse the evaluation results file and extract metrics."""
    metrics = []
    current_metrics = {}
    
    with open(results_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Lambda:'):
            if current_metrics:
                metrics.append(current_metrics)
            current_metrics = {}
            current_metrics['lambda'] = float(line.split(':')[1].strip())
        elif line.startswith('Average BPP:'):
            current_metrics['bpp'] = float(line.split(':')[1].strip())
        elif line.startswith('Average PSNR:'):
            current_metrics['psnr'] = float(line.split(':')[1].strip().replace(' dB', ''))
        elif line.startswith('Average MS-SSIM:'):
            current_metrics['ms_ssim'] = float(line.split(':')[1].strip())
    
    # Add the last set of metrics
    if current_metrics:
        metrics.append(current_metrics)
        
    return metrics

def plot_metrics(metrics, save_dir, output_name):
    """Create and save plots for PSNR vs BPP and MS-SSIM vs BPP."""
    if not metrics:
        print("No metrics found in the input file!")
        return
        
    # Extract data
    bpp_values = [m['bpp'] for m in metrics]
    psnr_values = [m['psnr'] for m in metrics]
    msssim_values = [m['ms_ssim'] for m in metrics]
    lambda_values = [m['lambda'] for m in metrics]
    
    # Sort by BPP for better visualization
    sorted_data = sorted(zip(bpp_values, psnr_values, msssim_values, lambda_values))
    bpp_values, psnr_values, msssim_values, lambda_values = zip(*sorted_data)
    
    print(f"Plotting {len(metrics)} data points:")
    for i, (bpp, psnr, msssim, lmbda) in enumerate(zip(bpp_values, psnr_values, msssim_values, lambda_values)):
        print(f"  λ={lmbda:.3f}: BPP={bpp:.4f}, PSNR={psnr:.2f}dB, MS-SSIM={msssim:.4f}")
    
    # Create figure with two subplots
    plt.figure(figsize=(15, 10))
    
    # Plot PSNR vs BPP
    plt.subplot(2, 1, 1)
    scatter = plt.scatter(bpp_values, psnr_values, c=np.log10(lambda_values), 
                         cmap='viridis', s=100)
    plt.colorbar(scatter, label='log10(λ)')
    plt.xlabel('Bits per pixel (BPP)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Curve: PSNR vs BPP')
    plt.grid(True)
    
    # Add lambda values as annotations
    for i, lambda_val in enumerate(lambda_values):
        plt.annotate(f'λ={lambda_val:.3f}', 
                    (bpp_values[i], psnr_values[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    # Plot MS-SSIM vs BPP
    plt.subplot(2, 1, 2)
    scatter = plt.scatter(bpp_values, msssim_values, c=np.log10(lambda_values),
                         cmap='viridis', s=100)
    plt.colorbar(scatter, label='log10(λ)')
    plt.xlabel('Bits per pixel (BPP)')
    plt.ylabel('MS-SSIM')
    plt.title('Rate-Distortion Curve: MS-SSIM vs BPP')
    plt.grid(True)
    
    # Add lambda values as annotations
    for i, lambda_val in enumerate(lambda_values):
        plt.annotate(f'λ={lambda_val:.3f}',
                    (bpp_values[i], msssim_values[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    
    # Save plots
    save_path = Path(save_dir) / output_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Plots saved to {save_path}')

def main():
    # Parse arguments
    args = parse_args()
    
    # Validate input file exists
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading evaluation results from: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Parse results and create plots
    metrics = parse_results_file(input_file)
    if metrics:
        plot_metrics(metrics, output_dir, args.output_name)
    else:
        print("No valid metrics found in the input file!")

if __name__ == '__main__':
    main() 