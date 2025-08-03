"""
FPN Feature Statistics Analysis Script

This script analyzes the statistical properties of FPN features extracted from the detection model
to understand value ranges, distributions, and whether normalization is needed for compression.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import argparse
import yaml
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from datetime import datetime
import pandas as pd

# Local imports
from model.detection import DetectionModel
from data.kitti_dataset import KITTIDetectionDataset
from data import ImageDataset
from data.transforms import create_transforms
from utils import get_project_path


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze FPN feature statistics')
    parser.add_argument('--config', type=str,
                       default=get_project_path('configs/train_factorized_prior_random_sample_fpn.yaml'),
                       help='Path to config file')
    parser.add_argument('--detection_checkpoint', type=str,
                       default=get_project_path('checkpoints/detection/run_0.002000_16/best_model.pth'),
                       help='Path to detection model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='fpn_feature_analysis',
                       help='Output directory for analysis results')
    return parser.parse_args()


def setup_logging(save_dir):
    """Setup logging for the analysis."""
    log_file = save_dir / 'fpn_analysis.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


class FPNFeatureAnalyzer:
    """Analyzer for FPN feature statistics."""
    
    def __init__(self, detection_model, device):
        self.detection_model = detection_model
        self.device = device
        self.fpn_levels = ['0', '1', '2', '3', '4']
        
        # Storage for statistics
        self.level_stats = {level: defaultdict(list) for level in self.fpn_levels}
        self.feature_samples = {level: [] for level in self.fpn_levels}
        
    def extract_fpn_features_batch(self, images):
        """Extract FPN features from a batch of images."""
        with torch.no_grad():
            images = images.to(self.device)
            image_list = [img for img in images]
            fpn_features = self.detection_model.get_fpn_features(image_list)
        return fpn_features
    
    def analyze_batch(self, images):
        """Analyze a batch of images and collect statistics."""
        fpn_features = self.extract_fpn_features_batch(images)
        
        batch_stats = {}
        for level in self.fpn_levels:
            if level in fpn_features:
                features = fpn_features[level]  # [B, C, H, W]
                
                # Compute statistics for this batch
                batch_stats[level] = self._compute_feature_stats(features)
                
                # Store sample features for detailed analysis (first sample only)
                if len(self.feature_samples[level]) < 10:  # Limit storage
                    self.feature_samples[level].append(features[0].cpu().numpy())
                
                # Accumulate statistics
                for stat_name, value in batch_stats[level].items():
                    self.level_stats[level][stat_name].append(value)
        
        return batch_stats
    
    def _compute_feature_stats(self, features):
        """Compute comprehensive statistics for feature tensor."""
        # features: [B, C, H, W]
        flat_features = features.flatten()
        
        stats = {
            'mean': flat_features.mean().item(),
            'std': flat_features.std().item(),
            'min': flat_features.min().item(),
            'max': flat_features.max().item(),
            'median': flat_features.median().item(),
            'mean_abs': flat_features.abs().mean().item(),
            'max_abs': flat_features.abs().max().item(),
            'percentile_1': torch.quantile(flat_features, 0.01).item(),
            'percentile_5': torch.quantile(flat_features, 0.05).item(),
            'percentile_95': torch.quantile(flat_features, 0.95).item(),
            'percentile_99': torch.quantile(flat_features, 0.99).item(),
            'num_positive': (flat_features > 0).sum().item(),
            'num_negative': (flat_features < 0).sum().item(),
            'num_zero': (flat_features == 0).sum().item(),
            'total_elements': flat_features.numel(),
            'shape': list(features.shape)
        }
        
        # Add derived statistics
        stats['positive_ratio'] = stats['num_positive'] / stats['total_elements']
        stats['negative_ratio'] = stats['num_negative'] / stats['total_elements']
        stats['zero_ratio'] = stats['num_zero'] / stats['total_elements']
        stats['range'] = stats['max'] - stats['min']
        
        return stats
    
    def get_summary_statistics(self):
        """Get summary statistics across all analyzed samples."""
        summary = {}
        
        for level in self.fpn_levels:
            if self.level_stats[level]:
                level_summary = {}
                
                for stat_name in ['mean', 'std', 'min', 'max', 'mean_abs', 'max_abs', 'range']:
                    values = self.level_stats[level][stat_name]
                    if values:
                        level_summary[stat_name] = {
                            'avg': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'median': np.median(values)
                        }
                
                # Average shape info
                shapes = self.level_stats[level]['shape']
                if shapes:
                    level_summary['avg_shape'] = {
                        'batch': np.mean([s[0] for s in shapes]),
                        'channels': int(np.mean([s[1] for s in shapes])),
                        'height': np.mean([s[2] for s in shapes]),
                        'width': np.mean([s[3] for s in shapes])
                    }
                
                # Sparsity info
                level_summary['sparsity'] = {
                    'avg_positive_ratio': np.mean(self.level_stats[level]['positive_ratio']),
                    'avg_negative_ratio': np.mean(self.level_stats[level]['negative_ratio']),
                    'avg_zero_ratio': np.mean(self.level_stats[level]['zero_ratio'])
                }
                
                summary[level] = level_summary
        
        return summary
    
    def save_analysis(self, save_dir, config):
        """Save comprehensive analysis results."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get summary statistics
        summary = self.get_summary_statistics()
        
        # Save detailed text report
        report_file = save_dir / f'fpn_feature_analysis_{timestamp}.txt'
        self._save_text_report(report_file, summary, config)
        
        # Save CSV data for further analysis
        csv_file = save_dir / f'fpn_feature_statistics_{timestamp}.csv'
        self._save_csv_data(csv_file)
        
        # Create visualizations
        self._create_visualizations(save_dir, timestamp)
        
        logging.info(f"Analysis saved to {save_dir}")
        logging.info(f"Report: {report_file}")
        logging.info(f"CSV data: {csv_file}")
    
    def _save_text_report(self, report_file, summary, config):
        """Save detailed text report."""
        with open(report_file, 'w') as f:
            f.write("FPN Feature Statistics Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {config}\n")
            f.write(f"Number of samples analyzed: {len(next(iter(self.level_stats.values()))['mean']) if self.level_stats else 0}\n\n")
            
            for level in self.fpn_levels:
                if level in summary:
                    f.write(f"FPN Level {level} Statistics:\n")
                    f.write("-" * 30 + "\n")
                    
                    level_data = summary[level]
                    
                    # Shape information
                    if 'avg_shape' in level_data:
                        shape = level_data['avg_shape']
                        f.write(f"  Average Shape: [B, {shape['channels']}, {shape['height']:.1f}, {shape['width']:.1f}]\n")
                    
                    # Basic statistics
                    for stat_name in ['mean', 'std', 'min', 'max', 'mean_abs', 'max_abs', 'range']:
                        if stat_name in level_data:
                            stat = level_data[stat_name]
                            f.write(f"  {stat_name.title()}:\n")
                            f.write(f"    Average: {stat['avg']:.6f}\n")
                            f.write(f"    Std Dev: {stat['std']:.6f}\n")
                            f.write(f"    Range: [{stat['min']:.6f}, {stat['max']:.6f}]\n")
                    
                    # Sparsity information
                    if 'sparsity' in level_data:
                        sparsity = level_data['sparsity']
                        f.write(f"  Value Distribution:\n")
                        f.write(f"    Positive values: {sparsity['avg_positive_ratio']:.3f}\n")
                        f.write(f"    Negative values: {sparsity['avg_negative_ratio']:.3f}\n")
                        f.write(f"    Zero values: {sparsity['avg_zero_ratio']:.3f}\n")
                    
                    f.write("\n")
            
            # Analysis recommendations
            f.write("Analysis Recommendations:\n")
            f.write("=" * 30 + "\n")
            
            # Check for normalization needs
            max_ranges = []
            max_means = []
            max_stds = []
            
            for level in self.fpn_levels:
                if level in summary:
                    if 'range' in summary[level]:
                        max_ranges.append(summary[level]['range']['max'])
                    if 'mean_abs' in summary[level]:
                        max_means.append(abs(summary[level]['mean_abs']['avg']))
                    if 'std' in summary[level]:
                        max_stds.append(summary[level]['std']['avg'])
            
            if max_ranges:
                max_range = max(max_ranges)
                max_mean_abs = max(max_means) if max_means else 0
                max_std = max(max_stds) if max_stds else 0
                
                f.write(f"Maximum value range across levels: {max_range:.6f}\n")
                f.write(f"Maximum mean absolute value: {max_mean_abs:.6f}\n")
                f.write(f"Maximum standard deviation: {max_std:.6f}\n\n")
                
                if max_range > 10.0:
                    f.write("⚠️  RECOMMENDATION: Large value ranges detected. Consider normalization.\n")
                elif max_range > 5.0:
                    f.write("⚠️  RECOMMENDATION: Moderate value ranges. Normalization may help.\n")
                else:
                    f.write("✅ RECOMMENDATION: Value ranges are reasonable. Normalization optional.\n")
                
                if max_mean_abs > 1.0:
                    f.write("⚠️  RECOMMENDATION: Large absolute values. Consider feature scaling.\n")
                
                if max_std > 2.0:
                    f.write("⚠️  RECOMMENDATION: High variance. Standardization may improve training.\n")
    
    def _save_csv_data(self, csv_file):
        """Save statistics as CSV for further analysis."""
        rows = []
        
        for level in self.fpn_levels:
            if self.level_stats[level]:
                level_data = self.level_stats[level]
                num_samples = len(level_data['mean'])
                
                for i in range(num_samples):
                    row = {'level': level, 'sample_idx': i}
                    for stat_name in ['mean', 'std', 'min', 'max', 'mean_abs', 'max_abs', 'range']:
                        if stat_name in level_data:
                            row[stat_name] = level_data[stat_name][i]
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
    
    def _create_visualizations(self, save_dir, timestamp):
        """Create visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create subplots for different statistics
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'FPN Feature Statistics Analysis - {timestamp}', fontsize=16)
            
            stats_to_plot = ['mean_abs', 'max_abs', 'std', 'range', 'positive_ratio', 'zero_ratio']
            
            for idx, stat_name in enumerate(stats_to_plot):
                row, col = idx // 3, idx % 3
                ax = axes[row, col]
                
                # Collect data for all levels
                data_dict = {}
                for level in self.fpn_levels:
                    if stat_name in self.level_stats[level]:
                        data_dict[f'Level {level}'] = self.level_stats[level][stat_name]
                
                if data_dict:
                    # Create box plot
                    data_list = list(data_dict.values())
                    labels = list(data_dict.keys())
                    
                    ax.boxplot(data_list, labels=labels)
                    ax.set_title(f'{stat_name.replace("_", " ").title()}')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'fpn_statistics_overview_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create distribution plots for feature values
            if any(self.feature_samples.values()):
                fig, axes = plt.subplots(1, len(self.fpn_levels), figsize=(20, 4))
                fig.suptitle('FPN Feature Value Distributions (Sample)', fontsize=14)
                
                for idx, level in enumerate(self.fpn_levels):
                    if self.feature_samples[level]:
                        sample = self.feature_samples[level][0].flatten()
                        # Sample subset for plotting
                        sample_subset = np.random.choice(sample, min(10000, len(sample)), replace=False)
                        
                        axes[idx].hist(sample_subset, bins=50, alpha=0.7, density=True)
                        axes[idx].set_title(f'Level {level}')
                        axes[idx].set_xlabel('Feature Value')
                        axes[idx].set_ylabel('Density')
                        axes[idx].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(save_dir / f'fpn_distributions_{timestamp}.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logging.info("Visualizations created successfully")
            
        except ImportError:
            logging.warning("Matplotlib/Seaborn not available. Skipping visualizations.")
        except Exception as e:
            logging.warning(f"Error creating visualizations: {e}")


def main():
    args = parse_args()
    
    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load detection model
    logging.info("Loading detection model...")
    if os.path.exists(args.detection_checkpoint):
        checkpoint = torch.load(args.detection_checkpoint, map_location=device)
        num_classes = len(KITTIDetectionDataset.CLASSES) + 1
        detection_model = DetectionModel(num_classes=num_classes, pretrained=False)
        detection_model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded detection model from {args.detection_checkpoint}")
    else:
        logging.error(f"Detection checkpoint not found: {args.detection_checkpoint}")
        return
    
    detection_model = detection_model.to(device)
    detection_model.eval()
    
    # Setup data loading (same as training script)
    train_transform = create_transforms(config['data']['transforms'], split='train')
    
    train_dataset = ImageDataset(
        txt_file=config['data']['train_list'],
        transform=train_transform
    )
    
    # Create data loader with small batch size for analysis
    data_loader = DataLoader(
        train_dataset,
        batch_size=min(1, config['training']['batch_size']),
        shuffle=True,
        num_workers=min(4, config['training']['num_workers'])
    )
    
    logging.info(f"Dataset size: {len(train_dataset)}")
    logging.info(f"Analyzing {args.num_samples} samples...")
    
    # Initialize analyzer
    analyzer = FPNFeatureAnalyzer(detection_model, device)
    
    # Analyze samples
    samples_analyzed = 0
    total_batches = 0
    
    for batch_idx, images in enumerate(data_loader):
        if samples_analyzed >= args.num_samples:
            break
        
        try:
            batch_stats = analyzer.analyze_batch(images)
            samples_analyzed += images.size(0)
            total_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logging.info(f"Processed {samples_analyzed}/{args.num_samples} samples...")
                
                # Log sample statistics
                for level, stats in batch_stats.items():
                    logging.info(f"  Level {level}: mean_abs={stats['mean_abs']:.4f}, "
                               f"max_abs={stats['max_abs']:.4f}, "
                               f"std={stats['std']:.4f}, "
                               f"shape={stats['shape']}")
                    
        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {e}")
            continue
    
    logging.info(f"Analysis completed. Processed {samples_analyzed} samples in {total_batches} batches.")
    
    # Save analysis results
    analyzer.save_analysis(save_dir, config)
    
    # Print summary to console
    summary = analyzer.get_summary_statistics()
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    
    for level in ['0', '1', '2', '3', '4']:
        if level in summary:
            data = summary[level]
            print(f"Level {level}:")
            if 'mean_abs' in data:
                print(f"  Mean Abs: {data['mean_abs']['avg']:.4f} ± {data['mean_abs']['std']:.4f}")
            if 'max_abs' in data:
                print(f"  Max Abs:  {data['max_abs']['avg']:.4f} ± {data['max_abs']['std']:.4f}")
            if 'std' in data:
                print(f"  Std Dev:  {data['std']['avg']:.4f} ± {data['std']['std']:.4f}")
            if 'range' in data:
                print(f"  Range:    {data['range']['avg']:.4f} ± {data['range']['std']:.4f}")
            
            # Check if normalization is recommended
            if 'range' in data and data['range']['avg'] > 5.0:
                print(f"  ⚠️  Consider normalization (large range)")
            elif 'mean_abs' in data and data['mean_abs']['avg'] > 2.0:
                print(f"  ⚠️  Consider normalization (large values)")
            else:
                print(f"  ✅ Values seem reasonable")
            print()


if __name__ == '__main__':
    main()