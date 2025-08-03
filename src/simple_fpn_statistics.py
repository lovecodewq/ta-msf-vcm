"""
Simple FPN Feature Statistics Analysis

A lightweight version that analyzes FPN feature statistics without requiring matplotlib/pandas.
This provides the core analysis needed to understand normalization requirements.
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import yaml
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Local imports
from model.detection import DetectionModel
from data.kitti_dataset import KITTIDetectionDataset
from data import ImageDataset
from data.transforms import create_transforms
from utils import get_project_path


def parse_args():
    parser = argparse.ArgumentParser(description='Simple FPN feature statistics analysis')
    parser.add_argument('--config', type=str,
                       default=get_project_path('configs/train_factorized_prior_random_sample_fpn.yaml'),
                       help='Path to config file')
    parser.add_argument('--detection_checkpoint', type=str,
                       default=get_project_path('checkpoints/detection/run_0.002000_16/best_model.pth'),
                       help='Path to detection model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='fpn_simple_analysis',
                       help='Output directory for analysis results')
    return parser.parse_args()


class SimpleFPNAnalyzer:
    """Simple FPN feature analyzer focusing on key statistics."""
    
    def __init__(self, detection_model, device):
        self.detection_model = detection_model
        self.device = device
        self.fpn_levels = ['0', '1', '2', '3', '4']
        
        # Storage for raw statistics - CHANNEL-WISE
        self.stats = {level: {} for level in self.fpn_levels}
        
    def extract_fpn_features(self, images):
        """Extract FPN features from images."""
        with torch.no_grad():
            images = images.to(self.device)
            image_list = [img for img in images]
            fpn_features = self.detection_model.get_fpn_features(image_list)
        return fpn_features
    
    def analyze_batch(self, images):
        """Analyze FPN features from a batch of images - CHANNEL-WISE analysis."""
        fpn_features = self.extract_fpn_features(images)
        
        batch_results = {}
        for level in self.fpn_levels:
            if level in fpn_features:
                features = fpn_features[level]  # [B, C, H, W]
                B, C, H, W = features.shape
                
                # Analyze per-channel statistics
                channel_stats = {
                    'mean': [],
                    'std': [],
                    'min': [],
                    'max': [],
                    'mean_abs': [],
                    'max_abs': []
                }
                
                for c in range(C):
                    channel_data = features[:, c, :, :].flatten()  # All spatial locations for this channel
                    
                    channel_stats['mean'].append(channel_data.mean().item())
                    channel_stats['std'].append(channel_data.std().item())
                    channel_stats['min'].append(channel_data.min().item())
                    channel_stats['max'].append(channel_data.max().item())
                    channel_stats['mean_abs'].append(channel_data.abs().mean().item())
                    channel_stats['max_abs'].append(channel_data.abs().max().item())
                
                # Convert to tensors for easier statistics
                for key in channel_stats:
                    channel_stats[key] = torch.tensor(channel_stats[key])
                
                # Compute overall statistics (across all channels and spatial locations)
                flat_features = features.flatten()
                
                # Aggregate statistics
                stats = {
                    # Global statistics (all channels together)
                    'global_mean': flat_features.mean().item(),
                    'global_std': flat_features.std().item(),
                    'global_min': flat_features.min().item(),
                    'global_max': flat_features.max().item(),
                    'global_mean_abs': flat_features.abs().mean().item(),
                    'global_max_abs': flat_features.abs().max().item(),
                    
                    # Channel-wise statistics (variability across channels)
                    'channel_mean_avg': channel_stats['mean'].mean().item(),
                    'channel_mean_std': channel_stats['mean'].std().item(),
                    'channel_mean_min': channel_stats['mean'].min().item(),
                    'channel_mean_max': channel_stats['mean'].max().item(),
                    
                    'channel_std_avg': channel_stats['std'].mean().item(),
                    'channel_std_std': channel_stats['std'].std().item(),
                    'channel_std_min': channel_stats['std'].min().item(),
                    'channel_std_max': channel_stats['std'].max().item(),
                    
                    'channel_max_abs_avg': channel_stats['max_abs'].mean().item(),
                    'channel_max_abs_std': channel_stats['max_abs'].std().item(),
                    'channel_max_abs_min': channel_stats['max_abs'].min().item(),
                    'channel_max_abs_max': channel_stats['max_abs'].max().item(),
                    
                    # Range statistics
                    'channel_range_avg': (channel_stats['max'] - channel_stats['min']).mean().item(),
                    'channel_range_std': (channel_stats['max'] - channel_stats['min']).std().item(),
                    'channel_range_max': (channel_stats['max'] - channel_stats['min']).max().item(),
                    
                    # Shape and metadata
                    'shape': list(features.shape),
                    'num_channels': C,
                    'num_elements': flat_features.numel()
                }
                
                # Store statistics
                for key, value in stats.items():
                    if key not in self.stats[level]:
                        self.stats[level][key] = []
                    self.stats[level][key].append(value)
                
                batch_results[level] = stats
        
        return batch_results
    
    def get_summary(self):
        """Get summary statistics across all analyzed samples - CHANNEL-WISE."""
        summary = {}
        
        for level in self.fpn_levels:
            if self.stats[level] and 'global_mean' in self.stats[level]:  # Check if we have data
                level_summary = {}
                
                # Global statistics (all channels together)
                for stat_name in ['global_mean', 'global_std', 'global_min', 'global_max', 'global_mean_abs', 'global_max_abs']:
                    values = self.stats[level][stat_name]
                    level_summary[stat_name] = {
                        'avg': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'range': np.max(values) - np.min(values)
                    }
                
                # Channel-wise variability statistics
                channel_stats = {}
                for stat_group in ['channel_mean', 'channel_std', 'channel_max_abs', 'channel_range']:
                    channel_stats[stat_group] = {}
                    for suffix in ['_avg', '_std', '_min', '_max']:
                        stat_name = stat_group + suffix
                        if stat_name in self.stats[level]:
                            values = self.stats[level][stat_name]
                            channel_stats[stat_group][suffix] = {
                                'avg': np.mean(values),
                                'std': np.std(values),
                                'min': np.min(values),
                                'max': np.max(values)
                            }
                
                level_summary['channel_variability'] = channel_stats
                
                # Shape information (take first sample)
                if self.stats[level]['shape']:
                    shape = self.stats[level]['shape'][0]
                    level_summary['typical_shape'] = shape
                    level_summary['spatial_size'] = f"{shape[2]}x{shape[3]}"
                    level_summary['channels'] = shape[1]
                
                # Extremes and key insights
                level_summary['key_insights'] = {
                    'global_value_range': np.max(self.stats[level]['global_max']) - np.min(self.stats[level]['global_min']),
                    'max_abs_across_all': np.max(self.stats[level]['global_max_abs']),
                    'max_channel_range': np.max(self.stats[level]['channel_range_max']),
                    'channel_variability_score': np.mean(self.stats[level]['channel_max_abs_std']),  # How much channels vary
                    'avg_channel_max_abs': np.mean(self.stats[level]['channel_max_abs_avg'])
                }
                
                summary[level] = level_summary
        
        return summary
    
    def save_report(self, save_dir, config, num_samples):
        """Save analysis report."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = save_dir / f'fpn_simple_analysis_{timestamp}.txt'
        
        summary = self.get_summary()
        
        with open(report_file, 'w') as f:
            f.write("FPN Feature Statistics - Simple Analysis\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Samples analyzed: {num_samples}\n")
            f.write(f"Detection checkpoint: {config.get('detection_checkpoint', 'N/A')}\n\n")
            
            # Overall summary table - CHANNEL-WISE
            f.write("QUICK OVERVIEW (Channel-wise Analysis):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Level':<6} {'Shape':<15} {'Global_Range':<12} {'Max_Abs':<10} {'Ch_Variability':<15}\n")
            f.write("-" * 80 + "\n")
            
            for level in self.fpn_levels:
                if level in summary:
                    data = summary[level]
                    shape = data.get('spatial_size', 'N/A')
                    val_range = data['key_insights']['global_value_range']
                    max_abs = data['key_insights']['max_abs_across_all']
                    ch_var = data['key_insights']['channel_variability_score']
                    
                    f.write(f"{level:<6} {shape:<15} {val_range:<12.3f} {max_abs:<10.3f} {ch_var:<15.4f}\n")
            
            f.write("\n" + "="*60 + "\n\n")
            
            # Detailed level-by-level analysis - CHANNEL-WISE
            for level in self.fpn_levels:
                if level in summary:
                    f.write(f"FPN LEVEL {level} DETAILED CHANNEL-WISE ANALYSIS:\n")
                    f.write("-" * 50 + "\n")
                    
                    data = summary[level]
                    
                    # Basic info
                    f.write(f"Typical shape: {data.get('typical_shape', 'N/A')}\n")
                    f.write(f"Channels: {data.get('channels', 'N/A')}\n")
                    f.write(f"Spatial size: {data.get('spatial_size', 'N/A')}\n\n")
                    
                    # Global statistics (all channels together)
                    f.write("Global Statistics (all 256 channels together):\n")
                    for stat in ['global_mean', 'global_std', 'global_mean_abs', 'global_max_abs']:
                        if stat in data:
                            s = data[stat]
                            clean_name = stat.replace('global_', '').replace('_', ' ').title()
                            f.write(f"  {clean_name:<12}: "
                                   f"{s['avg']:>8.4f} ± {s['std']:>6.4f} "
                                   f"[{s['min']:>7.4f}, {s['max']:>7.4f}]\n")
                    
                    # Channel variability analysis
                    f.write(f"\nChannel Variability Analysis:\n")
                    if 'channel_variability' in data:
                        cv = data['channel_variability']
                        
                        # Channel means analysis
                        if 'channel_mean' in cv:
                            cm = cv['channel_mean']
                            f.write(f"  Per-channel means:\n")
                            f.write(f"    Average:      {cm['_avg']['avg']:>8.4f} (avg across channels)\n")
                            f.write(f"    Variability:  {cm['_std']['avg']:>8.4f} (how much channels differ)\n")
                            f.write(f"    Range:        [{cm['_min']['avg']:>7.4f}, {cm['_max']['avg']:>7.4f}]\n")
                        
                        # Channel max_abs analysis
                        if 'channel_max_abs' in cv:
                            cma = cv['channel_max_abs']
                            f.write(f"  Per-channel max absolute values:\n")
                            f.write(f"    Average:      {cma['_avg']['avg']:>8.4f} (typical channel max)\n")
                            f.write(f"    Variability:  {cma['_std']['avg']:>8.4f} (spread across channels)\n")
                            f.write(f"    Range:        [{cma['_min']['avg']:>7.4f}, {cma['_max']['avg']:>7.4f}]\n")
                        
                        # Channel range analysis
                        if 'channel_range' in cv:
                            cr = cv['channel_range']
                            f.write(f"  Per-channel value ranges:\n")
                            f.write(f"    Average:      {cr['_avg']['avg']:>8.4f} (typical channel range)\n")
                            f.write(f"    Variability:  {cr['_std']['avg']:>8.4f} (consistency across channels)\n")
                            f.write(f"    Max range:    {cr['_max']['avg']:>8.4f} (worst-case channel)\n")
                    
                    # Key insights
                    f.write(f"\nKey Insights for Compression:\n")
                    insights = data['key_insights']
                    f.write(f"  Global value range:        {insights['global_value_range']:>8.4f}\n")
                    f.write(f"  Maximum absolute value:    {insights['max_abs_across_all']:>8.4f}\n")
                    f.write(f"  Worst channel range:       {insights['max_channel_range']:>8.4f}\n")
                    f.write(f"  Channel variability score: {insights['channel_variability_score']:>8.4f}\n")
                    f.write(f"  Average channel max_abs:   {insights['avg_channel_max_abs']:>8.4f}\n")
                    
                    # Channel uniformity assessment
                    ch_var_score = insights['channel_variability_score']
                    if ch_var_score < 0.1:
                        f.write(f"  📊 Assessment: Channels are very uniform (var={ch_var_score:.4f})\n")
                    elif ch_var_score < 0.5:
                        f.write(f"  📊 Assessment: Channels are moderately uniform (var={ch_var_score:.4f})\n")
                    else:
                        f.write(f"  📊 Assessment: Channels vary significantly (var={ch_var_score:.4f})\n")
                    
                    f.write("\n" + "-"*60 + "\n\n")
            
            # Channel-aware normalization recommendations
            f.write("CHANNEL-AWARE NORMALIZATION RECOMMENDATIONS:\n")
            f.write("=" * 50 + "\n")
            
            # Analyze across all levels for channel-wise insights
            max_global_range = 0
            max_abs = 0
            max_channel_variability = 0
            max_channel_range = 0
            level_insights = {}
            
            for level in self.fpn_levels:
                if level in summary:
                    insights = summary[level]['key_insights']
                    global_range = insights['global_value_range']
                    abs_val = insights['max_abs_across_all']
                    ch_var = insights['channel_variability_score']
                    ch_range = insights['max_channel_range']
                    
                    level_insights[level] = {
                        'global_range': global_range,
                        'max_abs': abs_val,
                        'channel_variability': ch_var,
                        'max_channel_range': ch_range
                    }
                    
                    max_global_range = max(max_global_range, global_range)
                    max_abs = max(max_abs, abs_val)
                    max_channel_variability = max(max_channel_variability, ch_var)
                    max_channel_range = max(max_channel_range, ch_range)
            
            f.write(f"Cross-level Analysis:\n")
            f.write(f"  Maximum global range:        {max_global_range:.4f}\n")
            f.write(f"  Maximum absolute value:      {max_abs:.4f}\n")
            f.write(f"  Maximum channel variability: {max_channel_variability:.4f}\n")
            f.write(f"  Maximum single-channel range: {max_channel_range:.4f}\n\n")
            
            # Determine normalization strategy
            f.write("RECOMMENDED NORMALIZATION STRATEGY:\n")
            f.write("-" * 40 + "\n")
            
            # Primary recommendation based on global range
            if max_global_range > 10.0:
                f.write("🚨 URGENT: Implement strong normalization\n")
                primary_rec = "strong"
            elif max_global_range > 5.0:
                f.write("⚠️  RECOMMENDED: Implement normalization\n")
                primary_rec = "moderate"
            elif max_global_range > 2.0:
                f.write("💡 SUGGESTED: Consider normalization\n")
                primary_rec = "optional"
            else:
                f.write("✅ OPTIONAL: Normalization not critical\n")
                primary_rec = "none"
            
            # Channel-wise specific recommendations
            f.write(f"\nChannel-wise Strategy:\n")
            if max_channel_variability > 1.0:
                f.write("🔄 CHANNEL-SPECIFIC: High channel variability detected\n")
                f.write("   → Recommended: Per-channel normalization (LayerNorm or ChannelNorm)\n")
                f.write("   → Each of the 256 channels should be normalized independently\n")
                channel_strategy = "per_channel"
            elif max_channel_variability > 0.5:
                f.write("📊 MODERATE CHANNEL VARIATION: Some channel differences\n")
                f.write("   → Recommended: Global normalization with channel awareness\n")
                f.write("   → Consider GroupNorm or global standardization\n")
                channel_strategy = "group"
            else:
                f.write("🎯 UNIFORM CHANNELS: Channels have similar characteristics\n")
                f.write("   → Recommended: Global normalization (all channels together)\n")
                f.write("   → Simple global mean/std normalization sufficient\n")
                channel_strategy = "global"
            
            # Implementation recommendations
            f.write(f"\nImplementation Recommendations:\n")
            
            if primary_rec in ["strong", "moderate"]:
                f.write("1. 🛠️  IMPLEMENT NORMALIZATION:\n")
                
                if channel_strategy == "per_channel":
                    f.write("   ```python\n")
                    f.write("   # Per-channel normalization for each FPN level\n")
                    f.write("   def normalize_fpn_features(features):\n")
                    f.write("       # features: [B, 256, H, W]\n")
                    f.write("       normalized = torch.zeros_like(features)\n")
                    f.write("       for c in range(256):\n")
                    f.write("           channel = features[:, c, :, :]\n")
                    f.write("           normalized[:, c, :, :] = (channel - channel.mean()) / (channel.std() + 1e-8)\n")
                    f.write("       return normalized\n")
                    f.write("   ```\n")
                
                elif channel_strategy == "group":
                    f.write("   ```python\n")
                    f.write("   # Group normalization (e.g., 32 groups of 8 channels)\n")
                    f.write("   group_norm = nn.GroupNorm(32, 256)  # 32 groups for 256 channels\n")
                    f.write("   normalized_features = group_norm(features)\n")
                    f.write("   ```\n")
                
                else:  # global
                    f.write("   ```python\n")
                    f.write("   # Global normalization\n")
                    f.write("   def normalize_fpn_features(features):\n")
                    f.write("       return (features - features.mean()) / (features.std() + 1e-8)\n")
                    f.write("   ```\n")
                
                f.write("2. 🔧 UPDATE TRAINING LOOP:\n")
                f.write("   - Apply normalization before compression\n")
                f.write("   - Re-tune lambda values (may need adjustment)\n")
                f.write("   - Monitor MSE scale changes\n\n")
                
                f.write("3. 📊 EXPECTED BENEFITS:\n")
                f.write("   - More stable MSE values across experiments\n")
                f.write("   - Better gradient flow during training\n")
                f.write("   - Consistent lambda parameter behavior\n")
                f.write("   - Improved convergence\n")
            
            else:
                f.write("✅ Current feature scales are reasonable for MSE-based training\n")
                f.write("   - No immediate normalization required\n")
                f.write("   - Monitor training stability\n")
                f.write("   - Consider normalization if issues arise\n")
            
            # Per-level summary
            f.write(f"\nPer-Level Summary:\n")
            for level, data in level_insights.items():
                f.write(f"  Level {level}: range={data['global_range']:.3f}, "
                       f"ch_var={data['channel_variability']:.3f}")
                if data['global_range'] > 5.0 or data['channel_variability'] > 1.0:
                    f.write(" ⚠️")
                else:
                    f.write(" ✅")
                f.write("\n")
        
        return report_file


def main():
    args = parse_args()
    
    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("="*60)
    print("FPN FEATURE STATISTICS ANALYSIS")
    print("="*60)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config loaded: {args.config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load detection model
    print("Loading detection model...")
    if os.path.exists(args.detection_checkpoint):
        checkpoint = torch.load(args.detection_checkpoint, map_location=device)
        num_classes = len(KITTIDetectionDataset.CLASSES) + 1
        detection_model = DetectionModel(num_classes=num_classes, pretrained=False)
        detection_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded detection model: {args.detection_checkpoint}")
    else:
        print(f"❌ Detection checkpoint not found: {args.detection_checkpoint}")
        return
    
    detection_model = detection_model.to(device)
    detection_model.eval()
    
    # Setup data loading
    print("Setting up data loading...")
    train_transform = create_transforms(config['data']['transforms'], split='train')
    
    train_dataset = ImageDataset(
        txt_file=config['data']['train_list'],
        transform=train_transform
    )
    
    data_loader = DataLoader(
        train_dataset,
        batch_size=min(4, config['training']['batch_size']),  # Small batch for analysis
        shuffle=True,
        num_workers=min(2, config['training']['num_workers'])
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Will analyze {args.num_samples} samples...")
    
    # Initialize analyzer
    analyzer = SimpleFPNAnalyzer(detection_model, device)
    
    # Analyze samples
    samples_analyzed = 0
    print("\nStarting analysis...")
    
    for batch_idx, images in enumerate(data_loader):
        if samples_analyzed >= args.num_samples:
            break
        
        try:
            # Analyze this batch
            batch_results = analyzer.analyze_batch(images)
            samples_analyzed += images.size(0)
            
            # Progress update
            if batch_idx % 5 == 0:
                print(f"Progress: {samples_analyzed}/{args.num_samples} samples")
                
                # Show sample results - CHANNEL-WISE
                for level, stats in batch_results.items():
                    print(f"  Level {level}: shape={stats['shape']}, "
                          f"global_range={stats['global_max'] - stats['global_min']:.4f}, "
                          f"ch_var={stats['channel_max_abs_std']:.4f}")
                    
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    print(f"\n✅ Analysis completed! Processed {samples_analyzed} samples.")
    
    # Save report
    print("Generating report...")
    report_file = analyzer.save_report(save_dir, {'detection_checkpoint': args.detection_checkpoint}, samples_analyzed)
    print(f"📊 Report saved: {report_file}")
    
    # Print quick channel-wise summary to console
    summary = analyzer.get_summary()
    print("\n" + "="*80)
    print("CHANNEL-WISE QUICK SUMMARY")
    print("="*80)
    
    for level in ['0', '1', '2', '3', '4']:
        if level in summary:
            data = summary[level]
            insights = data['key_insights']
            
            global_range = insights['global_value_range']
            max_abs = insights['max_abs_across_all']
            ch_var = insights['channel_variability_score']
            ch_range = insights['max_channel_range']
            
            print(f"Level {level}: global_range={global_range:.3f}, max_abs={max_abs:.3f}, ch_var={ch_var:.4f}")
            
            # Channel-aware recommendations
            needs_norm = global_range > 5.0 or max_abs > 3.0
            high_ch_var = ch_var > 0.5
            
            if needs_norm and high_ch_var:
                print(f"         🚨 High variability + large values → Per-channel normalization")
            elif needs_norm:
                print(f"         ⚠️  Large values → Global normalization") 
            elif high_ch_var:
                print(f"         📊 High channel variability → Consider channel-aware normalization")
            else:
                print(f"         ✅ Reasonable scales and uniform channels")
    
    # Overall channel-aware recommendation
    max_global_ranges = [summary[level]['key_insights']['global_value_range'] for level in summary.keys()]
    max_ch_vars = [summary[level]['key_insights']['channel_variability_score'] for level in summary.keys()]
    
    max_global_range = max(max_global_ranges) if max_global_ranges else 0
    max_ch_var = max(max_ch_vars) if max_ch_vars else 0
    
    print(f"\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    if max_global_range > 10.0:
        print(f"🚨 URGENT: Implement strong normalization (max_range={max_global_range:.3f})")
    elif max_global_range > 5.0:
        print(f"⚠️  RECOMMENDED: Implement normalization (max_range={max_global_range:.3f})")
    elif max_global_range > 2.0:
        print(f"💡 SUGGESTED: Consider normalization (max_range={max_global_range:.3f})")
    else:
        print(f"✅ OPTIONAL: Normalization not critical (max_range={max_global_range:.3f})")
    
    if max_ch_var > 1.0:
        print(f"🔄 Strategy: Per-channel normalization (high channel variability: {max_ch_var:.4f})")
    elif max_ch_var > 0.5:
        print(f"📊 Strategy: Group normalization (moderate channel variability: {max_ch_var:.4f})")
    else:
        print(f"🎯 Strategy: Global normalization (uniform channels: {max_ch_var:.4f})")
    
    print(f"\n📁 Full channel-wise analysis saved in: {save_dir}")
    print(f"🔬 This analysis considers how FactorizedPrior processes each of the 256 channels")


if __name__ == '__main__':
    main()