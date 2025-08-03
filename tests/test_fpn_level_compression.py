import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from model.factorized_prior import FactorizedPrior
from model.factorized_prior_features_compression import FactorizedPriorFPNCompression
from model.detection import DetectionModel
from model.gdn import GDN
from compressai.entropy_models import EntropyBottleneck
from data.kitti_dataset import KITTIDetectionDataset
from data import ImageDataset
from data.transforms import create_transforms
import yaml
import logging
from pathlib import Path
import numpy as np
from utils import get_project_path
from collections import OrderedDict
import torch.nn.functional as F
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Test FPN level-wise compression vs concatenated compression')
    parser.add_argument('--detection_checkpoint', type=str, 
                       default=get_project_path('checkpoints/detection/run_0.002000_16/best_model.pth'),
                       help='Path to detection model checkpoint')
    parser.add_argument('--fpn_compression_checkpoint', type=str,
                       default=None,
                       help='Path to FPN compression model checkpoint')
    parser.add_argument('--test_images', type=str,
                       default=get_project_path('data/val_list.txt'),
                       help='Path to test images list')
    parser.add_argument('--num_test_samples', type=int, default=10,
                       help='Number of test samples to analyze')
    parser.add_argument('--output_dir', type=str, default='./fpn_compression_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--pretrained_factorized_prior', type=str,
                       default=get_project_path('checkpoints/factorized_prior/model_lamdba_0.010.pth'),
                       help='Path to pretrained FactorizedPrior model')
    return parser.parse_args()

class FPNLevelCompressor(nn.Module):
    """Compress each FPN level separately using individual FactorizedPrior models."""
    
    def __init__(self, fpn_channels_per_level=256, n_hidden=128, n_channels=64, pretrained_path=None):
        """
        Initialize separate compression models for each FPN level.
        
        Args:
            fpn_channels_per_level: Number of channels in each FPN level (default 256)
            n_hidden: Hidden channels for factorized prior
            n_channels: Latent channels for factorized prior
            pretrained_path: Path to pretrained FactorizedPrior model
        """
        super().__init__()
        self.fpn_channels_per_level = fpn_channels_per_level
        self.num_levels = 5  # FPN levels 0-4
        
        # Create separate models for each FPN level
        self.level_compressors = nn.ModuleDict()
        for level in range(self.num_levels):
            # Each level gets its own factorized prior model
            # Input channels = fpn_channels_per_level for each level
            compressor = FactorizedPrior(
                n_hidden=n_hidden,
                n_channels=n_channels,
                input_channels=fpn_channels_per_level,
                output_channels=fpn_channels_per_level
            )
            
            # Load pretrained weights if available
            if pretrained_path and os.path.exists(pretrained_path):
                self._load_pretrained_weights(compressor, pretrained_path, level)
            
            self.level_compressors[str(level)] = compressor
    
    def _load_pretrained_weights(self, model, pretrained_path, level):
        """Load pretrained weights into the model, adapting for different input/output channels."""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            pretrained_state = checkpoint.get('model_state_dict', checkpoint)
            
            # Create a new state dict with adapted weights
            adapted_state = {}
            
            for name, param in model.state_dict().items():
                if name in pretrained_state:
                    pretrained_param = pretrained_state[name]
                    
                    # Handle input/output channel adaptation
                    if 'g_a.0.weight' in name:  # First conv layer (input adaptation)
                        # Pretrained: [n_hidden, 3, 5, 5] -> Target: [n_hidden, 256, 5, 5]
                        if pretrained_param.size(1) == 3 and param.size(1) == 256:
                            # Replicate RGB weights across all 256 channels
                            adapted_param = pretrained_param.repeat(1, 256//3, 1, 1)
                            # Handle remainder channels
                            remainder = 256 % 3
                            if remainder > 0:
                                extra = pretrained_param[:, :remainder, :, :]
                                adapted_param = torch.cat([adapted_param, extra], dim=1)
                            adapted_state[name] = adapted_param
                        else:
                            adapted_state[name] = pretrained_param
                    elif 'g_s.6.weight' in name:  # Last conv layer (output adaptation) 
                        # Pretrained: [3, n_hidden, 5, 5] -> Target: [256, n_hidden, 5, 5]
                        if pretrained_param.size(0) == 3 and param.size(0) == 256:
                            # Replicate RGB weights across all 256 output channels
                            adapted_param = pretrained_param.repeat(256//3, 1, 1, 1)
                            # Handle remainder channels
                            remainder = 256 % 3
                            if remainder > 0:
                                extra = pretrained_param[:remainder, :, :, :]
                                adapted_param = torch.cat([adapted_param, extra], dim=0)
                            adapted_state[name] = adapted_param
                        else:
                            adapted_state[name] = pretrained_param
                    elif param.shape == pretrained_param.shape:
                        # Same shape - direct copy
                        adapted_state[name] = pretrained_param
                    else:
                        # Shape mismatch - skip and use random initialization
                        logging.warning(f"Skipping {name} due to shape mismatch: {param.shape} vs {pretrained_param.shape}")
                        adapted_state[name] = param
                else:
                    # Not in pretrained - use random initialization
                    adapted_state[name] = param
            
            # Load adapted weights
            model.load_state_dict(adapted_state, strict=False)
            logging.info(f"Loaded pretrained weights for level {level} compressor")
            
        except Exception as e:
            logging.warning(f"Could not load pretrained weights for level {level}: {e}")
            logging.info(f"Using random initialization for level {level} compressor")
    

    
    def forward(self, fpn_features):
        """
        Compress each FPN level separately.
        
        Args:
            fpn_features: Dict with keys '0', '1', '2', '3', '4' and tensor values
            
        Returns:
            Dict with reconstructed features and likelihoods for each level
        """
        results = {
            'fpn_features_hat': OrderedDict(),
            'likelihoods_per_level': OrderedDict()
        }
        
        for level_key in ['0', '1', '2', '3', '4']:
            if level_key in fpn_features:
                level_features = fpn_features[level_key]  # [B, C, H, W]
                
                # Compress this level
                compressor = self.level_compressors[level_key]
                out = compressor(level_features)
                
                results['fpn_features_hat'][level_key] = out['x_hat']
                results['likelihoods_per_level'][level_key] = out['likelihoods']
        
        return results



def compute_fpn_level_bpp(fpn_features, likelihoods_per_level):
    """Compute BPP for each FPN level separately."""
    level_bpps = {}
    total_bpp = 0.0
    
    for level_key in fpn_features.keys():
        if level_key in likelihoods_per_level:
            # Get number of pixels for this level
            level_features = fpn_features[level_key]
            level_pixels = level_features.size(-2) * level_features.size(-1)  # H * W
            batch_size = level_features.size(0)
            
            # Compute BPP for this level
            level_likelihoods = likelihoods_per_level[level_key]
            level_bpp = -torch.log2(level_likelihoods).sum() / (level_pixels * batch_size)
            
            level_bpps[level_key] = level_bpp.item()
            total_bpp += level_bpp.item()
    
    return level_bpps, total_bpp

def compute_fpn_level_mse(fpn_features_orig, fpn_features_recon):
    """Compute MSE for each FPN level separately."""
    level_mses = {}
    total_mse = 0.0
    total_elements = 0
    
    for level_key in fpn_features_orig.keys():
        if level_key in fpn_features_recon:
            orig = fpn_features_orig[level_key]
            recon = fpn_features_recon[level_key]
            
            level_mse = F.mse_loss(orig, recon)
            level_mses[level_key] = level_mse.item()
            
            # Accumulate for total MSE
            total_mse += F.mse_loss(orig, recon, reduction='sum').item()
            total_elements += orig.numel()
    
    overall_mse = total_mse / total_elements if total_elements > 0 else 0.0
    return level_mses, overall_mse

def test_compression_methods(detection_model, fpn_compression_model, level_compressor, 
                           test_loader, device, output_dir, num_samples=10):
    """Test and compare both compression methods."""
    
    detection_model.eval()
    if fpn_compression_model is not None:
        fpn_compression_model.eval()
    level_compressor.eval()
    
    results = {
        'concatenated': {'bpps': [], 'mses': [], 'samples': []},
        'level_wise': {'bpps': [], 'mses': [], 'level_bpps': [], 'level_mses': [], 'samples': []}
    }
    
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # Extract FPN features
            images = images.to(device)
            image_list = [img for img in images]
            fpn_features = detection_model.get_fpn_features(image_list)
            
            sample_result = {
                'sample_id': i,
                'fpn_shapes': {k: list(v.shape) for k, v in fpn_features.items()}
            }
            
            # Log FPN spatial dimensions for first sample
            if i == 0:
                logging.info("FPN Feature Spatial Dimensions:")
                for level_key, features in fpn_features.items():
                    B, C, H, W = features.shape
                    logging.info(f"  Level {level_key}: [{B}, {C}, {H}, {W}] - Spatial: {H}×{W}")
                logging.info("")
            
            # Method 1: Concatenated compression (if model available)
            if fpn_compression_model is not None:
                concat_out = fpn_compression_model(fpn_features)
                concat_fpn_hat = concat_out['fpn_features_hat']
                concat_likelihoods = concat_out['likelihoods']
                
                # Compute metrics for concatenated method
                from train_factorized_prior_features_compression import compute_fpn_bpp, compute_fpn_mse
                concat_bpp = compute_fpn_bpp(fpn_features, concat_likelihoods).item()
                concat_mse = compute_fpn_mse(fpn_features, concat_fpn_hat).item()
                
                results['concatenated']['bpps'].append(concat_bpp)
                results['concatenated']['mses'].append(concat_mse)
                sample_result['concatenated'] = {
                    'bpp': concat_bpp,
                    'mse': concat_mse
                }
            
            # Method 2: Level-wise compression
            level_out = level_compressor(fpn_features)
            level_fpn_hat = level_out['fpn_features_hat']
            level_likelihoods_per_level = level_out['likelihoods_per_level']
            
            # Log likelihood dimensions for first sample
            if i == 0:
                logging.info("Likelihood Dimensions After Compression:")
                for level_key, likelihoods in level_likelihoods_per_level.items():
                    B, C, H, W = likelihoods.shape
                    logging.info(f"  Level {level_key}: [{B}, {C}, {H}, {W}] - Spatial: {H}×{W}")
                logging.info("")
            
            # Compute metrics for level-wise method
            level_bpps, total_level_bpp = compute_fpn_level_bpp(fpn_features, level_likelihoods_per_level)
            level_mses, total_level_mse = compute_fpn_level_mse(fpn_features, level_fpn_hat)
            
            results['level_wise']['bpps'].append(total_level_bpp)
            results['level_wise']['mses'].append(total_level_mse)
            results['level_wise']['level_bpps'].append(level_bpps)
            results['level_wise']['level_mses'].append(level_mses)
            
            sample_result['level_wise'] = {
                'total_bpp': total_level_bpp,
                'total_mse': total_level_mse,
                'level_bpps': level_bpps,
                'level_mses': level_mses
            }
            
            results['concatenated']['samples'].append(sample_result)
            results['level_wise']['samples'].append(sample_result)
            
            # Log progress
            logging.info(f"Sample {i}: Level-wise BPP: {total_level_bpp:.4f}, MSE: {total_level_mse:.6f}")
            if fpn_compression_model is not None:
                logging.info(f"Sample {i}: Concatenated BPP: {concat_bpp:.4f}, MSE: {concat_mse:.6f}")
            
            # Log per-level statistics
            level_stats = " | ".join([f"L{k}: BPP={v:.3f}" for k, v in level_bpps.items()])
            logging.info(f"Sample {i} Level BPPs: {level_stats}")
    
    return results

def save_analysis_results(results, output_dir):
    """Save comprehensive analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = output_dir / f'fpn_compression_analysis_{timestamp}.txt'
    
    with open(analysis_file, 'w') as f:
        f.write("FPN Compression Methods Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        # Level-wise analysis
        level_bpps = np.array(results['level_wise']['bpps'])
        level_mses = np.array(results['level_wise']['mses'])
        
        f.write("LEVEL-WISE COMPRESSION RESULTS:\n")
        f.write(f"  Average Total BPP: {level_bpps.mean():.4f} ± {level_bpps.std():.4f}\n")
        f.write(f"  Average Total MSE: {level_mses.mean():.6f} ± {level_mses.std():.6f}\n")
        f.write(f"  BPP Range: [{level_bpps.min():.4f}, {level_bpps.max():.4f}]\n")
        f.write(f"  MSE Range: [{level_mses.min():.6f}, {level_mses.max():.6f}]\n\n")
        
        # Per-level statistics
        f.write("PER-LEVEL STATISTICS:\n")
        level_keys = ['0', '1', '2', '3', '4']
        for level_key in level_keys:
            level_bpp_values = [sample[level_key] for sample in results['level_wise']['level_bpps'] if level_key in sample]
            level_mse_values = [sample[level_key] for sample in results['level_wise']['level_mses'] if level_key in sample]
            
            if level_bpp_values:
                level_bpp_array = np.array(level_bpp_values)
                level_mse_array = np.array(level_mse_values)
                
                f.write(f"  Level {level_key}:\n")
                f.write(f"    BPP: {level_bpp_array.mean():.4f} ± {level_bpp_array.std():.4f}\n")
                f.write(f"    MSE: {level_mse_array.mean():.6f} ± {level_mse_array.std():.6f}\n")
        
        f.write("\n")
        
        # Concatenated analysis (if available)
        if results['concatenated']['bpps']:
            concat_bpps = np.array(results['concatenated']['bpps'])
            concat_mses = np.array(results['concatenated']['mses'])
            
            f.write("CONCATENATED COMPRESSION RESULTS:\n")
            f.write(f"  Average Total BPP: {concat_bpps.mean():.4f} ± {concat_bpps.std():.4f}\n")
            f.write(f"  Average Total MSE: {concat_mses.mean():.6f} ± {concat_mses.std():.6f}\n")
            f.write(f"  BPP Range: [{concat_bpps.min():.4f}, {concat_bpps.max():.4f}]\n")
            f.write(f"  MSE Range: [{concat_mses.min():.6f}, {concat_mses.max():.6f}]\n\n")
            
            # Comparison
            f.write("COMPARISON:\n")
            f.write(f"  BPP Improvement (Level-wise vs Concatenated): {concat_bpps.mean() - level_bpps.mean():.4f}\n")
            f.write(f"  MSE Difference (Level-wise vs Concatenated): {level_mses.mean() - concat_mses.mean():.6f}\n")
            
            if level_bpps.mean() < concat_bpps.mean():
                f.write(f"  → Level-wise compression achieves {((concat_bpps.mean() - level_bpps.mean()) / concat_bpps.mean() * 100):.1f}% better compression\n")
            else:
                f.write(f"  → Concatenated compression achieves {((level_bpps.mean() - concat_bpps.mean()) / level_bpps.mean() * 100):.1f}% better compression\n")
        
        f.write("\n")
        
        # Detailed sample results
        f.write("DETAILED SAMPLE RESULTS:\n")
        for i, sample in enumerate(results['level_wise']['samples']):
            f.write(f"\nSample {i}:\n")
            f.write(f"  FPN Shapes: {sample['fpn_shapes']}\n")
            
            level_wise = sample['level_wise']
            f.write(f"  Level-wise: Total BPP={level_wise['total_bpp']:.4f}, Total MSE={level_wise['total_mse']:.6f}\n")
            f.write(f"    Per-level BPPs: {level_wise['level_bpps']}\n")
            f.write(f"    Per-level MSEs: {level_wise['level_mses']}\n")
            
            if 'concatenated' in sample:
                concat = sample['concatenated']
                f.write(f"  Concatenated: BPP={concat['bpp']:.4f}, MSE={concat['mse']:.6f}\n")
    
    logging.info(f"Analysis results saved to {analysis_file}")
    return analysis_file

def main():
    args = parse_args()
    
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
        logging.warning(f"Detection checkpoint not found: {args.detection_checkpoint}")
        detection_model = DetectionModel(num_classes=2)
    
    detection_model = detection_model.to(device)
    detection_model.eval()
    
    # Load concatenated FPN compression model (if available)
    fpn_compression_model = None
    if args.fpn_compression_checkpoint and os.path.exists(args.fpn_compression_checkpoint):
        logging.info("Loading concatenated FPN compression model...")
        checkpoint = torch.load(args.fpn_compression_checkpoint, map_location=device)
        fpn_compression_model = FactorizedPriorFPNCompression.from_state_dict(checkpoint['model_state_dict'])
        fpn_compression_model = fpn_compression_model.to(device)
        fpn_compression_model.eval()
        logging.info("Loaded concatenated FPN compression model")
    else:
        logging.info("No concatenated FPN compression model provided - testing level-wise only")
    
    # Create level-wise compressor with pretrained weights
    logging.info("Creating level-wise FPN compressor...")
    if os.path.exists(args.pretrained_factorized_prior):
        logging.info(f"Using pretrained FactorizedPrior: {args.pretrained_factorized_prior}")
        pretrained_path = args.pretrained_factorized_prior
    else:
        logging.warning(f"Pretrained model not found: {args.pretrained_factorized_prior}")
        logging.info("Using random initialization")
        pretrained_path = None
    
    level_compressor = FPNLevelCompressor(
        fpn_channels_per_level=256,
        n_hidden=128,
        n_channels=192,
        pretrained_path=pretrained_path
    ).to(device)
    level_compressor.eval()
    
    # Setup test data
    test_transform = transforms.Compose([
        transforms.Resize((480, 640)),  # Resize to consistent size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageDataset(args.test_images, test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for detailed analysis
        shuffle=False,
        num_workers=0
    )
    
    logging.info(f"Test dataset size: {len(test_dataset)}")
    
    # Run compression tests
    logging.info("Starting compression comparison tests...")
    results = test_compression_methods(
        detection_model=detection_model,
        fpn_compression_model=fpn_compression_model,
        level_compressor=level_compressor,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir,
        num_samples=args.num_test_samples
    )
    
    # Save and display results
    analysis_file = save_analysis_results(results, args.output_dir)
    
    # Print summary to console
    level_bpps = np.array(results['level_wise']['bpps'])
    level_mses = np.array(results['level_wise']['mses'])
    
    print("\n" + "="*60)
    print("FPN COMPRESSION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Level-wise Compression:")
    print(f"  Average BPP: {level_bpps.mean():.4f} ± {level_bpps.std():.4f}")
    print(f"  Average MSE: {level_mses.mean():.6f} ± {level_mses.std():.6f}")
    
    if results['concatenated']['bpps']:
        concat_bpps = np.array(results['concatenated']['bpps'])
        concat_mses = np.array(results['concatenated']['mses'])
        print(f"\nConcatenated Compression:")
        print(f"  Average BPP: {concat_bpps.mean():.4f} ± {concat_bpps.std():.4f}")
        print(f"  Average MSE: {concat_mses.mean():.6f} ± {concat_mses.std():.6f}")
        
        bpp_improvement = concat_bpps.mean() - level_bpps.mean()
        print(f"\nComparison:")
        if bpp_improvement > 0:
            print(f"  Level-wise is {bpp_improvement:.4f} BPP better ({bpp_improvement/concat_bpps.mean()*100:.1f}%)")
        else:
            print(f"  Concatenated is {-bpp_improvement:.4f} BPP better ({-bpp_improvement/level_bpps.mean()*100:.1f}%)")
    
    print(f"\nDetailed results saved to: {analysis_file}")
    print("="*60)

if __name__ == '__main__':
    main()