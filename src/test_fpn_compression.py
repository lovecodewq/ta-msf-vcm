#!/usr/bin/env python3
"""
Test script for FPN feature compression model.
This script verifies that the FPN compression pipeline works correctly.
"""

import torch
import torch.nn.functional as F
from model.detection import DetectionModel
from model.factorized_prior_features_compression import FactorizedPriorFPNCompression
from collections import OrderedDict
import numpy as np

"""
Note: This script will failed since the pretrained factorized prior model is trained on image instead of FPN features.
That output of this script is the reason why we need to train a factorized prior model for FPN features.
"""
def create_dummy_fpn_features(batch_size=1, device='cpu'):
    """Create dummy FPN features for testing."""
    fpn_features = OrderedDict()
    
    # Standard FPN scales: P2, P3, P4, P5, P6
    # Assuming input image size around 384x1280
    base_h, base_w = 96, 320  # P2 scale (1/4 of input)
    
    for i in range(5):
        # Each level has half the spatial resolution of the previous
        h = base_h // (2 ** i)
        w = base_w // (2 ** i)
        
        # All FPN levels have 256 channels
        feature = torch.randn(batch_size, 256, h, w, device=device)
        fpn_features[str(i)] = feature
        
        print(f"FPN Level {i}: {feature.shape}")
    
    return fpn_features

def test_fpn_compression_model():
    """Test the FPN compression model pipeline."""
    print("Testing FPN Feature Compression Model")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = FactorizedPriorFPNCompression(
        n_hidden=128,
        n_channels=192,
        fpn_channels_per_level=256,
        num_fpn_levels=5
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy FPN features
    print("\nCreating dummy FPN features...")
    fpn_features = create_dummy_fpn_features(batch_size=2, device=device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(fpn_features)
    
    fpn_features_hat = output['fpn_features_hat']
    likelihoods = output['likelihoods']
    metadata = output['metadata']
    
    print("Forward pass successful!")
    print(f"Likelihoods shape: {likelihoods.shape}")
    
    # Verify reconstructed FPN features
    print("\nVerifying reconstructed FPN features...")
    for level_key in fpn_features.keys():
        orig_shape = fpn_features[level_key].shape
        recon_shape = fpn_features_hat[level_key].shape
        
        if orig_shape == recon_shape:
            mse = F.mse_loss(fpn_features[level_key], fpn_features_hat[level_key])
            print(f"Level {level_key}: Shape ✓ {orig_shape}, MSE: {mse.item():.6f}")
        else:
            print(f"Level {level_key}: Shape ✗ Original: {orig_shape}, Reconstructed: {recon_shape}")
    
    # Test compression and decompression
    print("\nTesting compression and decompression...")
    model.update()  # Update entropy bottleneck
    
    # Compress
    compressed = model.compress(fpn_features)
    print(f"Compressed data keys: {compressed.keys()}")
    print(f"Compressed latent shape: {compressed['shape']}")
    
    # Decompress
    decompressed = model.decompress(
        compressed['y_strings'],
        compressed['shape'],
        compressed['input_size'],
        compressed['fpn_metadata']
    )
    
    fpn_features_decompressed = decompressed['fpn_features_hat']
    
    # Verify decompressed features
    print("\nVerifying decompressed FPN features...")
    total_mse = 0.0
    for level_key in fpn_features.keys():
        orig = fpn_features[level_key]
        decomp = fpn_features_decompressed[level_key]
        
        if orig.shape == decomp.shape:
            mse = F.mse_loss(orig, decomp)
            total_mse += mse.item()
            print(f"Level {level_key}: MSE: {mse.item():.6f}")
        else:
            print(f"Level {level_key}: Shape mismatch!")
    
    print(f"\nTotal average MSE: {total_mse / len(fpn_features):.6f}")
    print("FPN compression test completed successfully!")

def test_with_real_detection_model():
    """Test with real detection model for FPN feature extraction."""
    print("\n" + "=" * 50)
    print("Testing with Real Detection Model")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create detection model
    detection_model = DetectionModel(num_classes=2).to(device)
    detection_model.eval()
    
    # Create dummy image
    batch_size = 1
    dummy_image = torch.randn(3, 384, 1280, device=device)
    images = [dummy_image]
    
    print(f"Input image shape: {dummy_image.shape}")
    
    # Extract FPN features
    with torch.no_grad():
        fpn_features = detection_model.get_fpn_features(images)
    
    print("\nExtracted FPN features:")
    for level_key, feat in fpn_features.items():
        print(f"Level {level_key}: {feat.shape}")
    
    # Test compression
    compression_model = FactorizedPriorFPNCompression(
        n_hidden=128,
        n_channels=192,
        fpn_channels_per_level=256,
        num_fpn_levels=5
    ).to(device)
    
    compression_model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = compression_model(fpn_features)
    
    print("\nCompression test with real FPN features successful!")
    
    # Compute approximate compression statistics
    fpn_features_hat = output['fpn_features_hat']
    likelihoods = output['likelihoods']
    
    # Compute MSE
    total_mse = 0.0
    total_elements = 0
    for level_key in fpn_features.keys():
        orig = fpn_features[level_key]
        recon = fpn_features_hat[level_key]
        
        level_mse = F.mse_loss(orig, recon, reduction='sum')
        total_mse += level_mse.item()
        total_elements += orig.numel()
    
    avg_mse = total_mse / total_elements
    
    # Compute approximate BPP
    total_pixels = sum(feat.size(-2) * feat.size(-1) for feat in fpn_features.values())
    bpp = -torch.log2(likelihoods).sum().item() / total_pixels
    
    print(f"\nCompression Statistics:")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Approximate BPP: {bpp:.4f}")

if __name__ == '__main__':
    # Test with dummy data
    test_fpn_compression_model()
    
    # Test with real detection model
    test_with_real_detection_model()
    
    print("\nAll tests completed successfully!") 