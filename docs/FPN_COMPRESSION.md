# FPN Feature Compression

This guide covers training compression models for Feature Pyramid Network (FPN) features extracted from detection models.

## Overview

FPN feature compression enables:
- **Efficient Storage**: Compress intermediate FPN features instead of raw images
- **Detection Pipeline Integration**: Maintain detection performance with compressed features
- **Rate-Distortion Optimization**: Balance compression rate vs. detection accuracy
- **End-to-End Evaluation**: Complete pipeline from compressed features to detection results

## Architecture

The FPN compression model consists of:
- **Multi-Level Compression**: Separate compression for each FPN level (P2-P6)
- **Factorized Prior**: Advanced entropy modeling for better compression
- **Feature-Aware Design**: Optimized for high-dimensional feature representations
- **Rate-Distortion Control**: Lambda-based trade-off between compression and quality

## Quick Start

### 1. Prerequisites
First, train a detection model to extract FPN features:
```bash
# Train detection model (if not already done)
python src/train_detection.py --config configs/train_detection.yaml
```

### 2. Train FPN Compression
```bash
# Train FPN feature compression
python src/train_factorized_prior_features_compression.py \
    --config configs/train_factorized_prior_fpn.yaml \
    --detection_checkpoint checkpoints/detection/run_*/best_model.pth
```

## Configuration

Key settings in `configs/train_factorized_prior_fpn.yaml`:

### Model Architecture
```yaml
model:
  name: "factorized_prior_fpn"
  n_hidden: 128              # Hidden channels in transforms
  n_channels: 192            # Latent representation channels
  fpn_channels_per_level: 256  # FPN feature channels (typically 256)
  num_fpn_levels: 5          # Number of FPN levels (P2-P6)
```

### Training Configuration
```yaml
training:
  batch_size: 4              # Smaller due to large feature maps
  learning_rate: 5e-5        # Lower LR for feature compression
  lambda: 2.5e-4             # Rate-distortion trade-off
  epochs: 100
```

### Data Pipeline
```yaml
data:
  # Image transforms (applied before FPN extraction)
  transforms:
    resize:
      size: [384, 1280]      # Detection model input size
    normalize:
      mean: [0.485, 0.456, 0.406]  # ImageNet normalization
      std: [0.229, 0.224, 0.225]
```

## Training Process

### 1. Feature Extraction Pipeline
```
Raw Image → Detection Model → FPN Features → Compression Model → Reconstructed Features
```

### 2. Loss Function
The training optimizes:
```
Total Loss = MSE(FPN_original, FPN_reconstructed) + λ × Rate
```

Where:
- **MSE**: Mean squared error between original and reconstructed FPN features
- **Rate**: Bits per pixel equivalent for FPN features
- **λ (lambda)**: Controls rate-distortion trade-off

### 3. Multi-Level Processing
Each FPN level (P2, P3, P4, P5, P6) is processed independently:
- Different spatial resolutions (from 1/4 to 1/64 of input)
- Shared compression architecture across levels
- Level-specific rate-distortion analysis

## Monitoring Training

### Enhanced Logging
FPN compression training provides detailed metrics:

```bash
# Example log output
Train Batch: [10/250] Loss: 0.1234 MSE: 0.000123 BPP: 0.456 GradNorm: 1.23 LR: 5.00e-05 
FPN_MSE: [L0:0.00012 L1:0.00015 L2:0.00018 L3:0.00021 L4:0.00025]
```

### Key Metrics
- **Overall MSE**: Combined MSE across all FPN levels
- **Level-Specific MSE**: Individual MSE for each FPN level  
- **BPP**: Bits per pixel equivalent for FPN features
- **Rate-Distortion Ratio**: Balance between compression and quality
- **Gradient Norm**: Training stability indicator

### Warning System
Automatic alerts for:
- High gradient norms (potential instability)
- Unbalanced rate-distortion trade-off
- Excessive feature magnitudes

## Advanced Configuration

### Lambda Sweep Training
Train multiple models with different lambda values:

```yaml
# configs/train_fpn_lambda_sweep.yaml
training:
  lambda_values: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]  # Multiple rate points
```

### Mixed Precision
```yaml
mixed_precision:
  enabled: true
  dtype: "float16"
  scale_factor: 128
```

### Memory Optimization
```yaml
memory:
  gradient_accumulation:
    enabled: true
    steps: 2              # Effective batch size = 4 * 2 = 8
  
  # Reduce memory usage
  training:
    batch_size: 2         # Smaller batch
    num_workers: 2        # Fewer data workers
```

## Evaluation and Analysis

### Feature Analysis
The training automatically saves FPN feature analysis:

```
checkpoints/fpn_compression/run_*/fpn_analysis/lambda_0.000/
├── lambda_2.5e-4_epoch_10_analysis.txt
├── lambda_2.5e-4_epoch_20_analysis.txt
└── ...
```

### Rate-Distortion Curves
Generate comprehensive R-D analysis:

```bash
# Evaluate across multiple lambda values
python src/evaluation/evaluate_fpn_compression.py \
    --checkpoint_dir checkpoints/fpn_compression \
    --output_dir evaluation_results/fpn_rd_curves
```

### End-to-End Detection Evaluation
Test detection performance with compressed features:

```bash
# Evaluate detection pipeline with compression
python src/evaluation/evaluate_detection_with_compression.py \
    --detection_model checkpoints/detection/best_model.pth \
    --compression_model checkpoints/fpn_compression/best_model.pth \
    --test_data data/processed/kitti/test.txt
```

## Typical Results

### Compression Performance
Expected results on KITTI dataset:

| Lambda | BPP | MSE | Detection mAP | Notes |
|--------|-----|-----|---------------|-------|
| 1e-5   | 2.5 | 1e-5 | 0.74 | High quality |
| 5e-5   | 1.8 | 3e-5 | 0.72 | Balanced |
| 1e-4   | 1.2 | 6e-5 | 0.69 | Medium compression |
| 5e-4   | 0.8 | 2e-4 | 0.65 | High compression |
| 1e-3   | 0.5 | 5e-4 | 0.58 | Very high compression |

### Training Time
- **RTX 5090**: ~4-6 hours for 100 epochs
- **RTX 3090**: ~6-8 hours for 100 epochs
- **Memory Usage**: ~12-16GB VRAM with batch_size=4

## Troubleshooting

### Common Issues

**High MSE, Low Rate**
- Increase lambda value
- Check feature extraction pipeline
- Verify FPN feature normalization

**Low MSE, High Rate**  
- Decrease lambda value
- Enable mixed precision
- Check entropy bottleneck convergence

**Training Instability**
- Reduce learning rate
- Enable gradient clipping
- Check for feature magnitude explosions

**Memory Issues**
- Reduce batch size to 2 or 1
- Enable gradient accumulation
- Use smaller image input sizes

### Performance Tips

1. **Lambda Selection**: Start with 2.5e-4, adjust based on desired compression ratio
2. **Convergence**: Expect convergence around 50-80 epochs
3. **Feature Quality**: Monitor FPN-level MSE to identify problematic levels
4. **Rate Control**: Watch rate-distortion ratio warnings

## Integration with Detection Pipeline

### Feature Compression Workflow
```
1. Train Detection Model → Extract FPN Features
2. Train Compression Model → Compress FPN Features  
3. Evaluation → Detection Performance with Compressed Features
```

### Production Deployment
```python
# Example usage in production
detection_model = load_detection_model("detection_checkpoint.pth")
compression_model = load_compression_model("fpn_compression_checkpoint.pth")

# Process image
image = load_image("input.jpg")
fpn_features = detection_model.get_fpn_features(image)
compressed_features = compression_model.compress(fpn_features)

# Use compressed features for detection
detections = detection_model.detect_from_features(compressed_features)
```

## Next Steps

After training FPN compression:
1. **Rate-Distortion Analysis**: Generate R-D curves for different lambda values
2. **End-to-End Evaluation**: Test complete detection pipeline
3. **Model Optimization**: Quantization and deployment optimization
4. **Comparative Analysis**: Compare with other compression methods

See [EVALUATION.md](EVALUATION.md) for comprehensive evaluation procedures. 