# Evaluation Guide

This guide covers comprehensive evaluation procedures for all models in the neural image compression project.

## Overview

The evaluation framework supports:
- **Standard Image Compression**: PSNR, MS-SSIM, rate-distortion curves
- **Detection Models**: mAP, precision-recall curves, inference speed
- **FPN Feature Compression**: Feature fidelity, detection pipeline integration
- **Comparative Analysis**: Cross-model comparison and benchmarking

## Evaluation Types

### 1. Standard Image Compression Evaluation

#### Rate-Distortion Analysis
```bash
# Evaluate factorized prior models
python src/evaluation/evaluate_compression.py \
    --checkpoint_dir checkpoints/factorized_prior \
    --test_data data/processed/kitti/test.txt \
    --output_dir evaluation_results/compression
```

#### Metrics
- **PSNR**: Peak Signal-to-Noise Ratio
- **MS-SSIM**: Multi-Scale Structural Similarity
- **BPP**: Bits Per Pixel
- **Encoding/Decoding Time**: Performance benchmarks

#### Output Files
```
evaluation_results/compression/
├── rate_distortion_curves.png
├── psnr_vs_bpp.png
├── msssim_vs_bpp.png
├── detailed_metrics.csv
└── compression_summary.txt
```

### 2. Detection Model Evaluation

#### mAP Evaluation
```bash
# Evaluate detection performance
python src/evaluation/evaluate_detection.py \
    --model_path checkpoints/detection/best_model.pth \
    --test_data data/processed/kitti/test.txt \
    --output_dir evaluation_results/detection
```

#### Metrics
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.75**: Mean Average Precision at IoU 0.75
- **Per-Class AP**: Individual class performance
- **Inference Speed**: FPS and latency

#### Visualization
```bash
# Generate detection visualizations
python src/evaluation/visualize_detections.py \
    --model_path checkpoints/detection/best_model.pth \
    --test_images data/processed/kitti/test_samples/ \
    --output_dir evaluation_results/detection/visualizations
```

### 3. FPN Feature Compression Evaluation

#### Feature Fidelity Analysis
```bash
# Evaluate FPN compression quality
python src/evaluation/evaluate_fpn_compression.py \
    --compression_model checkpoints/fpn_compression/best_model.pth \
    --detection_model checkpoints/detection/best_model.pth \
    --test_data data/processed/kitti/test.txt \
    --output_dir evaluation_results/fpn_compression
```

#### End-to-End Detection Pipeline
```bash
# Evaluate detection with compressed FPN features
python src/evaluation/evaluate_detection_with_compression.py \
    --detection_model checkpoints/detection/best_model.pth \
    --compression_model checkpoints/fpn_compression/best_model.pth \
    --test_data data/processed/kitti/test.txt \
    --output_dir evaluation_results/end_to_end
```

## Automated Evaluation Scripts

### Batch Evaluation
```bash
# Evaluate all models in a directory
python src/evaluation/batch_evaluate.py \
    --base_dir checkpoints/ \
    --test_data data/processed/kitti/test.txt \
    --output_dir evaluation_results/batch_results
```

### Lambda Sweep Analysis
```bash
# Analyze multiple lambda values
python src/evaluation/lambda_sweep_analysis.py \
    --checkpoint_pattern "checkpoints/*/run_*lambda*" \
    --output_dir evaluation_results/lambda_analysis
```

## Rate-Distortion Curve Generation

### Standard Compression
```python
# Example: Generate R-D curves for compression models
import matplotlib.pyplot as plt
from src.evaluation.rd_curves import generate_rd_curves

models = [
    "checkpoints/factorized_prior/lambda_0.001/best_model.pth",
    "checkpoints/factorized_prior/lambda_0.01/best_model.pth", 
    "checkpoints/factorized_prior/lambda_0.1/best_model.pth"
]

generate_rd_curves(models, test_data="data/processed/kitti/test.txt")
```

### FPN Compression R-D Analysis
```bash
# Generate R-D curves for FPN compression
python src/evaluation/fpn_rd_analysis.py \
    --fpn_models "checkpoints/fpn_compression/run_*lambda*" \
    --detection_model checkpoints/detection/best_model.pth \
    --test_data data/processed/kitti/test.txt
```

## Performance Benchmarking

### Inference Speed Testing
```bash
# Benchmark inference speed
python src/evaluation/speed_benchmark.py \
    --model_path checkpoints/detection/best_model.pth \
    --input_size 800 1333 \
    --batch_sizes 1 4 8 16 \
    --num_runs 100
```

### Memory Usage Analysis
```bash
# Profile memory usage
python src/evaluation/memory_profiler.py \
    --model_path checkpoints/fpn_compression/best_model.pth \
    --input_size 384 1280 \
    --profile_cuda True
```

## Comparative Analysis

### Model Comparison
```bash
# Compare multiple model types
python src/evaluation/model_comparison.py \
    --models detection:checkpoints/detection/best_model.pth \
             compression:checkpoints/factorized_prior/lambda_0.01/best_model.pth \
             fpn:checkpoints/fpn_compression/best_model.pth \
    --test_data data/processed/kitti/test.txt \
    --metrics mAP PSNR MS-SSIM BPP
```

### Ablation Studies
```bash
# Run ablation studies
python src/evaluation/ablation_study.py \
    --base_model checkpoints/detection/best_model.pth \
    --variations backbone_frozen,backbone_unfrozen,mixed_precision \
    --test_data data/processed/kitti/test.txt
```

## Custom Evaluation Scripts

### Single Image Analysis
```python
# Detailed analysis of single image
from src.evaluation.single_image_analysis import analyze_image

results = analyze_image(
    image_path="test_image.jpg",
    models={
        "detection": "checkpoints/detection/best_model.pth",
        "compression": "checkpoints/factorized_prior/best_model.pth",
        "fpn_compression": "checkpoints/fpn_compression/best_model.pth"
    }
)

print(f"Detection mAP: {results['detection']['mAP']}")
print(f"Compression PSNR: {results['compression']['PSNR']}")
print(f"FPN compression rate: {results['fpn_compression']['BPP']}")
```

### Dataset Statistics
```bash
# Analyze dataset characteristics
python src/evaluation/dataset_analysis.py \
    --data_path data/processed/kitti \
    --output_dir evaluation_results/dataset_stats
```

## Evaluation Metrics Reference

### Image Quality Metrics

| Metric | Range | Higher Better | Description |
|--------|-------|---------------|-------------|
| PSNR | 0-∞ dB | ✓ | Peak Signal-to-Noise Ratio |
| MS-SSIM | 0-1 | ✓ | Multi-Scale Structural Similarity |
| LPIPS | 0-∞ | ✗ | Learned Perceptual Image Patch Similarity |

### Detection Metrics

| Metric | Range | Higher Better | Description |
|--------|-------|---------------|-------------|
| mAP@0.5 | 0-1 | ✓ | Mean Average Precision at IoU 0.5 |
| mAP@0.75 | 0-1 | ✓ | Mean Average Precision at IoU 0.75 |
| Recall | 0-1 | ✓ | True Positive Rate |
| Precision | 0-1 | ✓ | Positive Predictive Value |

### Compression Metrics

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| BPP | 0-∞ | Lower | Bits Per Pixel |
| Compression Ratio | 1-∞ | Higher | Original Size / Compressed Size |
| Encoding Time | 0-∞ ms | Lower | Time to compress |
| Decoding Time | 0-∞ ms | Lower | Time to decompress |

## Expected Results

### KITTI Dataset Benchmarks

#### Detection Performance
| Model | mAP@0.5 | mAP@0.75 | Inference Speed (FPS) |
|-------|---------|----------|-----------------------|
| Faster R-CNN (baseline) | 0.75 | 0.45 | 15 |
| With FPN Compression (λ=1e-4) | 0.69 | 0.41 | 18 |
| With FPN Compression (λ=1e-3) | 0.58 | 0.32 | 22 |

#### Compression Performance
| Model | Lambda | BPP | PSNR | MS-SSIM |
|-------|--------|-----|------|---------|
| Factorized Prior | 0.001 | 0.8 | 28.5 | 0.85 |
| Factorized Prior | 0.01 | 0.4 | 25.2 | 0.78 |
| Factorized Prior | 0.1 | 0.2 | 22.1 | 0.65 |

## Troubleshooting Evaluation

### Common Issues

**CUDA Out of Memory**
```bash
# Use smaller batch sizes
python evaluate.py --batch_size 1 --num_workers 0
```

**Slow Evaluation**
```bash
# Enable mixed precision
python evaluate.py --mixed_precision True
```

**Missing Dependencies**
```bash
# Install evaluation dependencies
pip install lpips pytorch-lightning tensorboard
```

### Performance Optimization

1. **Use GPU**: Ensure CUDA is available and models are moved to GPU
2. **Batch Processing**: Process multiple images simultaneously when memory allows
3. **Mixed Precision**: Enable FP16 for faster evaluation
4. **Pre-computed Features**: Cache extracted features for repeated evaluations

## Automated Reporting

### Generate Comprehensive Report
```bash
# Generate full evaluation report
python src/evaluation/generate_report.py \
    --checkpoint_dirs checkpoints/ \
    --test_data data/processed/kitti/test.txt \
    --output_dir evaluation_results/final_report \
    --include_visualizations True
```

The report includes:
- Performance summary tables
- Rate-distortion curves
- Detection visualization samples
- Model comparison charts
- Computational efficiency analysis

### Export Results
```bash
# Export results to different formats
python src/evaluation/export_results.py \
    --input_dir evaluation_results/ \
    --formats csv json latex \
    --output_dir exported_results/
```

## Integration with MLOps

### Weights & Biases Integration
```python
import wandb

# Log evaluation results to W&B
wandb.init(project="neural-compression-evaluation")
wandb.log({
    "test_mAP": 0.75,
    "test_PSNR": 28.5,
    "test_BPP": 0.8
})
```

### TensorBoard Logging
```bash
# View evaluation results in TensorBoard
tensorboard --logdir evaluation_results/tensorboard_logs
```

This comprehensive evaluation framework ensures thorough analysis of all model components and enables data-driven improvements to the compression and detection pipeline. 