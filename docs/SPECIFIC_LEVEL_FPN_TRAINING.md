# Specific Level FPN Feature Compression Training

This guide covers training compression models for individual FPN levels, allowing you to analyze level-specific compression characteristics and compare performance across different pyramid scales.

## Overview

Unlike the random level sampling approach that trains one model on randomly selected FPN levels, this specific level training approach:

- **Trains separate models for each FPN level** (0, 1, 2, 3, 4)
- **Specializes models for level-specific characteristics** (resolution, semantic content)
- **Enables detailed level-by-level analysis** of compression performance
- **Creates distinguishable checkpoints** for each level

## Quick Start

### Train a Single Level

```bash
# Train on FPN level 0 (highest resolution)
./run_train_factorized_prior_specific_level_fpn.sh --level 0

# Train on FPN level 2 (middle resolution)  
./run_train_factorized_prior_specific_level_fpn.sh --level 2

# Train on FPN level 4 (lowest resolution)
./run_train_factorized_prior_specific_level_fpn.sh --level 4
```

### Train All Levels

```bash
# Train models for all FPN levels
for level in 0 1 2 3 4; do
    echo "Training FPN level $level..."
    ./run_train_factorized_prior_specific_level_fpn.sh --level $level
done
```

### Custom Configuration

```bash
# Use custom config file
./run_train_factorized_prior_specific_level_fpn.sh \
    --level 1 \
    --config configs/my_custom_config.yaml

# Use different detection checkpoint
./run_train_factorized_prior_specific_level_fpn.sh \
    --level 2 \
    --detection_checkpoint checkpoints/my_detection_model.pth
```

## Output Structure

Each level creates its own checkpoint directory:

```
checkpoints/
├── fpn_specific_level_compression_level_0/
│   ├── run_20250101_120000_lambda_1.00e-03_lr_5.00e-05_bs_8/
│   │   ├── best_model.pth
│   │   ├── training_summary.txt
│   │   ├── training_diagnostics_epoch_*.txt
│   │   └── level_analysis/
│   │       └── level_0/
│   │           └── lambda_0.001/
├── fpn_specific_level_compression_level_1/
│   └── ...
├── fpn_specific_level_compression_level_2/
│   └── ...
└── ...
```

## Configuration

### Model Settings

The model architecture is consistent across levels but specializes for each:

```yaml
model:
  name: "factorized_prior_specific_level"
  n_hidden: 128      # Hidden channels in transforms
  n_channels: 192    # Latent representation channels
  fpn_channels_per_level: 256  # All FPN levels have 256 channels
```

### Training Configuration

Key settings optimized for specific level training:

```yaml
training:
  batch_size: 8      # Smaller batch due to FPN overhead
  epochs: 100        # Can train longer per level
  learning_rate: 5e-5
  lambda: 0.001      # Rate-distortion trade-off
  
  # Learning rate schedule
  lr_schedule:
    patience: 8      # More patient for single level
    
  # Early stopping  
  early_stopping:
    patience: 20     # More patience for convergence
```

## FPN Level Characteristics

Understanding what each level represents:

| Level | Resolution | Semantic Content | Typical Use |
|-------|------------|------------------|-------------|
| 0 | Highest (1/4 input) | Fine details, edges | Small object detection |
| 1 | High (1/8 input) | Local features | Medium object detection |
| 2 | Medium (1/16 input) | Balanced | General detection |
| 3 | Low (1/32 input) | Semantic features | Large object detection |
| 4 | Lowest (1/64 input) | Global context | Very large objects |

## Usage Examples

### Basic Training

```bash
# Train level 0 with default settings
./run_train_factorized_prior_specific_level_fpn.sh --level 0
```

### Parallel Training

```bash
# Train multiple levels in parallel (if you have multiple GPUs)
./run_train_factorized_prior_specific_level_fpn.sh --level 0 &
./run_train_factorized_prior_specific_level_fpn.sh --level 1 &
./run_train_factorized_prior_specific_level_fpn.sh --level 2 &
wait
```

### Different Lambda Values

```bash
# Create configs with different lambda values for level comparison
for level in 0 1 2 3 4; do
    # Modify config to use lambda=0.0005 and train
    sed 's/lambda: 0.001/lambda: 0.0005/' configs/train_factorized_prior_specific_level_fpn.yaml > temp_config.yaml
    ./run_train_factorized_prior_specific_level_fpn.sh --level $level --config temp_config.yaml
done
```

## Command Line Options

The training script supports these options:

### Required
- `--level <0|1|2|3|4>`: Specify which FPN level to train on

### Optional
- `--config CONFIG_FILE`: Path to configuration file
- `--detection_checkpoint CHECKPOINT`: Path to detection model weights

### Examples
```bash
# Minimal usage
./run_train_factorized_prior_specific_level_fpn.sh --level 2

# Full specification
./run_train_factorized_prior_specific_level_fpn.sh \
    --level 1 \
    --config configs/train_factorized_prior_specific_level_fpn.yaml \
    --detection_checkpoint checkpoints/detection/best_model.pth
```

## Analysis and Comparison

After training multiple levels, you can:

### 1. Compare Training Summaries
```bash
# View summaries for all levels
for level in 0 1 2 3 4; do
    echo "=== Level $level ==="
    find checkpoints/fpn_specific_level_compression_level_$level -name "training_summary.txt" -exec cat {} \;
    echo
done
```

### 2. Extract Key Metrics
```bash
# Extract best validation loss for each level
for level in 0 1 2 3 4; do
    summary_file=$(find checkpoints/fpn_specific_level_compression_level_$level -name "training_summary.txt")
    if [ -f "$summary_file" ]; then
        best_loss=$(grep "Best Validation Loss:" "$summary_file" | cut -d' ' -f4)
        echo "Level $level: Best Loss = $best_loss"
    fi
done
```

### 3. Level-Specific Analysis
Each level saves detailed analysis files showing:
- Compression performance (BPP, MSE)
- Feature statistics
- Reconstruction quality metrics
- Level-specific characteristics

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size in config
2. **Slow Training**: Ensure detection model checkpoint exists
3. **Convergence Issues**: Adjust learning rate or lambda value
4. **Disk Space**: Monitor checkpoint directory sizes

### Debugging

Enable more verbose logging:
```bash
# Add debug logging to Python script
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python -u src/train_factorized_prior_specific_level_fpn.py --level 0 --config configs/train_factorized_prior_specific_level_fpn.yaml
```

## Next Steps

After training specific level models:

1. **Performance Analysis**: Compare BPP and MSE across levels
2. **Compression Testing**: Test compression/decompression on validation set
3. **Detection Impact**: Evaluate how compression affects detection performance
4. **Optimization**: Fine-tune lambda values for each level
5. **Ensemble Methods**: Combine models for optimal multi-level compression

## Comparison with Random Sampling

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Specific Level** | Level specialization, detailed analysis, optimal per-level performance | More models to train, higher storage requirements | Research, level-specific optimization |
| **Random Sampling** | Single model, efficient training, general robustness | Less specialized, averaged performance | Production, general use |

Choose specific level training when you need to:
- Understand level-specific compression characteristics
- Optimize performance for particular FPN levels
- Conduct detailed research on multi-scale compression
- Compare compression efficiency across pyramid scales