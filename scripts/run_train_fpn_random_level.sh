#!/bin/bash

# Script to train single model FPN compression with random level sampling

# Default paths
DETECTION_CHECKPOINT="checkpoints/detection/run_0.002000_16/best_model.pth"
CONFIG_FILE="configs/train_factorized_prior_random_sample_fpn.yaml"

echo "Training Single Model FPN Compression with Random Level Sampling..."
echo "Detection checkpoint: $DETECTION_CHECKPOINT"
echo "Config file: $CONFIG_FILE"
echo ""

# Check if detection checkpoint exists
if [ ! -f "$DETECTION_CHECKPOINT" ]; then
    echo "⚠️  Warning: Detection checkpoint not found: $DETECTION_CHECKPOINT"
    echo "Using default pretrained detection model weights"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Train single model with random level sampling
echo "🚀 Starting training with random FPN level sampling..."
python src/train_factorized_prior_random_sample_fpn.py \
    --config "$CONFIG_FILE" \
    --detection_checkpoint "$DETECTION_CHECKPOINT"

echo ""
echo "✅ Training completed!"
echo "Check the results in checkpoints/fpn_random_level_compression/"