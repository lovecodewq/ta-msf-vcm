#!/bin/bash

# Script to run FPN level-wise compression test

# Default paths (adjust as needed)
DETECTION_CHECKPOINT="checkpoints/detection/run_0.002000_16/best_model.pth"
TEST_IMAGES="data/processed/kitti/val.txt"
OUTPUT_DIR="./fpn_compression_analysis"
NUM_SAMPLES=10

# Pretrained FactorizedPrior model for initialization
PRETRAINED_FACTORIZED_PRIOR="checkpoints/factorized_prior/model_lambda_0.010.pth"

# Optional: Add FPN compression checkpoint if available
FPN_CHECKPOINT=""  # Set this if you have a trained FPN compression model

echo "Running FPN Level-wise Compression Test..."
echo "Detection checkpoint: $DETECTION_CHECKPOINT"
echo "Test images: $TEST_IMAGES"
echo "Output directory: $OUTPUT_DIR"
echo "Number of samples: $NUM_SAMPLES"
echo "Pretrained FactorizedPrior: $PRETRAINED_FACTORIZED_PRIOR"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the test
if [ -n "$FPN_CHECKPOINT" ] && [ -f "$FPN_CHECKPOINT" ]; then
    echo "Using FPN compression checkpoint: $FPN_CHECKPOINT"
    python test_fpn_level_compression.py \
        --detection_checkpoint "$DETECTION_CHECKPOINT" \
        --fpn_compression_checkpoint "$FPN_CHECKPOINT" \
        --test_images "$TEST_IMAGES" \
        --num_test_samples $NUM_SAMPLES \
        --output_dir "$OUTPUT_DIR" \
        --pretrained_factorized_prior "$PRETRAINED_FACTORIZED_PRIOR"
else
    echo "Running level-wise compression test only (no concatenated comparison)"
    python test_fpn_level_compression.py \
        --detection_checkpoint "$DETECTION_CHECKPOINT" \
        --test_images "$TEST_IMAGES" \
        --num_test_samples $NUM_SAMPLES \
        --output_dir "$OUTPUT_DIR" \
        --pretrained_factorized_prior "$PRETRAINED_FACTORIZED_PRIOR"
fi

echo "Test completed. Results saved to: $OUTPUT_DIR"