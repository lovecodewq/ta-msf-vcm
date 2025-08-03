#!/bin/bash

# Script to train specific-level FPN compression model

# Default paths
DETECTION_CHECKPOINT="checkpoints/detection/run_0.002000_16/best_model.pth"
CONFIG_FILE="configs/train_factorized_prior_specific_level_fpn.yaml"

# Parse command line arguments
FPN_LEVEL=""

# Function to show usage
show_usage() {
    echo "Usage: $0 --level <0|1|2|3|4> [--config CONFIG_FILE] [--detection_checkpoint CHECKPOINT_PATH]"
    echo ""
    echo "Required arguments:"
    echo "  --level <0|1|2|3|4>     Specific FPN level to train on"
    echo ""
    echo "Optional arguments:"
    echo "  --config CONFIG_FILE    Path to config file (default: $CONFIG_FILE)"
    echo "  --detection_checkpoint  Path to detection checkpoint (default: $DETECTION_CHECKPOINT)"
    echo ""
    echo "Examples:"
    echo "  $0 --level 0                           # Train on FPN level 0"
    echo "  $0 --level 2 --config my_config.yaml  # Train on level 2 with custom config"
    echo ""
    echo "This will create separate checkpoints for each level:"
    echo "  checkpoints/fpn_specific_level_compression_level_0/"
    echo "  checkpoints/fpn_specific_level_compression_level_1/"
    echo "  ... etc"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --level)
            FPN_LEVEL="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --detection_checkpoint)
            DETECTION_CHECKPOINT="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$FPN_LEVEL" ]; then
    echo "❌ Error: FPN level is required"
    show_usage
    exit 1
fi

# Validate FPN level
if [[ ! "$FPN_LEVEL" =~ ^[0-4]$ ]]; then
    echo "❌ Error: FPN level must be 0, 1, 2, 3, or 4. Got: $FPN_LEVEL"
    exit 1
fi

echo "Training Specific-Level FPN Compression Model..."
echo "FPN Level: $FPN_LEVEL"
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

# Expected output directory
OUTPUT_DIR="checkpoints/fpn_specific_level_compression_level_$FPN_LEVEL"

# Train specific level model
echo "🚀 Starting training for FPN level $FPN_LEVEL..."
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

python src/train_factorized_prior_specific_level_fpn.py \
    --config "$CONFIG_FILE" \
    --detection_checkpoint "$DETECTION_CHECKPOINT" \
    --fpn_level "$FPN_LEVEL"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Training summary:"
    if [ -f "$OUTPUT_DIR"/*/training_summary.txt ]; then
        echo "📄 Summary file: $(find "$OUTPUT_DIR" -name "training_summary.txt" | head -1)"
    fi
    if [ -f "$OUTPUT_DIR"/*/best_model.pth ]; then
        echo "💾 Best model: $(find "$OUTPUT_DIR" -name "best_model.pth" | head -1)"
    fi
    echo ""
    echo "Next steps:"
    echo "1. Train other levels: $0 --level {0,1,2,3,4}"
    echo "2. Compare results across levels using analysis scripts"
    echo "3. Run level-specific compression tests"
else
    echo "❌ Training failed with exit code $exit_code"
    echo "Check the logs for details"
    exit $exit_code
fi