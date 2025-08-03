#!/bin/bash

# Script to train compression models for all FPN levels sequentially
# This creates specialized models for each level (0-4)

echo "🚀 Training FPN Compression Models for All Levels"
echo "=================================================="
echo ""

# Configuration
CONFIG_FILE="configs/train_factorized_prior_specific_level_fpn.yaml"
DETECTION_CHECKPOINT="checkpoints/detection/run_0.002000_16/best_model.pth"

# Check prerequisites
echo "📋 Checking prerequisites..."

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "Please ensure the config file exists"
    exit 1
fi

if [ ! -f "$DETECTION_CHECKPOINT" ]; then
    echo "⚠️  Detection checkpoint not found: $DETECTION_CHECKPOINT"
    echo "Will use default pretrained weights"
fi

echo "✅ Prerequisites checked"
echo ""

# Training summary
echo "📊 Training Plan:"
echo "  Config: $CONFIG_FILE"
echo "  Detection checkpoint: $DETECTION_CHECKPOINT"
echo "  Levels to train: 0, 1, 2, 3, 4"
echo "  Output directories:"
for level in 0 1 2 3 4; do
    echo "    Level $level: checkpoints/fpn_specific_level_compression_level_$level/"
done
echo ""

# Confirm before starting
read -p "Continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 0
fi

echo ""

# Train each level
SUCCESS_LEVELS=()
FAILED_LEVELS=()
START_TIME=$(date +%s)

for level in 0 1 2 3 4; do
    echo "🎯 Training FPN Level $level"
    echo "=============================="
    
    level_start_time=$(date +%s)
    
    # Run training for this level
    ./run_train_factorized_prior_specific_level_fpn.sh \
        --level "$level" \
        --config "$CONFIG_FILE" \
        --detection_checkpoint "$DETECTION_CHECKPOINT"
    
    exit_code=$?
    level_end_time=$(date +%s)
    level_duration=$((level_end_time - level_start_time))
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "✅ Level $level completed successfully (${level_duration}s)"
        SUCCESS_LEVELS+=($level)
    else
        echo "❌ Level $level failed (${level_duration}s)"
        FAILED_LEVELS+=($level)
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "🏁 Training Complete!"
echo "===================="
echo "Total time: ${TOTAL_DURATION}s"
echo ""

if [ ${#SUCCESS_LEVELS[@]} -gt 0 ]; then
    echo "✅ Successfully trained levels: ${SUCCESS_LEVELS[*]}"
fi

if [ ${#FAILED_LEVELS[@]} -gt 0 ]; then
    echo "❌ Failed levels: ${FAILED_LEVELS[*]}"
fi

echo ""
echo "📂 Results:"
for level in "${SUCCESS_LEVELS[@]}"; do
    checkpoint_dir="checkpoints/fpn_specific_level_compression_level_$level"
    if [ -d "$checkpoint_dir" ]; then
        echo "  Level $level: $checkpoint_dir"
        
        # Find training summary
        summary_file=$(find "$checkpoint_dir" -name "training_summary.txt" 2>/dev/null | head -1)
        if [ -f "$summary_file" ]; then
            best_loss=$(grep "Best Validation Loss:" "$summary_file" 2>/dev/null | cut -d' ' -f4)
            echo "    Best validation loss: $best_loss"
        fi
        
        # Find best model
        model_file=$(find "$checkpoint_dir" -name "best_model.pth" 2>/dev/null | head -1)
        if [ -f "$model_file" ]; then
            model_size=$(du -h "$model_file" | cut -f1)
            echo "    Model size: $model_size"
        fi
    fi
done

echo ""
echo "📊 Next Steps:"
echo "1. Compare performance across levels:"
echo "   grep 'Best Validation Loss:' checkpoints/fpn_specific_level_compression_level_*/*/training_summary.txt"
echo ""
echo "2. Run compression tests:"
echo "   python test_fpn_level_compression.py --specific_level_checkpoints checkpoints/fpn_specific_level_compression_level_*"
echo ""
echo "3. Analyze level-specific characteristics:"
echo "   ls checkpoints/fpn_specific_level_compression_level_*/*/level_analysis/"

# Exit with error if any level failed
if [ ${#FAILED_LEVELS[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi