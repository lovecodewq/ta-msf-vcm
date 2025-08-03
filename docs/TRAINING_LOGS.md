# Training Logs Management

## Overview
This document explains the improved logging system that prevents log overwriting and helps organize training experiments.

## Directory Structure
Each training run now creates a unique directory with the following naming convention:
```
checkpoints/fpn_compression/
├── run_20241216_143022_lambda_2.50e-04_lr_5.00e-05_bs_4/
│   ├── train_20241216_143022.log              # Training logs
│   ├── config_used.yaml                       # Configuration backup
│   ├── command_args.txt                       # Command line arguments
│   ├── best_model.pth                         # Best model (easy access)
│   ├── best_model_epoch_15_loss_0.1234_timestamp.pth  # Detailed checkpoint
│   ├── training_diagnostics_epoch_*.txt       # Per-epoch diagnostics
│   └── fpn_analysis/                          # FPN feature analysis
│       └── lambda_0.000/
│           └── analysis_files...
├── run_20241216_145030_lambda_1.00e-03_lr_5.00e-05_bs_4/
│   └── ...
└── ...
```

## Features

### 1. Unique Run Directories
- **Timestamp**: `YYYYMMDD_HHMMSS` format
- **Key Parameters**: Lambda, learning rate, batch size
- **Example**: `run_20241216_143022_lambda_2.50e-04_lr_5.00e-05_bs_4`

### 2. Comprehensive Logging
- **Training Logs**: `train_YYYYMMDD_HHMMSS.log`
- **Configuration Backup**: Exact config used for training
- **Command Arguments**: Command line arguments and execution time
- **Model Checkpoints**: Both detailed and easy-access versions

### 3. Reproducibility
- All runs are self-contained with their configuration
- Easy to identify and compare different experiments
- Complete audit trail of training parameters

## Usage

### Running Training
```bash
# Training will automatically create a unique run directory
python src/train_factorized_prior_features_compression.py --config configs/train_factorized_prior_fpn.yaml
```

### Listing Training Runs
```bash
# List all training runs
python src/utils/list_training_runs.py

# List runs in specific directory
python src/utils/list_training_runs.py --base-dir checkpoints/detection

# Show detailed information about a specific run
python src/utils/list_training_runs.py --show-details checkpoints/fpn_compression/run_20241216_143022_lambda_2.50e-04_lr_5.00e-05_bs_4
```

### Finding Best Models
```bash
# Each run directory contains:
# - best_model.pth (latest best model for easy access)
# - best_model_epoch_X_loss_Y_timestamp.pth (detailed checkpoint)
```

## Best Practices

### 1. Experiment Organization
- Use meaningful lambda values for rate-distortion experiments
- Document experiment goals in your notes
- Use the `--show-details` command to review past experiments

### 2. Disk Space Management
- Regularly review old training runs
- Archive or delete unsuccessful experiments
- Keep best performing models for comparison

### 3. Hyperparameter Tracking
- Run directory names encode key parameters
- Configuration files are automatically backed up
- Easy to compare settings across experiments

## Migration from Old System
If you have existing logs in the old format (single `train.log` files), they will not be overwritten. The new system creates separate directories for each run, so your old logs remain safe.

## Example Workflow
```bash
# 1. Start training experiment
python src/train_factorized_prior_features_compression.py --config configs/train_factorized_prior_fpn.yaml

# 2. Monitor progress (logs are in the timestamped directory)
tail -f checkpoints/fpn_compression/run_*/train_*.log

# 3. List all experiments when done
python src/utils/list_training_runs.py

# 4. Review specific experiment
python src/utils/list_training_runs.py --show-details checkpoints/fpn_compression/run_20241216_143022_lambda_2.50e-04_lr_5.00e-05_bs_4
```

## Testing

You can test the logging utilities to ensure they work correctly:

```bash
# Run logging utilities test (from project root)
python tests/test_logging_utils.py
```
