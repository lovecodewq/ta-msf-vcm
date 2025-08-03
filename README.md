# Neural Image Compression with Detection Models

This project implements learned image compression models with support for both standard image compression and feature-level compression for object detection pipelines. The project includes multiple architectures and training modes.

## 🎯 Project Overview

This repository contains:
- **Standard Image Compression**: Factorized Prior model for end-to-end image compression
- **Detection Model Training**: Faster R-CNN training on KITTI dataset  
- **FPN Feature Compression**: Compress intermediate FPN features for detection pipelines
- **Comprehensive Evaluation**: Rate-distortion analysis and performance metrics
- **Training Management**: Advanced logging and experiment tracking utilities

## 📁 Project Structure

```
.
├── src/
│   ├── model/
│   │   ├── factorized_prior.py                    # Standard image compression
│   │   ├── factorized_prior_features_compression.py  # FPN feature compression
│   │   ├── detection.py                           # Detection model wrapper
│   │   └── gdn.py                                 # GDN layers
│   ├── data/
│   │   ├── kitti_dataset.py                       # KITTI detection dataset
│   │   ├── transforms.py                          # Data augmentation
│   │   └── __init__.py                            # Dataset implementations
│   ├── utils/
│   │   ├── logging_utils.py                       # Training logging utilities
│   │   ├── list_training_runs.py                  # Experiment management
│   │   ├── metrics.py                             # Evaluation metrics
│   │   └── paths.py                               # Path utilities
│   ├── evaluation/                                # Evaluation scripts
│   ├── train_detection.py                         # Detection model training
│   ├── train_factorized_prior.py                  # Standard compression training
│   ├── train_factorized_prior_features_compression.py  # FPN compression training
│   └── data_preprocess.py                         # Data preprocessing
├── configs/                                       # Training configurations
├── docs/                                          # Detailed documentation
├── tests/                                         # Unit tests
├── checkpoints/                                   # Model checkpoints
└── data/                                          # Dataset storage
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone [repository-url]
cd image_compress
pip install -r requirements.txt
```

### 2. Choose Your Training Mode

#### **Detection Model Training**
```bash
python src/train_detection.py --config configs/train_detection.yaml
```

#### **Standard Image Compression**
```bash
python src/train_factorized_prior.py --config configs/train_factorized_prior.yaml
```

#### **FPN Feature Compression**
```bash
python src/train_factorized_prior_features_compression.py --config configs/train_factorized_prior_fpn.yaml
```

### 3. Monitor Training

```bash
# List all training runs
python src/utils/list_training_runs.py

# View specific run details
python src/utils/list_training_runs.py --show-details path/to/run_directory
```

## 📚 Documentation

| Topic | Description | Link |
|-------|-------------|------|
| **Training Management** | Logging, experiment tracking, avoiding overwrites | [docs/TRAINING_LOGS.md](docs/TRAINING_LOGS.md) |
| **Detection Training** | KITTI dataset, Faster R-CNN setup | [docs/DETECTION_TRAINING.md](docs/DETECTION_TRAINING.md) |
| **FPN Compression** | Feature-level compression for detection | [docs/FPN_COMPRESSION.md](docs/FPN_COMPRESSION.md) |
| **Evaluation** | Rate-distortion analysis, metrics | [docs/EVALUATION.md](docs/EVALUATION.md) |

## 🏗️ Model Architectures

### 1. **Factorized Prior Compression**
End-to-end image compression optimizing rate-distortion trade-off
- Analysis/synthesis transforms
- Entropy bottleneck for quantization
- Lambda-based rate control

### 2. **Detection Models** 
Faster R-CNN with ResNet50 FPN backbone
- KITTI dataset support
- Mixed precision training
- Comprehensive evaluation metrics

### 3. **FPN Feature Compression**
Compress intermediate FPN features while preserving detection performance
- Feature-level rate-distortion optimization
- Detection pipeline integration
- Specialized entropy modeling

## ⚙️ Key Features

- **🔄 No Log Overwriting**: Unique timestamped directories for each training run
- **📊 Comprehensive Metrics**: PSNR, MS-SSIM, mAP, rate-distortion curves
- **🎛️ Flexible Configuration**: YAML-based configuration system
- **🚀 Mixed Precision**: Optimized training with automatic mixed precision
- **📈 Advanced Scheduling**: Learning rate scheduling and early stopping
- **🧪 Testing Suite**: Comprehensive unit tests and utilities

## 📊 Evaluation

```bash
# Generate rate-distortion curves
python src/evaluation/evaluate_compression.py --checkpoint_dir checkpoints/

# Evaluate detection performance
python src/evaluation/evaluate_detection.py --model_path checkpoints/detection/best_model.pth
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Test specific component
python tests/test_logging_utils.py
```

## 🎯 Experiment Management

This project includes advanced experiment tracking:
- Automatic timestamped run directories
- Configuration backups for reproducibility  
- Command argument logging
- Easy experiment comparison and analysis

See [docs/TRAINING_LOGS.md](docs/TRAINING_LOGS.md) for detailed information.

## 📋 Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- PyYAML >= 6.0
- pytorch-msssim >= 0.2.1
- matplotlib >= 3.7.0
- numpy >= 1.24.0
- tqdm >= 4.65.0
- Pillow >= 9.5.0

## 🏆 Results

*Benchmark results and comparisons will be added as experiments complete.*

## 📖 References

- Ballé, J., et al. (2017). End-to-end optimized image compression. ICLR 2017.
- Ballé, J., et al. (2018). Variational image compression with a scale hyperprior. ICLR 2018.
- Ren, S., et al. (2015). Faster R-CNN: Towards real-time object detection. NIPS 2015.

## 📝 License

[Specify your license here]
