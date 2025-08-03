# Detection Model Training

This guide covers training Faster R-CNN models on the KITTI dataset for object detection.

## Overview

The detection training pipeline uses:
- **Architecture**: Faster R-CNN with ResNet50 FPN backbone
- **Dataset**: KITTI object detection dataset
- **Classes**: Car, Pedestrian, Cyclist (+ Background)
- **Features**: Mixed precision training, advanced scheduling, comprehensive evaluation

## Quick Start

```bash
# Train detection model
python src/train_detection.py --config configs/train_detection.yaml
```

## Configuration

Key configuration options in `configs/train_detection.yaml`:

### Model Settings
```yaml
model:
  name: "faster_rcnn"
  backbone: "resnet50_fpn_v2"
  pretrained: true
  freeze_backbone: true  # Initially freeze backbone
```

### Training Settings
```yaml
training:
  batch_size: 16
  epochs: 20
  learning_rate: 0.002
  optimizer:
    type: "SGD"
    momentum: 0.9
    weight_decay: 0.0001
```

### Data Augmentation
```yaml
data:
  transforms:
    resize:
      size: [800, 1333]  # Standard Faster R-CNN input size
    horizontal_flip:
      probability: 0.5
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
```

## Dataset Preparation

### KITTI Dataset Structure
```
data/processed/kitti/
├── train.txt        # Training image list
├── val.txt          # Validation image list  
├── test.txt         # Test image list
├── images/          # Image files
│   ├── 000001.png
│   └── ...
└── labels/          # Annotation files (KITTI format)
    ├── 000001.txt
    └── ...
```

### Preprocessing
```bash
# Preprocess KITTI dataset
python src/data_preprocess.py --dataset kitti --input_dir raw_kitti --output_dir data/processed/kitti
```

## Training Features

### Mixed Precision Training
- Automatic mixed precision with FP16
- Gradient scaling for numerical stability
- Faster training on modern GPUs

### Advanced Scheduling
- Cosine annealing with warmup
- Learning rate reduction on plateau
- Early stopping with patience

### Gradient Accumulation
- Effective larger batch sizes
- Memory-efficient training
- Configurable accumulation steps

## Monitoring Training

### Training Logs
Each run creates a unique directory:
```
checkpoints/detection/run_20241216_143022_lambda_2.00e-03_lr_2.00e-03_bs_16/
├── train_20241216_143022.log
├── config_used.yaml
├── command_args.txt
├── best_model.pth
└── training_diagnostics_*.txt
```

### Key Metrics
- **Loss Components**: Classification, regression, RPN losses
- **mAP**: Mean Average Precision at IoU 0.5
- **Learning Rate**: Current and scheduled rates
- **Memory Usage**: GPU memory consumption

### Visualization
- Precision-recall curves
- Loss curves over training
- Sample predictions with bounding boxes

## Evaluation

### Validation Metrics
```bash
# Evaluate trained model
python src/evaluation/evaluate_detection.py \
    --model_path checkpoints/detection/best_model.pth \
    --test_data data/processed/kitti/test.txt \
    --output_dir evaluation_results
```

### Performance Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.75**: Mean Average Precision at IoU threshold 0.75  
- **Per-class AP**: Average Precision for each object class
- **Inference Speed**: FPS and latency measurements

## Advanced Usage

### Transfer Learning
```yaml
# Fine-tune from checkpoint
training:
  resume_from: "path/to/pretrained/model.pth"
  freeze_backbone: false  # Unfreeze for fine-tuning
```

### Multi-Scale Training
```yaml
data:
  transforms:
    resize:
      size: [[600, 1000], [800, 1333], [1000, 1666]]  # Multi-scale
```

### Custom Classes
Modify `src/data/kitti_dataset.py` to add custom object classes:
```python
CLASSES = ['Car', 'Pedestrian', 'Cyclist', 'Truck']  # Add custom classes
```

## Troubleshooting

### Common Issues

**GPU Memory Issues**
- Reduce batch size
- Enable gradient accumulation
- Use smaller image sizes

**Training Instability**
- Lower learning rate
- Increase warmup epochs
- Check gradient clipping

**Poor mAP Performance**
- Verify data preprocessing
- Check annotation format
- Increase training epochs
- Adjust anchor scales

### Performance Tips

1. **Optimal Batch Size**: Start with batch_size=16, adjust based on GPU memory
2. **Learning Rate**: Use 0.002 for batch_size=16, scale linearly for other sizes  
3. **Training Time**: Expect 2-4 hours on RTX 5090 for 20 epochs
4. **Convergence**: mAP should reach >0.7 on KITTI validation set

## Results

Expected performance on KITTI validation set:

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.75+ |
| Car AP | 0.85+ |
| Pedestrian AP | 0.70+ |
| Cyclist AP | 0.65+ |

## Next Steps

After training a detection model:
1. **Feature Extraction**: Use trained model for FPN feature extraction
2. **FPN Compression**: Train feature compression models on extracted features
3. **End-to-End Evaluation**: Evaluate detection performance with compressed features

See [FPN_COMPRESSION.md](FPN_COMPRESSION.md) for the next step in the pipeline. 