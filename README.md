## TA-MSF-VCM

Task-aware multi-scale feature compression for Visual Coding for Machines (VCM). This repo contains training and evaluation code for compressing features from an object detection backbone (FPN) and comparing against image/feature codec anchors (VTM) and a third-party L-MSFC baseline on the KITTI dataset.

### Features
- **Anchors**: VTM image anchor, VTM feature anchor, and L-MSFC feature compressor.
- **Reproducible scripts**: Shell scripts for training, evaluation, RD plotting, and results merging.

## Setup

### Requirements
- Python 3.9+
- PyTorch 2.0+ and TorchVision 0.15+

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Optional: VTM binaries are needed for the VTM anchors (see scripts for expected paths). L-MSFC code is vendored under `thirparty/L-MSFC`.

## Data

This repo uses KITTI detection. Prepare processed images/lists according to your setup and update paths in the YAML configs under `configs/`.

- Expected root (example): `data/processed/kitti`
- Train/val/test lists are referenced in `configs/*.yaml`

## Quickstart

1. Preprocess KITTI into train/val/test splits

```bash
python src/data_preprocess.py
```

2. Train the object detector on KITTI

```bash
python -m src.train_detection --config configs/train_detection.yaml
```

3. Train TA-MSF (feature compression with detection loss)

```bash
bash scripts/run_train_joint_auto_regressive_fused_feature_with_detect_loss.sh
```

4. Train L-MSFC feature anchor

```bash
bash scripts/run_train_lmsfc_anchor.sh
```

5. Train image compressor baseline (factorized prior)

```bash
bash scripts/run_train_factorized_prior.sh
```

6. Run end-to-end evaluation pipeline

```bash
bash scripts/run_full_eval.sh
```

7. Plot and aggregate results

```bash
python -m src.evaluation.visualize_results --help
python -m src.evaluation.merge_rd_results --help
```

Notes:
- The detection checkpoint used for FPN-based training defaults to `checkpoints/detection/run_0.002000_16/best_model.pth` (override via script args/env).
- Edit `configs/*.yaml` to match your data and paths.git 

### VTM anchors

- Image anchor:

```bash
bash scripts/run_vtm_imager_anchor.sh both
```

- Feature anchor:

```bash
bash scripts/run_vtm_feature_anchor.sh
```

Configure encoder/decoder binaries and cfg paths via environment variables at the top of each script if needed.

## Repository structure

- `src/`
  - `model/`: detection model wrapper and compression models
  - `utils/`: training utilities, metrics, dataset helpers
  - `evaluation/`: VTM anchors, evaluation, visualization, RD plotting
  - Training entry points: `train_factorized_prior.py`, `train_joint_autoregress_prior_fused_feature.py`, `train_joint_autoregress_prior_fused_feature_with_detloss.py`, `train_lmsfc_anchor.py`
- `configs/`: YAML configs for training/eval
- `scripts/`: One-click scripts to train/evaluate and aggregate results
- `checkpoints/`: Saved models and logs (created at runtime)
- `evaluation_results/`: Outputs from experiments and plots
- `thirparty/L-MSFC/`: Third-party L-MSFC baseline code

## Tips
- Some scripts set `MKL_THREADING_LAYER=GNU` to avoid threading issues.
- Use the provided scripts as references; you can pass custom flags to the underlying Python modules if needed.

## Acknowledgements
- L-MSFC code is included under `thirparty/L-MSFC`.


