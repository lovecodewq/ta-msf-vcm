python -m src.train_joint_autoregress_prior_fused_feature_with_detloss \
  --config configs/train_joint_autoregress_prior_fused_feature_detect_loss.yaml \
  --detection_checkpoint checkpoints/detection/run_0.002000_16/best_model.pth
  # --init_checkpoint checkpoints/fpn_joint_auto_compression_fused_feature/run_20250809_135205_lambda_5.00e+01_lr_1.00e-04_bs_8/best_model.pth