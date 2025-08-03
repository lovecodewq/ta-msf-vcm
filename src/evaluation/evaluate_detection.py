"""
Evaluation script for measuring the impact of neural image compression on object detection.
Uses Faster R-CNN fine-tuned on KITTI dataset to evaluate detection performance
on original vs compressed images.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import logging
import yaml
import cv2
from PIL import Image
import argparse
from model.factorized_prior import FactorizedPrior
from model.detection import DetectionModel
from data.kitti_dataset import KITTIDetectionDataset
from data.transforms import create_detection_transforms

def setup_logging(save_dir):
    """Setup logging configuration."""
    log_file = save_dir / 'detection_evaluation.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_model(num_classes):
    """Create Faster R-CNN model with pretrained backbone."""
    return DetectionModel(num_classes=num_classes, pretrained=True)

def evaluate_detections(model, data_loader, device):
    """Evaluate detection performance."""
    model.eval()
    results = []
    targets_list = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = [img.to(device) for img in images]
            
            # Get predictions
            predictions = model(images)
            
            # Convert predictions to numpy for metric calculation
            for pred in predictions:
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                
                results.append({
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels
                })
            
            # Store targets for metric calculation
            targets_list.extend(targets)
    
    return results, targets_list

def calculate_ap(predictions, targets, iou_threshold):
    """Calculate Average Precision for a single class following PASCAL VOC metrics."""
    # If no ground truth objects, return 0
    if not any(len(t['boxes']) > 0 for t in targets):
        return 0.0
        
    # Concatenate all predictions and sort by confidence
    all_pred_boxes = []
    all_pred_scores = []
    all_target_boxes = []
    n_positives = 0  # Total number of ground truth objects
    
    for pred, target in zip(predictions, targets):
        all_pred_boxes.append(pred['boxes'])
        all_pred_scores.append(pred['scores'])
        all_target_boxes.append(target['boxes'])
        n_positives += len(target['boxes'])
    
    if n_positives == 0:
        logging.debug('No positive samples found for this class')
        return 0.0
        
    # Concatenate all predictions
    if len(all_pred_boxes) > 0:
        pred_boxes = np.concatenate(all_pred_boxes, axis=0)
        pred_scores = np.concatenate(all_pred_scores, axis=0)
        logging.debug(f'Total predictions: {len(pred_boxes)}, Total ground truth: {n_positives}')
    else:
        logging.debug('No predictions found for this class')
        return 0.0
    
    # Sort predictions by confidence
    sort_idx = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sort_idx]
    pred_scores = pred_scores[sort_idx]
    
    # Initialize precision/recall
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    # For each prediction
    for i, pred_box in enumerate(pred_boxes):
        max_iou = 0.0
        max_idx = -1
        max_gt_idx = -1
        
        # Find the best matching ground truth box
        for j, target_boxes in enumerate(all_target_boxes):
            if len(target_boxes) == 0:
                continue
                
            # Calculate IoU with all ground truth boxes in this image
            ious = calculate_iou(pred_box, target_boxes)
            max_iou_idx = np.argmax(ious)
            if ious[max_iou_idx] > max_iou:
                max_iou = ious[max_iou_idx]
                max_idx = j
                max_gt_idx = max_iou_idx
        
        # Log matching information for debugging
        if max_iou >= iou_threshold:
            logging.debug(f'Prediction {i} matched with IoU {max_iou:.3f} '
                        f'(image {max_idx}, gt_box {max_gt_idx})')
            logging.debug(f'GT boxes in image {max_idx}: {len(all_target_boxes[max_idx])}')
            logging.debug(f'Pred box: {pred_box}')
            if max_idx >= 0:
                logging.debug(f'Matched GT box: {all_target_boxes[max_idx][max_gt_idx]}')
        
        # If IoU exceeds threshold and ground truth box hasn't been matched
        if max_iou >= iou_threshold:
            tp[i] = 1
            # Remove the matched ground truth box to prevent multiple matches
            if max_idx >= 0:
                try:
                    all_target_boxes[max_idx] = np.delete(all_target_boxes[max_idx], max_gt_idx, axis=0)
                except IndexError as e:
                    logging.error(f'IndexError during box deletion:')
                    logging.error(f'max_idx: {max_idx}, max_gt_idx: {max_gt_idx}')
                    logging.error(f'all_target_boxes[max_idx] shape: {all_target_boxes[max_idx].shape}')
                    logging.error(f'Original error: {str(e)}')
                    raise
        else:
            fp[i] = 1
            if max_iou > 0:
                logging.debug(f'False positive with IoU {max_iou:.3f} (below threshold {iou_threshold})')
    
    # Compute cumulative precision and recall
    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)
    recalls = cumsum_tp / n_positives
    precisions = cumsum_tp / (cumsum_tp + cumsum_fp)
    
    # Log final statistics
    logging.debug(f'Final statistics:')
    logging.debug(f'True positives: {int(np.sum(tp))}')
    logging.debug(f'False positives: {int(np.sum(fp))}')
    logging.debug(f'Total ground truth: {n_positives}')
    
    # Compute average precision using 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    logging.debug(f'Average Precision: {ap:.4f}')
    return ap

def calculate_map(predictions, targets, iou_threshold=0.5):
    """Calculate mean Average Precision."""
    aps = []  # Average precision for each class
    
    for class_id in range(1, len(KITTIDetectionDataset.CLASSES) + 1):
        logging.info(f'Calculating AP for class {KITTIDetectionDataset.CLASSES[class_id-1]}')
        # Get predictions for this class
        class_preds = []
        class_targets = []
        
        for pred, target in zip(predictions, targets):
            # Get predictions for this class
            mask = pred['labels'] == class_id
            class_preds.append({
                'boxes': pred['boxes'][mask],
                'scores': pred['scores'][mask]
            })
            
            # Get ground truth for this class
            target_mask = target['labels'] == class_id
            class_targets.append({
                'boxes': target['boxes'][target_mask]
            })
        
        # Calculate AP for this class
        ap = calculate_ap(class_preds, class_targets, iou_threshold)
        aps.append(ap)
        logging.info(f'AP for {KITTIDetectionDataset.CLASSES[class_id-1]}: {ap:.4f}')
    
    map_score = np.mean(aps)
    logging.info(f'Mean Average Precision (mAP): {map_score:.4f}')
    return map_score

def calculate_iou(box, boxes):
    """Calculate IoU between a box and an array of boxes."""
    # Calculate intersection coordinates
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    # Calculate intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    
    return intersection / union

def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a normalized image tensor back to [0,1] range."""
    mean = torch.tensor(mean, device=image.device).view(3, 1, 1)
    std = torch.tensor(std, device=image.device).view(3, 1, 1)
    return image * std + mean

def normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalize an image tensor with ImageNet mean and std."""
    mean = torch.tensor(mean, device=image.device).view(3, 1, 1)
    std = torch.tensor(std, device=image.device).view(3, 1, 1)
    return (image - mean) / std

def save_comparison_images(original_images, compressed_images, original_preds, compressed_preds, 
                         save_dir, batch_idx, confidence_threshold=0.5):
    """Save comparison images with detection boxes.
    
    Args:
        original_images: List of original images (already denormalized)
        compressed_images: List of compressed images (already denormalized)
        original_preds: Original predictions
        compressed_preds: Compressed predictions
        save_dir: Directory to save visualizations
        batch_idx: Batch index
        confidence_threshold: Confidence threshold for showing detections
    """
    # Create directory for visualizations
    vis_dir = save_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Define colors for different classes
    colors = {
        1: 'red',     # Car
        2: 'blue',    # Pedestrian
    }
    
    for i, (orig_img, comp_img, orig_pred, comp_pred) in enumerate(
            zip(original_images, compressed_images, original_preds, compressed_preds)):
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Convert to numpy and clip to [0,1] range
        orig_img_np = torch.clamp(orig_img, 0, 1).cpu().permute(1, 2, 0).numpy()
        comp_img_np = torch.clamp(comp_img, 0, 1).cpu().permute(1, 2, 0).numpy()
        
        # Plot original image with detections
        ax1.imshow(orig_img_np)
        ax1.set_title('Original Image')
        
        # Draw boxes for original predictions
        orig_boxes = orig_pred['boxes'].cpu().numpy()
        orig_scores = orig_pred['scores'].cpu().numpy()
        orig_labels = orig_pred['labels'].cpu().numpy()
        
        for box, score, label in zip(orig_boxes, orig_scores, orig_labels):
            if score > confidence_threshold:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2,
                    edgecolor=colors[label],
                    facecolor='none'
                )
                ax1.add_patch(rect)
                ax1.text(
                    x1, y1-5,
                    f'{KITTIDetectionDataset.CLASSES[label-1]} {score:.2f}',
                    color=colors[label],
                    bbox=dict(facecolor='white', alpha=0.8)
                )
        
        # Plot compressed image with detections
        ax2.imshow(comp_img_np)
        ax2.set_title('Compressed Image')
        
        # Draw boxes for compressed predictions
        comp_boxes = comp_pred['boxes'].cpu().numpy()
        comp_scores = comp_pred['scores'].cpu().numpy()
        comp_labels = comp_pred['labels'].cpu().numpy()
        
        for box, score, label in zip(comp_boxes, comp_scores, comp_labels):
            if score > confidence_threshold:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2,
                    edgecolor=colors[label],
                    facecolor='none'
                )
                ax2.add_patch(rect)
                ax2.text(
                    x1, y1-5,
                    f'{KITTIDetectionDataset.CLASSES[label-1]} {score:.2f}',
                    color=colors[label],
                    bbox=dict(facecolor='white', alpha=0.8)
                )
        
        # Remove axes
        ax1.axis('off')
        ax2.axis('off')
        
        # Add title with compression info
        plt.suptitle(f'Detection Results - Original vs Compressed (Sample {batch_idx}_{i})', y=0.95)
        
        # Save figure
        plt.savefig(vis_dir / f'comparison_{batch_idx:04d}_{i:02d}.png', 
                   bbox_inches='tight', dpi=150)
        plt.close(fig)

def get_lambda_from_checkpoint(checkpoint_path):
    """Extract lambda value from compression model checkpoint path."""
    path = Path(checkpoint_path)
    # Expected format: model_lambda_0.010.pth
    try:
        lambda_str = path.stem.split('_')[-1]  # Get the last part after splitting by '_'
        return float(lambda_str)
    except (IndexError, ValueError):
        return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection on compressed images')
    parser.add_argument('--data_dir', type=str, default='data/processed/kitti',
                      help='Path to processed KITTI dataset (containing train/val/test splits)')
    parser.add_argument('--detection_model', type=str, required=True,
                      help='Path to trained detection model')
    parser.add_argument('--compression_checkpoint', type=str, required=True,
                      help='Path to compression model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/detection_results',
                      help='Base output directory')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size')
    parser.add_argument('--num_samples', type=int, default=20,
                      help='Number of sample images to save for visualization')
    args = parser.parse_args()
    
    # Extract lambda value from compression checkpoint path
    lambda_val = get_lambda_from_checkpoint(args.compression_checkpoint)
    if lambda_val is None:
        logging.warning('Could not extract lambda value from checkpoint path. Using default directory.')
        results_dir = 'unknown_lambda'
    else:
        results_dir = f'lambda_{lambda_val:.3f}'
    
    # Create output directory with lambda value
    save_dir = Path(args.output_dir) / results_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir)
    
    # Log the configuration
    logging.info(f'Evaluation Configuration:')
    logging.info(f'  Detection Model: {args.detection_model}')
    logging.info(f'  Compression Model: {args.compression_checkpoint} (λ={lambda_val if lambda_val is not None else "unknown"})')
    logging.info(f'  Results Directory: {save_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load detection model checkpoint to get the config
    detection_checkpoint = torch.load(args.detection_model, map_location=device)
    config = detection_checkpoint['config']
    
    # Create test transforms using the same transforms as validation
    # Best practice: Use validation transforms for testing (no augmentations)
    test_transform_config = {
        'normalize': config['data']['val_transforms']['normalize'],
        'resize': config['data']['val_transforms']['resize']
    }
    logging.info(f'Using test transforms: {test_transform_config}')
    test_transform = create_detection_transforms(test_transform_config)
    
    # Create test dataset with detection transforms
    test_dataset = KITTIDetectionDataset(
        args.data_dir,
        split='test',
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Load detection model
    model = create_model(num_classes=len(KITTIDetectionDataset.CLASSES) + 1)
    model.load_state_dict(detection_checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load compression model
    compression_checkpoint = torch.load(args.compression_checkpoint, map_location=device)
    compression_model = FactorizedPrior(
        n_hidden=compression_checkpoint['config']['model']['n_hidden'],
        n_channels=compression_checkpoint['config']['model']['n_channels']
    ).to(device)
    compression_model.load_state_dict(compression_checkpoint['model_state_dict'])
    compression_model.eval()
    
    # Evaluate on original images
    logging.info('Evaluating on original images...')
    original_results, original_targets = evaluate_detections(model, test_loader, device)
    
    # Evaluate on compressed images
    logging.info('Evaluating on compressed images...')
    compressed_results = []
    compressed_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc='Evaluating compressed')):
            # Compress and decompress images
            images = [img.to(device) for img in images]
            compressed_images = []
            for img in images:
                # Denormalize before compression
                img_denorm = denormalize_image(img, 
                    mean=config['data']['val_transforms']['normalize']['mean'],
                    std=config['data']['val_transforms']['normalize']['std']
                )
                
                # Compress and decompress
                compressed = compression_model.compress(img_denorm.unsqueeze(0))
                decompressed = compression_model.decompress(
                    compressed['y_strings'],
                    compressed['shape'],
                    input_size=compressed['input_size']
                )
                
                # Normalize again for detection
                img_comp_norm = normalize_image(
                    decompressed['x_hat'].squeeze(0),
                    mean=config['data']['val_transforms']['normalize']['mean'],
                    std=config['data']['val_transforms']['normalize']['std']
                )
                compressed_images.append(img_comp_norm)
            
            # Get predictions on compressed images
            compressed_preds = model(compressed_images)
            
            # Save comparison visualizations for the first num_samples batches
            if batch_idx < args.num_samples:
                # For visualization, use denormalized images
                orig_images_denorm = [denormalize_image(img, 
                    mean=config['data']['val_transforms']['normalize']['mean'],
                    std=config['data']['val_transforms']['normalize']['std']
                ) for img in images]
                
                comp_images_denorm = [denormalize_image(img,
                    mean=config['data']['val_transforms']['normalize']['mean'],
                    std=config['data']['val_transforms']['normalize']['std']
                ) for img in compressed_images]
                
                save_comparison_images(
                    orig_images_denorm, comp_images_denorm,
                    model(images), compressed_preds,
                    save_dir, batch_idx,
                    confidence_threshold=config['validation']['metrics']['confidence_threshold']
                )
            
            # Convert predictions to numpy
            for pred in compressed_preds:
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                
                compressed_results.append({
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels
                })
            
            compressed_targets.extend(targets)
    
    # Calculate metrics
    original_map = calculate_map(original_results, original_targets)
    compressed_map = calculate_map(compressed_results, compressed_targets)
    
    # Calculate per-class AP
    class_aps_original = []
    class_aps_compressed = []
    
    for class_id in range(1, len(KITTIDetectionDataset.CLASSES) + 1):
        # Get predictions for this class
        class_preds_orig = []
        class_preds_comp = []
        class_targets = []
        
        for pred_orig, pred_comp, target in zip(original_results, compressed_results, original_targets):
            # Original predictions
            mask_orig = pred_orig['labels'] == class_id
            class_preds_orig.append({
                'boxes': pred_orig['boxes'][mask_orig],
                'scores': pred_orig['scores'][mask_orig]
            })
            
            # Compressed predictions
            mask_comp = pred_comp['labels'] == class_id
            class_preds_comp.append({
                'boxes': pred_comp['boxes'][mask_comp],
                'scores': pred_comp['scores'][mask_comp]
            })
            
            # Ground truth
            target_mask = target['labels'] == class_id
            class_targets.append({
                'boxes': target['boxes'][target_mask]
            })
        
        # Calculate AP for this class
        ap_orig = calculate_ap(class_preds_orig, class_targets, iou_threshold=0.5)
        ap_comp = calculate_ap(class_preds_comp, class_targets, iou_threshold=0.5)
        
        class_aps_original.append(ap_orig)
        class_aps_compressed.append(ap_comp)
    
    # Log results
    logging.info(f'Overall Results:')
    logging.info(f'Original mAP: {original_map:.4f}')
    logging.info(f'Compressed mAP: {compressed_map:.4f}')
    logging.info(f'\nPer-class Results:')
    
    for i, class_name in enumerate(KITTIDetectionDataset.CLASSES):
        logging.info(f'{class_name}:')
        logging.info(f'  Original AP: {class_aps_original[i]:.4f}')
        logging.info(f'  Compressed AP: {class_aps_compressed[i]:.4f}')
        logging.info(f'  Relative Change: {((class_aps_compressed[i] / class_aps_original[i]) - 1) * 100:.1f}%')
    
    # Save results
    results = {
        'overall': {
            'original_map': float(original_map),
            'compressed_map': float(compressed_map),
            'relative_change': float((compressed_map / original_map - 1) * 100)
        },
        'per_class': {
            class_name: {
                'original_ap': float(ap_orig),
                'compressed_ap': float(ap_comp),
                'relative_change': float((ap_comp / ap_orig - 1) * 100)
            }
            for class_name, ap_orig, ap_comp in zip(
                KITTIDetectionDataset.CLASSES,
                class_aps_original,
                class_aps_compressed
            )
        },
        'config': {
            'detection_model': args.detection_model,
            'compression_model': args.compression_checkpoint,
            'kitti_root': str(args.data_dir) # Changed from args.kitti_root to args.data_dir
        }
    }
    
    with open(save_dir / 'detection_results.yaml', 'w') as f:
        yaml.dump(results, f)
    
    logging.info(f'Results saved to {save_dir}')

if __name__ == '__main__':
    main() 