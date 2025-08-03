"""
Data Preprocessing Script for KITTI to YOLO Conversion

This script processes the KITTI dataset for object detection tasks by:
- Converting KITTI label format to YOLO format (normalized bounding boxes and class indices).
- Filtering for specific classes (e.g., 'Car', 'Pedestrian').
- Splitting the dataset into training, validation, and test sets according to a specified ratio.
- Organizing the output into a directory structure compatible with YOLO training pipelines.

Usage:
    python src/data_preprocess.py

Before running, ensure the KITTI dataset is available at the following locations (relative to the project root):
    - Images: data/kitti/training/image_02
    - Labels: data/kitti/training/label_02

The processed data will be saved to:
    - data/processed/kitti

Dependencies:
    - OpenCV (cv2)
    - tqdm
    - numpy (if used elsewhere)

"""
import os
import glob
import random
import shutil
import cv2
from tqdm import tqdm
from utils.paths import get_project_path

def convert_kitti_to_yolo(data,img_height,img_width, class_to_idx):
    """Convert KITTI format to YOLO format."""
    data = data.strip().split(' ')
    class_idx = class_to_idx[data[2]]
    bbox = list(map(float, data[6:10]))  # bbox coordinates are at indices 6-9
        
    # Convert to YOLO format (x_center, y_center, width, height)
    x_center = (bbox[0] + bbox[2]) / 2 / img_width
    y_center = (bbox[1] + bbox[3]) / 2 / img_height
    w = (bbox[2] - bbox[0]) / img_width
    h = (bbox[3] - bbox[1]) / img_height
        
    yolo_label = f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
    return yolo_label

def split_dataset(image_dir, label_dir, output_dir, split_ratio=(0.7, 0.15, 0.15)):
    """Split dataset into train, validation and test sets."""
    classes = ['Car', 'Pedestrian']    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Load labels
    image_file_to_lables = {}
    for label_file_name in os.listdir(label_dir):
        sequence_id = label_file_name.split('.')[0]
        print(f"Loading sequence {sequence_id}")
        label_file_path = os.path.join(label_dir, label_file_name)
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
        lines = [line for line in lines if line.strip().split(' ')[2] in classes]
        first_line = lines[0]
        frame_number = first_line.strip().split(' ')[0]
        first_image_file_path = os.path.join(image_dir, sequence_id, f"{int(frame_number):06d}.png")
        img = cv2.imread(first_image_file_path)
        height, width = img.shape[:2]
        for line in lines:
            data = line.strip().split(' ')
            frame_number = data[0]
            image_file_path = os.path.join(image_dir, sequence_id, f"{int(frame_number):06d}.png")
            if not os.path.exists(image_file_path):
                print(f"ERROR: Image file not found: {image_file_path}")
                exit(0)
            if image_file_path not in image_file_to_lables:
                image_file_to_lables[image_file_path] = []
            yolo_label = convert_kitti_to_yolo(line, height, width, class_to_idx)
            image_file_to_lables[image_file_path].append(yolo_label)

    image_files = sorted(image_file_to_lables.keys())
    random.shuffle(image_files)
    n_total = len(image_files)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    n_test = n_total - n_train - n_val
    train_image_files = image_files[:n_train]
    val_image_files = image_files[n_train:n_train + n_val]
    test_image_files = image_files[n_train + n_val:]
    print(f"Total number of images: {n_total}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
    splits = {
        'train': train_image_files,
        'val': val_image_files,
        'test': test_image_files
    }
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    

    # Create each split
    for split_name, split_files in splits.items():
        with open(os.path.join(output_dir, f'{split_name}.txt'), 'w') as f:
            img_idx = 0
            for img_path in tqdm(split_files, desc=f'Processing {split_name} set'):
                yolo_labels = image_file_to_lables[img_path]
                output_img_path = os.path.join(output_dir, split_name, 'images', f'{img_idx}.png')
                # Copy image
                shutil.copy2(img_path, output_img_path)
                f.write(f'{output_img_path}\n')
                output_label_path = os.path.join(output_dir, split_name, 'labels', f'{img_idx}.txt')
                with open(output_label_path, 'w') as lf:
                    lf.write('\n'.join(yolo_labels))
                img_idx += 1

def main():
    # Set paths using project root
    image_dir = get_project_path('data/kitti/training/image_02')
    label_dir = get_project_path('data/kitti/training/label_02')
    output_dir = get_project_path('data/processed/kitti')
    
    # Create dataset splits
    split_dataset(image_dir, label_dir, output_dir)
    
    print('Dataset preprocessing completed!')

if __name__ == '__main__':
    main() 