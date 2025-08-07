"""
Dataset utilities for FPN feature compression.
"""
import torch
import random
from data import ImageDataset

class FPNDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, transform):
        self.image_dataset = ImageDataset(txt_file, transform)

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image = self.image_dataset[idx]
        return image

class FPNRandomLevelDataset(torch.utils.data.Dataset):
    """Dataset that randomly selects FPN level features from images during training."""
    
    def __init__(self, txt_file, transform, training=True):
        """
        Args:
            txt_file: Path to image list file
            transform: Image transforms (applied to input images)
            training: If True, randomly select level per sample. If False, cycle through levels.
        """
        self.image_dataset = ImageDataset(txt_file, transform)
        self.training = training
        self.fpn_levels = ['0', '1', '2', '3', '4']
    
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        # Get preprocessed image
        image = self.image_dataset[idx]  # [C, H, W] - already preprocessed
        
        if self.training:
            # Randomly select FPN level for this sample
            selected_level = random.choice(self.fpn_levels)
        else:
            # For validation, cycle through levels for consistent evaluation
            selected_level = self.fpn_levels[idx % len(self.fpn_levels)]
        
        return image, selected_level