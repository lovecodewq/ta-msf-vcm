"""
Faster R-CNN detection model implementation with support for distributed inference using FPN features.
"""
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Optional, Tuple, Union
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from collections import OrderedDict

class DetectionModel(nn.Module):
    """Wrapper for Faster R-CNN with support for distributed inference using FPN features."""
    
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Create base model
        if pretrained:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        else:
            self.model = fasterrcnn_resnet50_fpn_v2(weights=None)
        
        # Get number of input features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, images: List[torch.Tensor], 
                targets: Optional[List[Dict[str, torch.Tensor]]] = None,
                features: Optional[Dict[str, torch.Tensor]] = None) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """Forward pass supporting both standard and feature-based inference.
        
        For compression scenarios, use get_fpn_features() to extract features for compression,
        then pass them to this method for efficient inference.
        """
        if features is not None:
            return self.forward_from_features(images, features, targets)
        return self.model(images, targets)
    
    def forward_from_features(self, 
                            images: List[torch.Tensor],
                            features: Dict[str, torch.Tensor],
                            targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """Forward pass using pre-computed FPN features.
        
        Args:
            images: Input images
            features: Pre-computed FPN features with string keys 'p2', 'p3', 'p4', 'p5', 'p6'
            targets: Ground truth targets (for training)
        """
        # Get original image sizes for proper scaling
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
        
        # Transform images to get proper image list and sizes
        images, _ = self.model.transform(images)

        # Convert FPN features to expected format (0-4), p2, p3, p4, p5, p6 -> 0, 1, 2, 3, 4
        features = OrderedDict([(str(i), features[f'p{i+2}']) for i in range(5)])

        # Generate proposals
        proposals, proposal_losses = self.model.rpn(images, features, targets)
        
        # Get detections
        detections, detector_losses = self.model.roi_heads(features, proposals, images.image_sizes, targets)
        
        # Post-process detections
        detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        
        # Return losses during training
        if targets is not None:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
            
        return detections
    
    def get_fpn_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract FPN features from images - ideal for compression and distributed inference."""
        # conver batched image tensor to list of images
        with torch.no_grad():
            image_list = [img for img in images]
        images, _ = self.model.transform(image_list)
        
        # Extract backbone features (before FPN)
        x = images.tensors
        
        # Process through ResNet stages
        x = self.model.backbone.body.conv1(x)
        x = self.model.backbone.body.bn1(x)
        x = self.model.backbone.body.relu(x)
        x = self.model.backbone.body.maxpool(x)

        # Save intermediate features for all layers
        layer1 = self.model.backbone.body.layer1(x)
        layer2 = self.model.backbone.body.layer2(layer1)
        layer3 = self.model.backbone.body.layer3(layer2)
        layer4 = self.model.backbone.body.layer4(layer3)
        
        # Create ResNet feature dict with correct naming for FPN
        resnet_features = OrderedDict([
            ('feat1', layer1),  # 256 channels
            ('feat2', layer2),  # 512 channels
            ('feat3', layer3),  # 1024 channels
            ('feat4', layer4)   # 2048 channels
        ])
        
        # Apply FPN to get multi-scale features
        fpn_features = self.model.backbone.fpn(resnet_features)
        # convert feat1, feat2, feat3, feat4 to p2, p3, p4, p5
        fpn_features = OrderedDict([
            ('p2', fpn_features['feat1']),
            ('p3', fpn_features['feat2']),
            ('p4', fpn_features['feat3']),
            ('p5', fpn_features['feat4']),
            ('p6', fpn_features['pool'])
        ])
        return fpn_features
    
    def train(self, mode=True):
        """Sets the module in training mode."""
        self.training = mode
        self.model.train(mode)
        return self
    
    def eval(self):
        """Sets the module in evaluation mode."""
        self.training = False
        self.model.eval()
        return self
    
    def to(self, device):
        """Moves and/or casts the parameters and buffers."""
        self.model = self.model.to(device)
        return super().to(device)
    
    def state_dict(self):
        """Returns a dictionary containing whole state of the module."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from state_dict into this module."""
        self.model.load_state_dict(state_dict) 