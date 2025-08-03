"""
Enhanced Detection Model with Semantic FPN Naming.
This demonstrates how to implement consistent FPN naming while maintaining backward compatibility.
"""

import torch
import torch.nn as nn
import torchvision.models.detection as detection
from collections import OrderedDict
from typing import Dict, Union, Optional


class DetectionModelSemanticFPN(nn.Module):
    """
    Detection model with semantic FPN naming support.
    
    Supports both legacy numeric keys (0,1,2,3,4) and semantic keys (p2,p3,p4,p5,p6)
    for backward compatibility.
    """
    
    def __init__(self, num_classes, pretrained=True, use_semantic_keys=True):
        super().__init__()
        
        # Load pretrained Faster R-CNN model
        if pretrained:
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=False)
            
        # Modify classifier for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, num_classes)
        self.model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, 4 * num_classes)
        
        # FPN naming configuration
        self.use_semantic_keys = use_semantic_keys
        self.fpn_level_names = ['p2', 'p3', 'p4', 'p5', 'p6']
        
        # Key conversion mappings
        self.numeric_to_semantic = {str(i): f'p{i+2}' for i in range(5)}
        self.semantic_to_numeric = {f'p{i+2}': str(i) for i in range(5)}
        
    def get_fpn_features(self, image_list, use_semantic_keys=None):
        """
        Extract FPN features from images.
        
        Args:
            image_list: Input images
            use_semantic_keys: Override instance setting for key format
            
        Returns:
            OrderedDict with FPN features using either numeric or semantic keys
        """
        # Use parameter override or instance setting
        use_semantic = use_semantic_keys if use_semantic_keys is not None else self.use_semantic_keys
        
        # Process images through ResNet backbone
        x = image_list
        x = self.model.backbone.body.conv1(x)
        x = self.model.backbone.body.bn1(x)
        x = self.model.backbone.body.relu(x)
        x = self.model.backbone.body.maxpool(x)
        
        # Extract features from ResNet stages
        layer1 = self.model.backbone.body.layer1(x)
        layer2 = self.model.backbone.body.layer2(layer1)
        layer3 = self.model.backbone.body.layer3(layer2)
        layer4 = self.model.backbone.body.layer4(layer3)
        
        # Create ResNet feature dict with correct naming for FPN
        resnet_features = OrderedDict([
            ('feat1', layer1),  # 256 channels -> becomes p2
            ('feat2', layer2),  # 512 channels -> becomes p3
            ('feat3', layer3),  # 1024 channels -> becomes p4
            ('feat4', layer4)   # 2048 channels -> becomes p5
        ])
        
        # Apply FPN to get multi-scale features
        fpn_features = self.model.backbone.fpn(resnet_features)
        
        # Convert to desired key format
        features = OrderedDict()
        for idx, (_, feat) in enumerate(fpn_features.items()):
            if use_semantic:
                # Use semantic FPN naming (p2, p3, p4, p5, p6)
                key = self.fpn_level_names[idx]
            else:
                # Use legacy numeric naming (0, 1, 2, 3, 4)
                key = str(idx)
            features[key] = feat
            
        return features
    
    def convert_fpn_keys(self, features: Dict[str, torch.Tensor], to_format='semantic') -> Dict[str, torch.Tensor]:
        """Convert between numeric and semantic FPN keys."""
        if to_format == 'semantic':
            conversion_map = self.numeric_to_semantic
        else:
            conversion_map = self.semantic_to_numeric
            
        return OrderedDict([
            (conversion_map.get(k, k), v) for k, v in features.items()
        ])
    
    def parse_fpn_level(self, level_input: Union[int, str]) -> str:
        """Parse FPN level from various input formats to semantic format."""
        if isinstance(level_input, int):
            return f'p{level_input + 2}'
        elif isinstance(level_input, str):
            if level_input.isdigit():
                return f'p{int(level_input) + 2}'
            elif level_input.startswith('p') and level_input[1:].isdigit():
                return level_input
            else:
                raise ValueError(f"Invalid FPN level: {level_input}")
        else:
            raise ValueError(f"FPN level must be int or str, got {type(level_input)}")
    
    def get_fpn_level_info(self, level_key: str) -> Dict[str, Union[str, int]]:
        """Get information about a specific FPN level."""
        if level_key.startswith('p'):
            level_num = int(level_key[1:])
            numeric_key = str(level_num - 2)
        else:
            numeric_key = level_key
            level_num = int(level_key) + 2
            
        resolution_factor = 2 ** level_num
        
        return {
            'semantic_key': f'p{level_num}',
            'numeric_key': numeric_key,
            'resolution_factor': resolution_factor,
            'description': f'1/{resolution_factor} resolution'
        }
    
    def forward(self, images):
        """Forward pass for inference."""
        return self.model(images)
    
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


def demo_semantic_detection_model():
    """Demonstrate the semantic FPN detection model."""
    print("=== SEMANTIC FPN DETECTION MODEL DEMO ===")
    
    # Create model with semantic keys
    model = DetectionModelSemanticFPN(num_classes=10, pretrained=False, use_semantic_keys=True)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 384, 1280)
    
    print("1. Extract FPN features with semantic keys:")
    semantic_features = model.get_fpn_features(dummy_input, use_semantic_keys=True)
    for k, v in semantic_features.items():
        info = model.get_fpn_level_info(k)
        print(f"   {k}: {v.shape} ({info['description']})")
    
    print("\n2. Extract FPN features with legacy keys:")
    legacy_features = model.get_fpn_features(dummy_input, use_semantic_keys=False)
    for k, v in legacy_features.items():
        info = model.get_fpn_level_info(k)
        print(f"   {k}: {v.shape} ({info['description']})")
    
    print("\n3. Convert between key formats:")
    converted_to_semantic = model.convert_fpn_keys(legacy_features, to_format='semantic')
    converted_to_numeric = model.convert_fpn_keys(semantic_features, to_format='numeric')
    
    print(f"   Legacy keys: {list(legacy_features.keys())}")
    print(f"   Converted to semantic: {list(converted_to_semantic.keys())}")
    print(f"   Semantic keys: {list(semantic_features.keys())}")
    print(f"   Converted to numeric: {list(converted_to_numeric.keys())}")
    
    print("\n4. Parse different level input formats:")
    test_levels = [0, '2', 'p3', 4, 'p6']
    for level in test_levels:
        try:
            parsed = model.parse_fpn_level(level)
            info = model.get_fpn_level_info(parsed)
            print(f"   {level} -> {parsed} ({info['description']})")
        except Exception as e:
            print(f"   {level} -> ERROR: {e}")


if __name__ == '__main__':
    demo_semantic_detection_model()