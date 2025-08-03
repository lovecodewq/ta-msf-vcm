import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from model.gdn import GDN
from typing import Dict, Tuple
from collections import OrderedDict

class FPNFeaturePreprocessor(nn.Module):
    """Preprocessor for FPN features to handle multi-scale concatenation using interpolation."""
    
    def __init__(self, fpn_channels_per_level=256, num_levels=5):
        super().__init__()
        self.fpn_channels_per_level = fpn_channels_per_level
        self.num_levels = num_levels
        self.total_channels = fpn_channels_per_level * num_levels
    
    def forward(self, fpn_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess FPN features: interpolate to same spatial size and concatenate.
        
        Args:
            fpn_features: Dict with keys '0', '1', '2', '3', '4' and tensor values
            
        Returns:
            concatenated_tensor: Single tensor with all FPN levels concatenated
            metadata: Information needed for reconstruction
        """
        # Get all feature levels
        feature_levels = []
        original_shapes = {}
        
        for level_idx in range(self.num_levels):
            key = str(level_idx)
            if key in fpn_features:
                feature_levels.append(fpn_features[key])
                original_shapes[key] = fpn_features[key].shape[-2:]  # H, W
            else:
                raise ValueError(f"Missing FPN level {key}")
        
        # Find maximum spatial dimensions as target size
        max_h = max(feat.size(-2) for feat in feature_levels)
        max_w = max(feat.size(-1) for feat in feature_levels)
        target_size = (max_h, max_w)
        
        # Interpolate all features to target size
        interpolated_features = []
        
        for i, feat in enumerate(feature_levels):
            current_size = feat.shape[-2:]
            
            if current_size != target_size:
                # Use bilinear interpolation to resize to target size
                # align_corners=False is recommended for better gradients
                interpolated_feat = F.interpolate(
                    feat, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                interpolated_feat = feat
            
            interpolated_features.append(interpolated_feat)
        
        # Concatenate along channel dimension
        concatenated = torch.cat(interpolated_features, dim=1)
        
        metadata = {
            'original_shapes': original_shapes,
            'target_size': target_size,
            'num_levels': self.num_levels,
            'channels_per_level': self.fpn_channels_per_level
        }
        
        return concatenated, metadata

class FPNFeaturePostprocessor(nn.Module):
    """Postprocessor to reconstruct FPN features from concatenated tensor using interpolation."""
    
    def __init__(self, fpn_channels_per_level=256, num_levels=5):
        super().__init__()
        self.fpn_channels_per_level = fpn_channels_per_level
        self.num_levels = num_levels
    
    def forward(self, concatenated_tensor: torch.Tensor, metadata: Dict) -> Dict[str, torch.Tensor]:
        """
        Reconstruct FPN features from concatenated tensor using interpolation.
        
        Args:
            concatenated_tensor: Reconstructed concatenated features
            metadata: Information from preprocessing
            
        Returns:
            fpn_features: Dict with reconstructed FPN features
        """
        # Split concatenated tensor back to individual levels
        feature_chunks = torch.split(concatenated_tensor, self.fpn_channels_per_level, dim=1)
        
        fpn_features = OrderedDict()
        original_shapes = metadata['original_shapes']
        
        for i, chunk in enumerate(feature_chunks):
            level_key = str(i)
            target_shape = original_shapes[level_key]  # (H, W)
            
            current_size = chunk.shape[-2:]
            
            if current_size != target_shape:
                # Use bilinear interpolation to resize back to original size
                reconstructed_feat = F.interpolate(
                    chunk,
                    size=target_shape,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                reconstructed_feat = chunk
            
            fpn_features[level_key] = reconstructed_feat
        
        return fpn_features

class FactorizedPriorFPNCompression(nn.Module):
    """Factorized Prior model for compressing FPN features."""
    
    def __init__(self, n_hidden=128, n_channels=192, fpn_channels_per_level=256, num_fpn_levels=5):
        super().__init__()
        
        # FPN preprocessing and postprocessing
        self.preprocessor = FPNFeaturePreprocessor(fpn_channels_per_level, num_fpn_levels)
        self.postprocessor = FPNFeaturePostprocessor(fpn_channels_per_level, num_fpn_levels)
        
        # Total input channels from FPN
        input_channels = fpn_channels_per_level * num_fpn_levels
        
        # Entropy bottleneck
        self.entropy_bottleneck = EntropyBottleneck(n_channels)
        
        # Analysis transform (encoder)
        self.g_a = nn.Sequential(
            nn.Conv2d(input_channels, n_hidden, kernel_size=5, padding=2, stride=2),
            GDN(n_hidden),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=5, padding=2, stride=2),
            GDN(n_hidden),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=5, padding=2, stride=2),
            GDN(n_hidden),
            nn.Conv2d(n_hidden, n_channels, kernel_size=5, padding=2, stride=2),
        )
        
        # Synthesis transform (decoder)
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(n_channels, n_hidden, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(n_hidden, inverse=True),
            nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(n_hidden, inverse=True),
            nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(n_hidden, inverse=True),
            nn.ConvTranspose2d(n_hidden, input_channels, kernel_size=5, padding=2, output_padding=1, stride=2),
        )
        
        # Store parameters
        self.n_hidden = n_hidden
        self.n_channels = n_channels
        self.fpn_channels_per_level = fpn_channels_per_level
        self.num_fpn_levels = num_fpn_levels
    
    def _pad_crop_size(self, size):
        """Calculate the padded size that's divisible by downsample_factor.
        
        Args:
            size (tuple): Input size (H, W)
            
        Returns:
            tuple: New size divisible by downsample_factor
            tuple: Crop slices to recover original size
        """
        new_h = ((size[0] + self.downsample_factor - 1) // self.downsample_factor) * self.downsample_factor
        new_w = ((size[1] + self.downsample_factor - 1) // self.downsample_factor) * self.downsample_factor
        
        # Calculate crop slices to recover original size
        h_slice = slice(0, size[0])
        w_slice = slice(0, size[1])
        
        return (new_h, new_w), (h_slice, w_slice)
    
    @property
    def downsample_factor(self):
        """Downsample factor of the model."""
        return 2**4  # 16 total downsampling
    
    def forward(self, fpn_features: Dict[str, torch.Tensor]):
        """Forward pass of the model.
        
        Args:
            fpn_features: Dict with FPN features
            
        Returns:
            Dict with reconstructed features and likelihoods
        """
        # Preprocess FPN features
        x, metadata = self.preprocessor(fpn_features)
        
        # Get input size and calculate padded size
        input_size = x.size()[-2:]
        padded_size, crop_slices = self._pad_crop_size(input_size)
        
        # Pad input if necessary
        if input_size != padded_size:
            x = F.pad(x, (0, padded_size[1] - input_size[1], 0, padded_size[0] - input_size[0]))
        
        # Encoder
        y = self.g_a(x)
        
        # Entropy bottleneck
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        # Decoder
        x_hat = self.g_s(y_hat)
        
        # Crop to original size
        x_hat = x_hat[..., crop_slices[0], crop_slices[1]]
        
        # Postprocess to FPN features
        fpn_features_hat = self.postprocessor(x_hat, metadata)
        
        return {
            'fpn_features_hat': fpn_features_hat,
            'likelihoods': y_likelihoods,
            'metadata': metadata
        }
    
    @classmethod
    def from_state_dict(cls, state_dict):
        """Create a new model instance and load state dict into it."""
        # Infer parameters from state dict
        first_conv_weight = state_dict['g_a.0.weight']  # [out_channels, in_channels, h, w]
        input_channels = first_conv_weight.size(1)
        n_hidden = first_conv_weight.size(0)
        n_channels = state_dict['g_a.6.weight'].size(0)
        
        # Assume standard FPN configuration
        fpn_channels_per_level = 256
        num_fpn_levels = input_channels // fpn_channels_per_level
        
        model = cls(
            n_hidden=n_hidden,
            n_channels=n_channels,
            fpn_channels_per_level=fpn_channels_per_level,
            num_fpn_levels=num_fpn_levels
        )
        model.load_state_dict(state_dict)
        return model
    
    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.
        
        Args:
            force: If True, will update the CDF values even if already initialized.
        
        Returns:
            bool: True if at least one of the entropy bottlenecks was updated.
        """
        updated = self.entropy_bottleneck.update(force=force)
        return updated
    
    def compress(self, fpn_features: Dict[str, torch.Tensor]):
        """Compress FPN features to strings."""
        # Preprocess FPN features
        x, metadata = self.preprocessor(fpn_features)
        
        # Get input size and calculate padded size
        input_size = x.size()[-2:]
        padded_size, _ = self._pad_crop_size(input_size)
        
        # Pad input if necessary
        if input_size != padded_size:
            x = F.pad(x, (0, padded_size[1] - input_size[1], 0, padded_size[0] - input_size[0]))
        
        # Always update entropy bottleneck before compression
        self.update()
            
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        
        return {
            "y_strings": y_strings, 
            "shape": y.size()[-2:], 
            "input_size": input_size,
            "fpn_metadata": metadata
        }
    
    def decompress(self, strings, shape, input_size=None, fpn_metadata=None):
        """Decompress strings to reconstructed FPN features.
        
        Args:
            strings: Compressed strings
            shape: Shape of the latent representation
            input_size: Original concatenated input size (H, W)
            fpn_metadata: Metadata for FPN reconstruction
        """
        # Always update entropy bottleneck before decompression
        self.update()
            
        y_hat = self.entropy_bottleneck.decompress(strings, shape)
        x_hat = self.g_s(y_hat)
        
        # Crop to original size if provided
        if input_size is not None:
            _, crop_slices = self._pad_crop_size(input_size)
            x_hat = x_hat[..., crop_slices[0], crop_slices[1]]
        
        # Postprocess to FPN features if metadata provided
        if fpn_metadata is not None:
            fpn_features_hat = self.postprocessor(x_hat, fpn_metadata)
            return {"fpn_features_hat": fpn_features_hat}
        else:
            return {"x_hat": x_hat} 