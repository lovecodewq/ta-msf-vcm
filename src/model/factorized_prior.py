import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
from model.gdn import GDN

class FactorizedPrior(nn.Module):
    def __init__(self, n_hidden, n_channels, input_channels=3, output_channels=None):
        super().__init__()
        if output_channels is None:
            output_channels = input_channels
        
        self.entropy_bottleneck = EntropyBottleneck(n_channels)
        self.g_a = nn.Sequential(
            nn.Conv2d(input_channels, n_hidden, kernel_size=5, padding=2, stride=2),
            GDN(n_hidden),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=5, padding=2, stride=2),
            GDN(n_hidden),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=5, padding=2, stride=2),
            GDN(n_hidden),
            nn.Conv2d(n_hidden, n_channels, kernel_size=5, padding=2, stride=2),
        )
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(n_channels, n_hidden, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(n_hidden, inverse=True),
            nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(n_hidden, inverse=True),
            nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(n_hidden, inverse=True),
            nn.ConvTranspose2d(n_hidden, output_channels, kernel_size=5, padding=2, output_padding=1, stride=2),
        )
    
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
    
    def forward(self, x):
        """Forward pass of the model."""
        # Get input size and calculate padded size
        input_size = x.size()[-2:]
        padded_size, crop_slices = self._pad_crop_size(input_size)
        
        # Pad input if necessary
        if input_size != padded_size:
            x = torch.nn.functional.pad(x, (0, padded_size[1] - input_size[1], 0, padded_size[0] - input_size[0]))
        
        # Encoder
        y = self.g_a(x)
        
        # Entropy bottleneck
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        # Decoder
        x_hat = self.g_s(y_hat)
        
        # Crop to original size
        x_hat = x_hat[..., crop_slices[0], crop_slices[1]]
        
        return {
            'x_hat': x_hat,
            'likelihoods': y_likelihoods,
        }

    @property
    def downsample_factor(self):
        """Downsample factor of the model."""
        return 2*4  # 2^4 = 16 total downsampling

    @classmethod
    def from_state_dict(cls, state_dict):
        """Create a new model instance and load state dict into it."""
        n_hidden = state_dict['g_a.0.weight'].size(0)  
        n_channels = state_dict['g_a.6.weight'].size(0)  
        model = cls(n_hidden=n_hidden, n_channels=n_channels)
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

    def compress(self, x):
        """Compress input tensor to strings."""
        # Get input size and calculate padded size
        input_size = x.size()[-2:]
        padded_size, _ = self._pad_crop_size(input_size)
        
        # Pad input if necessary
        if input_size != padded_size:
            x = torch.nn.functional.pad(x, (0, padded_size[1] - input_size[1], 0, padded_size[0] - input_size[0]))
        
        # Always update entropy bottleneck before compression
        self.update()
            
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"y_strings": y_strings, "shape": y.size()[-2:], "input_size": input_size}

    def decompress(self, strings, shape, input_size=None):
        """Decompress strings to reconstructed image.
        
        Args:
            strings: Compressed strings
            shape: Shape of the latent representation
            input_size: Original input size (H, W)
        """
        # Always update entropy bottleneck before decompression
        self.update()
            
        y_hat = self.entropy_bottleneck.decompress(strings, shape)
        x_hat = self.g_s(y_hat)
        
        # Crop to original size if provided
        if input_size is not None:
            _, crop_slices = self._pad_crop_size(input_size)
            x_hat = x_hat[..., crop_slices[0], crop_slices[1]]
        
        return {"x_hat": x_hat}
