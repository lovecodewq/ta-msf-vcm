import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Local imports
from utils.padding_utils import cal_feature_padding_size, feature_padding, feature_unpadding

# CompressAI imports
from compressai.models import JointAutoregressiveHierarchicalPriors
from compressai.layers import GDN
from compressai.layers import conv3x3, subpel_conv3x3
from model.fpn_featuer_fusion import FPNFeatureFusion, FeatureDecoder
from data import ImageDataset

def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)

class JointAutoregressFPNCompressor(JointAutoregressiveHierarchicalPriors):
    """Adapted JointAutoregressiveHierarchicalPriors for FPN feature compression."""
    
    def __init__(self, N, M, input_channels=256, output_channels=256, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        
        # Replace first and last convolutions to match FPN feature dimensions
        # Use wider hidden layers (n_hidden) in the middle of the network
        self.g_a = FPNFeatureFusion(input_channels, N)
        self.g_s = FeatureDecoder(output_channels, M)

        self.h_a = nn.Sequential(
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(M, M, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M * 3 // 2, M * 2),
        )

        self.p6Decoder = nn.Sequential(nn.MaxPool2d(1, stride=2))

    def forward(self, x):
        # conver ordered dict to list
        features = list(x.values())
        features = features[:-1]
        _, _, p2_h, p2_w = features[0].shape
        pad_info = cal_feature_padding_size((p2_h, p2_w))
        features = feature_padding(features, pad_info)
        y = self.g_a(features)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        reconstructed_features = feature_unpadding(
            x_hat, pad_info
        )
        p6 = self.p6Decoder(
            reconstructed_features[3]
        )  # p6 is generated from p5 directly

        reconstructed_features.append(p6)
        reconstructed_features = OrderedDict(zip(x.keys(), reconstructed_features))
        return {
            'features': reconstructed_features,
            'likelihoods': {
                'y': y_likelihoods,
                'z': z_likelihoods,
            }
        }

    def compress(self, x):
        features = x[:-1]
        _, _, p2_h, p2_w = features[0].shape
        pad_info = cal_feature_padding_size((p2_h, p2_w))
        features = feature_padding(features, pad_info)
        y = self.g_a(features)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))
        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, p2_h, p2_w):
        assert isinstance(strings, list) and len(strings) == 2
        pad_info = cal_feature_padding_size((p2_h, p2_w))
        padded_p2_h = pad_info["padded_size"][0][0]
        padded_p2_w = pad_info["padded_size"][0][1]
        z_shape = get_downsampled_shape(padded_p2_h, padded_p2_w, 64)

        z_hat = self.entropy_bottleneck.decompress(strings[1], z_shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))

        recon_p_layer_features = self.g_s(y_hat)
        recon_p_layer_features = feature_unpadding(
            recon_p_layer_features, pad_info
        )
        p6 = self.p6Decoder(
            recon_p_layer_features[3]
        )  # p6 is generated from p5 directly
        recon_p_layer_features.append(p6)
        return {"features": recon_p_layer_features, "y_hat": y_hat}