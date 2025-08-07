import torch.nn as nn
import torch

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

class FPNFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.p2_encoder = nn.Sequential(
            ResidualBlockWithStride(in_channels, out_channels, stride=2),
            ResidualBlock(out_channels, out_channels),
        )
        self.p3_encoder = nn.Sequential(
            ResidualBlockWithStride(in_channels + out_channels, out_channels, stride=2),
            AttentionBlock(out_channels),
            ResidualBlock(out_channels, out_channels),
        )
        self.p4_encoder = nn.Sequential(
            ResidualBlockWithStride(in_channels + out_channels, out_channels, stride=2),
            AttentionBlock(out_channels),
            ResidualBlock(out_channels, out_channels),
        )
        self.p5_encoder = nn.Sequential(
            ResidualBlockWithStride(in_channels + out_channels, out_channels, stride=2),
            AttentionBlock(out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, fpn_features):
        p2, p3, p4, p5 = tuple(fpn_features)
        out = self.p2_encoder(p2)
        out = self.p3_encoder(torch.cat([out, p3], dim=1))
        out = self.p4_encoder(torch.cat([out, p4], dim=1))
        out = self.p5_encoder(torch.cat([out, p5], dim=1))
        return out


class FeatureDecoder(nn.Module):
    def __init__(self, N, M) -> None:
        super().__init__()

        class FeatureMixingBlock(nn.Module):
            def __init__(self, N) -> None:
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2), nn.LeakyReLU()
                )

                self.conv2 = nn.Sequential(
                    nn.Conv2d(N * 2, N, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                )

            def forward(self, high, low):
                high = self.conv1(high)
                return self.conv2(torch.cat([high, low], dim=1)) + low

        self.p5Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, N, 2),
        )

        self.p4Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, N, 2),
        )

        self.p3Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            AttentionBlock(M),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, N, 2),
        )
        self.p2Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            AttentionBlock(M),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            subpel_conv3x3(M, N, 2),
        )

        self.decoder_attention = AttentionBlock(M)

        self.fmb23 = FeatureMixingBlock(N)
        self.fmb34 = FeatureMixingBlock(N)
        self.fmb45 = FeatureMixingBlock(N)

    def forward(self, y_hat):
        y_hat = self.decoder_attention(y_hat)
        p2 = self.p2Decoder(y_hat)
        p3 = self.fmb23(p2, self.p3Decoder(y_hat))
        p4 = self.fmb34(p3, self.p4Decoder(y_hat))
        p5 = self.fmb45(p4, self.p5Decoder(y_hat))
        return [p2, p3, p4, p5]