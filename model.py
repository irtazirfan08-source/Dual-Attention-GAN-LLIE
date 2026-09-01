"""
Dual-Attention GAN for Low-Light Image Enhancement
Author: Irtaz Irfan (East Delta University)
Supervisor: Golam Moktader Daiyan (Assistant Professor, Dept. of CSE, EDU)
Thesis Title: Dual-Attention GAN: A Robust Framework for Low-Light Image 
              Enhancement using Efficient Residual Blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ==============================================================================
# 1. DEPTHWISE SEPARABLE CONVOLUTION (Efficient Residual Component)
# ==============================================================================
class DepthwiseSeparableConv(nn.Module):
    """
    Decouples spatial filtering from channel mixing to reduce computational
    complexity while preserving rich hierarchical features.
    """
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise spatial filtering
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, 
            groups=in_channels, bias=False
        )
        # Pointwise channel mixing (1x1 conv)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.relu(x)


# ==============================================================================
# 2. DUAL-ATTENTION MECHANISM (Channel-Spatial + SE Block)
# ==============================================================================
class EnhancedChannelSpatialAttention(nn.Module):
    """
    Combines Channel Attention (MLP-guided feature recalibration) and Spatial Attention
    (7x7 Conv-guided spatial weighting) via element-wise addition.
    """
    def __init__(self, channels, reduction=16):
        super(EnhancedChannelSpatialAttention, self).__init__()
        # Channel Attention Branch
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        # Spatial Attention Branch
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. Channel Attention
        avg_pool_c = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        max_pool_c = F.adaptive_max_pool2d(x, (1, 1)).view(b, c)
        channel_out = self.mlp(avg_pool_c) + self.mlp(max_pool_c)
        channel_att = self.sigmoid(channel_out).view(b, c, 1, 1)

        # 2. Spatial Attention
        avg_pool_s = torch.mean(x, dim=1, keepdim=True)
        max_pool_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_pool_s, max_pool_s], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_cat))

        # 3. Hybrid Addition Fusion & Gating
        hybrid_att = channel_att + spatial_att
        return x * hybrid_att


class SqueezeAndExcitation(nn.Module):
    """
    Recalibrates channel-wise feature responses using Global Average Pooling.
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeAndExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class EfficientResidualBlock(nn.Module):
    """
    9 sequential bottleneck blocks combining Depthwise Separable Convolutions,
    Enhanced Channel-Spatial Attention, and Squeeze-and-Excitation Refinement.
    """
    def __init__(self, channels):
        super(EfficientResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.dual_attention = EnhancedChannelSpatialAttention(channels)
        self.se_refine = SqueezeAndExcitation(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dual_attention(out)
        out = self.se_refine(out)
        return residual + out


# ==============================================================================
# 3. GENERATOR ARCHITECTURE (Symmetric U-Net with Bottleneck)
# ==============================================================================
class DualAttentionGenerator(nn.Module):
    """
    U-Net Generator:
      - 4 Downsampling Stages (Encoder)
      - 9 Efficient Residual Bottleneck Blocks with Dual Attention
      - 4 Upsampling Stages with Skip Connections (Decoder)
      - Output Layer with Tanh activation mapping to [-1, 1]
    """
    def __init__(self, in_channels=3, out_channels=3, num_residuals=9):
        super(DualAttentionGenerator, self).__init__()

        # Initial Feature Expansion (256x256x3 -> 256x256x64)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )

        # Encoder (Downsampling Blocks 1 to 4)
        self.down1 = self._make_down_block(64, 128)    # 256x256 -> 128x128
        self.down2 = self._make_down_block(128, 256)   # 128x128 -> 64x64
        self.down3 = self._make_down_block(256, 512)   # 64x64 -> 32x32
        self.down4 = self._make_down_block(512, 512)   # 32x32 -> 16x16

        # Bottleneck (9 Sequential Efficient Residual Blocks)
        self.bottleneck = nn.Sequential(*[
            EfficientResidualBlock(512) for _ in range(num_residuals)
        ])

        # Decoder (Upsampling Blocks 1 to 4 with Nearest-Neighbor + Conv)
        self.up1 = self._make_up_block(512, 512)       # 16x16 -> 32x32
        self.up2 = self._make_up_block(1024, 256)      # 32x32 -> 64x64 (512 skip + 512 up)
        self.up3 = self._make_up_block(512, 128)       # 64x64 -> 128x128 (256 skip + 256 up)
        self.up4 = self._make_up_block(256, 64)        # 128x128 -> 256x256 (128 skip + 128 up)

        # Output Layer (256x256x128 [64 skip + 64 up] -> 256x256x3)
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def _make_down_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True)
        )

    def _make_up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Initial Conv
        x0 = self.initial_conv(x)          # [B, 64, 256, 256]

        # Encoder Forward
        d1 = self.down1(x0)                # [B, 128, 128, 128]
        d2 = self.down2(d1)                # [B, 256, 64, 64]
        d3 = self.down3(d2)                # [B, 512, 32, 32]
        d4 = self.down4(d3)                # [B, 512, 16, 16]

        # Attention-Guided Bottleneck
        bn = self.bottleneck(d4)           # [B, 512, 16, 16]

        # Decoder Forward with Skip-Connection Concatenations
        u1 = self.up1(bn)                  # [B, 512, 32, 32]
        u2 = self.up2(torch.cat([u1, d3], dim=1))  # [B, 256, 64, 64]
        u3 = self.up3(torch.cat([u2, d2], dim=1))  # [B, 128, 128, 128]
        u4 = self.up4(torch.cat([u3, d1], dim=1))  # [B, 64, 256, 256]

        # Output Reconstruction
        out = self.final_conv(torch.cat([u4, x0], dim=1))  # [B, 3, 256, 256]
        return out


# ==============================================================================
# 4. DISCRIMINATOR ARCHITECTURE (Spectrally Normalized PatchGAN)
# ==============================================================================
class PatchGANDiscriminator(nn.Module):
    """
    5-Layer Fully Convolutional PatchGAN Discriminator with Spectral Normalization
    Classifies local 70x70 patches as Real vs. Fake.
    Output: 30x30 Validity Map.
    """
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()

        # Layer 1: 256x256 -> 128x128x64
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 2: 128x128 -> 64x64x128
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 3: 64x64 -> 32x32x256
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 4: 32x32 -> 31x31x512
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 5: 31x31 -> 30x30x1 (Final validity map)
        self.conv5 = spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        validity = self.conv5(x)
        return validity


# ==============================================================================
# 5. SANITY CHECK & VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Initializing models on: {device}")

    # Instantiate Generator
    gen = DualAttentionGenerator(in_channels=3, out_channels=3, num_residuals=9).to(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    enhanced_output = gen(dummy_input)
    print(f"[✓] Generator Forward Pass Successful: Input {dummy_input.shape} -> Output {enhanced_output.shape}")

    # Instantiate Discriminator
    disc = PatchGANDiscriminator(in_channels=3).to(device)
    validity_map = disc(enhanced_output)
    print(f"[✓] Discriminator Forward Pass Successful: Input {enhanced_output.shape} -> Validity Map {validity_map.shape}")