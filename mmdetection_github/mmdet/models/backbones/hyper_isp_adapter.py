import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmdet.registry import MODELS
except ImportError:
    from mmdet.models.builder import BACKBONES as MODELS


class SceneContextHead(nn.Module):
    """
    The Hyper-Network 'Brain'.

    Scientific Goal:
    Standard ISPs use fixed lookup tables (LUTs) based on exposure time.
    We replace this with a content-aware predictor that looks at the
    actual distribution of the RAW signal (Mean, Std, Max) to predict
    the optimal Dynamic Range Compression factor (Mu) and Denoising Gain.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # Input: Mean, Std, Max, Min per channel (4 stats * 3 channels = 12)
        # We use a tiny MLP to keep latency < 0.5ms
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )
        # Heads
        self.mu_pred = nn.Linear(64, 1)  # Predicts Tone Mapping strength (Mu)
        self.noise_pred = nn.Linear(64, 1)  # Predicts Denoising strength (Gain)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 1. Extract Global Statistics (Vectorized)
        # Reshape to [B, C, N]
        x_flat = x.view(B, C, -1)

        # Compute stats across spatial dimensions
        mean = x_flat.mean(dim=2)
        std = x_flat.std(dim=2)
        max_val = x_flat.max(dim=2)[0]
        # Approximate min via negative max (faster than sorting)
        min_val = -1.0 * (-x_flat).max(dim=2)[0]

        # Concat stats: [B, 12]
        stats = torch.cat([mean, std, max_val, min_val], dim=1)

        # 2. Predict Parameters
        feat = self.fc(stats)

        # Predict Mu: Base 5000 (from Proposition 1), learnable shift
        # Sigmoid gives [0, 1] -> scaled to [1, 10000]
        pred_mu = 10000.0 * torch.sigmoid(self.mu_pred(feat)) + 1.0

        # Predict Noise Gain: Range [0, 2] (0 = No denoise, 2 = Heavy denoise)
        pred_gain = 2.0 * torch.sigmoid(self.noise_pred(feat))

        return pred_mu, pred_gain


class DynamicMuLaw(nn.Module):
    """
    Stage 3: Physics-Informed, Instance-Adaptive Tone Mapping.
    Applies the logarithmic curve using the predicted Mu.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mu):
        # x: [B, C, H, W]
        # mu: [B, 1] (Predicted per image)

        # Reshape mu for broadcasting to image dimensions [B, 1, 1, 1]
        mu = mu.view(-1, 1, 1, 1)

        # Ensure non-negative input for log
        x = torch.clamp(x, min=0.0)

        # Formula: log(1 + mu*x) / log(1 + mu)
        numerator = torch.log1p(mu * x)
        denominator = torch.log1p(mu)

        # Add epsilon to denominator for stability
        return numerator / (denominator + 1e-7)


class DynamicSpatialDenoiser(nn.Module):
    """
    Stage 1: Adaptive Spatial Denoising.
    Instead of a fixed Gaussian blur (RAW-Adapter), we learn a spatial
    filter whose intensity is modulated by the predicted 'noise_gain'.
    """

    def __init__(self, channels):
        super().__init__()
        # Depthwise conv acts as a learnable spatial filter per channel
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        # Attention mask determines *where* to denoise (e.g., sky vs edges)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, noise_gain):
        # x: Feature map
        # noise_gain: [B, 1] predicted scalar

        smooth = self.dw_conv(x)
        mask = self.attn(x)

        # Reshape gain for broadcasting
        gain = noise_gain.view(-1, 1, 1, 1)

        # Residual Denoising: Original + (Filter * Attention * DynamicGain)
        # If gain is 0 (clean image), this reduces to Identity.
        return x + (mask * smooth * gain)


class ChannelColorAdapter(nn.Module):
    """
    Stage 2: Color / WB (Intrinsically Adaptive).
    Uses Squeeze-and-Excitation to perform dynamic White Balance.
    """

    def __init__(self, channels):
        super().__init__()
        self.ccm = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.wb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Color Correction Matrix (Linear)
        x = self.ccm(x)
        # Dynamic White Balance (Non-linear gating)
        return x * self.wb(x)


@MODELS.register_module()
class HyperISPAdapter(nn.Module):
    """
    The Main Module: Hyper-ISP.
    Replaces the standard ISP with a self-tuning neural pipeline.
    """

    def __init__(self, bit_depth=14, output_channels=3):
        super().__init__()
        self.max_value = float(2 ** bit_depth - 1)

        # 0. The Brain (Context Head)
        self.context_head = SceneContextHead(in_channels=3)

        # 1. Feature Stem
        self.stem = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)

        # 2. Dynamic Pipeline Stages
        self.denoiser = DynamicSpatialDenoiser(32)
        self.color = ChannelColorAdapter(32)
        self.tone = DynamicMuLaw()

        # 3. Output Reconstruction
        self.proj = nn.Conv2d(32, output_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: [B, 3, H, W] - Linear RGB Input

        # 1. Normalize if necessary
        # (Assuming PASCAL RAW might be 14-bit integer or float)
        if self.max_value > 1.0 and x.max() > 1.0:
            x = x / self.max_value

        # 2. Hyper-Inference: Predict Parameters from Raw Image
        # mu: [B, 1], gain: [B, 1]
        pred_mu, pred_gain = self.context_head(x)

        # 3. Feature Extraction
        feat = self.stem(x)

        # 4. Apply Dynamic ISP Stages
        feat = self.denoiser(feat, pred_gain)  # Apply predicted noise gain
        feat = self.color(feat)  # Apply WB
        feat = self.tone(feat, pred_mu)  # Apply predicted Mu-Law

        # 5. Project to Backbone Space (e.g. ResNet Input)
        out = self.proj(feat)

        # Return list to match MMDetection interface (expects a list of features)
        return [out]