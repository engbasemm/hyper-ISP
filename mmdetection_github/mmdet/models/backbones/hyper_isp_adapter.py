import torch
import torch.nn as nn

try:
    from mmdet.registry import MODELS
except ImportError:
    from mmdet.models.builder import BACKBONES as MODELS


class SceneContextHead(nn.Module):
    """
    The Hyper-Network 'Brain'.
    Analyzes global scene statistics to predict instance-adaptive ISP parameters.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # Input: Mean, Std, Max, Min per channel (4 stats * 3 channels = 12)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )
        # Heads
        self.mu_pred = nn.Linear(64, 1)  # Dynamic Mu (Tone Mapping)
        self.noise_pred = nn.Linear(64, 1)  # Dynamic Gain (Denoising)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)

        # Vectorized stats computation
        mean = x_flat.mean(dim=2)
        std = x_flat.std(dim=2)
        max_val = x_flat.max(dim=2)[0]
        min_val = -1.0 * (-x_flat).max(dim=2)[0]

        stats = torch.cat([mean, std, max_val, min_val], dim=1)
        feat = self.fc(stats)

        # Predict Mu: Base 5000, learnable shift [1, 10001]
        pred_mu = 10000.0 * torch.sigmoid(self.mu_pred(feat)) + 1.0

        # Predict Noise Gain: [0, 2]
        pred_gain = 2.0 * torch.sigmoid(self.noise_pred(feat))

        return pred_mu, pred_gain


class DynamicMuLaw(nn.Module):
    """Stage 3: Physics-Informed, Instance-Adaptive Tone Mapping."""

    def __init__(self):
        super().__init__()

    def forward(self, x, mu):
        # Reshape mu for broadcasting [B, 1, 1, 1]
        mu = mu.view(-1, 1, 1, 1)
        x = torch.clamp(x, min=0.0)

        # Logarithmic Companding: y = log(1 + mu*x) / log(1 + mu)
        # Added eps for numerical stability
        numerator = torch.log1p(mu * x)
        denominator = torch.log1p(mu)
        return numerator / (denominator + 1e-7)


class DynamicSpatialDenoiser(nn.Module):
    """Stage 1: Adaptive Spatial Denoising."""

    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self, x, noise_gain):
        smooth = self.dw_conv(x)
        mask = self.attn(x)
        gain = noise_gain.view(-1, 1, 1, 1)
        # Residual Denoising modulated by predicted gain
        return x + (mask * smooth * gain)


class ChannelColorAdapter(nn.Module):
    """Stage 2: Color / WB (Intrinsically Adaptive via SE-Block)."""

    def __init__(self, channels):
        super().__init__()
        self.ccm = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.wb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.ccm(x)
        return x * self.wb(x)


@MODELS.register_module()
class HyperISPAdapter(nn.Module):
    """
    Hyper-ISP Main Module with Ablation Controls.

    Args:
        isp_mode (str): Control flag for scientific ablation.
            - 'dynamic': Full model (Self-Tuning).
            - 'static': Fixed Mu=5000, Gain=1.0.
            - 'linear': No ISP physics, pure CNN baseline.
    """

    def __init__(self,
                 bit_depth=14,
                 output_channels=3,
                 isp_mode='dynamic'
                 ):
        super().__init__()
        self.max_value = float(2 ** bit_depth - 1)
        self.isp_mode = isp_mode

        # Components
        self.context_head = SceneContextHead(in_channels=3)
        self.stem = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.denoiser = DynamicSpatialDenoiser(32)
        self.color = ChannelColorAdapter(32)
        self.tone = DynamicMuLaw()
        self.proj = nn.Conv2d(32, output_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 1. Normalize
        if self.max_value > 1.0 and x.max() > 1.0:
            x = x / self.max_value

        # --- ABLATION MODE: LINEAR ---
        # Strictly bypass all physics/hypernet logic.
        # This acts as the scientific "Null Hypothesis" baseline.
        if self.isp_mode == 'linear':
            x = self.stem(x)
            # Skip Denoise/Color/Tone
            out = self.proj(x)
            return [out]

        # 2. Parameter Prediction (or Static Override)
        if self.isp_mode == 'dynamic':
            pred_mu, pred_gain = self.context_head(x)
        elif self.isp_mode == 'static':
            # Force optimal static values found in literature
            B = x.size(0)
            device = x.device
            pred_mu = torch.ones((B, 1), device=device) * 5000.0
            pred_gain = torch.ones((B, 1), device=device) * 1.0
        else:
            raise ValueError(f"Unknown isp_mode: {self.isp_mode}")

        # 3. Pipeline Application
        x = self.stem(x)
        x = self.denoiser(x, pred_gain)
        x = self.color(x)
        x = self.tone(x, pred_mu)  # <--- Mu-Law Gradient Boost happens here

        # 4. Output
        out = self.proj(x)
        return [out]