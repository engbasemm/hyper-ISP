import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmdet.registry import MODELS
except ImportError:
    from mmdet.models.builder import BACKBONES as MODELS


# =============================================================================
# Simplified Multi-Region Tone Mapping
# =============================================================================

class SimpleToneMap(nn.Module):
    """
    Simple multi-region tone mapping with 4 regions.
    Each region gets a learned gain factor.
    """

    def __init__(self):
        super().__init__()

        # Learnable gains for each region (initialized near 1.0 for stability)
        self.shadow_gain = nn.Parameter(torch.tensor([1.5]))  # [0.0-0.25]
        self.midshadow_gain = nn.Parameter(torch.tensor([1.1]))  # [0.25-0.50]
        self.midhigh_gain = nn.Parameter(torch.tensor([1.1]))  # [0.50-0.75]
        self.highlight_gain = nn.Parameter(torch.tensor([0.8]))  # [0.75-1.0]

    def soft_region_mask(self, x, low, high, smooth=0.05):
        """Create smooth transition masks between regions."""
        lower_gate = torch.sigmoid((x - low) / smooth)
        upper_gate = torch.sigmoid((high - x) / smooth)
        return lower_gate * upper_gate

    def forward(self, x):
        """Apply region-specific gains."""
        # Create soft masks
        mask1 = self.soft_region_mask(x, 0.00, 0.25)  # Shadows
        mask2 = self.soft_region_mask(x, 0.25, 0.50)  # Mid-shadows
        mask3 = self.soft_region_mask(x, 0.50, 0.75)  # Mid-highlights
        mask4 = self.soft_region_mask(x, 0.75, 1.00)  # Highlights

        # Normalize masks
        total = mask1 + mask2 + mask3 + mask4 + 1e-7
        mask1, mask2, mask3, mask4 = mask1 / total, mask2 / total, mask3 / total, mask4 / total

        # Clamp gains to reasonable range [0.5, 3.0]
        g1 = torch.clamp(self.shadow_gain, 0.5, 3.0)
        g2 = torch.clamp(self.midshadow_gain, 0.5, 3.0)
        g3 = torch.clamp(self.midhigh_gain, 0.5, 3.0)
        g4 = torch.clamp(self.highlight_gain, 0.5, 3.0)

        # Apply gains
        out = (x * g1 * mask1 +
               x * g2 * mask2 +
               x * g3 * mask3 +
               x * g4 * mask4)

        return torch.clamp(out, 0.0, 1.0)


# =============================================================================
# Hyper-ISP Adapter - Simplified Version with Mu-Law
# =============================================================================

@MODELS.register_module()
class HyperISPAdapter(nn.Module):
    """
    Simplified Hyper-ISP with Global Mu-Law + Multi-Region Mapping.

    Pipeline:
    1. Global Mu-Law (Expands dark range globally)
    2. Multi-region tone mapping (Fine-tunes specific bands)
    3. Contrast enhancement
    4. Denoising
    """

    def __init__(
            self,
            bit_depth=14,
            output_channels=3,
            print_interval=200,
            enable_diagnostics=True,
            use_denoise=True,
            use_contrast_boost=True,
            use_mulaw=True,  # New Flag
    ):
        super().__init__()
        self.max_value = float(2 ** bit_depth - 1)
        self.print_interval = print_interval
        self.enable_diagnostics = enable_diagnostics
        self.use_denoise = use_denoise
        self.use_contrast_boost = use_contrast_boost
        self.use_mulaw = use_mulaw
        self.iter_count = 0

        # --- Stage 1: Global Mu-Law ---
        if self.use_mulaw:
            # We learn 'rho'. Actual mu = exp(rho) to ensure positivity.
            # Init at 3.0 -> exp(3.0) approx 20.0 (Moderate expansion)
            self.mu_param = nn.Parameter(torch.tensor([1.3]))

            # --- Stage 2: Multi-region tone mapping ---
        self.tone_map = SimpleToneMap()

        # --- Stage 3: Contrast enhancement ---
        if use_contrast_boost:
            self.contrast_gain = nn.Parameter(torch.tensor([1.2]))

        # --- Stage 4: Denoising ---
        if use_denoise:
            self.blur = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
            nn.init.constant_(self.blur.weight, 1.0 / 9.0)
            self.blur.weight.requires_grad = False
            self.denoise_strength = nn.Parameter(torch.tensor([0.3]))

        self._reset_stats()

    # ---------------------------------------------------------------------

    def _reset_stats(self):
        self.stats = {k: [] for k in [
            "in_mean", "in_std", "in_min", "in_max",
            "mu_val", "tone_mean", "tone_std",
            "contrast_mean", "contrast_std",
            "out_mean", "out_std", "out_min", "out_max",
            "shadow_gain", "midshadow_gain", "midhigh_gain", "highlight_gain",
            "contrast_gain", "denoise_strength"
        ]}

    # ---------------------------------------------------------------------

    def forward(self, x):
        # Normalize to [0, 1]
        if self.max_value > 1.0 and x.max() > 1.0:
            x = x / self.max_value
        x = torch.clamp(x, 0.0, 1.0)

        if self.training and self.enable_diagnostics:
            with torch.no_grad():
                self.stats["in_mean"].append(x.mean().item())
                self.stats["in_std"].append(x.std().item())
                self.stats["in_min"].append(x.min().item())
                self.stats["in_max"].append(x.max().item())

        # ====================================================
        # Stage 0: Global Mu-Law (Pre-conditioner)
        # ====================================================
        feat = x
        if self.use_mulaw:
            # Enforce mu > 0
            mu = torch.exp(self.mu_param)
            # Apply: log(1 + mu*x) / log(1 + mu)
            feat = torch.log1p(mu * feat) / torch.log1p(mu)

            if self.training and self.enable_diagnostics:
                with torch.no_grad():
                    self.stats["mu_val"].append(mu.item())

        # ====================================================
        # Stage 1: Multi-region tone mapping
        # ====================================================
        feat = self.tone_map(feat)

        if self.training and self.enable_diagnostics:
            with torch.no_grad():
                self.stats["tone_mean"].append(feat.mean().item())
                self.stats["tone_std"].append(feat.std().item())
                self.stats["shadow_gain"].append(self.tone_map.shadow_gain.item())
                self.stats["midshadow_gain"].append(self.tone_map.midshadow_gain.item())
                self.stats["midhigh_gain"].append(self.tone_map.midhigh_gain.item())
                self.stats["highlight_gain"].append(self.tone_map.highlight_gain.item())

        # ====================================================
        # Stage 2: Contrast enhancement
        # ====================================================
        if self.use_contrast_boost:
            feat_mean = feat.mean(dim=(2, 3), keepdim=True)
            contrast_gain = torch.clamp(self.contrast_gain, 0.8, 2.0)
            feat = (feat - feat_mean) * contrast_gain + feat_mean
            feat = torch.clamp(feat, 0.0, 1.0)

            if self.training and self.enable_diagnostics:
                with torch.no_grad():
                    self.stats["contrast_mean"].append(feat.mean().item())
                    self.stats["contrast_std"].append(feat.std().item())
                    self.stats["contrast_gain"].append(contrast_gain.item())

        # ====================================================
        # Stage 3: Denoising
        # ====================================================
        if self.use_denoise:
            denoise_str = torch.clamp(torch.sigmoid(self.denoise_strength), 0.0, 0.8)
            blurred = self.blur(feat)
            feat = feat + (blurred - feat) * denoise_str

            if self.training and self.enable_diagnostics:
                with torch.no_grad():
                    self.stats["denoise_strength"].append(denoise_str.item())

        out = torch.clamp(feat, 0.0, 1.0)

        if self.training and self.enable_diagnostics:
            with torch.no_grad():
                self.stats["out_mean"].append(out.mean().item())
                self.stats["out_std"].append(out.std().item())
                self.stats["out_min"].append(out.min().item())
                self.stats["out_max"].append(out.max().item())

            self.iter_count += 1
            if self.iter_count % self.print_interval == 0:
                self._print_diagnostics()
                self._reset_stats()

        return [out]

    # ---------------------------------------------------------------------

    def _print_diagnostics(self):
        def avg(k):
            vals = self.stats.get(k, [])
            return sum(vals) / max(len(vals), 1) if vals else 0.0

        in_mean = avg("in_mean")
        tone_mean = avg("tone_mean")
        contrast_mean = avg("contrast_mean") if self.use_contrast_boost else tone_mean
        out_mean = avg("out_mean")

        in_std = avg("in_std")
        tone_std = avg("tone_std")
        contrast_std = avg("contrast_std") if self.use_contrast_boost else tone_std
        out_std = avg("out_std")

        brightness_ratio = out_mean / (in_mean + 1e-6)
        contrast_ratio = out_std / (in_std + 1e-6)

        print("\n" + "=" * 80)
        print(f"[Hyper-ISP Minimal + Mu-Law] Iteration {self.iter_count}")
        print("=" * 80)

        print("Pipeline:")
        print(f"  Input                 : {in_mean:.4f} (std: {in_std:.4f})")
        if self.use_mulaw:
            print(f"  Mu-Law Strength (μ)   : {avg('mu_val'):.1f}")
        print(f"  After Tone Mapping    : {tone_mean:.4f} (std: {tone_std:.4f})")
        if self.use_contrast_boost:
            print(f"  After Contrast Boost  : {contrast_mean:.4f} (std: {contrast_std:.4f})")
        print(f"  Final Output          : {out_mean:.4f} (std: {out_std:.4f})")
        print(f"  Total Brightness      : {brightness_ratio:.2f}x", end="")

        if 1.1 <= brightness_ratio <= 1.8:
            print(" ✓ EXCELLENT")
        elif 0.95 <= brightness_ratio <= 2.2:
            print(" ✓ GOOD")
        else:
            print(" ⚠️  Check values")

        print("-" * 80)
        print("Contrast:")
        print(f"  Ratio       : {contrast_ratio:.2f}x", end="")

        if 0.9 <= contrast_ratio <= 1.4:
            print(" ✓ EXCELLENT (preserved/enhanced)")
        else:
            print(" ⚠️  Check values")

        print("-" * 80)
        print("Learned Region Gains:")
        print(f"  Shadows [0.0-0.25]    : {avg('shadow_gain'):.3f}x")
        print(f"  Mid-shadows [0.25-0.50]: {avg('midshadow_gain'):.3f}x")
        print(f"  Mid-high [0.50-0.75]   : {avg('midhigh_gain'):.3f}x")
        print(f"  Highlights [0.75-1.0]  : {avg('highlight_gain'):.3f}x")

        if self.use_contrast_boost:
            print(f"  Contrast Gain          : {avg('contrast_gain'):.3f}x")

        print("=" * 80 + "\n")