import torch
import torch.nn as nn
import os
import torchvision.utils as vutils
from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet
from .hyper_isp_adapter import HyperISPAdapter


@MODELS.register_module()
class HyperISPResNet(ResNet):
    def __init__(self,
                 bit_depth=14,
                 isp_out_channels=3,
                 depth=18,
                 vis_subdir=None,
                 **kwargs):

        # 1. Clean legacy arguments (from old RAW-Adapter config)
        legacy_keys = ['light_mode', 'lut_dim', 'ada_c_s', 'fea_c_s',
                       'mid_c_s', 'k_size', 'merge_ratio', 'w_lut']
        for key in legacy_keys:
            if key in kwargs:
                kwargs.pop(key)

        # 2. Extract Hyper-ISP specific args from kwargs
        # We use .pop() so they are REMOVED from kwargs and not passed to ResNet
        isp_mode = kwargs.pop('isp_mode', 'dynamic')

        # 3. Initialize Standard ResNet
        # Now kwargs only contains valid ResNet arguments (like num_stages, etc.)
        super(HyperISPResNet, self).__init__(depth=depth, in_channels=isp_out_channels, **kwargs)

        # 4. Initialize Hyper-ISP
        self.hyper_isp = HyperISPAdapter(
            bit_depth=bit_depth,
            output_channels=isp_out_channels,
            isp_mode=isp_mode,
        )

        # --- SMART VISUALIZATION PATH ---
        self.vis_count = 0
        self.vis_interval = 200

        # Try to use the provided subdir, or fallback to a generic name
        if vis_subdir:
            self.vis_dir = os.path.join('debug_vis', vis_subdir)
        else:
            self.vis_dir = os.path.join('debug_vis', f'resnet{depth}_{isp_mode}')

        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            os.makedirs(self.vis_dir, exist_ok=True)
            # print(f"[HyperISP] Visualization Enabled. Saving to: {self.vis_dir}")

    def visualize_internals(self, raw, isp_out):
        """
        Saves a comparison of RAW input vs ISP Output.
        """
        # Only rank 0 visualizes
        if int(os.environ.get('LOCAL_RANK', 0)) != 0:
            return

        if self.vis_count % self.vis_interval == 0:
            with torch.no_grad():
                # Normalize RAW [0,1] for vis
                vis_raw = raw[0].detach().cpu()
                vis_raw = (vis_raw - vis_raw.min()) / (vis_raw.max() - vis_raw.min() + 1e-6)

                # Normalize ISP Output [0,1]
                vis_isp = isp_out[0].detach().cpu()
                vis_isp = (vis_isp - vis_isp.min()) / (vis_isp.max() - vis_isp.min() + 1e-6)

                # Stack: Raw (Left), ISP (Right)
                # Concatenate along width (dim=2) for side-by-side
                comparison = torch.cat([vis_raw, vis_isp], dim=2)

                filename = os.path.join(self.vis_dir, f'step_{self.vis_count}.png')
                vutils.save_image(comparison, filename)

        self.vis_count += 1

    def forward(self, x):
        # 1. Apply Neural ISP
        x_isp = self.hyper_isp(x)[0]

        # 2. Visualize (Training only)
        if self.training:
            self.visualize_internals(x, x_isp)

        # 3. Apply ResNet
        outs = super(HyperISPResNet, self).forward(x_isp)

        return outs