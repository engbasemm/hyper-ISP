_base_ = ['./Normal_Light_raw_adapter_res18.py']

model = dict(
    backbone=dict(
        # We replace the backbone type with our new Wrapper Class
        # This wrapper handles BOTH the ISP logic AND the ResNet feature extraction
        type='HyperISPResNet',

        # --- Args for the Hyper-ISP part ---
        bit_depth=14,
        isp_out_channels=3,
        isp_mode='dynamic',
        vis_subdir='RES18_PASCAL_Normal_Light_hyper',

        # --- Args for the ResNet part ---
        # The wrapper passes these down to the ResNet superclass
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),

        # --- Cleanup ---
        # We explicitly nullify arguments specific to the OLD adapter (RAW-Adapter)
        # to ensure they don't cause confusion or errors if passed via **kwargs
        ada_c_s=None,
        fea_c_s=None,
        mid_c_s=None,
        lut_dim=None,
        light_mode=None
    )
)

# Standard SGD Optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

# Standard Schedule
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1
    )
]

train_cfg = dict(max_epochs=35, val_interval=1)