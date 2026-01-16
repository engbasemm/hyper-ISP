# Inherit from the ResNet-50 baseline if available. 
# If not, inheriting from Res18 and overriding is fine, but we MUST override everything.
_base_ = ['./Over_Exp_raw_adapter_res50.py']
# Note: Ensure 'Low_Light_raw_adapter_res50.py' exists in this folder. 
# If it doesn't, check if it's named 'LowLight_...' or similar.

model = dict(
    backbone=dict(
        # 1. Use our Wrapper
        type='HyperISPResNet',

        # 2. Hyper-ISP Args
        bit_depth=14,
        isp_out_channels=3,
        isp_mode='dynamic',
        vis_subdir='RES50_PASCAL_Over_Exp_hyper',
        # 3. ResNet-50 Specifics
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',

        # 4. Correct Checkpoint
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),  # <--- Was resnet18

        # 5. Cleanup Legacy Args
        ada_c_s=None,
        fea_c_s=None,
        mid_c_s=None,
        lut_dim=None,
        light_mode=None
    ),
    # Ensure Neck matches ResNet-50 output channels (Expansion=4)
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # ResNet-50 outputs
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5
    )
)

# ResNet-50 is heavier, so we often adjust batch size or LR, 
# but for fair comparison, keep settings as close to baseline as possible.
#optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)



# Optimizer with gradient clipping
# === STABILITY FIXES ===
optim_wrapper = dict(
    _delete_=True,  # <--- THIS IS CRITICAL. It deletes the base SGD config.
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'hyper_isp.lut.lut': dict(lr_mult=0.1, decay_mult=1.0)
        }
    )
)