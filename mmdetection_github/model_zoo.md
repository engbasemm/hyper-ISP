# Model Zoo

This document provides pretrained models and their performance on various benchmarks.

---

## 📊 Available Models

### LOD Dataset (Low-Light Object Detection)

#### RetinaNet with ResNet50

| Backbone | Epochs | mAP | Config | Checkpoint | Log |
|----------|--------|-----|--------|------------|-----|
| ResNet50 | 15 | **62.9** | [config](configs/LOD/hyper_isp_retinanet_lod.py) | [model](https://github.com/yourusername/releases/download/v1.0/lod_retinanet_r50_epoch15.pth) | [log](logs/lod_retinanet_r50.log) |
| ResNet50 | 50 | 63.1 | [config](configs/LOD/hyper_isp_retinanet_lod.py) | [model](https://github.com/yourusername/releases/download/v1.0/lod_retinanet_r50_epoch50.pth) | [log](logs/lod_retinanet_r50_full.log) |

**Performance Details:**
```
Class         | Recall | AP
──────────────────────────────
bicycle       | 84.0%  | 75.4%
car           | 92.7%  | 89.8%
motorbike     | 86.5%  | 56.7%
chair         | 87.9%  | 78.8%
diningtable   | 62.5%  | 35.8%
bottle        | 28.3%  | 23.0%
tvmonitor     | 68.8%  | 56.3%
bus           | 73.1%  | 64.6%
──────────────────────────────
mAP           |        | 62.9%
```

#### SP-RCNN with ResNet50

| Backbone | Epochs | mAP | Config | Checkpoint | Log |
|----------|--------|-----|--------|------------|-----|
| ResNet50 | 20 | **62.2** | [config](configs/LOD/hyper_isp_sprcnn_lod.py) | [model](https://github.com/yourusername/releases/download/v1.0/lod_sprcnn_r50.pth) | [log](logs/lod_sprcnn_r50.log) |

---

### PASCAL RAW - Low-Light Conditions

#### RetinaNet with ResNet18

| Backbone | Epochs | mAP | Config | Checkpoint | Log |
|----------|--------|-----|--------|------------|-----|
| ResNet18 | 9 | **84.5** | [config](configs/PASCALRAW_Res18/Low_Light_hyper_isp_res18.py) | [model](https://github.com/yourusername/releases/download/v1.0/pascal_ll_r18.pth) | [log](logs/pascal_ll_r18.log) |

**Performance Details:**
```
Class    | Recall | AP
─────────────────────────
car      | 90.5%  | 86.7%
person   | 88.6%  | 81.3%
bicycle  | 93.7%  | 85.3%
─────────────────────────
mAP      |        | 84.5%
```

#### RetinaNet with ResNet50

| Backbone | Epochs | mAP | Config | Checkpoint | Log |
|----------|--------|-----|--------|------------|-----|
| ResNet50 | 5 | **86.7** | [config](configs/PASCALRAW_Res50/Low_Light_hyper_isp_res50.py) | [model](https://github.com/yourusername/releases/download/v1.0/pascal_ll_r50.pth) | [log](logs/pascal_ll_r50.log) |

**Performance Details:**
```
Class    | Recall | AP
─────────────────────────
car      | 91.9%  | 89.7%
person   | 88.3%  | 81.4%
bicycle  | 94.9%  | 88.9%
─────────────────────────
mAP      |        | 86.7%
```

---

### PASCAL RAW - Normal-Light Conditions

#### RetinaNet with ResNet18

| Backbone | Epochs | mAP | Config | Checkpoint | Log |
|----------|--------|-----|--------|------------|-----|
| ResNet18 | 10 | **88.9** | [config](configs/PASCALRAW_Res18/Normal_Light_hyper_isp_res18.py) | [model](https://github.com/yourusername/releases/download/v1.0/pascal_nl_r18.pth) | [log](logs/pascal_nl_r18.log) |

**Performance Details:**
```
Class    | Recall | AP
─────────────────────────
car      | 91.9%  | 89.8%
person   | 90.9%  | 87.6%
bicycle  | 95.1%  | 89.4%
─────────────────────────
mAP      |        | 88.9%
```

#### RetinaNet with ResNet50

| Backbone | Epochs | mAP | Config | Checkpoint | Log |
|----------|--------|-----|--------|------------|-----|
| ResNet50 | 9 | **90.3** | [config](configs/PASCALRAW_Res50/Normal_Light_hyper_isp_res50.py) | [model](https://github.com/yourusername/releases/download/v1.0/pascal_nl_r50.pth) | [log](logs/pascal_nl_r50.log) |

**Performance Details:**
```
Class    | Recall | AP
─────────────────────────
car      | 93.3%  | 90.7%
person   | 91.3%  | 89.9%
bicycle  | 97.1%  | 90.3%
─────────────────────────
mAP      |        | 90.3%
```

---

### PASCAL RAW - Over-Exposure Conditions

#### RetinaNet with ResNet18

| Backbone | Epochs | mAP | Config | Checkpoint | Log |
|----------|--------|-----|--------|------------|-----|
| ResNet18 | 12 | **89.0** | [config](configs/PASCALRAW_Res18/Over_Exp_hyper_isp_res18.py) | [model](https://github.com/yourusername/releases/download/v1.0/pascal_oe_r18.pth) | [log](logs/pascal_oe_r18.log) |

**Performance Details:**
```
Class    | Recall | AP
─────────────────────────
car      | 92.2%  | 90.5%
person   | 90.1%  | 87.1%
bicycle  | 93.1%  | 89.3%
─────────────────────────
mAP      |        | 89.0%
```

---

## 📥 How to Use Pretrained Models

### Download Model

```bash
# Using wget
wget https://github.com/engbasemm/hyper-ISP/releases/download/v1.0/lod_retinanet_r50_epoch15.pth \
  -O checkpoints/lod_retinanet_r50_epoch15.pth

# Or using curl
curl -L https://github.com/engbasemm/hyper-ISP/releases/download/v1.0/lod_retinanet_r50_epoch15.pth \
  -o checkpoints/lod_retinanet_r50_epoch15.pth
```

### Inference

```python
from mmdet.apis import init_detector, inference_detector

# Initialize model
config = 'configs/LOD/hyper_isp_retinanet_lod.py'
checkpoint = 'checkpoints/lod_retinanet_r50_epoch15.pth'
model = init_detector(config, checkpoint, device='cuda:0')

# Run inference
img = 'path/to/your/image.png'
result = inference_detector(model, img)

# Visualize
from mmdet.apis import show_result_pyplot
show_result_pyplot(model, img, result, score_thr=0.3)
```

### Evaluation

```bash
# Evaluate on LOD test set
python tools/test.py \
  configs/LOD/hyper_isp_retinanet_lod.py \
  checkpoints/lod_retinanet_r50_epoch15.pth \
  --work-dir work_dirs/eval_lod

# Evaluate on PASCAL RAW Low-Light
python tools/test.py \
  configs/PASCALRAW_Res50/Low_Light_hyper_isp_res50.py \
  checkpoints/pascal_ll_r50.pth \
  --work-dir work_dirs/eval_pascal_ll
```

---

## 🔍 Model Comparison

### vs. RAW-Adapter (SOTA)

| Dataset | Condition | Backbone | RAW-Adapter | Ours | Gain |
|---------|-----------|----------|-------------|------|------|
| LOD | Low-light | R50 | 62.1 | **62.9** | +0.8 |
| LOD | Low-light | SP-RCNN-R50 | 59.2 | **62.2** | +3.0 🚀 |
| PASCAL | Low-light | R18 | 82.5 | **84.5** | +2.0 🚀 |
| PASCAL | Low-light | R50 | 86.6 | **86.7** | +0.1 |
| PASCAL | Normal | R18 | 88.7 | **88.9** | +0.2 |
| PASCAL | Normal | R50 | 89.7 | **90.3** | +0.6 |
| PASCAL | Over-exp | R18 | 88.7 | **89.0** | +0.3 |

**Win Rate: 7/7 (100%)**

---

## 🎓 Training Tips

### For Best Results

1. **Batch Size**: Start with 8, reduce to 4 if OOM
2. **Learning Rate**: 1e-3 works well for most cases
3. **Warmup**: Use 500 iterations for stable training
4. **Epochs**: 
   - PASCAL: 5-15 epochs sufficient
   - LOD: 15-50 epochs recommended

### Hyperparameter Settings

Our best models use these settings:

```python
# Optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500  # Warmup
    ),
]

# Gradient clipping
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
)
```

### ISP Parameters (Learned during training)

Initial values that work well:

```python
shadow_gain = 1.5        # Boost shadows strongly
midshadow_gain = 1.1     # Moderate boost
midhigh_gain = 1.1       # Moderate boost
highlight_gain = 0.8     # Compress highlights
mu_param = 1.3           # μ = exp(1.3) ≈ 3.7
contrast_gain = 1.2      # 20% contrast boost
denoise_strength = 0.3   # Moderate denoising
```

---

## 📈 Training Curves

Example training curve (LOD, ResNet50):

```
Epoch    Loss    mAP
  1     1.789   45.2%
  5     0.845   58.7%
 10     0.421   61.5%
 15     0.289   62.9% ← Converged
 20     0.267   63.0%
```

Notice the fast convergence (< 15 epochs)!

---

## 🔧 Fine-tuning on Custom Dataset

To fine-tune on your own dataset:

1. **Start from LOD checkpoint**:
```bash
python tools/train.py \
  configs/your_custom_config.py \
  --load-from checkpoints/lod_retinanet_r50_epoch15.pth
```

2. **Reduce learning rate**:
```python
optimizer = dict(lr=0.0001)  # 10x smaller
```

3. **Train for fewer epochs**: 5-10 usually sufficient

---

## 📞 Support

For model-specific questions:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/simplified-isp/issues)
- **Email**: your.email@university.edu

---

## 📜 License

All pretrained models are released under [MIT License](LICENSE).

---

<p align="center">
  Happy detecting! 🎯
</p>
