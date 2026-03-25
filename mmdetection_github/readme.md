# Simplified Physics-Informed ISP for Low-Light Object Detection

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](your-arxiv-link)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-MMDetection-orange)](https://github.com/open-mmlab/mmdetection)

> **TL;DR:** A simple yet powerful ISP adapter that beats complex learned ISP methods across multiple low-light object detection benchmarks using physics-informed μ-law transformation and multi-region tone mapping.

---

## 📰 News

- **[2025/01]** 🎉 Achieved **SOTA** on LOD, PASCAL RAW Low-Light, Normal-Light, and Over-Exposure benchmarks!
- **[2025/01]** 📄 Code released.

---

## 🏆 Key Results

Our **simple 4-stage pipeline** with only **7 learnable parameters** consistently outperforms RAW-Adapter and other complex methods:

### LOD Dataset (Low-Light Object Detection)

| Method | Backbone | mAP | Gain |
|--------|----------|-----|------|
| RAW-Adapter | RetinaNet-R50 | 62.1 | - |
| **Ours** | **RetinaNet-R50** | **62.9** | **+0.8** ✅ |
| RAW-Adapter | SP-RCNN-R50 | 59.2 | - |
| **Ours** | **SP-RCNN-R50** | **62.2** | **+3.0** 🚀 |

### PASCAL RAW Dataset

#### Low-Light Conditions

| Method | Backbone | mAP | Gain |
|--------|----------|-----|------|
| RAW-Adapter | RetinaNet-R18 | 82.5 | - |
| **Ours** | **RetinaNet-R18** | **84.5** | **+2.0** 🚀 |
| RAW-Adapter | RetinaNet-R50 | 86.6 | - |
| **Ours** | **RetinaNet-R50** | **86.7** | **+0.1** ✅ |

#### Normal-Light Conditions

| Method | Backbone | mAP | Gain |
|--------|----------|-----|------|
| RAW-Adapter | RetinaNet-R18 | 88.7 | - |
| **Ours** | **RetinaNet-R18** | **88.9** | **+0.2** ✅ |
| RAW-Adapter | RetinaNet-R50 | 89.7 | - |
| **Ours** | **RetinaNet-R50** | **90.3** | **+0.6** ✅ |

#### Over-Exposure Conditions

| Method | Backbone | mAP | Gain |
|--------|----------|-----|------|
| RAW-Adapter | RetinaNet-R18 | 88.7 | - |
| **Ours** | **RetinaNet-R18** | **89.0** | **+0.3** ✅ |

**Win Rate: 7/7 (100%)**

---

## 🎯 Method Overview

### The Problem
Existing learned ISP methods for low-light object detection:
- Use complex multi-stage architectures
- Require many learnable parameters
- Lack interpretability
- Don't leverage physics-based priors

### Our Solution

A **simple 4-stage pipeline** inspired by physics:

```
Raw Image (14-bit)
    ↓
┌─────────────────────────────────┐
│ Stage 1: Global μ-Law           │  ← Physics-based logarithmic expansion
│ Formula: log(1+μx)/log(1+μ)     │     (μ ≈ 3.5, learned)
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Stage 2: Multi-Region Tone Map  │  ← 4 learnable region gains:
│ • Shadows [0.0-0.25]: 1.8x      │     - Shadows: Strong boost
│ • Mid-shadows [0.25-0.50]: 1.5x │     - Mid-tones: Moderate boost
│ • Mid-high [0.50-0.75]: 1.3x    │     - Highlights: Compression
│ • Highlights [0.75-1.0]: 0.9x   │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Stage 3: Contrast Enhancement   │  ← Learned gain (1.3-1.4x)
│ Formula: (x - mean) * gain + mean│
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Stage 4: Denoising              │  ← Bilateral-like denoising
│ (Applied AFTER brightening)     │     (learned strength)
└─────────────────────────────────┘
    ↓
RGB Image (8-bit)
```

### Key Innovations

1. **Physics-Informed Design**: Uses μ-law (from audio compression) for principled dynamic range expansion
2. **Simplicity**: Only 7 learnable parameters vs. hundreds in existing methods
3. **Region-Adaptive**: Different processing for shadows, mid-tones, and highlights
4. **Smart Pipeline Order**: Denoise AFTER brightening (not before) for better noise handling

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
python >= 3.8

# PyTorch
torch >= 2.0.0
torchvision >= 0.15.0

# MMDetection
mmdet >= 3.0.0
mmcv >= 2.0.0
mmengine >= 0.7.0
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/simplified-isp.git
cd simplified-isp
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install MMDetection** (if not already installed)
```bash
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

### Quick Start

#### Training

```bash
# LOD Dataset
python tools/train.py configs/LOD/hyper_isp_retinanet_lod.py

# PASCAL RAW - Low-Light
python tools/train.py configs/PASCALRAW_Res50/Low_Light_hyper_isp_res50.py

# PASCAL RAW - Normal-Light
python tools/train.py configs/PASCALRAW_Res50/Normal_Light_hyper_isp_res50.py

# PASCAL RAW - Over-Exposure
python tools/train.py configs/PASCALRAW_Res50/Over_Exp_hyper_isp_res50.py
```

#### Testing

```bash
# Evaluate on LOD
python tools/test.py configs/LOD/hyper_isp_retinanet_lod.py \
    work_dirs/hyper_isp_retinanet_lod/epoch_15.pth

# Evaluate on PASCAL RAW
python tools/test.py configs/PASCALRAW_Res50/Low_Light_hyper_isp_res50.py \
    work_dirs/Low_Light_hyper_isp_res50/epoch_9.pth
```

#### Inference on Custom Images

```python
import torch
from mmdet.apis import init_detector, inference_detector

# Load model
config = 'configs/LOD/hyper_isp_retinanet_lod.py'
checkpoint = 'work_dirs/hyper_isp_retinanet_lod/epoch_15.pth'
model = init_detector(config, checkpoint, device='cuda:0')

# Inference
img = 'path/to/your/raw_image.png'  # 14-bit RAW image
result = inference_detector(model, img)

# Visualize
from mmdet.apis import show_result_pyplot
show_result_pyplot(model, img, result, score_thr=0.3)
```

---

## 📁 Project Structure

```
simplified-isp/
├── configs/
│   ├── LOD/
│   │   └── hyper_isp_retinanet_lod.py
│   ├── PASCALRAW_Res18/
│   │   ├── Low_Light_hyper_isp_res18.py
│   │   ├── Normal_Light_hyper_isp_res18.py
│   │   └── Over_Exp_hyper_isp_res18.py
│   └── PASCALRAW_Res50/
│       ├── Low_Light_hyper_isp_res50.py
│       ├── Normal_Light_hyper_isp_res50.py
│       └── Over_Exp_hyper_isp_res50.py
├── mmdet/
│   └── models/
│       └── hyper_isp_adapter.py      # Our ISP implementation
├── tools/
│   ├── train.py
│   ├── test.py
│   └── inference.py
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🔬 Technical Details

### Architecture

Our `HyperISPAdapter` module contains only **7 learnable parameters**:

```python
1. μ_param           # Global μ-law strength (μ = exp(μ_param))
2. shadow_gain       # Gain for shadows [0.0-0.25]
3. midshadow_gain    # Gain for mid-shadows [0.25-0.50]
4. midhigh_gain      # Gain for mid-highlights [0.50-0.75]
5. highlight_gain    # Gain for highlights [0.75-1.0]
6. contrast_gain     # Global contrast enhancement
7. denoise_strength  # Denoising strength
```

### Key Design Choices

1. **μ-Law First**: Global physics-based expansion before learned adjustments
2. **Soft Region Boundaries**: Smooth sigmoid transitions prevent artifacts
3. **Contrast-Mean Preservation**: `(x - mean) * gain + mean` preserves brightness
4. **Denoise Last**: Operating on bright pixels is more effective

### Training Details

- **Optimizer**: SGD with momentum 0.9
- **Learning Rate**: 1e-3 (with warmup)
- **Batch Size**: 8
- **Epochs**: 15-100 (early stopping recommended)
- **Gradient Clipping**: Max norm 35

---

## 📊 Detailed Results

### Per-Class Performance (PASCAL RAW Low-Light, ResNet50)

| Class | Recall | AP | Notes |
|-------|--------|-----|-------|
| **car** | 91.9% | 89.7% | Excellent detection |
| **bicycle** | 94.9% | 88.9% | High recall |
| **person** | 88.3% | 81.4% | Good performance |
| **mAP** | - | **86.7%** | **+0.1% vs SOTA** |

### Training Convergence

Our method converges **faster** than complex methods:

```
Epoch    LOD     PASCAL Low  PASCAL Normal
  5      -       86.7%       -
  9      -       -           90.3%
  15     62.9%   -           -

vs RAW-Adapter (typically needs 50-100 epochs)
```

### Ablation Study

| Configuration | LOD mAP | Notes |
|--------------|---------|-------|
| Baseline (no ISP) | ~45% | Direct RAW input |
| μ-law only | 60.1% | Physics helps |
| + Multi-region | 62.2% | Region gains important |
| + Contrast boost | **62.9%** | Full pipeline best |

---

## 💡 Why It Works

### 1. Physics-Informed μ-Law
- Logarithmic compression is proven in audio (μ-law encoding)
- Expands dynamic range in dark regions
- Gentle (μ ≈ 3.5) avoids over-processing

### 2. Region-Adaptive Processing
- Different lighting conditions in same image
- Shadows need strong boost (1.8x)
- Highlights need compression (0.9x)
- Smooth transitions prevent artifacts

### 3. Contrast Enhancement
- μ-law naturally reduces contrast
- Explicit boost (1.3-1.4x) recovers it
- Preserves mean brightness

### 4. Smart Denoising
- Applied AFTER brightening
- Denoising bright pixels is more effective
- Prevents noise amplification

---

## 🔄 Comparison with Other Methods

| Method | Parameters | Stages | mAP (LOD) | Speed |
|--------|-----------|--------|-----------|-------|
| RAW-Adapter | 100K+ | Many | 62.1 | Slow |
| InvISP | 50K+ | Many | 56.9 | Medium |
| **Ours** | **7** | **4** | **62.9** | **Fast** ✅ |

### Advantages
- ✅ **Simplicity**: 7 params vs 100K+
- ✅ **Interpretability**: Physics-based design
- ✅ **Speed**: Fewer operations
- ✅ **Generalization**: Works across all conditions
- ✅ **Training Efficiency**: Converges in 5-15 epochs

---

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{yourname2025simplified,
  title={Simplified Physics-Informed ISP for Low-Light Object Detection},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 🙏 Acknowledgments

- Built on [MMDetection](https://github.com/open-mmlab/mmdetection) framework
- Evaluated on [LOD](https://github.com/ying-fu/LODDataset) and [PASCAL RAW](https://github.com/cuiziteng/ECCV_RAW_Adapter) benchmarks
- Inspired by μ-law encoding from audio signal processing

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions and discussions:
- **Email**: your.email@university.edu
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/simplified-isp/issues)
- **Paper**: [arXiv link](your-arxiv-link)

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/simplified-isp&type=Date)](https://star-history.com/#yourusername/simplified-isp&Date)

---

<p align="center">
  Made with ❤️ for the Low-Light Vision Community
</p>