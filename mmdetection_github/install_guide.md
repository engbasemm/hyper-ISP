# Installation Guide

This document provides detailed installation instructions for the Simplified Physics-Informed ISP project.

---

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 18.04/20.04/22.04), macOS, or Windows with WSL2
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **CUDA**: 11.1 or higher
- **Python**: 3.8, 3.9, or 3.10

### Hardware Requirements
- **Minimum**: 8GB RAM, 4GB VRAM
- **Recommended**: 16GB RAM, 8GB+ VRAM
- **For Training**: 32GB RAM, 12GB+ VRAM

---

## 🔧 Step-by-Step Installation

### Option 1: Installation with Conda (Recommended)

#### Step 1: Create Conda Environment

```bash
# Create a new conda environment
conda create -n simplified-isp python=3.8 -y
conda activate simplified-isp
```

#### Step 2: Install PyTorch

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to get the installation command for your system.

**For Linux with CUDA 11.8:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**For Linux with CUDA 12.1:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Verify installation:**
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

#### Step 3: Install MMCV and MMDetection

```bash
# Install OpenMIM
pip install -U openmim

# Install mmengine and mmcv
mim install mmengine
mim install "mmcv>=2.0.0"

# Install mmdet
mim install "mmdet>=3.0.0"
```

#### Step 4: Clone and Install This Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/simplified-isp.git
cd simplified-isp

# Install other dependencies
pip install -r requirements.txt
```

#### Step 5: Verify Installation

```bash
# Check MMDetection installation
python -c "import mmdet; print(mmdet.__version__)"

# Run a simple test
python tools/test_installation.py
```

---

### Option 2: Installation with pip (Alternative)

#### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 2: Install PyTorch

```bash
# For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended for training)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 3: Install MMDetection Stack

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
```

#### Step 4: Install Project

```bash
git clone https://github.com/yourusername/simplified-isp.git
cd simplified-isp
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

### LOD Dataset

1. **Download LOD Dataset**
   - Visit: [LOD Dataset GitHub](https://github.com/ying-fu/LODDataset)
   - Download the RAW_dark images and annotations

2. **Organize Directory Structure**
```
data/LOD_BMVC2021/
├── RAW_dark/
│   ├── train/
│   └── val/
├── RAW-dark-Annotations/
│   ├── train/
│   └── val/
└── trainval/
    ├── train.txt
    └── val.txt
```

3. **Update Config Path**
```python
# In configs/LOD/hyper_isp_retinanet_lod.py
data_root = '../data/LOD_BMVC2021/'
```

### PASCAL RAW Dataset

1. **Download PASCAL RAW**
   - Visit: [PASCAL RAW GitHub](https://github.com/cuiziteng/ECCV_RAW_Adapter)
   - Download Normal-Light, Low-Light, and Over-Exposure splits

2. **Organize Directory Structure**
```
data/PASCAL_RAW/
├── original/
│   ├── demosaic_ll/      # Low-light
│   ├── demosaic_nl/      # Normal-light
│   └── demosaic_oe/      # Over-exposure
├── annotations/
└── trainval/
    ├── train.txt
    └── val.txt
```

3. **Update Config Paths**
```python
# In configs/PASCALRAW_Res50/Low_Light_hyper_isp_res50.py
data_root = '../data/PASCAL_RAW/'
img_subdir = 'original/demosaic_ll'
```

---

## 🧪 Testing Installation

### Quick Test Script

Create `tools/test_installation.py`:

```python
import torch
import mmdet
import mmcv
import mmengine

print("=" * 50)
print("Installation Test")
print("=" * 50)

# Check PyTorch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check MMDetection stack
print(f"\nMMEngine version: {mmengine.__version__}")
print(f"MMCV version: {mmcv.__version__}")
print(f"MMDetection version: {mmdet.__version__}")

# Test HyperISPAdapter
try:
    from mmdet.models import build_backbone
    print("\n✅ HyperISPAdapter can be imported")
except ImportError as e:
    print(f"\n❌ HyperISPAdapter import failed: {e}")

print("\n" + "=" * 50)
print("Installation test complete!")
print("=" * 50)
```

Run:
```bash
python tools/test_installation.py
```

Expected output:
```
==================================================
Installation Test
==================================================
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 4070 Ti SUPER

MMEngine version: 0.10.7
MMCV version: 2.1.0
MMDetection version: 3.2.0

✅ HyperISPAdapter can be imported

==================================================
Installation test complete!
==================================================
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Solution**: Reduce batch size in config
```python
# In your config file
train_dataloader = dict(
    batch_size=4,  # Reduce from 8
    ...
)
```

### Issue 2: MMDetection Import Error

**Solution**: Reinstall MMDetection
```bash
pip uninstall mmdet -y
mim install "mmdet>=3.0.0"
```

### Issue 3: CUDA Version Mismatch

**Solution**: Reinstall PyTorch with correct CUDA version
```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 4: HyperISPAdapter Not Found

**Solution**: Ensure the module is in the correct path
```bash
# The file should be at:
mmdetection/mmdet/models/backbones/hyper_isp_adapter.py

# And registered in:
mmdetection/mmdet/models/backbones/__init__.py
```

---

## 🔄 Updating

To update to the latest version:

```bash
cd simplified-isp
git pull origin main
pip install -r requirements.txt --upgrade
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check Issues**: [GitHub Issues](https://github.com/yourusername/simplified-isp/issues)
2. **MMDetection Docs**: [MMDetection Documentation](https://mmdetection.readthedocs.io/)
3. **Create New Issue**: Include:
   - Python version
   - PyTorch version
   - CUDA version
   - Error messages
   - Steps to reproduce

---

## ✅ Next Steps

After successful installation:

1. **Prepare Datasets**: Follow the Dataset Preparation section
2. **Run Training**: See [README.md](README.md) for training commands
3. **Evaluate Models**: See [EVALUATION.md](EVALUATION.md) for testing
4. **Customize**: See [CUSTOMIZATION.md](CUSTOMIZATION.md) for advanced usage

---

<p align="center">
  Good luck with your installation! 🚀
</p>
