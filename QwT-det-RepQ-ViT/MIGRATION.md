# Migration Guide: Python 3.10/3.11 Upgrade

This guide helps you migrate from the older version of this project (Python 3.6-3.9, mmcv 1.x, mmdet 2.x) to the new version (Python 3.10+, mmcv 2.x, mmdet 3.x).

## Overview of Changes

### Major Version Updates

| Component | Old Version | New Version | Reason |
|-----------|-------------|-------------|--------|
| Python | 3.6-3.9 | 3.10-3.11 | Security updates, modern features, better performance |
| mmcv | 1.2.4-1.4.0 | 2.0.0-2.1.0 | Python 3.10+ compatibility, bug fixes |
| mmdet | 2.11.0 | 3.x (compatible) | API improvements, Python 3.10+ support |
| PyTorch | <1.13.0 | ≥1.13.0 | Python 3.10+ support, performance improvements |
| numpy | unspecified | 1.20.0-1.26.x | Python 3.10+ compatibility |
| onnx | 1.7.0 | ≥1.12.0 | Python 3.10+ support |
| onnxruntime | 1.5.1 | ≥1.13.0 | Python 3.10+ support |
| isort | 4.3.21 | ≥5.10.0 | Python 3.10+ compatibility |

## Breaking Changes

### 1. Python Version Requirement

**Old**: Python 3.6, 3.7, 3.8, or 3.9  
**New**: Python 3.10 or 3.11 (minimum 3.10)

**Impact**: You must upgrade your Python environment. Python 3.9 and below are no longer supported.

**Action Required**: Create a new environment with Python 3.10 or 3.11.

### 2. MMCV Package Name and Version

**Old**: `mmcv-full==1.2.4` to `1.4.0`  
**New**: `mmcv>=2.0.0,<2.2.0`

**Impact**: 
- The package name changed from `mmcv-full` to `mmcv`
- Major version bump means API changes
- Installation method changed to use `mim` (recommended)

**Action Required**: Uninstall old mmcv and install new version using mim.

### 3. PyTorch Version Requirement

**Old**: PyTorch version unspecified (likely <1.13.0)  
**New**: PyTorch ≥1.13.0

**Impact**: Older PyTorch versions don't support Python 3.10+

**Action Required**: Upgrade PyTorch to 1.13.0 or higher.

### 4. Dependency Version Updates

Several dependencies have been updated to support Python 3.10+:
- `onnx`: 1.7.0 → ≥1.12.0
- `onnxruntime`: 1.5.1 → ≥1.13.0
- `isort`: 4.3.21 → ≥5.10.0
- `pytest`: unspecified → ≥7.0.0

**Impact**: Older versions may not work with Python 3.10+

**Action Required**: These will be automatically updated when you reinstall dependencies.

### 5. Collections Module Changes

**Old**: `from collections import Iterable, Mapping`  
**New**: `from collections.abc import Iterable, Mapping`

**Impact**: This is handled internally by the codebase. No user action required unless you've written custom extensions.

**Action Required**: If you have custom code that imports from `collections`, update to `collections.abc`.

## Migration Steps

### Option 1: Clean Installation (Recommended)

This is the safest approach and avoids dependency conflicts.

#### Step 1: Backup Your Work

```bash
# Backup your config files, checkpoints, and any custom code
cp -r configs/ configs_backup/
cp -r pretrained_weights/ pretrained_weights_backup/
```

#### Step 2: Create New Environment

```bash
# Deactivate current environment
conda deactivate

# Create new environment with Python 3.10
conda create -n qwt-py310 python=3.10
conda activate qwt-py310
```

Or with Python 3.11:

```bash
conda create -n qwt-py311 python=3.11
conda activate qwt-py311
```

#### Step 3: Install PyTorch

Install PyTorch 1.13.0 or higher. Choose the appropriate command for your system from [PyTorch's website](https://pytorch.org/get-started/locally/).

Example for CUDA 11.8:
```bash
pip install torch>=1.13.0 torchvision>=0.14.0 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only:
```bash
pip install torch>=1.13.0 torchvision>=0.14.0
```

#### Step 4: Install MMCV

```bash
pip install -U openmim
mim install "mmcv>=2.0.0,<2.2.0"
```

#### Step 5: Install MMDetection

```bash
cd QwT/detection
pip install -v -e .
```

#### Step 6: Verify Installation

```bash
# Check Python version
python --version  # Should show 3.10.x or 3.11.x

# Check PyTorch version
python -c "import torch; print(torch.__version__)"  # Should be ≥1.13.0

# Check mmcv version
python -c "import mmcv; print(mmcv.__version__)"  # Should be ≥2.0.0

# Test import
python -c "import mmdet; print(mmdet.__version__)"
```

#### Step 7: Restore Your Data

```bash
# Your config files and checkpoints should still work
# No changes needed to config files or checkpoint formats
```

### Option 2: In-Place Upgrade (Advanced)

This approach upgrades your existing environment. Use with caution as it may cause conflicts.

#### Step 1: Backup Your Environment

```bash
# Export current environment
conda env export > environment_backup.yml
```

#### Step 2: Upgrade Python

```bash
# This may not work in all cases; clean installation is safer
conda install python=3.10
```

#### Step 3: Uninstall Old Packages

```bash
pip uninstall mmcv mmcv-full mmdet torch torchvision
```

#### Step 4: Install New Versions

Follow steps 3-6 from Option 1 above.

## API Changes

### MMCV 2.x API Changes

Most mmcv 1.x APIs are preserved in 2.x, but some have been deprecated or changed:

#### Config System

**Old**:
```python
from mmcv import Config
cfg = Config.fromfile('config.py')
```

**New** (still works, but new API available):
```python
from mmcv import Config
cfg = Config.fromfile('config.py')
# Or use the new API:
from mmengine.config import Config
cfg = Config.fromfile('config.py')
```

**Impact**: Existing code should continue to work. The new `mmengine.config` API is recommended for new code.

#### Registry System

The registry system has been updated but maintains backward compatibility for most use cases.

**Impact**: Custom registered modules should continue to work without changes.

### MMDetection API Changes

The quantization functionality (QwT) has been preserved and should work identically to the previous version.

**No changes required** for:
- Model configuration files
- Checkpoint loading
- Evaluation scripts
- Quantization parameters (--w_bit, --a_bit)

## Compatibility Matrix

### Supported Configurations

| Python | PyTorch | mmcv | mmdet | Status |
|--------|---------|------|-------|--------|
| 3.10 | 1.13.0+ | 2.0.0-2.1.0 | 3.x | ✅ Fully Supported |
| 3.11 | 1.13.0+ | 2.0.0-2.1.0 | 3.x | ✅ Fully Supported |
| 3.9 | Any | Any | Any | ❌ Not Supported |
| 3.10 | <1.13.0 | Any | Any | ❌ Not Supported |
| 3.10 | 1.13.0+ | <2.0.0 | Any | ❌ Not Supported |

### Platform Support

- **Linux**: Fully supported (Ubuntu 18.04+, CentOS 7+)
- **macOS**: Supported (CPU-only or with MPS acceleration on Apple Silicon)
- **Windows**: Supported (with appropriate C++ compiler for extensions)

## Troubleshooting Migration Issues

### Issue: "No module named 'mmcv.cnn'"

**Cause**: mmcv not properly installed or wrong version

**Solution**:
```bash
pip uninstall mmcv mmcv-full
mim install "mmcv>=2.0.0,<2.2.0"
```

### Issue: "ImportError: cannot import name 'Config' from 'mmcv'"

**Cause**: mmcv version mismatch

**Solution**: Ensure mmcv 2.0.0+ is installed:
```bash
python -c "import mmcv; print(mmcv.__version__)"
```

### Issue: "CUDA extension compilation failed"

**Cause**: Incompatible CUDA toolkit or compiler

**Solution**:
1. Ensure CUDA toolkit matches PyTorch CUDA version
2. Install appropriate C++ compiler
3. Or install without CUDA extensions:
```bash
pip install -v -e . --no-build-isolation
```

### Issue: "RuntimeError: MMCV version is incompatible"

**Cause**: mmcv version outside supported range

**Solution**: Install the correct mmcv version:
```bash
mim install "mmcv>=2.0.0,<2.2.0"
```

### Issue: Evaluation results differ from previous version

**Cause**: Potential numerical differences due to PyTorch/CUDA version changes

**Solution**: 
- Verify you're using the same checkpoint files
- Check that COCO dataset is correctly configured
- Small numerical differences (<0.1 mAP) are expected due to library updates
- Large differences indicate a configuration issue

### Issue: "AttributeError" when loading checkpoints

**Cause**: Checkpoint format incompatibility (rare)

**Solution**: Checkpoints should be compatible. If you encounter this:
1. Verify the checkpoint file is not corrupted
2. Ensure you're using the correct config file for the checkpoint
3. Check that the model architecture hasn't changed

## Rollback Instructions

If you need to rollback to the previous version:

### Step 1: Restore Old Environment

```bash
# If you backed up your environment
conda env create -f environment_backup.yml -n qwt-old

# Or create from scratch
conda create -n qwt-old python=3.8
conda activate qwt-old
```

### Step 2: Install Old Versions

```bash
# Install PyTorch (older version)
pip install torch==1.10.0 torchvision==0.11.0

# Install old mmcv
pip install mmcv-full==1.3.0

# Install old mmdet
cd QwT/detection
git checkout <old-commit-hash>  # If you have version control
pip install -v -e .
```

## Testing Your Migration

After migration, verify everything works:

### 1. Import Test

```bash
python -c "import mmdet; import mmcv; print('Success')"
```

### 2. Config Loading Test

```bash
python -c "from mmcv import Config; cfg = Config.fromfile('configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'); print('Config loaded successfully')"
```

### 3. Model Instantiation Test

```python
from mmcv import Config
from mmdet.models import build_detector

cfg = Config.fromfile('configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py')
model = build_detector(cfg.model)
print("Model built successfully")
```

### 4. Evaluation Test

Run a quick evaluation to ensure everything works:

```bash
# Use a small subset of data for quick testing
CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh \
    configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py \
    pretrained_weights/mask_rcnn_swin_tiny_patch4_window7.pth \
    1 --eval bbox segm --w_bit 4 --a_bit 4
```

## Getting Help

If you encounter issues during migration:

1. **Check this guide**: Review the troubleshooting section above
2. **Check documentation**: 
   - [mmcv 2.x documentation](https://mmcv.readthedocs.io/)
   - [mmdetection documentation](https://mmdetection.readthedocs.io/)
3. **Search existing issues**: Check if others have encountered the same problem
4. **Open an issue**: Provide:
   - Python version
   - PyTorch version
   - mmcv version
   - mmdet version
   - Full error message and stack trace
   - Steps to reproduce

## Summary

The migration to Python 3.10/3.11 and mmcv 2.x brings:

**Benefits**:
- ✅ Modern Python features and security updates
- ✅ Better performance
- ✅ Long-term support
- ✅ Compatibility with latest libraries

**Preserved**:
- ✅ All quantization functionality (QwT)
- ✅ Model checkpoints compatibility
- ✅ Config file format
- ✅ Evaluation scripts and commands
- ✅ Results reproducibility

**Required Changes**:
- ⚠️ Python 3.10+ required
- ⚠️ PyTorch 1.13.0+ required
- ⚠️ mmcv 2.0.0+ required
- ⚠️ New environment recommended

The migration is straightforward if you follow the clean installation approach. Your existing checkpoints, config files, and quantization functionality will continue to work without modification.
