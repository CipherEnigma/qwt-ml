# Validation Scripts for Python 3.10/3.11 Upgrade

This directory contains validation scripts to verify the Python 3.10/3.11 upgrade for mmdetection.

## Scripts

### 1. validate_installation.py

Validates that the installation environment is correctly set up.

**Checks:**
- Python version (3.10 or 3.11)
- Core dependencies (torch, torchvision, mmcv, numpy, etc.)
- Optional dependencies (onnx, onnxruntime, isort, pytest)
- CUDA availability
- mmdet installation

**Usage:**
```bash
python tools/validate_installation.py
```

**Expected Output:**
- ✓ PASS: All checks passed
- ❌ FAIL: Some checks failed (with details)

### 2. validate_imports.py

Validates that all mmdet modules can be imported without errors.

**Checks:**
- mmcv version compatibility
- Core mmdet module imports
- Model submodule imports
- Specific model implementations (Swin Transformer, ResNet, etc.)
- Quantization support
- Deprecation warnings

**Usage:**
```bash
python tools/validate_imports.py
```

**Expected Output:**
- ✓ PASS: All imports working correctly
- ❌ FAIL: Some imports failed (with details)

## Testing on Different Python Versions

To test the upgrade on Python 3.10 and 3.11:

### Using conda:

```bash
# Test on Python 3.10
conda create -n mmdet-py310 python=3.10
conda activate mmdet-py310
pip install torch>=1.13.0 torchvision>=0.14.0
pip install -U openmim
mim install "mmcv>=2.0.0,<2.2.0"
pip install -v -e .
python tools/validate_installation.py
python tools/validate_imports.py

# Test on Python 3.11
conda create -n mmdet-py311 python=3.11
conda activate mmdet-py311
pip install torch>=1.13.0 torchvision>=0.14.0
pip install -U openmim
mim install "mmcv>=2.0.0,<2.2.0"
pip install -v -e .
python tools/validate_installation.py
python tools/validate_imports.py
```

### Using pyenv:

```bash
# Test on Python 3.10
pyenv install 3.10.13
pyenv virtualenv 3.10.13 mmdet-py310
pyenv activate mmdet-py310
pip install torch>=1.13.0 torchvision>=0.14.0
pip install -U openmim
mim install "mmcv>=2.0.0,<2.2.0"
pip install -v -e .
python tools/validate_installation.py
python tools/validate_imports.py

# Test on Python 3.11
pyenv install 3.11.7
pyenv virtualenv 3.11.7 mmdet-py311
pyenv activate mmdet-py311
pip install torch>=1.13.0 torchvision>=0.14.0
pip install -U openmim
mim install "mmcv>=2.0.0,<2.2.0"
pip install -v -e .
python tools/validate_installation.py
python tools/validate_imports.py
```

## Interpreting Results

### Installation Validation

**All checks passed:**
- Environment is correctly set up
- All dependencies are installed with correct versions
- Ready to use mmdet

**Some checks failed:**
- Review the error messages
- Install missing dependencies
- Check version constraints
- Refer to README.md for installation instructions

### Import Validation

**All checks passed:**
- All modules import successfully
- No deprecation warnings
- Quantization support is available
- Ready to run models

**Some checks failed:**
- Review import errors
- Check for missing dependencies
- Look for deprecation warnings
- Refer to MIGRATION.md for upgrade guidance

## Troubleshooting

### Python version check fails
- Ensure you're using Python 3.10 or 3.11
- Create a new virtual environment with the correct Python version

### mmcv not found
- Install mmcv: `mim install "mmcv>=2.0.0,<2.2.0"`
- Or: `pip install mmcv>=2.0.0,<2.2.0`

### mmdet not installed
- Install in development mode: `pip install -v -e .`
- Or: `python setup.py develop`

### CUDA not available
- This is expected on CPU-only systems
- mmdet will work in CPU-only mode
- For GPU support, install CUDA-enabled PyTorch

### Deprecation warnings
- Review the warning messages
- Check if any deprecated Python features are used
- Update code to use modern equivalents

## CI/CD Integration

These scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Validate Installation
  run: python tools/validate_installation.py

- name: Validate Imports
  run: python tools/validate_imports.py
```

## Requirements

Both scripts require:
- Python 3.10 or 3.11 (or higher with warning)
- packaging library (for version comparison)

The import validation script additionally requires:
- mmdet to be installed
- mmcv to be installed
