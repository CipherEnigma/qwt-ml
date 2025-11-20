# Design Document: Python 3.10/3.11 Upgrade

## Overview

This design document outlines the technical approach for upgrading the mmdetection-based quantization project to support Python 3.10 and 3.11. The upgrade involves migrating from mmcv 1.2.4-1.4.0 to mmcv 2.x, updating all dependencies, fixing deprecated Python features, and ensuring the quantization functionality remains intact.

The key challenge is that mmcv 1.x is incompatible with Python 3.10+ due to removed features (e.g., collections.Iterable moved to collections.abc.Iterable) and dependency conflicts. The solution requires upgrading to mmcv 2.x (specifically mmcv>=2.0.0) and mmdet 3.x, which are designed for Python 3.7-3.11 compatibility.

## Architecture

### Current Architecture
```
mmdet 2.11.0
├── mmcv 1.2.4-1.4.0 (incompatible with Python 3.10+)
├── torch (version unspecified, likely <1.13)
├── numpy
├── mmpycocotools
└── Other dependencies (some with outdated versions)
```

### Target Architecture
```
mmdet 3.x (compatible fork or updated version)
├── mmcv>=2.0.0,<2.2.0
├── torch>=1.13.0 (Python 3.10+ support)
├── numpy>=1.20.0,<2.0.0
├── mmpycocotools
├── Updated dependencies (onnx>=1.12.0, onnxruntime>=1.13.0, isort>=5.0.0)
└── Quantization modules (preserved from original)
```

### Migration Strategy

The upgrade will follow a **compatibility-first approach**:

1. **Phase 1**: Update dependency specifications
2. **Phase 2**: Fix mmcv version compatibility checks
3. **Phase 3**: Update deprecated Python code patterns
4. **Phase 4**: Modernize build system
5. **Phase 5**: Validate quantization functionality

## Components and Interfaces

### 1. Dependency Management

#### setup.py
- **Current**: Imports torch at module level, causing installation failures if torch not pre-installed
- **Design**: Wrap torch import in try-except block, make it optional during setup
- **Changes**:
  - Move torch import inside functions that need it
  - Add fallback for when torch is not available during setup
  - Update Python version classifiers to include 3.10 and 3.11
  - Add python_requires='>=3.10' to enforce minimum version

#### requirements/runtime.txt
- **Current**: No version constraints, relies on mmcv 1.x
- **Design**: Add explicit version constraints for Python 3.10+ compatibility
- **Changes**:
  ```
  matplotlib>=3.3.0
  mmpycocotools
  numpy>=1.20.0,<2.0.0
  six
  terminaltables
  timm>=0.6.0
  torch>=1.13.0
  torchvision>=0.14.0
  ```

#### requirements/tests.txt
- **Current**: Uses outdated versions (isort==4.3.21, onnx==1.7.0, onnxruntime==1.5.1)
- **Design**: Update to Python 3.10+ compatible versions
- **Changes**:
  ```
  asynctest; python_version<'3.8'
  codecov
  flake8
  interrogate
  isort>=5.10.0
  kwarray
  onnx>=1.12.0
  onnxruntime>=1.13.0
  pytest>=7.0.0
  ubelt
  xdoctest>=0.10.0
  yapf
  ```

### 2. MMCV Version Compatibility

#### mmdet/__init__.py
- **Current**: Enforces mmcv 1.2.4-1.4.0
- **Design**: Update to accept mmcv 2.x versions
- **Changes**:
  ```python
  mmcv_minimum_version = '2.0.0'
  mmcv_maximum_version = '2.2.0'
  ```
- **Rationale**: mmcv 2.x is the first version series with Python 3.10+ support

#### Version Detection Logic
- **Current**: Simple digit_version function
- **Design**: Enhanced version parsing to handle mmcv 2.x versioning scheme
- **Changes**: Update digit_version to handle rc, post, and dev versions correctly

### 3. Build System Updates

#### setup.py - CUDA Extension Building
- **Current**: Uses torch.utils.cpp_extension directly at import time
- **Design**: Lazy import of torch, graceful degradation
- **Implementation**:
  ```python
  def get_extensions():
      try:
          import torch
          from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
      except ImportError:
          # Allow installation without torch pre-installed
          return []
      
      # Build extensions only if torch is available
      return [...]
  ```

#### Extension Compilation
- **Current**: May have compatibility issues with PyTorch 1.13+
- **Design**: Update compiler flags for modern PyTorch
- **Changes**:
  - Remove deprecated CUDA flags if present
  - Add compatibility checks for CUDA version
  - Ensure C++14 or C++17 standard is used

### 4. Code Compatibility Fixes

#### Collections ABC Imports
- **Pattern**: Any code using `collections.Iterable`, `collections.Mapping`, etc.
- **Fix**: Replace with `collections.abc.Iterable`, `collections.abc.Mapping`
- **Scope**: Search entire codebase for deprecated collections imports

#### Type Hints and Annotations
- **Pattern**: String-based forward references that may cause issues
- **Fix**: Use `from __future__ import annotations` where needed
- **Scope**: Check all type-hinted code

#### Deprecated setuptools Features
- **Pattern**: Use of deprecated setuptools APIs
- **Fix**: Update to modern setuptools patterns
- **Scope**: setup.py and any build scripts

## Data Models

### Version Compatibility Matrix

| Component | Current Version | Target Version | Python 3.10 Support | Python 3.11 Support |
|-----------|----------------|----------------|---------------------|---------------------|
| Python | 3.6-3.8 | 3.10-3.11 | ✓ | ✓ |
| mmcv | 1.2.4-1.4.0 | 2.0.0-2.1.0 | ✓ | ✓ |
| mmdet | 2.11.0 | 3.x (or patched 2.x) | ✓ | ✓ |
| torch | unspecified | ≥1.13.0 | ✓ | ✓ |
| torchvision | unspecified | ≥0.14.0 | ✓ | ✓ |
| numpy | unspecified | 1.20.0-1.26.x | ✓ | ✓ |
| onnx | 1.7.0 | ≥1.12.0 | ✓ | ✓ |
| onnxruntime | 1.5.1 | ≥1.13.0 | ✓ | ✓ |
| isort | 4.3.21 | ≥5.10.0 | ✓ | ✓ |

### Configuration Files

No changes to configuration file formats are expected. The existing config files in `configs/` should work with upgraded mmdet, though some API changes may require minor adjustments.

## Error Handling

### Installation Errors

1. **Python Version Check**
   - **Error**: Installation attempted on Python <3.10
   - **Handling**: setup.py will fail with clear message: "Python 3.10 or higher is required"
   - **Implementation**: `python_requires='>=3.10'` in setup.py

2. **MMCV Version Mismatch**
   - **Error**: Incompatible mmcv version installed
   - **Handling**: Clear error message with installation instructions
   - **Implementation**: Enhanced error message in mmdet/__init__.py

3. **CUDA Extension Build Failure**
   - **Error**: CUDA extensions fail to compile
   - **Handling**: Graceful fallback to CPU-only mode with warning
   - **Implementation**: Try-except in setup.py with informative messages

### Runtime Errors

1. **Import Errors**
   - **Error**: Missing or incompatible dependencies
   - **Handling**: Clear error messages indicating which package needs updating
   - **Implementation**: Add dependency checks in __init__.py

2. **Deprecated API Usage**
   - **Error**: Code uses deprecated mmcv/mmdet APIs
   - **Handling**: Update code to use new APIs, add compatibility shims if needed
   - **Implementation**: API mapping layer for critical functions

## Testing Strategy

### Unit Tests

1. **Import Tests**
   - Test that all core modules can be imported
   - Verify version compatibility checks work correctly
   - Validate that quantization modules load properly

2. **Dependency Tests**
   - Verify all required packages are installed
   - Check version constraints are satisfied
   - Test torch/CUDA availability detection

### Integration Tests

1. **Model Loading**
   - Load a sample Swin Transformer model
   - Verify config parsing works
   - Test checkpoint loading

2. **Quantization Tests**
   - Test W4/A4 quantization can be applied
   - Test W6/A6 quantization can be applied
   - Verify quantized model can be created (no need to run full evaluation)

### Compatibility Tests

1. **Python Version Tests**
   - Test installation on Python 3.10
   - Test installation on Python 3.11
   - Verify no deprecation warnings

2. **Platform Tests**
   - Test on Linux (primary platform)
   - Test with CUDA available
   - Test with CPU-only

### Validation Tests

1. **Smoke Tests**
   - Run dist_test.sh with --help to verify script works
   - Load a config file and verify it parses
   - Import all major modules

2. **Regression Tests**
   - Compare model outputs before/after upgrade (if feasible)
   - Verify quantization produces similar results
   - Check that no functionality is lost

## Migration Path

### For Users

1. **Clean Installation** (Recommended)
   ```bash
   # Create new environment
   conda create -n mmdet-py310 python=3.10
   conda activate mmdet-py310
   
   # Install PyTorch first
   pip install torch>=1.13.0 torchvision>=0.14.0
   
   # Install mmcv
   pip install -U openmim
   mim install "mmcv>=2.0.0,<2.2.0"
   
   # Install mmdet
   cd QwT/detection
   pip install -v -e .
   ```

2. **Upgrade Existing Environment** (Advanced)
   ```bash
   # Uninstall old versions
   pip uninstall mmcv mmcv-full mmdet
   
   # Follow clean installation steps
   ```

### Breaking Changes

1. **MMCV API Changes**
   - Some mmcv 1.x APIs may be deprecated in 2.x
   - Config file format may have minor changes
   - Custom operators may need updates

2. **Python API Changes**
   - Minimum Python version is now 3.10
   - Some deprecated Python patterns removed

### Backward Compatibility

- **Not Maintained**: Python 3.6-3.9 support will be dropped
- **Maintained**: All quantization functionality will work identically
- **Maintained**: Config file format compatibility (with minor exceptions)
- **Maintained**: Checkpoint format compatibility

## Implementation Notes

### Critical Files to Modify

1. **setup.py** - Build system and dependency declarations
2. **mmdet/__init__.py** - Version compatibility checks
3. **requirements/*.txt** - Dependency specifications
4. **README.md** - Installation instructions
5. **Any files with deprecated imports** - Collections, typing, etc.

### Files to Review (May Need Changes)

1. **mmdet/models/backbones/swin_transformer.py** - May use deprecated APIs
2. **configs/_base_/*.py** - May need minor config updates
3. **tools/*.py** - Scripts may need updates for new APIs

### Files Unlikely to Need Changes

1. **configs/swin/*.py** - Config files should mostly work as-is
2. **demo/*.py** - Demo scripts should work with minimal changes
3. **tests/*.py** - Test files may need updates but structure is fine

## Risk Assessment

### High Risk
- **MMCV API compatibility**: mmcv 2.x has breaking changes from 1.x
- **Mitigation**: Thorough testing, create compatibility shims if needed

### Medium Risk
- **CUDA extension compilation**: May fail on some platforms
- **Mitigation**: Make extensions optional, provide pre-built wheels if possible

### Low Risk
- **Python syntax compatibility**: Python 3.10/3.11 are mostly backward compatible
- **Mitigation**: Run automated syntax checkers

### Very Low Risk
- **Quantization functionality**: Core algorithms are independent of mmcv version
- **Mitigation**: Preserve all quantization code unchanged

## Success Criteria

1. ✓ Project installs successfully on Python 3.10 and 3.11
2. ✓ All core modules import without errors
3. ✓ Sample model can be loaded and evaluated
4. ✓ Quantization functionality works (W4/A4, W6/A6)
5. ✓ No deprecation warnings during normal usage
6. ✓ Documentation is updated and accurate
7. ✓ Existing checkpoints can be loaded
8. ✓ Config files work without modification (or with documented changes)
