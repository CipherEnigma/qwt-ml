# Python 3.10/3.11 Compatibility Test Suite

This directory contains comprehensive tests for verifying Python 3.10 and 3.11 compatibility after the upgrade from Python 3.6-3.8 and mmcv 1.x to mmcv 2.x.

## Test Files

### test_python_version.py
Tests for Python version compatibility and deprecated feature removal.

**Tests:**
- `test_python_version_minimum`: Verifies Python version is at least 3.10
- `test_python_version_supported`: Verifies Python version is 3.10 or higher
- `test_no_deprecation_warnings_on_import`: Ensures no deprecation warnings when importing mmdet
- `test_collections_abc_imports`: Verifies collections.abc is used instead of deprecated collections imports
- `test_setup_py_python_requires`: Checks setup.py declares correct python_requires
- `test_setup_py_classifiers`: Verifies setup.py declares Python 3.10 and 3.11 support

**Requirements Validated:** 1.1, 1.2, 1.3, 1.4, 7.1

### test_dependencies.py
Tests for dependency version compatibility.

**Tests:**
- `test_mmcv_version_installed`: Verifies mmcv is installed
- `test_mmcv_version_minimum`: Checks mmcv version is >= 2.0.0
- `test_mmcv_version_maximum`: Checks mmcv version is < 2.2.0
- `test_mmcv_version_check_in_mmdet`: Verifies mmdet's version check logic works
- `test_torch_version_minimum`: Checks torch version is >= 1.13.0
- `test_torchvision_version_minimum`: Checks torchvision version is >= 0.14.0
- `test_numpy_version_compatible`: Verifies numpy version is compatible (>= 1.20.0, < 2.0.0)
- `test_onnx_version_compatible`: Checks onnx version is >= 1.12.0 (optional)
- `test_onnxruntime_version_compatible`: Checks onnxruntime version is >= 1.13.0 (optional)
- `test_isort_version_compatible`: Checks isort version is >= 5.0.0 (optional)
- `test_pytest_version_compatible`: Checks pytest version is >= 7.0.0
- `test_all_required_packages_installed`: Verifies all required packages are present
- `test_version_constraints_enforced`: Tests that version checking logic works correctly

**Requirements Validated:** 2.1, 2.2, 2.4, 7.2

### test_model_loading.py
Smoke tests for model loading and config parsing.

**Tests:**
- `test_swin_config_exists`: Verifies Swin Transformer config files exist
- `test_load_swin_config`: Tests loading a Swin config file
- `test_parse_swin_config_without_errors`: Verifies config parsing works without errors
- `test_instantiate_swin_model_without_weights`: Tests model instantiation without loading weights
- `test_no_import_errors_for_core_modules`: Verifies core mmdet modules can be imported
- `test_config_base_files_exist`: Checks base config files exist and can be loaded
- `test_model_registry_available`: Verifies model registry is populated
- `test_build_simple_config`: Tests building a detector from a minimal config

**Requirements Validated:** 7.3, 7.4

## Running the Tests

### Run all compatibility tests:
```bash
pytest tests/test_compatibility/ -v
```

### Run specific test file:
```bash
pytest tests/test_compatibility/test_python_version.py -v
pytest tests/test_compatibility/test_dependencies.py -v
pytest tests/test_compatibility/test_model_loading.py -v
```

### Run specific test:
```bash
pytest tests/test_compatibility/test_python_version.py::test_python_version_minimum -v
```

### Run with detailed output:
```bash
pytest tests/test_compatibility/ -v --tb=short
```

### Skip xdoctest if not installed:
If you encounter errors about xdoctest, you can temporarily disable it:
```bash
pytest tests/test_compatibility/ -v -p no:xdoctest
```

Or install xdoctest:
```bash
pip install xdoctest
```

## Expected Results

### In a properly configured Python 3.10/3.11 environment with all dependencies:
- All tests in `test_python_version.py` should pass
- All tests in `test_dependencies.py` should pass
- All tests in `test_model_loading.py` should pass

### In an environment without mmcv installed:
- Tests in `test_python_version.py` that don't require mmcv should pass
- Tests in `test_dependencies.py` will skip or fail for missing packages
- Tests in `test_model_loading.py` will fail or skip

### In a Python 3.9 or earlier environment:
- `test_python_version_minimum` will fail (expected behavior)
- This validates that the upgrade enforces Python 3.10+ requirement

## Test Coverage

These tests validate the following aspects of the Python 3.10/3.11 upgrade:

1. **Python Version Enforcement**: Ensures the codebase requires Python 3.10+
2. **Dependency Versions**: Validates all dependencies are at compatible versions
3. **Deprecated Features**: Confirms no deprecated Python features are used
4. **Import Compatibility**: Verifies all modules can be imported without errors
5. **Config Parsing**: Tests that config files can be loaded and parsed
6. **Model Building**: Validates that models can be instantiated from configs
7. **Version Checking**: Ensures mmcv version compatibility checks work correctly

## Integration with CI/CD

These tests should be run as part of the CI/CD pipeline to ensure:
- New code doesn't introduce Python 3.9 or earlier dependencies
- All required packages remain at compatible versions
- No deprecated features are reintroduced
- Model loading continues to work after changes

## Troubleshooting

### ModuleNotFoundError: No module named 'mmcv'
This is expected if mmcv is not installed. Install it with:
```bash
pip install -U openmim
mim install "mmcv>=2.0.0,<2.2.0"
```

### AssertionError: Python 3.10 or higher is required
You're running on Python 3.9 or earlier. Upgrade to Python 3.10 or 3.11:
```bash
conda create -n mmdet-py310 python=3.10
conda activate mmdet-py310
```

### Tests fail with xdoctest errors
Either install xdoctest or run tests with `-p no:xdoctest`:
```bash
pip install xdoctest
# OR
pytest tests/test_compatibility/ -v -p no:xdoctest
```

## Maintenance

When adding new dependencies or changing version requirements:
1. Update the corresponding test in `test_dependencies.py`
2. Update the version compatibility matrix in the design document
3. Run the full test suite to ensure compatibility
4. Update this README if new test files are added
