"""Tests for dependency version compatibility."""
import importlib.metadata
import sys

import pytest


def get_package_version(package_name):
    """Get the version of an installed package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_version_tuple(version_string):
    """Parse version string into tuple of integers for comparison."""
    # Remove any pre-release suffixes (rc, dev, post, etc.)
    import re
    version_clean = re.split(r'[a-zA-Z]', version_string)[0]
    parts = version_clean.split('.')
    return tuple(int(p) for p in parts if p.isdigit())


def test_mmcv_version_installed():
    """Test that mmcv is installed."""
    mmcv_version = get_package_version('mmcv')
    assert mmcv_version is not None, "mmcv is not installed"


def test_mmcv_version_minimum():
    """Test that mmcv version is at least 2.0.0."""
    mmcv_version = get_package_version('mmcv')
    if mmcv_version is None:
        pytest.skip("mmcv is not installed")
    
    version_tuple = parse_version_tuple(mmcv_version)
    assert version_tuple >= (2, 0, 0), (
        f"mmcv version should be >= 2.0.0, but got {mmcv_version}"
    )


def test_mmcv_version_maximum():
    """Test that mmcv version is less than 2.2.0."""
    mmcv_version = get_package_version('mmcv')
    if mmcv_version is None:
        pytest.skip("mmcv is not installed")
    
    version_tuple = parse_version_tuple(mmcv_version)
    assert version_tuple < (2, 2, 0), (
        f"mmcv version should be < 2.2.0, but got {mmcv_version}"
    )


def test_mmcv_version_check_in_mmdet():
    """Test that mmdet's mmcv version check works correctly."""
    import mmdet
    from mmdet import digit_version
    
    # Test that the version check function works
    assert digit_version('2.0.0') > digit_version('1.4.0')
    assert digit_version('2.1.0') > digit_version('2.0.0')
    assert digit_version('2.0.0rc1') < digit_version('2.0.0')
    
    # Check that mmdet has the correct version constraints
    assert hasattr(mmdet, 'mmcv_minimum_version')
    assert hasattr(mmdet, 'mmcv_maximum_version')
    
    # Verify the constraints are for mmcv 2.x
    min_version = digit_version(mmdet.mmcv_minimum_version)
    max_version = digit_version(mmdet.mmcv_maximum_version)
    
    assert min_version >= digit_version('2.0.0'), (
        f"mmcv_minimum_version should be >= 2.0.0, but got {mmdet.mmcv_minimum_version}"
    )
    assert max_version <= digit_version('2.2.0'), (
        f"mmcv_maximum_version should be <= 2.2.0, but got {mmdet.mmcv_maximum_version}"
    )


def test_torch_version_minimum():
    """Test that torch version is at least 1.13.0."""
    torch_version = get_package_version('torch')
    if torch_version is None:
        pytest.skip("torch is not installed")
    
    version_tuple = parse_version_tuple(torch_version)
    assert version_tuple >= (1, 13, 0), (
        f"torch version should be >= 1.13.0 for Python 3.10+ support, but got {torch_version}"
    )


def test_torchvision_version_minimum():
    """Test that torchvision version is at least 0.14.0."""
    torchvision_version = get_package_version('torchvision')
    if torchvision_version is None:
        pytest.skip("torchvision is not installed")
    
    version_tuple = parse_version_tuple(torchvision_version)
    assert version_tuple >= (0, 14, 0), (
        f"torchvision version should be >= 0.14.0 for Python 3.10+ support, but got {torchvision_version}"
    )


def test_numpy_version_compatible():
    """Test that numpy version is compatible with Python 3.10+."""
    numpy_version = get_package_version('numpy')
    if numpy_version is None:
        pytest.skip("numpy is not installed")
    
    version_tuple = parse_version_tuple(numpy_version)
    
    # numpy >= 1.20.0 is required for Python 3.10+
    assert version_tuple >= (1, 20, 0), (
        f"numpy version should be >= 1.20.0 for Python 3.10+ support, but got {numpy_version}"
    )
    
    # numpy < 2.0.0 for compatibility
    assert version_tuple < (2, 0, 0), (
        f"numpy version should be < 2.0.0 for compatibility, but got {numpy_version}"
    )


def test_onnx_version_compatible():
    """Test that onnx version is compatible with Python 3.10+."""
    onnx_version = get_package_version('onnx')
    if onnx_version is None:
        pytest.skip("onnx is not installed (optional dependency)")
    
    version_tuple = parse_version_tuple(onnx_version)
    
    # onnx >= 1.12.0 is required for Python 3.10+
    assert version_tuple >= (1, 12, 0), (
        f"onnx version should be >= 1.12.0 for Python 3.10+ support, but got {onnx_version}"
    )


def test_onnxruntime_version_compatible():
    """Test that onnxruntime version is compatible with Python 3.10+."""
    onnxruntime_version = get_package_version('onnxruntime')
    if onnxruntime_version is None:
        pytest.skip("onnxruntime is not installed (optional dependency)")
    
    version_tuple = parse_version_tuple(onnxruntime_version)
    
    # onnxruntime >= 1.13.0 is required for Python 3.10+
    assert version_tuple >= (1, 13, 0), (
        f"onnxruntime version should be >= 1.13.0 for Python 3.10+ support, but got {onnxruntime_version}"
    )


def test_isort_version_compatible():
    """Test that isort version is compatible with Python 3.10+."""
    isort_version = get_package_version('isort')
    if isort_version is None:
        pytest.skip("isort is not installed (optional dependency)")
    
    version_tuple = parse_version_tuple(isort_version)
    
    # isort >= 5.0.0 is required for Python 3.10+
    assert version_tuple >= (5, 0, 0), (
        f"isort version should be >= 5.0.0 for Python 3.10+ support, but got {isort_version}"
    )


def test_pytest_version_compatible():
    """Test that pytest version is compatible with Python 3.10+."""
    pytest_version = get_package_version('pytest')
    if pytest_version is None:
        pytest.skip("pytest is not installed")
    
    version_tuple = parse_version_tuple(pytest_version)
    
    # pytest >= 7.0.0 is recommended for Python 3.10+
    assert version_tuple >= (7, 0, 0), (
        f"pytest version should be >= 7.0.0 for Python 3.10+ support, but got {pytest_version}"
    )


def test_all_required_packages_installed():
    """Test that all required packages are installed."""
    required_packages = [
        'mmcv',
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'terminaltables',
        'mmpycocotools',
    ]
    
    missing_packages = []
    for package in required_packages:
        if get_package_version(package) is None:
            missing_packages.append(package)
    
    assert len(missing_packages) == 0, (
        f"Missing required packages: {', '.join(missing_packages)}"
    )


def test_version_constraints_enforced():
    """Test that version constraints are properly enforced."""
    # This test verifies that the version checking logic in mmdet/__init__.py works
    import mmdet
    
    # If we got here, mmdet imported successfully, which means version checks passed
    # Now verify that the version check would fail for incompatible versions
    from mmdet import digit_version
    
    # Simulate checking an incompatible mmcv version
    incompatible_version = digit_version('1.4.0')
    min_version = digit_version(mmdet.mmcv_minimum_version)
    
    assert incompatible_version < min_version, (
        "Version check should reject mmcv 1.4.0 as incompatible"
    )
