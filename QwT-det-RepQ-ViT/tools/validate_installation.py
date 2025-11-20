#!/usr/bin/env python3
"""
Installation validation script for mmdetection Python 3.10/3.11 upgrade.

This script validates that:
1. Python version is 3.10 or 3.11
2. All required dependencies are installed with correct versions
3. Version constraints are satisfied
4. CUDA availability is detected correctly
"""

import sys
import subprocess
from importlib.metadata import version, PackageNotFoundError


def check_python_version():
    """Check if Python version is 3.10 or 3.11."""
    print("=" * 60)
    print("Checking Python version...")
    print("=" * 60)
    
    py_version = sys.version_info
    print(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major != 3:
        print("❌ FAIL: Python 3.x is required")
        return False
    
    if py_version.minor < 10:
        print("❌ FAIL: Python 3.10 or higher is required")
        return False
    
    if py_version.minor in [10, 11]:
        print("✓ PASS: Python version is compatible")
        return True
    
    # Python 3.12+ - may work but not officially tested
    print(f"⚠ WARNING: Python 3.{py_version.minor} detected (officially supports 3.10/3.11)")
    print("  The upgrade targets Python 3.10/3.11, but may work on newer versions")
    return True


def check_package_version(package_name, min_version=None, max_version=None):
    """Check if a package is installed and meets version constraints."""
    try:
        installed_version = version(package_name)
        print(f"  {package_name}: {installed_version}", end="")
        
        if min_version or max_version:
            from packaging.version import parse
            installed = parse(installed_version)
            
            if min_version:
                min_v = parse(min_version)
                if installed < min_v:
                    print(f" ❌ (requires >={min_version})")
                    return False
            
            if max_version:
                max_v = parse(max_version)
                if installed >= max_v:
                    print(f" ❌ (requires <{max_version})")
                    return False
        
        print(" ✓")
        return True
    except PackageNotFoundError:
        print(f"  {package_name}: ❌ NOT INSTALLED")
        return False


def check_dependencies():
    """Check all required dependencies."""
    print("\n" + "=" * 60)
    print("Checking dependencies...")
    print("=" * 60)
    
    # Core dependencies with version constraints
    dependencies = {
        'torch': ('1.13.0', None),
        'torchvision': ('0.14.0', None),
        'mmcv': ('2.0.0', '2.2.0'),
        'numpy': ('1.20.0', '2.0.0'),
        'matplotlib': ('3.3.0', None),
        'mmpycocotools': (None, None),
        'six': (None, None),
        'terminaltables': (None, None),
    }
    
    all_passed = True
    for package, (min_ver, max_ver) in dependencies.items():
        if not check_package_version(package, min_ver, max_ver):
            all_passed = False
    
    if all_passed:
        print("\n✓ PASS: All core dependencies are installed with correct versions")
    else:
        print("\n❌ FAIL: Some dependencies are missing or have incorrect versions")
    
    return all_passed


def check_optional_dependencies():
    """Check optional dependencies (test dependencies)."""
    print("\n" + "=" * 60)
    print("Checking optional dependencies...")
    print("=" * 60)
    
    optional_deps = {
        'onnx': ('1.12.0', None),
        'onnxruntime': ('1.13.0', None),
        'isort': ('5.10.0', None),
        'pytest': ('7.0.0', None),
    }
    
    all_passed = True
    for package, (min_ver, max_ver) in optional_deps.items():
        if not check_package_version(package, min_ver, max_ver):
            all_passed = False
    
    if all_passed:
        print("\n✓ PASS: All optional dependencies are installed")
    else:
        print("\n⚠ WARNING: Some optional dependencies are missing (tests may not work)")
    
    return all_passed


def check_cuda_availability():
    """Check if CUDA is available."""
    print("\n" + "=" * 60)
    print("Checking CUDA availability...")
    print("=" * 60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA is not available (CPU-only mode)")
        
        return True
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def check_mmdet_installation():
    """Check if mmdet is properly installed."""
    print("\n" + "=" * 60)
    print("Checking mmdet installation...")
    print("=" * 60)
    
    try:
        import mmdet
        print(f"  mmdet version: {mmdet.__version__} ✓")
        return True
    except ImportError as e:
        print(f"❌ mmdet is not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error importing mmdet: {e}")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("MMDetection Installation Validation")
    print("Python 3.10/3.11 Upgrade")
    print("=" * 60)
    
    results = []
    
    # Run all checks
    results.append(("Python version", check_python_version()))
    results.append(("Core dependencies", check_dependencies()))
    results.append(("Optional dependencies", check_optional_dependencies()))
    results.append(("CUDA availability", check_cuda_availability()))
    results.append(("mmdet installation", check_mmdet_installation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for check_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("Installation is valid for Python 3.10/3.11")
    else:
        print("❌ SOME CHECKS FAILED")
        print("Please review the errors above and fix the installation")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
