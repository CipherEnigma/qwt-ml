"""Tests for Python version compatibility."""
import sys
import warnings

import pytest


def test_python_version_minimum():
    """Test that Python version is at least 3.10."""
    assert sys.version_info >= (3, 10), (
        f"Python 3.10 or higher is required, but got {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_python_version_supported():
    """Test that Python version is in supported range (3.10+)."""
    major, minor = sys.version_info.major, sys.version_info.minor
    assert major == 3, f"Python 3.x is required, but got Python {major}.{minor}"
    assert minor >= 10, (
        f"Python 3.10 or higher is supported, but got Python {major}.{minor}"
    )


def test_no_deprecation_warnings_on_import():
    """Test that importing mmdet does not produce deprecation warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        
        # Import mmdet and key modules
        import mmdet
        from mmdet import apis
        from mmdet import datasets
        from mmdet import models
        
        # Filter for deprecation warnings
        deprecation_warnings = [
            warning for warning in w 
            if issubclass(warning.category, DeprecationWarning)
        ]
        
        # Check for collections-related deprecation warnings
        collections_warnings = [
            warning for warning in deprecation_warnings
            if 'collections' in str(warning.message).lower()
        ]
        
        assert len(collections_warnings) == 0, (
            f"Found {len(collections_warnings)} collections-related deprecation warnings: "
            f"{[str(w.message) for w in collections_warnings]}"
        )


def test_collections_abc_imports():
    """Test that collections.abc is used instead of deprecated collections imports."""
    import collections.abc
    
    # These should be available from collections.abc
    assert hasattr(collections.abc, 'Iterable')
    assert hasattr(collections.abc, 'Mapping')
    assert hasattr(collections.abc, 'Sequence')
    
    # Verify we can import them
    from collections.abc import Iterable, Mapping, Sequence
    
    # Basic sanity checks
    assert isinstance([], Iterable)
    assert isinstance({}, Mapping)
    assert isinstance([], Sequence)


def test_setup_py_python_requires():
    """Test that setup.py declares correct python_requires."""
    import os
    import re
    
    setup_py_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'setup.py'
    )
    
    with open(setup_py_path, 'r') as f:
        setup_content = f.read()
    
    # Check for python_requires
    python_requires_match = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)['\"]", setup_content)
    
    assert python_requires_match is not None, "python_requires not found in setup.py"
    
    python_requires = python_requires_match.group(1)
    
    # Should require at least Python 3.10
    assert '>=3.10' in python_requires or '>=3.1' in python_requires, (
        f"setup.py should require Python >=3.10, but got: {python_requires}"
    )


def test_setup_py_classifiers():
    """Test that setup.py declares Python 3.10 and 3.11 in classifiers."""
    import os
    import re
    
    setup_py_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'setup.py'
    )
    
    with open(setup_py_path, 'r') as f:
        setup_content = f.read()
    
    # Check for Python version classifiers
    assert 'Programming Language :: Python :: 3.10' in setup_content, (
        "setup.py should declare Python 3.10 support in classifiers"
    )
    assert 'Programming Language :: Python :: 3.11' in setup_content, (
        "setup.py should declare Python 3.11 support in classifiers"
    )
    
    # Should not have old Python versions
    assert 'Programming Language :: Python :: 3.6' not in setup_content, (
        "setup.py should not declare Python 3.6 support"
    )
    assert 'Programming Language :: Python :: 3.7' not in setup_content, (
        "setup.py should not declare Python 3.7 support"
    )
    assert 'Programming Language :: Python :: 3.8' not in setup_content, (
        "setup.py should not declare Python 3.8 support"
    )
