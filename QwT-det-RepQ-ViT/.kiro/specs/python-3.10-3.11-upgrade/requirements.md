# Requirements Document

## Introduction

This document outlines the requirements for upgrading the mmdetection-based quantization project to support Python 3.10 and 3.11. The current implementation uses mmcv 1.2.4-1.4.0 and mmdet 2.11.0, which have compatibility issues with Python 3.10+ due to deprecated features and dependency conflicts. The upgrade must maintain backward compatibility with the quantization functionality (QwT) while modernizing the codebase to work with newer Python versions.

## Requirements

### Requirement 1: Python Version Compatibility

**User Story:** As a developer, I want the codebase to run on Python 3.10 and 3.11, so that I can use modern Python features and security updates.

#### Acceptance Criteria

1. WHEN the project is installed on Python 3.10 THEN all dependencies SHALL install without errors
2. WHEN the project is installed on Python 3.11 THEN all dependencies SHALL install without errors
3. WHEN setup.py is executed THEN it SHALL declare support for Python 3.10 and 3.11 in classifiers
4. IF the Python version is below 3.10 THEN the installation SHALL fail with a clear error message

### Requirement 2: MMCV and MMDetection Upgrade

**User Story:** As a developer, I want to upgrade mmcv and mmdetection to compatible versions, so that the project works with Python 3.10/3.11.

#### Acceptance Criteria

1. WHEN dependencies are installed THEN mmcv-full SHALL be replaced with mmcv>=2.0.0
2. WHEN mmdet is imported THEN the version compatibility check SHALL accept mmcv 2.x versions
3. WHEN the project is built THEN all CUDA extensions SHALL compile successfully on supported platforms
4. IF mmcv version is incompatible THEN a clear error message SHALL be displayed with upgrade instructions

### Requirement 3: Dependency Updates

**User Story:** As a developer, I want all project dependencies to be compatible with Python 3.10/3.11, so that there are no installation or runtime conflicts.

#### Acceptance Criteria

1. WHEN requirements are parsed THEN all packages SHALL have versions compatible with Python 3.10+
2. WHEN onnx and onnxruntime are installed THEN they SHALL be updated to versions supporting Python 3.10+
3. WHEN isort is installed THEN it SHALL be updated from 4.3.21 to a version compatible with Python 3.10+
4. WHEN torch is imported THEN it SHALL be version 1.13.0 or higher for Python 3.10+ support

### Requirement 4: Code Compatibility Fixes

**User Story:** As a developer, I want the codebase to be free of deprecated Python features, so that it runs without warnings on Python 3.10/3.11.

#### Acceptance Criteria

1. WHEN Python code is executed THEN there SHALL be no deprecation warnings related to collections.abc
2. WHEN setup.py is executed THEN it SHALL handle torch imports gracefully if torch is not pre-installed
3. WHEN any module is imported THEN there SHALL be no syntax errors or import failures
4. IF deprecated features are found THEN they SHALL be replaced with modern equivalents

### Requirement 5: Quantization Functionality Preservation

**User Story:** As a researcher, I want the quantization functionality (QwT) to work exactly as before, so that I can reproduce published results.

#### Acceptance Criteria

1. WHEN quantization is applied THEN the W4/A4 and W6/A6 precision modes SHALL work correctly
2. WHEN evaluation is run THEN the results SHALL match the original implementation within acceptable tolerance
3. WHEN custom quantization modules are imported THEN they SHALL be compatible with upgraded mmdet
4. IF any quantization-specific code breaks THEN it SHALL be fixed while maintaining algorithmic equivalence

### Requirement 6: Build System Modernization

**User Story:** As a developer, I want a modern build system, so that installation is reliable and follows current best practices.

#### Acceptance Criteria

1. WHEN setup.py is executed THEN it SHALL use modern setuptools features
2. WHEN CUDA extensions are built THEN they SHALL compile with PyTorch 1.13+ build system
3. WHEN installation fails THEN clear error messages SHALL guide the user to resolve issues
4. IF CUDA is not available THEN the project SHALL still install with CPU-only support

### Requirement 7: Testing and Validation

**User Story:** As a developer, I want to verify that the upgrade works correctly, so that I can confidently use the upgraded codebase.

#### Acceptance Criteria

1. WHEN the project is installed THEN import tests SHALL pass for all core modules
2. WHEN dist_test.sh is executed THEN it SHALL run without Python version errors
3. WHEN a sample model is loaded THEN it SHALL load successfully with upgraded dependencies
4. IF any tests fail THEN the root cause SHALL be identified and documented

### Requirement 8: Documentation Updates

**User Story:** As a user, I want updated documentation, so that I know how to install and use the upgraded project.

#### Acceptance Criteria

1. WHEN README.md is read THEN it SHALL include Python 3.10/3.11 installation instructions
2. WHEN installation instructions are followed THEN they SHALL work on fresh Python 3.10/3.11 environments
3. WHEN dependency conflicts occur THEN troubleshooting guidance SHALL be provided
4. IF breaking changes exist THEN they SHALL be clearly documented in a migration guide
