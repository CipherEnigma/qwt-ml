#!/usr/bin/env python
import os
from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'mmdet/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def make_cuda_ext(name, module, sources, sources_cuda=[]):
    """Build CUDA extension with graceful fallback.
    
    This function attempts to build CUDA extensions if torch is available.
    If torch is not installed or CUDA is not available, it falls back to
    CPU-only extensions or skips extension building entirely.
    
    Compatible with PyTorch 1.13+ build system.
    
    Args:
        name (str): Extension name
        module (str): Module path
        sources (list): List of source files
        sources_cuda (list): List of CUDA source files
        
    Returns:
        Extension object or None if building fails
    """
    try:
        import torch
        from torch.utils.cpp_extension import CppExtension, CUDAExtension
    except ImportError:
        print(f'Warning: torch not available, skipping {name} extension')
        return None

    define_macros = []
    extra_compile_args = {'cxx': ['-std=c++14']}  # Ensure C++14 standard for PyTorch 1.13+

    try:
        if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
            define_macros += [('WITH_CUDA', None)]
            extension = CUDAExtension
            
            # CUDA compiler flags compatible with PyTorch 1.13+
            # Removed deprecated flags that may cause issues
            extra_compile_args['nvcc'] = [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
            
            # Add CUDA sources
            sources = sources + sources_cuda
            
            print(f'Building {name} with CUDA support')
        else:
            print(f'Building {name} without CUDA (CPU only)')
            extension = CppExtension
    except Exception as e:
        print(f'Warning: Error checking CUDA availability for {name}: {e}')
        print(f'Falling back to CPU-only extension for {name}')
        extension = CppExtension

    try:
        return extension(
            name=f'{module}.{name}',
            sources=[os.path.join(*module.split('.'), p) for p in sources],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
    except Exception as e:
        print(f'Error: Failed to create extension {name}: {e}')
        print(f'Extension {name} will be skipped. The package will install but this extension will not be available.')
        return None


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def get_extensions():
    """Get list of extensions to build.
    
    Returns empty list if torch is not available, allowing installation
    to proceed without CUDA extensions. Filters out any extensions that
    failed to build.
    
    Returns:
        tuple: (list of extensions, cmdclass dict)
    """
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension
        
        # Verify PyTorch version for compatibility
        torch_version = torch.__version__.split('+')[0]  # Remove CUDA version suffix
        major, minor = map(int, torch_version.split('.')[:2])
        
        if major < 1 or (major == 1 and minor < 13):
            print(f'Warning: PyTorch {torch_version} detected. PyTorch >= 1.13.0 is recommended for Python 3.10+')
        
    except ImportError:
        print('Warning: torch not available, skipping extension building')
        print('The package will install but CUDA extensions will not be available.')
        return [], {}
    except Exception as e:
        print(f'Warning: Error checking torch version: {e}')
        print('Proceeding with extension building anyway...')
    
    # Currently no extensions are defined in ext_modules
    # This function is here for future extension support
    # When extensions are added, use make_cuda_ext() to create them
    extensions = []
    
    # Example of how to add extensions (currently none defined):
    # ext = make_cuda_ext(
    #     name='example_ext',
    #     module='mmdet.ops',
    #     sources=['src/example.cpp'],
    #     sources_cuda=['src/example_cuda.cu']
    # )
    # if ext is not None:
    #     extensions.append(ext)
    
    # Filter out None values from failed extension builds
    extensions = [ext for ext in extensions if ext is not None]
    
    cmdclass = {'build_ext': BuildExtension}
    
    return extensions, cmdclass


if __name__ == '__main__':
    ext_modules, cmdclass = get_extensions()
    
    setup(
        name='mmdet',
        version=get_version(),
        description='OpenMMLab Detection Toolbox and Benchmark',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='OpenMMLab',
        author_email='openmmlab@gmail.com',
        keywords='computer vision, object detection',
        url='https://github.com/open-mmlab/mmdetection',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
        ],
        python_requires='>=3.10',
        license='Apache License 2.0',
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        zip_safe=False)
