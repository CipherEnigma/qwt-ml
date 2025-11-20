import mmcv

from .version import __version__, short_version


def digit_version(version_str):
    """Parse version string to list of integers for comparison.
    
    Handles version formats like:
    - '2.0.0' -> [2, 0, 0, 0, 0]
    - '2.0.0rc1' -> [2, 0, 0, -1, 1]  (rc versions are pre-release)
    - '2.0.0.post1' -> [2, 0, 0, 1, 1]  (post versions are post-release)
    - '2.0.0.dev1' -> [2, 0, 0, -2, 1]  (dev versions are pre-release)
    
    The last two elements represent release type and number:
    - [0, 0] for final releases
    - [-2, N] for dev releases (earliest)
    - [-1, N] for rc releases (before final)
    - [1, N] for post releases (after final)
    
    Args:
        version_str: Version string to parse
        
    Returns:
        List of integers representing version for comparison
    """
    digit_version = []
    parts = version_str.split('.')
    has_suffix = False
    
    for i, x in enumerate(parts):
        if x.isdigit():
            digit_version.append(int(x))
        else:
            # Handle rc (release candidate), post, and dev suffixes
            if 'rc' in x:
                # rc versions are pre-release, so they come before the actual release
                base_and_suffix = x.split('rc')
                if base_and_suffix[0]:
                    digit_version.append(int(base_and_suffix[0]))
                # Add marker for rc (-1) and rc number
                digit_version.append(-1)
                digit_version.append(int(base_and_suffix[1]) if base_and_suffix[1] else 0)
                has_suffix = True
                break
            elif 'post' in x:
                # post versions come after the release
                base_and_suffix = x.split('post')
                if base_and_suffix[0]:
                    digit_version.append(int(base_and_suffix[0]))
                # Add marker for post (1) and post number
                digit_version.append(1)
                digit_version.append(int(base_and_suffix[1]) if base_and_suffix[1] else 0)
                has_suffix = True
                break
            elif 'dev' in x:
                # dev versions are pre-release, even before rc
                base_and_suffix = x.split('dev')
                if base_and_suffix[0]:
                    digit_version.append(int(base_and_suffix[0]))
                # Add marker for dev (-2) and dev number
                digit_version.append(-2)
                digit_version.append(int(base_and_suffix[1]) if base_and_suffix[1] else 0)
                has_suffix = True
                break
            else:
                # Try to extract any leading digits
                num_str = ''
                for char in x:
                    if char.isdigit():
                        num_str += char
                    else:
                        break
                if num_str:
                    digit_version.append(int(num_str))
                break
    
    # Ensure all versions have the same length for proper comparison
    # Final releases get [0, 0] suffix to be greater than rc/dev but less than post
    if not has_suffix:
        digit_version.extend([0, 0])
    
    return digit_version


mmcv_minimum_version = '2.0.0'
mmcv_maximum_version = '2.2.0'
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.\n' \
    f'To install a compatible version, run:\n' \
    f'  pip install -U openmim\n' \
    f'  mim install "mmcv>={mmcv_minimum_version},<={mmcv_maximum_version}"'

__all__ = ['__version__', 'short_version']
