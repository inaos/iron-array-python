import os
from sys import platform
from skbuild import setup

BUILD_WHEELS = True if 'BUILD_WHEELS' in os.environ else False

DESCRIPTION = 'The Math Array Accelerator for Python'


with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

# Libraries to copy as 'data' in package
# Copying just 'libarray' seems good enough
if platform == "linux" or platform == "linux2":
    copy_libs = ['libiarray.so']
elif platform == "darwin":
    copy_libs = ['libiarray.dylib']
elif platform == "win32":
    copy_libs = ['iarray.dll']  # TODO: not sure about this

if BUILD_WHEELS:
    print("BUILD_WHEELS mode is ON!")
    package_info = dict(
        package_dir={'iarray': 'iarray'},
        packages=['iarray', 'iarray.py2llvm', 'iarray.tests'],
        package_data={'iarray': copy_libs},
    )
else:
    # For some reason this is necessary for inplace compilation
    # One can avoid using this if we nuke _skbuild/ next to iarray/
    package_info = dict(
        package_dir={'': '.'},
    )
setup(
    name="iarray",
    description=DESCRIPTION,
    # long_description=LONG_DESCRIPTION,
    python_requires=">=3.6",
    extras_require={
        'doc': [
            'sphinx >= 1.5',
            'sphinx_rtd_theme',
            'numpydoc',
        ],
        'examples': [
            'matplotlib',
            'numexpr',
            'numba',
        ]},
    **package_info,
)
