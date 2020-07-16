import os
from sys import platform
from skbuild import setup

BUILD_WHEELS = True if 'BUILD_WHEELS' in os.environ else False
if not BUILD_WHEELS:
    if os.path.exists('BUILD_WHEELS'):
        BUILD_WHEELS = True

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
    copy_libs = ['iarray.dll', 'svml_dispmd.dll']
else:
    copy_libs = []

doc_deps = [
    'sphinx >= 1.5',
    'sphinx_rtd_theme',
    'numpydoc',
    ]
examples_deps = [
    'matplotlib',
    'numexpr',
    'numba',
    ]

if BUILD_WHEELS:
    print("BUILD_WHEELS mode is ON!")
    runtime_deps = open("requirements-runtime.txt").read().split()
    package_info = dict(
        package_dir={'iarray': 'iarray'},
        packages=['iarray', 'iarray.py2llvm', 'iarray.tests'],
        package_data={'iarray': copy_libs},
        extras_require = {
           'runtime': runtime_deps,
            'doc': doc_deps,
            'examples': examples_deps},
    )
else:
    # For some reason this is necessary for inplace compilation
    # One can avoid using this if we nuke _skbuild/ next to iarray/
    package_info = dict(
        package_dir={'': '.'},
        extras_require = {
            'doc': doc_deps,
            'examples': examples_deps},
    )
setup(
    name="iarray",
    description=DESCRIPTION,
    # long_description=LONG_DESCRIPTION,
    python_requires=">=3.6",
    **package_info,
)
