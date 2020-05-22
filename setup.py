from sys import platform
from skbuild import setup

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
    copy_lib = ['iarray.dll']  # TODO: not sure about this

setup(
    name="iarray",
    description=DESCRIPTION,
    # long_description=LONG_DESCRIPTION,
    python_requires=">=3.6",
    package_dir={'iarray': 'iarray'},
    packages=['iarray', 'iarray.py2llvm', 'iarray.tests'],
    package_data={
        'iarray': copy_libs,
    },
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
)
