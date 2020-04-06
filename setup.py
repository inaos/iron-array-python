import os
from setuptools import Extension, find_packages
from skbuild import setup
from Cython.Build import cythonize
import numpy

DESCRIPTION = 'A Python wrapper of the IronArray (N-dimensional arrays) C library for Python.'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="iarray",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    use_scm_version={
        'version_scheme': 'guess-next-dev',
        'local_scheme': 'dirty-tag',
        'write_to': 'iarray/version.py'
    },
    setup_requires=[
        'setuptools>18.0',
        'setuptools-scm>3.0'
        'numpy>=1.15',
        'cython>=0.23',
        'scikit-build',
    ],
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.15',
        'numexpr>=2.6',
        'numba>=0.42',
        'llvmlite',
        'pytest',
        'hypothesis',
    ],
    tests_require=['numpy', 'numexpr'],
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
    packages = ['iarray'],
)
