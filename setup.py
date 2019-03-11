import os
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

DESCRIPTION = 'A Python wrapper of the IronArray (N-dimensional arrays) C library for Python.'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

# Compiler & linker flags
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()

# Sources & libraries
include_dirs = [numpy.get_include()]
library_dirs = []
libraries = []
define_macros = []
sources = ['iarray/iarray_ext.pyx']

BLOSC_DIR = os.environ.get('BLOSC_DIR', '')
if BLOSC_DIR != '':
    library_dirs += [os.path.join(BLOSC_DIR, 'build/blosc')]
    include_dirs += [os.path.join(BLOSC_DIR, 'blosc')]
    libraries += ['blosc']

IARRAY_DIR = os.environ.get('IARRAY_DIR', '')
if IARRAY_DIR != '':
    library_dirs += [os.path.join(IARRAY_DIR, 'build')]
    include_dirs += [os.path.join(IARRAY_DIR, 'include')]
    libraries += ['iarray']

INAC_DIR = os.environ.get('INAC_DIR', '')
if INAC_DIR != '':
    library_dirs += [os.path.join(INAC_DIR, 'lib')]
    include_dirs += [os.path.join(INAC_DIR, 'include')]
    libraries += ['inac']

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
    ],
    install_requires=[
        'numpy>=1.7',
        'cython>=0.23',
        'pytest',
        'matplotlib',
    ],
    package_dir={'': '.'},
    packages=find_packages(),
    ext_modules=cythonize([
        Extension(
            "iarray.iarray_ext", sources,
            define_macros=define_macros,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=CFLAGS,
            extra_link_args=LFLAGS,
            )
    ],
    )
)
