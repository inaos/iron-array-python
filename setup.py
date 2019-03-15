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
libraries = ['blosc', 'iarray', 'inac']
define_macros = []
sources = ['iarray/iarray_ext.pyx']

if 'IARRAY_DEVELOP_MODE' in os.environ:
    print("*** Entering iarray develop mode ***")
    BLOSC_DIR = os.environ.get('BLOSC_DIR', '../iron-array/contribs/c-blosc2')
    print("Looking at blosc library at:", BLOSC_DIR, ".  If not correct, use the `BLOSC_DIR` envvar")
    IARRAY_DIR = os.environ.get('IARRAY_DIR', '../iron-array')
    print("Looking at iarray library at:", IARRAY_DIR, ".  If not correct, use the `IARRAY_DIR` envvar")

    if BLOSC_DIR != '':
        library_dirs += [os.path.join(BLOSC_DIR, 'build/blosc')]
        include_dirs += [os.path.join(BLOSC_DIR, 'blosc')]

    if IARRAY_DIR != '':
        library_dirs += [os.path.join(IARRAY_DIR, 'build')]
        include_dirs += [os.path.join(IARRAY_DIR, 'include')]
else:
    print("*** Entering iarray production mode ***")
    BLOSC_DIR = os.environ.get('BLOSC_DIR', '/usr/local')
    print("Looking at blosc library at:", BLOSC_DIR, ".  If not correct, use the `BLOSC_DIR` envvar")
    IARRAY_DIR = os.environ.get('IARRAY_DIR', '/usr/local')
    print("Looking at iarray library at:", IARRAY_DIR, ".  If not correct, use the `IARRAY_DIR` envvar")

    if BLOSC_DIR != '':
        library_dirs += [os.path.join(BLOSC_DIR, 'lib')]
        include_dirs += [os.path.join(BLOSC_DIR, 'include')]

    if IARRAY_DIR != '':
        library_dirs += [os.path.join(IARRAY_DIR, 'lib')]
        include_dirs += [os.path.join(IARRAY_DIR, 'include')]

INAC_DIR = os.environ.get('INAC_DIR', '../INAOS')
print("Looking at the inac library at:", INAC_DIR, ".  If not correct, use the `INAC_DIR` envvar")
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
        'numpy>=1.15',
        'cython>=0.23',
    ],
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.15',
        'pytest',
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
    ),
    extras_require={
        'doc': [
            'sphinx >= 1.5',
            'sphinx_rtd_theme',
            'numpydoc'],
        'examples': [
            'matplotlib']},
)
