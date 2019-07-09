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
libraries = ['iarray', 'inac']
define_macros = []
sources = ['iarray/iarray_ext.pyx']

if 'IARRAY_DEVELOP_MODE' in os.environ:
    print("*** Entering iarray develop mode ***")
    IARRAY_DIR = os.environ.get('IARRAY_DIR', '../iron-array')
    IARRAY_DIR = os.path.expanduser(IARRAY_DIR)
    print("Looking at iarray sources at:", IARRAY_DIR, ".  If not correct, use the `IARRAY_DIR` envvar")

    if IARRAY_DIR != '':
        IARRAY_BUILD_DIR = os.environ.get('IARRAY_BUILD_DIR', os.path.join(IARRAY_DIR, 'build'))
        print("Looking at iarray library at:", IARRAY_BUILD_DIR, ".  If not correct, use the `IARRAY_BUILD_DIR` envvar")
        library_dirs += [IARRAY_BUILD_DIR]
        include_dirs += [os.path.join(IARRAY_DIR, 'include')]
else:
    print("*** Entering iarray production mode ***")
    IARRAY_DIR = os.environ.get('IARRAY_DIR', '/usr/local')
    print("Looking at iarray library at:", IARRAY_DIR, ".  If not correct, use the `IARRAY_DIR` envvar")

    if IARRAY_DIR != '':
        library_dirs += [os.path.join(IARRAY_DIR, 'lib')]
        include_dirs += [os.path.join(IARRAY_DIR, 'include')]

INAC_DIR = os.environ.get('INAC_DIR', '../INAOS')
INAC_DIR = os.path.expanduser(INAC_DIR)
print("Looking at the inac library at:", INAC_DIR, ".  If not correct, use the `INAC_DIR` envvar")
if INAC_DIR != '':
    library_dirs += [os.path.join(INAC_DIR, 'lib')]
    include_dirs += [os.path.join(INAC_DIR, 'include')]
    libraries += ['inac']


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if l]

    return requires


INSTALL_REQUIRES = parse_requirements_file('requirements/default.txt')
extras_require = {
    dep : parse_requirements_file('requirements/' + dep + '.txt')
    for dep in ['docs', 'optional', 'test']
}

# requirements for those browsing PyPI
REQUIRES = [r.replace('>=', ' (>= ') + ')' for r in INSTALL_REQUIRES]
REQUIRES = [r.replace('==', ' (== ') for r in REQUIRES]
REQUIRES = [r.replace('[array]', '') for r in REQUIRES]

setup(
    name="iarray",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    use_scm_version={
        'version_scheme': 'guess-next-dev',
        'local_scheme': 'dirty-tag',
        'write_to': 'iarray/version.py'
    },
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    requires=REQUIRES,
    extras_require=extras_require,
    packages=find_packages(exclude=['doc', 'benchmarks']),
    package_dir={'': '.'},
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
)
