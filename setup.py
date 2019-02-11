from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
ext_modules = cythonize([
    Extension("iarray", ["iarray.pyx"],
              library_dirs=['/Users/aleix11alcacer/Documents/Francesc Alted/IronArray/iron-array/cmake-build-release'],
              libraries=["iarray"])
    ])
)