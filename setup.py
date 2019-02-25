from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
ext_modules = cythonize([
    Extension("iarray", ["iarray.pyx"],
              include_dirs=["/Users/aleix11alcacer/INAOS/inac-darwin-x86_64-release-1.0.2/include", "/Users/aleix11alcacer/Documents/Francesc Alted/IronArray/iron-array/include", numpy.get_include()],
              extra_objects=["/Users/aleix11alcacer/Documents/Francesc Alted/IronArray/iron-array/cmake-build-relwithdebinfo/libiarray.dylib"])
    ])
)
