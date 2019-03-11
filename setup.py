from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(name="iarray",
ext_modules = cythonize([
    Extension("iarray.iarray_ext", ["iarray/iarray_ext.pyx"],
              include_dirs=["/Users/aleix11alcacer/Documents/Francesc Alted/IronArray/iron-array/include",
                            numpy.get_include()],
              extra_objects=["/Users/aleix11alcacer/Documents/Francesc Alted/IronArray/iron-array/cmake-build-relwithdebinfo/libiarray.dylib"])
    ])
)
