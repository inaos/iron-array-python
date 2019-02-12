from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
ext_modules = cythonize([
    Extension("iarray", ["iarray.pyx"],
              include_dirs=["/home/aleix/INAOS/inac-linux-x86_64-debug-1.0.2/include", "/home/aleix/iron-array/include"],
              extra_objects=["/home/aleix/iron-array/build/libiarray.a"])
    ])
)
