import os
from ctypes import cdll
from llvmlite import binding
import platform

# This is the source of truth for version
# https://packaging.python.org/guides/single-sourcing-package-version/
__version__ = "1.0.0-beta.1"

binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

install_dir = os.path.dirname(__file__)
platform_system = platform.system()
if platform_system == 'Linux':
    # Force the load of shared libs with no need to mess with LD_LIBRARY_PATH
    # https://stackoverflow.com/questions/6543847/setting-ld-library-path-from-inside-python
    # We can disable this when/if we can package iron-array into its own wheel
    # and make a dependency of it.  The same goes for other platforms.
    binding.load_library_permanently(os.path.join(install_dir, "libintlc.so.5"))
    binding.load_library_permanently(os.path.join(install_dir, "libsvml.so"))
    lib0 = cdll.LoadLibrary(os.path.join(install_dir, 'libiarray.so'))
elif platform_system == 'Darwin':
    lib0 = cdll.LoadLibrary(os.path.join(install_dir, 'libiarray.dylib'))
    binding.load_library_permanently("libsvml.dylib")
else:
    binding.load_library_permanently(os.path.join(install_dir, "svml_dispmd.dll"))
    lib1 = cdll.LoadLibrary(os.path.join(install_dir, "iarray.dll"))

# Codecs
BLOSCLZ = 0
LZ4 = 1
LZ4HC = 2
ZLIB = 4
ZSTD = 5
LIZARD = 6

# Filters
NOFILTER = 0
SHUFFLE = 1
BITSHUFFLE = 2
DELTA = 4
TRUNC_PREC = 8

# Storage types
PLAINBUFFER_STORAGE = 'plainbuffer'
BLOSC_STORAGE = 'blosc'

# Eval method

EVAL_AUTO = 'auto'
EVAL_ITERBLOSC = 'iterblosc'
EVAL_ITERCHUNK = 'iterchunk'
RANDOM_SEED = 0

from . import iarray_ext as ext

from .high_level import (IArray, dtshape, StorageProperties, Config, RandomContext, Expr, LazyExpr,
                         empty, arange, linspace, zeros, ones, full, load, save,
                         cmp_arrays, iarray2numpy, numpy2iarray, matmul,
                         # random constructors
                         random_set_seed,
                         random_rand, random_randn, random_beta, random_lognormal, random_exponential,
                         random_uniform, random_normal, random_bernoulli, random_binomial, random_poisson,
                         random_kstest,
                         # ufuncs
                         abs, arccos, arcsin, arctan, arctan2, ceil, cos, cosh,
                         exp, floor, log, log10, negative, power, sin, sinh,
                         sqrt, tan, tanh,
                         UFUNC_LIST,
                         # utils
                         get_ncores,
                         )


ext.IArrayInit()
