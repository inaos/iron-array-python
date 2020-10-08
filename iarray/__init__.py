import os
from enum import Enum, auto
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
if platform_system == "Linux":
    # Force the load of shared libs with no need to mess with LD_LIBRARY_PATH
    # https://stackoverflow.com/questions/6543847/setting-ld-library-path-from-inside-python
    # We can disable this when/if we can package iron-array into its own wheel
    # and make a dependency of it.  The same goes for other platforms.
    try:
        binding.load_library_permanently("libintlc.so.5")
        binding.load_library_permanently("libsvml.so")
    except RuntimeError:
        # Runtime libraries are not in the path.  Probably we are runing from wheels,
        # and wheels ensure than libraries are in the same directory than this file.
        binding.load_library_permanently(os.path.join(install_dir, "libintlc.so.5"))
        binding.load_library_permanently(os.path.join(install_dir, "libsvml.so"))
    lib0 = cdll.LoadLibrary(os.path.join(install_dir, "libiarray.so"))
elif platform_system == "Darwin":
    binding.load_library_permanently("libsvml.dylib")
    lib0 = cdll.LoadLibrary(os.path.join(install_dir, "libiarray.dylib"))
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


# Eval method
class Eval(Enum):
    AUTO = auto()
    ITERBLOSC = auto()
    ITERCHUNK = auto()


RANDOM_SEED = 0

# List of all know universal functions
UFUNC_LIST = (
    "abs",
    "arccos",
    "arcsin",
    "arctan",
    "arctan2",
    "ceil",
    "cos",
    "cosh",
    "exp",
    "floor",
    "log",
    "log10",
    "negative",
    "power",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
)

# That must come here so as to avoid circular import errors
from .high_level import (
    IArray,
    DTShape,
    StorageProperties,
    Config,
    RandomContext,
    create_expr,
    Expr,
    LazyExpr,
    empty,
    arange,
    linspace,
    zeros,
    ones,
    full,
    load,
    save,
    cmp_arrays,
    iarray2numpy,
    numpy2iarray,
    matmul,
    # random constructors
    random_set_seed,
    random_rand,
    random_randn,
    random_beta,
    random_lognormal,
    random_exponential,
    random_uniform,
    random_normal,
    random_bernoulli,
    random_binomial,
    random_poisson,
    random_kstest,
    # ufuncs
    abs,
    arccos,
    arcsin,
    arctan,
    arctan2,
    ceil,
    cos,
    cosh,
    exp,
    floor,
    log,
    log10,
    negative,
    power,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    # utils
    get_ncores,
    partition_advice,
)

from . import iarray_ext as ext

ext.IArrayInit()
