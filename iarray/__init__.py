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
        # Runtime libraries are not in the path.  Probably we are running from wheels,
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

# Compression codecs
class Codecs(Enum):
    BLOSCLZ = 0
    LZ4 = 1
    LZ4HC = 2
    ZLIB = 4
    ZSTD = 5
    LIZARD = 6


# Filters
class Filters(Enum):
    NOFILTER = 0
    SHUFFLE = 1
    BITSHUFFLE = 2
    DELTA = 4
    TRUNC_PREC = 8


# Random generators
class RandomGen(Enum):
    MERSENNE_TWISTER = 0
    SOBOL = 1


# Eval method
class Eval(Enum):
    """
    Available eval methods
    """

    AUTO = auto()
    ITERBLOSC = auto()  # Iterblosc method
    ITERCHUNK = auto()  # Iterchunk method


class Reduce(Enum):
    MAX = 0
    MIN = 1
    SUM = 2
    PROD = 3
    MEAN = 4


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

from .config_params import (
    Config,
    Storage,
    set_config,
    get_config,
    config,
    reset_config_defaults,
    get_ncores,
    partition_advice,
)

from .iarray_container import (
    IArray,
    matmul,
    transpose,
    # reductions
    max,
    min,
    sum,
    prod,
    mean,
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
)

from .constructors import (
    DTShape,
    empty,
    arange,
    linspace,
    zeros,
    ones,
    full,
)

from .utils import (
    load,
    open,
    save,
    cmp_arrays,
    iarray2numpy,
    numpy2iarray,
)

# random constructors (follow NumPy convention)
from . import irandom

from .expression import (
    Expr,
    expr_from_string,
    expr_from_udf,
)

from .lazy_expr import (
    LazyExpr,
)

# For some reason this needs to go to the end, else matmul function does not work.
from . import iarray_ext as ext

ext.IArrayInit()
