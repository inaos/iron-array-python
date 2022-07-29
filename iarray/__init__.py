###########################################################################################
# Copyright ironArray SL 2021.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of ironArray SL
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement.
###########################################################################################

import os
from enum import Enum, auto
from ctypes import cdll
from llvmlite import binding
import platform
import pytest


# This is the source of truth for version
# https://packaging.python.org/guides/single-sourcing-package-version/
# __version__ = "1.0.0-$IA_BUILD_VER"
# Change to use a YEAR.MINOR-BUILD_VER
__version__ = "2022.2-$IA_BUILD_VER"


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
class Codec(Enum):
    """
    Available codecs.
    """

    BLOSCLZ = 0
    LZ4 = 1
    LZ4HC = 2
    ZLIB = 4
    ZSTD = 5
    ZFP_FIXED_ACCURACY = 6
    ZFP_FIXED_PRECISION = 7
    ZFP_FIXED_RATE = 8


# Filter
class Filter(Enum):
    """
    Available filters.
    """

    NOFILTER = 0
    SHUFFLE = 1
    BITSHUFFLE = 2
    DELTA = 4
    TRUNC_PREC = 8


# Favor.  Select to favor compression speed, compression ratio or a balance between them.
class Favor(Enum):
    """
    Which computer resource to be favored.
    """

    BALANCE = 0
    SPEED = 1
    CRATIO = 2


# Split mode
class SplitMode(Enum):
    """
    Available split modes.
    """

    ALWAYS_SPLIT = 1
    NEVER_SPLIT = 2
    AUTO_SPLIT = 3
    FORWARD_COMPAT_SPLIT = 4


# Random generators
class RandomGen(Enum):
    """
    Available random generators.
    """

    MRG32K3A = 0


# Eval method
class Eval(Enum):
    """
    Available eval methods.
    """

    AUTO = auto()
    ITERBLOSC = auto()  # Iterblosc method
    ITERCHUNK = auto()  # Iterchunk method


class Reduce(Enum):
    """
    Available reduction operations.
    """

    MAX = 0
    MIN = 1
    SUM = 2
    PROD = 3
    MEAN = 4
    VAR = 5
    STD = 6
    MEDIAN = 7
    NAN_MAX = 8
    NAN_MIN = 9
    NAN_SUM = 10
    NAN_PROD = 11
    NAN_MEAN = 12
    NAN_VAR = 13
    NAN_STD = 14
    NAN_MEDIAN = 15


# List of all know universal functions
MATH_FUNC_LIST = (
    "abs",
    "absolute",
    "arccos",
    "acos",
    "arcsin",
    "asin",
    "arctan",
    "atan",
    "arctan2",
    "atan2",
    "ceil",
    "cos",
    "cosh",
    "exp",
    "floor",
    "log",
    "log10",
    "negative",
    "negate",
    "power",
    "pow",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
)


# That must come here so as to avoid circular import errors

from .config_params import (
    Config,
    set_config_defaults,
    get_config_defaults,
    config,
    reset_config_defaults,
    get_ncores,
    get_l2_size,
    partition_advice,
    defaults as _defaults,
)

from .iarray_container import (
    IArray,
    matmul_params,
    matmul,
    transpose,
    # reductions
    max,
    min,
    sum,
    prod,
    mean,
    std,
    var,
    median,
    nanmax,
    nanmin,
    nansum,
    nanprod,
    nanmean,
    nanstd,
    nanvar,
    nanmedian,
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
    uninit,
    zarr_proxy,
)

from .utils import (
    load,
    open,
    save,
    cmp_arrays,
    iarray2numpy,
    numpy2iarray,
    _check_access_mode,
    remove_urlpath,
    list_arrays,
)

# random constructors (follow NumPy convention)
from . import random

from .expression import (
    Expr,
    UdfRegistry,
    ULib,
    expr_from_string,
    expr_from_udf,
    expr_get_operands,
    expr_get_ops_funcs,
)

from .lazy_expr import (
    LazyExpr,
)

from .attrs import (
    Attributes,
)

# Global catalog
global_catalog = {}
HTTP_PORT = 28800

from . import http_server

# For some reason this needs to go to the end, else matmul function does not work.
from . import iarray_ext as ext

from .iarray_ext import (
    udf_lookup_func,
    IArrayError,
)

# Keep the reference so as to avoid calling the destroyer immediately
_init_object = ext.IArrayInit()

# Global registry for scalar UDFs
udf_registry = UdfRegistry()
# Default scalar UDF library (for lazy expressions)
dflt_ulib = "ulib"
# Accessor for scalar UDF registry (for lazy expressions)
ulib = ULib(dflt_ulib)

# Whether a IArray.__eq__ should be bypassed or not (for private use)
_disable_overloaded_equal = False

from . import tests


def test():
    retcode = pytest.main(["-x", os.path.dirname(tests.__file__)])
    return retcode
