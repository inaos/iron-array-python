
# TODO: move this to the extension and define them from iarray/blosc symbols
IARRAY_BLOSCLZ = 0
IARRAY_LZ4 = 1
IARRAY_LZ4HC = 2
IARRAY_ZLIB = 4
IARRAY_LIZARD = 5

from .iarray_ext import *  # the order of the import is important: extensions first
from .container import IArray, LazyExpr, empty2, arange2, linspace2, from_file2, iarray2numpy2, numpy2iarray2
from .version import version as __version__

IarrayInit()
