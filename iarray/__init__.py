
# TODO: move this to the extension and define them from iarray/blosc symbols
IARRAY_BLOSCLZ = 0
IARRAY_LZ4 = 1
IARRAY_LZ4HC = 2
IARRAY_ZLIB = 4
IARRAY_LIZARD = 5

from .iarray_ext import *
from .version import version as __version__

IarrayInit()
