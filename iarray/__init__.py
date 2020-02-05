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
DELTA = 3
TRUNC_PREC = 4


from . import iarray_ext as ext
from .high_level import (IArray, dtshape, Config, RandomContext, Expr, LazyExpr,
                         empty, arange, linspace, zeros, ones, full, load, save,
                         iarray2numpy, numpy2iarray, matmul,
                         # random constructors
                         random_rand, random_randn, random_beta, random_lognormal, random_exponential,
                         random_uniform, random_normal, random_bernoulli, random_binomial, random_poisson,
                         random_kstest,
                         # ufuncs
                         abs, arccos, arcsin, arctan, arctan2, ceil, cos, cosh, exp, floor, log, log10, negative, power,
                         sin, sinh, sqrt, tan, tanh
                         )

from .expression import Parser
from .version import version as __version__

ext.IArrayInit()
