# Codecs
BLOSCLZ = 0
LZ4 = 1
LZ4HC = 2
ZLIB = 4
LIZARD = 5

# Filters
NOFILTER = 0
SHUFFLE = 1
BITSHUFFLE = 2
DELTA = 3
TRUNC_PREC = 4


from . import iarray_ext as ext

from .container import (IArray, Config, RandomContext, Expr, LazyExpr,
                        empty, arange, linspace, zeros, ones, full, from_file,
                        iarray2numpy, numpy2iarray, matmul,
                        random_rand, random_randn, random_beta, random_lognormal, random_exponential,
                        random_uniform, random_normal, random_bernoulli, random_binomial, random_poisson,
                        random_kstest
                        )
from .version import version as __version__

ext.IArrayInit()
