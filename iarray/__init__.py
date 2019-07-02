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


from .iarray_ext import IarrayInit, Expression  # the order of the import is important: extensions first
from .container import (IArray, LazyExpr, empty2, arange2, linspace2, zeros2, ones2, full2, from_file2, iarray2numpy2, numpy2iarray2,
                        random_rand2, random_randn2, random_beta2, random_lognormal2, random_exponential2, random_uniform2,
                        random_normal2, random_bernoulli2, random_binomial2, random_poisson2,
                        random_kstest2,
                        matmul2, Config)
from .version import version as __version__

IarrayInit()
