# Example of filling a triangular array (similar to numpy.tri()) using UDFs

import numpy as np

import iarray as ia
from iarray.udf import jit, Array, float64, int64


# Number of iterations per benchmark
NITER = 5

# Define array params
shape = [10, 10]
dtype = np.float64

# Let's favor speed during computations
ia.set_config_defaults(favor=ia.Favor.SPEED, dtype=dtype)


@jit(verbose=0)
def tri(out: Array(float64, 2), k: int64) -> int64:
    n = out.window_shape[0]
    m = out.window_shape[1]
    row_start = out.window_start[0]
    col_start = out.window_start[1]
    for i in range(n):
        for j in range(m):
            if (row_start + i) >= (col_start + j - k):
                out[i, j] = 1
            else:
                out[i, j] = 0
    return 0


# Build an Expr() instance with not input arrays, and a single scalar param
# As there are no input arrays, we need to specify a shape!
ia_in = ia.empty(shape)
expr = ia.expr_from_udf(tri, [], [1], shape=shape)
ia_out = expr.eval()
print(ia_out.info)
out = ia.iarray2numpy(ia_out)
print(out)
