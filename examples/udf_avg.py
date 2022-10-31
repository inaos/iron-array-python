# Performance of User Defined Functions (UDF) compared with internal compiler.
# This exercises conditionals inside of a UDF function.

from functools import reduce
from time import time

import numpy as np

import iarray as ia
from iarray.udf import jit, Array, float64


# Number of iterations per benchmark
NITER = 1

# Define array params
shape = [20_000_000]
dtype = np.float64

# Most of modern computers can reach 8 threads
ia.set_config_defaults(nthreads=8, dtype=dtype)


@jit
def f(out: Array(float64, 1), x: Array(float64, 1)) -> int:
    n = x.window_shape[0]
    # base = x.window_start[0]
    for i in range(n):
        # i_abs = base + i  # Absolute i(ndex) within the whole array
        value = x[i]
        value += x[i - 1] if i > 0 else x[i]
        value += x[i + 1] if i < n - 1 else x[i]
        out[i] = value / 3

    return 0


# Create input arrays
ia_in = ia.linspace(0, 10, int(np.prod(shape)), shape=shape)
np_in = np.linspace(0, 10, reduce(lambda x, y: x * y, shape), dtype=dtype).reshape(shape)
ia.cmp_arrays(np_in, ia_in)
# print(np_in)

# iarray UDF evaluation
expr = ia.expr_from_udf(f, [ia_in])
ia_out = None  # fix a warning
t0 = time()
for i in range(NITER):
    ia_out = expr.eval()
print("Time for UDF eval:", round((time() - t0) / NITER, 3))
ia_out = ia.iarray2numpy(ia_out)
# print(ia_out)
