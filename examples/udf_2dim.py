# Performance of User Defined Functions (UDF) compared with internal compiler.
# This is for 2-dim arrays.

from functools import reduce
import math
from time import time

import numpy as np

import iarray as ia
from iarray.udf import jit, Array, float64, int64


# Number of iterations per benchmark
NITER = 5

# Define array params
shape = [1000, 10_000]
dtype = np.float64

# Let's favor speed during computations
ia.set_config_defaults(favor=ia.Favor.SPEED, dtype=dtype)


@jit
def f(out: Array(float64, 2), x: Array(float64, 2)) -> int64:
    n = x.window_shape[0]
    m = x.window_shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = (math.sin(x[i, j]) - 1.35) * (x[i, j] - 4.45) * (x[i, j] - 8.5)

    return 0


# Create input arrays
ia_in = ia.arange(shape, 0, np.prod(shape), 1)
np_in = np.arange(0, reduce(lambda x, y: x * y, shape), 1, dtype=dtype).reshape(shape)
ia.cmp_arrays(np_in, ia_in)

# numpy evaluation
print("numpy evaluation...")
np_out = None
t0 = time()
for i in range(NITER):
    np_out = (np.sin(np_in) - 1.35) * (np_in - 4.45) * (np_in - 8.5)
print("Time for numpy eval:", round((time() - t0) / NITER, 3))

# iarray udf evaluation
print("iarray evaluation ...")
expr = ia.expr_from_udf(f, [ia_in])
ia_out = None
t0 = time()
for i in range(NITER):
    ia_out = expr.eval()
print("Time for UDF eval:", round((time() - t0) / NITER, 3))
print(f"CRatio for result: {ia_out.cratio:.3f}")
ia_out = ia.iarray2numpy(ia_out)

# compare
ia.cmp_arrays(np_out, ia_out, success="OK. Results are the same.")
