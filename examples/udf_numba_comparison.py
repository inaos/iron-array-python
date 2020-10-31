# Example showing different combinations of complex expression, including transcendental functions.
# It compares iArray UDF expressions with numba, so you will need numba installed.

import math
import iarray as ia
import numpy as np
from time import time
from iarray.udf import Array, jit, float64, int64
import numba as nb


# Functions to compute

# For the internal iarray computational engine
str_expr = "sin(x) * arctan(x)"
# str_expr = "sin(x) * pow(x, 0.5)"  # try this!

# Define array params
shape = [10 * 1000 * 1000]
dtshape = ia.DTShape(shape)
nthreads = 8
clevel = 9
nrep = 5
ia.set_config(nthreads=nthreads, clevel=clevel)

# Create the iarray object
a1 = ia.linspace(dtshape, 0, 10)
np_a1 = np.linspace(0, 10, shape[0])


@jit
def func_udf(y: Array(float64, 1), x: Array(float64, 1)) -> int64:
    n = x.shape[0]
    for i in range(n):
        s = math.sin(x[i])
        a = math.atan(x[i])
        # a = math.pow(x[i], 0.5)  # try this!
        y[i] = s * a  # try this!

    return 0


@nb.njit(parallel=True)
def func_numba(x, y):
    for i in nb.prange(len(x)):
        s = math.sin(x[i])
        a = math.atan(x[i])
        # a = math.pow(x[i], 0.5)  # try this!
        y[i] = s * a


x = np_a1
# numba
nb.set_num_threads(nthreads)
np0 = np.empty(x.shape, x.dtype)
t0 = time()
func_numba(np_a1, np0)
t1 = time()
print("Time for numba:", round(t1 - t0, 3))

# iarray UDF
t0 = time()
# expr = func_udf.create_expr([a1], dtshape)
expr = ia.create_expr(func_udf, [a1], dtshape)
b1 = expr.eval()
t1 = time()
print("Time to evaluate expression with iarray.udf:", round(t1 - t0, 3))
# Compare results
np1 = ia.iarray2numpy(b1)
np.testing.assert_almost_equal(np1, np0)

# iarray internal engine
t0 = time()
expr = ia.create_expr(str_expr, {"x": a1}, dtshape)
b1 = expr.eval()
t1 = time()
print("Time for iarray (high-level API, internal engine):", round(t1 - t0, 3))
# Compare results
np1 = ia.iarray2numpy(b1)
np.testing.assert_almost_equal(np1, np0)

# iarray internal engine (lazy expressions)
t0 = time()
b1 = (a1.sin() * a1.atan()).eval(dtshape)
t1 = time()
print("Time for iarray (lazy eval, internal engine):", round(t1 - t0, 3))
# Compare results
np1 = ia.iarray2numpy(b1)
np.testing.assert_almost_equal(np1, np0)
