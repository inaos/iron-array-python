# Example showing different combinations of complex expression, including transcendental functions.
# It compares iArray UDF expressions with numba.

import math
import iarray as ia
import numpy as np
from time import time
from iarray.udf import Array, jit, float64, int64
import numexpr as ne
import numba as nb


# Functions to compute

# For the internal iarray computational engine
str_expr = "sin(x) * arctan(x)"
#str_expr = "sin(x) * pow(x, 0.5)"  # try this!

# Define array params
shape = [10 * 1000 * 1000]
dtshape = ia.dtshape(shape)
size = int(np.prod(shape))
nthreads = 8
clevel = 9
nrep = 5

# Create the iarray object
kwargs = dict(nthreads=nthreads, clevel=clevel)
a1 = ia.linspace(dtshape, 0, 10, **kwargs)
np_a1 = np.linspace(0, 10, shape[0])


@jit
def func_udf(y: Array(float64, 1), x: Array(float64, 1)) -> int64:
    n = x.shape[0]
    for i in range(n):
        s = math.sin(x[i])
        a = math.atan(x[i])
        #a = math.pow(x[i], 0.5)  # try this!
        y[i] = s * a  # try this!

    return 0

@nb.njit(parallel=True)
def func_numba(x):
    y = np.empty(x.shape, x.dtype)
    for i in nb.prange(len(x)):
        s = math.sin(x[i])
        a = math.atan(x[i])
        #a = math.pow(x[i], 0.5)  # try this!
        y[i] = s * a
    return y

# iarray UDF
t0 = time()
expr = func_udf.create_expr([a1], dtshape, **kwargs)
b1 = expr.eval()
t1 = time()
print("Time to evaluate expression with iarray.udf:", round(t1 - t0, 3))

# iarray internal engine (low level API)
t0 = time()
expr = ia.Expr(**kwargs)
expr.bind("x", a1)
expr.bind_out_properties(dtshape)
expr.compile(str_expr)
b2 = expr.eval()
t1 = time()
print("Time to evaluate expression with iarray (internal engine, low-level API):", round(t1 - t0, 3))

# iarray internal engine (high level API)
t0 = time()
expr = ia.create_expr(str_expr, {"x": a1}, dtshape, **kwargs)
b3 = expr.eval()
t1 = time()
print("Time to evaluate expression with iarray (internal engine, high-level API):", round(t1 - t0, 3))

# iarray internal engine (lazy expressions)
t0 = time()
x = a1
(a1.sin() * a1.atan()).eval(**kwargs)
t1 = time()
print("Time to evaluate expression with iarray (internal engine, lazy eval):", round(t1 - t0, 3))

# numba
nb.set_num_threads(nthreads)
t0 = time()
np0 = func_numba(np_a1)
t1 = time()
print("Time to evaluate expression with numba:", round(t1 - t0, 3))

# Compare results
np1 = ia.iarray2numpy(b1)
np.testing.assert_almost_equal(np1, np0)
np2 = ia.iarray2numpy(b2)
np.testing.assert_almost_equal(np2, np0)
np3 = ia.iarray2numpy(b3)
np.testing.assert_almost_equal(np3, np0)
