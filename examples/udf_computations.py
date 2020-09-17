
import math
import iarray as ia
import numpy as np
from time import time
from iarray.udf import Array, jit, float64, int64
import numba as nb

max_num_threads = 8
nrep = 5

# Functions to compute

# For juggernaut
str_expr = "sin(x) * atan(x)"  # this works!
#str_expr = "sin(x) * atan2(x, 0.5)"  # try this!
#str_expr = "sin(x) * pow(x, 0.5)"  # try this!

@jit
def poly_udf(y: Array(float64, 1), x: Array(float64, 1)) -> int64:
    n = x.shape[0]
    for i in range(n):
        s = math.sin(x[i])
        a = math.atan(x[i])  # this works!
        #a = math.atan2(x[i], 0.5)  # try this!
        #a = math.pow(x[i], 0.5)  # try this!
        y[i] = s * a  # try this!

    return 0

@nb.njit(parallel=True)
def poly_numba(x):
    y = np.empty(x.shape, x.dtype)
    for i in nb.prange(len(x)):
        s = math.sin(x[i])
        a = math.atan(x[i])   # this works!
        #a = math.atan2(x[i], 0.5)  # try this!
        #a = math.pow(x[i], 0.5)  # try this!
        y[i] = s * a
    return y

# Define array params
shape = [10 * 1000 * 1000]
chunkshape = [1 * 1000 * 1000]
blockshape = [8 * 1000]
dtshape = ia.dtshape(shape)
size = int(np.prod(shape))
nthreads = 6
clevel = 5

# iarray UDF
bstorage = ia.StorageProperties("blosc", chunkshape, blockshape)
kwargs = dict(nthreads=nthreads, clevel=clevel, storage=bstorage)
a1 = ia.linspace(dtshape, 0, 10, **kwargs)
expr = poly_udf.create_expr([a1], dtshape, method="auto", **kwargs)
t0 = time()
b1 = expr.eval()
t1 = time()
print("Time to evaluate expression with iarray.udf:", round(t1 - t0, 3))

# iarray juggernaut
expr = ia.Expr(**kwargs)
expr.bind("x", a1)
expr.bind_out_properties(dtshape, storage=bstorage)
expr.compile(str_expr)
t0 = time()
b2 = expr.eval()
t1 = time()
print("Time to evaluate expression with iarray (juggernaut):", round(t1 - t0, 3))

# numba
a1 = np.linspace(0, 10, size).reshape(shape)
nb.set_num_threads(nthreads)
t0 = time()
np3 = poly_numba(a1)
t1 = time()
print("Time to evaluate expression with numba:", round(t1 - t0, 3))

# Compare results.  The regular juggernaut works.
np2 = ia.iarray2numpy(b2)
np.testing.assert_almost_equal(np2, np3, decimal=5)

# The UDF result fails
np1 = ia.iarray2numpy(b1)
np.testing.assert_almost_equal(np1, np3, decimal=5)
