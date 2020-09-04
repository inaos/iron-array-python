# Performance of User Defined Functions (UDF) compared with internal compiler.
# This is for 2-dim arrays.

from functools import reduce
import math
from time import time

import numpy as np

import iarray as ia
from iarray.udf import jit, Array, float64, int64


# Number of iterations per benchmark
NITER = 10

# Define array params
shape = [1000, 1000]
cshape = [100, 90]
bshape = [14, 11]

dtype = np.float64

cparams = dict(clevel=5, nthreads=8)
storage = ia.StorageProperties("blosc", cshape, bshape)
dtshape = ia.dtshape(shape, dtype)


@jit(verbose=0)
def f(out: Array(float64, 2), x: Array(float64, 2)) -> int64:
    n = x.window_shape[0]
    m = x.window_shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = (math.sin(x[i, j]) - 1.35) * (x[i, j] - 4.45) * (x[i, j] - 8.5)

    return 0


# Create input arrays
ia_in = ia.arange(dtshape, 0, np.prod(shape), 1, storage=storage, **cparams)
np_in = np.arange(0, reduce(lambda x, y: x * y, shape), 1, dtype=dtype).reshape(shape)
ia.cmp_arrays(np_in, ia_in)

# iarray udf evaluation
print("iarray evaluation ...")
expr = f.create_expr([ia_in], dtshape, storage=storage, **cparams)
ia_out = None
t0 = time()
for i in range(NITER):
    ia_out = expr.eval()
print("Time for UDF eval:", round((time() - t0) / NITER, 3))
ia_out = ia.iarray2numpy(ia_out)

# numpy evaluation
print("numpy evaluation...")
np_out = None
t0 = time()
for i in range(NITER):
    np_out = (np.sin(np_in) - 1.35) * (np_in - 4.45) * (np_in - 8.5)
print("Time for numpy eval:", round((time() - t0) / NITER, 3))

# compare
ia.cmp_arrays(np_out, ia_out, success='OK. Results are the same.')
