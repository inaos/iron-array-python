# Performance of User Defined Functions (UDF) compared with internal compiler.
# This exercises conditionals inside of a UDF function.

from functools import reduce
from time import time

import numpy as np

import iarray as ia
from iarray.udf import jit, Array, float64, int64


# Number of iterations per benchmark
NITER = 1

# Define array params
shape = [100]
pshape = [6]
dtype = np.float64

blocksize = reduce(lambda x, y: x * y, pshape) * dtype(0).itemsize
cparams = dict(clib=ia.LZ4, clevel=5, nthreads=16, blocksize=blocksize)
storage = ia.StorageProperties("blosc", pshape, pshape)
dtshape = ia.dtshape(shape, dtype)


@jit(verbose=0)
def f(out: Array(float64, 1), x: Array(float64, 1)) -> int64:
    n = x.window_shape[0]
    # base = x.window_start[0]
    for i in range(n):
        # i_abs = base + i # Absolute i(ndex) within the whole array
        value = x[i]
        value += x[i - 1] if i > 0 else x[i]
        value += x[i + 1] if i < n - 1 else x[i]
        out[i] = value / 3

    return 0


# Create input arrays
ia_in = ia.linspace(dtshape, 0, 10, storage=storage, **cparams)
np_in = np.linspace(0, 10, reduce(lambda x, y: x * y, shape), dtype=dtype).reshape(shape)
ia.cmp_arrays(np_in, ia_in)

print(np_in)

# iarray UDF evaluation
print("iarray UDF evaluation ...")
expr = f.create_expr([ia_in], dtshape, storage=storage, **cparams)
ia_out = None  # fix a warning
t0 = time()
for i in range(NITER):
    ia_out = expr.eval()
print("Time for UDF eval:", round((time() - t0) / NITER, 3))
ia_out = ia.iarray2numpy(ia_out)
print(ia_out)
