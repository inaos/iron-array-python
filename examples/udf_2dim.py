from functools import reduce
import math
from time import time

import numpy as np

import iarray as ia
from iarray.udf import jit, Array
from iarray.py2llvm import float64, int64


# Number of iterations per benchmark
NITER = 10

# Define array params
shape = [20, 20]
chunkshape = [20, 19]

dtype = np.float64


cparams = dict(clib=ia.LZ4, clevel=5)
storage = ia.StorageProperties("blosc", chunkshape, chunkshape)
dtshape = ia.dtshape(shape, dtype)


@jit(verbose=0)
def f(out: Array(float64, 2), x: Array(float64, 2)) -> int64:
    n = x.window_shape[0]
    m = x.window_shape[1]
    for i in range(n):
        for j in range(m):
            out[i,j] = (math.sin(x[i,j]) - 1.35) * (x[i,j] - 4.45) * (x[i,j] - 8.5)

    return 0


if __name__ == '__main__':
    # Create input arrays
    ia_in = ia.linspace(dtshape, 0, 10, storage=storage, **cparams)
    np_in = np.linspace(0, 10, reduce(lambda x, y: x * y, shape), dtype=dtype).reshape(shape)
    ia.cmp_arrays(np_in, ia_in)


    # iarray udf evaluation
    print("iarray evaluation ...")
    expr = f.create_expr([ia_in], dtshape, storage=storage, **cparams)
    t0 = time()
    for i in range(NITER):
        ia_out = expr.eval()
    print("Time for py2llvm eval:", round((time() - t0) / NITER, 3))
    ia_out = ia.iarray2numpy(ia_out)
    print(ia_out)

    # numpy evaluation
    print("numpy evaluation...")
    t0 = time()
    for i in range(NITER):
        np_out = (np.sin(np_in) - 1.35) * (np_in - 4.45) * (np_in - 8.5)
    print("Time for numpy eval:", round((time() - t0) / NITER, 3))
    print(np_out)

    # compare
    ia.cmp_arrays(np_out, ia_out, success='OK. Results are the same.')
