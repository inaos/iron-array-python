from functools import reduce
from time import time

import numpy as np

import iarray as ia
from iarray.udf import jit, Array
from py2llvm import float64, int64


# Number of iterations per benchmark
NITER = 1

# Define array params
shape = [100]
pshape = [6]
dtype = np.float64

blocksize = reduce(lambda x, y: x * y, pshape) * dtype(0).itemsize
cparams = dict(clib=ia.LZ4, clevel=5, nthreads=16, blocksize=blocksize)


def cmp(a, b, success=None):
    if type(a) is ia.high_level.IArray:
        a = ia.iarray2numpy(a)

    if type(b) is ia.high_level.IArray:
        b = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(a, b)
    if success is not None:
        print(success)


@jit(verbose=1)
def f(out: Array(float64, 1), x: Array(float64, 1)) -> int64:
    n = x.window_shape[0]
    #base = x.window_start[0]
    for i in range(n):
        #i_abs = base + i # Absolute i(ndex) within the whole array
        value = x[i]
        value += x[i-1] if i > 0 else x[i]
        value += x[i+1] if i < n-1 else x[i]
        out[i] = value / 3

    return 0


if __name__ == '__main__':
    # Create input arrays
    ia_in = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 10, **cparams)
    np_in = np.linspace(0, 10, reduce(lambda x,y: x*y, shape), dtype=dtype).reshape(shape)
    cmp(np_in, ia_in)

    print(np_in)

    # iarray udf evaluation
    print("iarray evaluation ...")
    expr = f.create_expr([ia_in], ia.dtshape(shape, pshape, dtype), **cparams)
    t0 = time()
    for i in range(NITER):
        ia_out = expr.eval()
    print("Time for py2llvm eval:", round((time() - t0) / NITER, 3))
    ia_out = ia.iarray2numpy(ia_out)
    print(ia_out)

#   # numpy evaluation
#   print("numpy evaluation...")
#   t0 = time()
#   for i in range(NITER):
#       np_out = (np.sin(np_in) - 1.35) * (np_in - 4.45) * (np_in - 8.5)
#   print("Time for numpy eval:", round((time() - t0) / NITER, 3))
#   print(np_out)

#   # compare
#   cmp(np_out, ia_out, success='OK. Results are the same.')
