# Performance of User Defined Functions (UDF) compared with internal compiler.
# This is for 1-dim arrays.

import math
from time import time

import numpy as np

import iarray as ia
from iarray.udf import jit, Array
from iarray.py2llvm import float64, int64


# Number of iterations per benchmark
NITER = 10

# Define array params
shape = [20 * 1000 * 1000]
cshape = [1000 * 1000]
bshape = [10 * 1000]
dtype = np.float64

cparams = dict(clevel=5, nthreads=8)
storage = ia.StorageProperties("blosc", cshape, bshape)
dtshape = ia.dtshape(shape, dtype)


@jit(verbose=0)
def f(out: Array(float64, 1), x: Array(float64, 1)) -> int64:
    n = out.shape[0]
    for i in range(n):
        out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


# Create initial containers
a1 = ia.linspace(dtshape, 0, 10, storage=storage, **cparams)
a2 = np.linspace(0, 10, shape[0], dtype=dtype).reshape(shape)

print("numpy evaluation...")
t0 = time()
for i in range(NITER):
    bn = (np.sin(a2) - 1.35) * (a2 - 4.45) * (a2 - 8.5)
print("Time for numpy eval:", round((time() - t0) / NITER, 3))
print(bn)

print("iarray evaluation ...")
expr = f.create_expr([a1], dtshape, storage=storage, **cparams)
t0 = time()
for i in range(NITER):
    b1 = expr.eval()
print("Time for UDF eval:", round((time() - t0) / NITER, 3))
b1_n = ia.iarray2numpy(b1)
print(b1_n)

ia.cmp_arrays(bn, b1_n, success='OK. Results are the same.')

eval_flags = ia.EvalFlags(method="auto", engine="auto")
expr = ia.Expr(eval_flags=eval_flags, **cparams)
expr.bind('x', a1)
expr.bind_out_properties(dtshape, storage)
expr.compile('(sin(x) - 1.35) * (x - 4.45) * (x - 8.5)')
t0 = time()
for i in range(NITER):
    b2 = expr.eval()
print("Time for internal compiler eval:", round((time() - t0) / NITER, 3))
b2_n = ia.iarray2numpy(b2)
print(b2_n)

ia.cmp_arrays(bn, b2_n, success='OK. Results are the same.')
