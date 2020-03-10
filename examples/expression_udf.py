import math
from time import time

import numpy as np

import iarray as ia
from iarray.udf import jit, Array
from py2llvm import float64, int64


# Number of iterations per benchmark
NITER = 10

# Define array params
# shape = [10000, 2000]
# pshape = [1000, 200]
shape = [20 * 1000 * 1000]
pshape = [4 * 1000 * 1000]
dtype = np.float64

cparams = dict(clib=ia.LZ4, clevel=5, nthreads=16) #, blocksize=1024)


@jit(verbose=0)
def f(out: Array(float64, 1), x: Array(float64, 1)) -> int64:
    n = out.shape[0]
    for i in range(n):
        out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


# Create initial containers
a1 = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 10, **cparams)
a2 = np.linspace(0, 10, shape[0], dtype=dtype).reshape(shape)


print("iarray evaluation ...")
expr = f.create_expr([a1], ia.dtshape(shape, pshape, dtype), **cparams)
# expr = ia.Expr(eval_flags="iterblosc", **cparams)
# expr.bind('x', a1)
# expr.compile_udf(f)
t0 = time()
for i in range(NITER):
    b1 = expr.eval()
print("Time for py2llvm eval:", round((time() - t0) / NITER, 3))
b1_n = ia.iarray2numpy(b1)
print(b1_n)

eval_flags = ia.EvalFlags(method="iterblosc", engine="juggernaut")
expr = ia.Expr(eval_flags=eval_flags, **cparams)
expr.bind('x', a1)
expr.compile('(sin(x) - 1.35) * (x - 4.45) * (x - 8.5)')
t0 = time()
for i in range(NITER):
    b2 = expr.eval(shape, pshape, dtype)
print("Time for juggernaut eval:", round((time() - t0) / NITER, 3))
b2_n = ia.iarray2numpy(b2)
print(b2_n)

print("numpy evaluation...")
t0 = time()
for i in range(NITER):
    # bn = eval("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": a2})
    bn = (np.sin(a2) - 1.35) * (a2 - 4.45) * (a2 - 8.5)
print("Time for numpy eval:", round((time() - t0) / NITER, 3))
print(bn)

try:
    np.testing.assert_almost_equal(bn, b1_n)
    np.testing.assert_almost_equal(bn, b2_n)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")