from time import time
import iarray as ia
from iarray import udf
import numpy as np

from py2llvm import Array, float64, int64

# Number of iterations per benchmark
NITER = 10

cparams = dict(clib=ia.LZ4, clevel=5, nthreads=20) #, blocksize=1024)


@udf.jit(verbose=0)
def f(out: Array(float64, 1), inputs: Array(float64, 1)) -> int64:
    n = out.shape[0]
    x = inputs[0]
    #y = inputs[1]

    for i in range(n):
        out[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


# Define array params
# shape = [10000, 2000]
# pshape = [1000, 200]
shape = [20 * 1000 * 1000]
pshape = [2000 * 1000]
dtype = np.float64


# Create initial containers
a1 = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 10)
#a2 = ia.iarray2numpy(a1)
a2 = np.linspace(0, 10, shape[0], dtype=dtype).reshape(shape)


print("iarray evaluation via py2llvm...")
expr = f.create_expr([a1], **cparams)
t0 = time()
for i in range(NITER):
    b1 = expr.eval(shape, pshape, dtype)
b1_n = ia.iarray2numpy(b1)
print("Time for py2llvm eval:", round((time() - t0) / NITER, 3))
print(b1_n)

print("iarray evaluation via juggernaut (expr -> LLVM)...")
expr = ia.Expr(eval_flags="iterblosc", **cparams)
expr.bind('x', a1)
expr.compile('(x - 1.35) * (x - 4.45) * (x - 8.5)')
t0 = time()
for i in range(NITER):
    b2 = expr.eval(shape, pshape, dtype)
b2_n = ia.iarray2numpy(b2)
print("Time for juggernaut eval:", round((time() - t0) / NITER, 3))
print(b2_n)

print("numpy evaluation...")
t0 = time()
for i in range(NITER):
    bn = eval("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": a2})
print("Time for numpy eval:", round((time() - t0) / NITER, 3))
print(bn)

try:
    np.testing.assert_almost_equal(bn, b1_n)
    np.testing.assert_almost_equal(bn, b2_n)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
