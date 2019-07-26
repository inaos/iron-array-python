from time import time
import iarray as ia
import numpy as np

# Number of iterations per benchmark
NITER = 10

# Vector sizes and partitions
shape = [10 * 1000 * 1000]
N = int(np.prod(shape))
pshape = [100 * 1000]

block_size = pshape
expression = '(x - 1.35) * (x - 4.45) * (x - 8.5)'
clevel = 1   # compression level
clib = ia.BLOSCLZ  # compression codec
nthreads = 2  # number of threads for the evaluation and/or compression

x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)

# Reference to compare to
y0 = (x - 1.35) * (x - 4.45) * (x - 8.5)

cparams = dict(clib=clib, clevel=clevel, nthreads=nthreads)
xa = ia.linspace(ia.dtshape(shape=shape, pshape=pshape), 0., 10., **cparams)

ya = None

t0 = time()
expr = ia.Expr(eval_flags="iterblosc", **cparams)
# expr = ia.Expr(eval_flags="iterchunk", **cparams)
expr.bind(b'x', xa)
expr.compile(b'(x - 1.35) * (x - 4.45) * (x - 8.5)')
for i in range(NITER):
    ya = expr.eval(shape, pshape, np.float64)
print("Block evaluate via iarray.eval:", round((time() - t0) / NITER, 4))
y1 = ia.iarray2numpy(ya)
np.testing.assert_almost_equal(y0, y1)

t0 = time()
x = xa
for i in range(NITER):
    ya = ((x - 1.35) * (x - 4.45) * (x - 8.5)).eval(method="iarray_eval", eval_flags="iterblosc", **cparams)
    # ya = ((x - 1.35) * (x - 4.45) * (x - 8.5)).eval(method="iarray_eval", eval_flags="iterchunk")
print("Block evaluate via iarray.LazyExpr.eval('iarray_eval')):", round((time() - t0) / NITER, 4))
y1 = ia.iarray2numpy(ya)
np.testing.assert_almost_equal(y0, y1)
