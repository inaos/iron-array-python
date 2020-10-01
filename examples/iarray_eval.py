# Evaluation of complex expressions via low-level and high-level (lazy) API of iarray.

from time import time
import iarray as ia
import numpy as np

# Number of iterations per benchmark
NITER = 10

# Vector sizes and partitions
shape = [100 * 1000 * 1000]
N = int(np.prod(shape))
dtype = np.float64

expression = '(x - 1.35) * (x - 4.45) * (x - 8.5)'
clevel = 1   # compression level
clib = ia.LZ4  # compression codec
nthreads = 8  # number of threads for the evaluation and/or compression

x = np.linspace(0, 10, N, dtype=dtype).reshape(shape)

# Reference to compare to
y0 = None
t0 = time()
for i in range(NITER):
    y0 = (x - 1.35) * (x - 4.45) * (x - 8.5)
print("Regular evaluate via numpy:", round((time() - t0) / NITER, 4))

cparams = dict(clib=clib, clevel=clevel, nthreads=nthreads)
dtshape = ia.dtshape(shape=shape, dtype=dtype)
xa = ia.linspace(dtshape, 0., 10., **cparams)
print("Operand cratio:", round(xa.cratio, 2))

ya = None

t0 = time()
expr = ia.Expr(**cparams)
expr.bind('x', xa)
expr.bind_out_properties(dtshape)
expr.compile('(x - 1.35) * (x - 4.45) * (x - 8.5)')
for i in range(NITER):
    ya = expr.eval()
print("Result cratio:", round(ya.cratio, 2))
print("Block evaluate via iarray.eval:", round((time() - t0) / NITER, 4))
y1 = ia.iarray2numpy(ya)
np.testing.assert_almost_equal(y0, y1)

t0 = time()
x = xa
for i in range(NITER):
    ya = ((x - 1.35) * (x - 4.45) * (x - 8.5)).eval(dtshape, **cparams)
print("Block evaluate via iarray.LazyExpr.eval('iarray_eval')):", round((time() - t0) / NITER, 4))
y1 = ia.iarray2numpy(ya)
np.testing.assert_almost_equal(y0, y1)
