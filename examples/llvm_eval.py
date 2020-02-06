from time import time
import iarray as ia
import numpy as np

# Number of iterations per benchmark
NITER = 10

# Vector sizes and partitions
shape = [10 * 1000 * 1000]
N = int(np.prod(shape))
pshape = [200 * 1000]
# pshape = None  # for enforcing a plain buffer
dtype = np.float64

block_size = pshape
expression = '(x - 1.35) * (x - 4.45) * (x - 8.5)'
clevel = 1   # compression level
clib = ia.BLOSCLZ  # compression codec
nthreads = 4  # number of threads for the evaluation and/or compression

x = np.linspace(0, 10, N, dtype=dtype).reshape(shape)

# Reference to compare to
t0 = time()
for i in range(NITER):
    y0 = (x - 1.35) * (x - 4.45) * (x - 8.5)
print("Regular evaluate via numpy:", round((time() - t0) / NITER, 4))

cparams = dict(clib=clib, clevel=clevel, nthreads=nthreads)
xa = ia.linspace(ia.dtshape(shape=shape, pshape=pshape, dtype=dtype), 0., 10., **cparams)
print("Operand cratio:", round(xa.cratio, 2))

ya = None

if pshape is None:
    eval_method = "iterblock"
else:
    eval_method = "iterblosc"

t0 = time()
expr = ia.Expr(eval_flags=eval_method, **cparams)
expr.bind('x', xa)
expr.compile('(x - 1.35) * (x - 4.45) * (x - 8.5)')
for i in range(NITER):
    ya = expr.eval(ia.dtshape(shape, pshape, dtype=dtype))
print("Result cratio:", round(ya.cratio, 2))
print("Block evaluate via iarray.eval:", round((time() - t0) / NITER, 4))
y1 = ia.iarray2numpy(ya)
np.testing.assert_almost_equal(y0, y1)

t0 = time()
x = xa
for i in range(NITER):
    ya = ((x - 1.35) * (x - 4.45) * (x - 8.5)).eval(method="iarray_eval", pshape=pshape,
                                                    dtype=dtype, eval_flags=eval_method, **cparams)
print("Block evaluate via iarray.LazyExpr.eval('iarray_eval')):", round((time() - t0) / NITER, 4))
y1 = ia.iarray2numpy(ya)
np.testing.assert_almost_equal(y0, y1)
