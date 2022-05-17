# Evaluation of reductions inside (lazy) expressions.

from time import time
import iarray as ia
import numpy as np

# Number of iterations per benchmark
NITER = 3

# Vector sizes and partitions
shape = [10_000, 1_000]
N = int(np.prod(shape))

x = ia.linspace(shape, 0.0, 10.0)
# ia.set_config_defaults(btune=False, clevel=0) #, fp_mantissa_bits=20)

y = None
t0 = time()
for i in range(NITER):
    y = ((x.sum(axis=1) - 1.35) *  x[:,1]).eval()
print("Block evaluate via iarray.LazyExpr.eval('iarray_eval')):", round((time() - t0) / NITER, 4))
print("- Result cratio:", round(y.cratio, 2))
print(y.data)

# check
xnp = x.data
ynp = None
t0 = time()
for i in range(NITER):
    ynp = (xnp.sum(axis=1) - 1.35) * xnp[:,1]
print("Block evaluate via numpy:", round((time() - t0) / NITER, 4))
np.testing.assert_almost_equal(ynp, y.data, decimal=3)
