# Evaluation of reductions inside (lazy) expressions.

from time import time
import iarray as ia
import numpy as np

# Number of iterations per benchmark
NITER = 1

# Vector sizes and partitions
#shape = [10_000, 1_000]
shape = [10, 1_000]
N = int(np.prod(shape))

x = ia.linspace(shape, 0.0, 10.0)
y = ia.arange(shape)
# ia.set_config_defaults(btune=False, clevel=0) #, fp_mantissa_bits=20)

res = None
t0 = time()
for i in range(NITER):
    # res = ((x.sum(axis=1) - 1.35) *  x[:,1]).eval()
    res = x[y < 4].eval()
print("Block evaluate via iarray.LazyExpr.eval('iarray_eval')):", round((time() - t0) / NITER, 4))
print("- Result cratio:", round(res.cratio, 2))
print(y.data)

# check
resnp = None
xnp = x.data
ynp = y.data
t0 = time()
for i in range(NITER):
    # resnp = (xnp.sum(axis=1) - 1.35) * xnp[:,1]
    resnp = np.where(ynp < 4, xnp, np.nan)
print("Block evaluate via numpy:", round((time() - t0) / NITER, 4))
np.testing.assert_almost_equal(resnp, res.data, decimal=3)
