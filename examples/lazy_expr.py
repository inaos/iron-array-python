# Evaluation of reductions inside (lazy) expressions.

from time import time
import iarray as ia
import numpy as np

# Number of iterations per benchmark
NITER = 1

# Vector sizes and partitions
# shape = [10_000, 1_000]
shape = [10, 1_000]
N = int(np.prod(shape))

x = ia.linspace(shape, 0.0, 10.0)
y = ia.arange(shape)
z = ia.linspace(shape, 0.0, 10.0)
# ia.set_config_defaults(btune=False, clevel=0) #, fp_mantissa_bits=20)

res = None
t0 = time()
for i in range(NITER):
    lazyexpr = (x[(y == 30) & ~(z == 4)])
    res = lazyexpr.eval()
print("Block evaluate via iarray.LazyExpr.eval('iarray_eval')):", round((time() - t0) / NITER, 4))
print("- Result cratio:", round(res.cratio, 2))
print(y.data)

# check
resnp = None
xnp = x.data
ynp = y.data
znp = z.data
t0 = time()
for i in range(NITER):
    resnp = np.where((ynp == 30) & ~(znp == 4), xnp, np.nan)
print("Block evaluate via numpy:", round((time() - t0) / NITER, 4))
np.testing.assert_almost_equal(resnp, res.data, decimal=3)
