# Evaluation of complex expressions via low-level and high-level (lazy) API of iarray.

from time import time
import iarray as ia
import numpy as np

# Number of iterations per benchmark
NITER = 5

# Do lossy compression for improved compression ratio
ia.set_config(clevel=9, fp_mantissa_bits=20)

# Vector sizes and partitions
shape = [10_000_000]
N = int(np.prod(shape))
dtype = np.float64

expression = "(x - 1.35) * (x - 4.45) * (x - 8.5)"

x = np.linspace(0, 10, N, dtype=dtype).reshape(shape)

# Reference to compare to
y0 = None
t0 = time()
for i in range(NITER):
    y0 = (x - 1.35) * (x - 4.45) * (x - 8.5)
print("Regular evaluate via numpy:", round((time() - t0) / NITER, 4))

dtshape = ia.DTShape(shape=shape, dtype=dtype)
xa = ia.linspace(dtshape, 0.0, 10.0)
print("Operand cratio:", round(xa.cratio, 2))

ya = None

t0 = time()
expr = ia.create_expr("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": xa}, dtshape)
for i in range(NITER):
    ya = expr.eval()
print("Result cratio:", round(ya.cratio, 2))
print("Block evaluate via iarray.eval:", round((time() - t0) / NITER, 4))
y1 = ia.iarray2numpy(ya)
np.testing.assert_almost_equal(y0, y1, decimal=3)

t0 = time()
x = xa
for i in range(NITER):
    ya = ((x - 1.35) * (x - 4.45) * (x - 8.5)).eval(dtshape)
print("Block evaluate via iarray.LazyExpr.eval('iarray_eval')):", round((time() - t0) / NITER, 4))
y1 = ia.iarray2numpy(ya)
np.testing.assert_almost_equal(y0, y1, decimal=3)
