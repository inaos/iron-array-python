import iarray as ia
import numpy as np

# Number of iterations per benchmark
NITER = 10

# Define array params
# shape = [10000, 2000]
# pshape = [1000, 200]
shape = [10 * 1000 * 1000]
pshape = [200 * 1000]
dtype = np.float64

# Create initial containers
a1 = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 10)
#a2 = ia.iarray2numpy(a1)
a2 = np.linspace(0, 10, shape[0], dtype=dtype).reshape(shape)


print("iarray evaluation...")

# And now, the expression
expr = ia.Expr(eval_flags="iterblosc", nthreads=1)
expr.bind("x", a1)
expr.compile("(x - 1.35) * (x - 4.45) * (x - 8.5)")
for i in range(NITER):
    b1 = expr.eval(ia.dtshape(shape, pshape, np.float64))
b1_n = ia.iarray2numpy(b1)
print(b1_n)

print("numpy evaluation...")
for i in range(NITER):
    b2 = eval("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": a2})
print(b2)

try:
    np.testing.assert_almost_equal(b2, b1_n)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
