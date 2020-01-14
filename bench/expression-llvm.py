from time import time
import os
import iarray as ia
from iarray import udf
import numpy as np

from py2llvm import Array, float64, int64

# Number of iterations per benchmark
NITER = 10
PROFILE = False

cparams = dict(clib=ia.LZ4, clevel=5, nthreads=16) #, blocksize=1024)


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
if PROFILE:
    a1_fname = "a1.iarray"
    if not os.path.isfile(a1_fname):
        print(f"Creating {a1_fname}")
        a1 = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 10, filename=a1_fname, **cparams)
    else:
        print(f"Reading {a1_fname}")
        a1 = ia.from_file(a1_fname, load_in_mem=True, **cparams)
else:
    a1 = ia.linspace(ia.dtshape(shape, pshape, dtype), 0, 10, **cparams)

a2 = np.linspace(0, 10, shape[0], dtype=dtype).reshape(shape)

print("numpy evaluation...")
x = a2.copy()
y = a2.copy()
z = a2.copy()
t0 = time()
for i in range(NITER):
    bn = eval("(x - 1.35) * (y - 4.45) * (z - 8.5)", {"x": x, "y": y, "z": z})
print("Time for numpy eval:", round((time() - t0) / NITER, 3))
print(bn)

print("iarray evaluation...")
cparams2 = cparams.copy()
# cparams2.update(dict(fp_mantissa_bits=3, clevel=5))
# cparams2.update(dict(clevel=5))
expr = f.create_expr([a1], **cparams2)
# And now, the expression
t0 = time()
for i in range(NITER):
    b1 = expr.eval(shape, pshape, dtype)
print("Time for llvm eval:", round((time() - t0) / NITER, 3))
b1_n = ia.iarray2numpy(b1)
print(b1_n)

t0 = time()
x = a1.copy(view=False)
y = a1.copy(view=False)
z = a1.copy(view=False)
expr = ia.Expr(eval_flags="iterblosc", **cparams2)
# expr.bind('x', a1)
# expr.compile('(x - 1.35) * (x - 4.45) * (x - 8.5)')
expr.bind('x', x)
expr.bind('y', y)
expr.bind('z', z)
expr.compile('(x - 1.35) * (y - 4.45) * (z - 8.5)')
for i in range(NITER):
    b2 = expr.eval(shape, pshape, dtype)
print("Time for juggernaut eval:", round((time() - t0) / NITER, 3))
b2_n = ia.iarray2numpy(b2)
print(b2_n)

try:
    np.testing.assert_almost_equal(bn, b1_n)
    np.testing.assert_almost_equal(bn, b2_n)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
