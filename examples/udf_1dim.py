# Performance of User Defined Functions (UDF) compared with internal compiler.
# This is for 1-dim arrays.

import math
from time import time

import numpy as np

import iarray as ia
from iarray.udf import jit, Array, float64


# Number of iterations per benchmark
NITER = 5

# Define array params
shape = [20_000_000]
dtype = np.float64
# Let's favor speed during computations
ia.set_config_defaults(favor=ia.Favor.SPEED, dtype=dtype)


@jit
def f(out: Array(float64, 1), x: Array(float64, 1)) -> int:
    n = out.shape[0]
    for i in range(n):
        out[i] = (math.sin(x[i]) - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


# Create initial containers
a1 = ia.linspace(0, 10, int(np.prod(shape)), shape=shape)
a2 = np.linspace(0, 10, shape[0], dtype=dtype).reshape(shape)

print("numpy evaluation...")
bn = None
t0 = time()
for i in range(NITER):
    bn = (np.sin(a2) - 1.35) * (a2 - 4.45) * (a2 - 8.5)
print("Time for numpy eval:", round((time() - t0) / NITER, 3))
print(bn)

print("iarray evaluation ...")
expr = ia.expr_from_udf(f, [a1])
b1 = None
t0 = time()
for i in range(NITER):
    b1 = expr.eval()
print("Time for UDF eval:", round((time() - t0) / NITER, 3))
print(f"CRatio for result: {b1.cratio:.3f}")
b1_n = ia.iarray2numpy(b1)
print(b1_n)

ia.cmp_arrays(bn, b1_n, success="OK. Results are the same.")

expr = ia.expr_from_string("(sin(x) - 1.35) * (x - 4.45) * (x - 8.5)", {"x": a1})
b2 = None
t0 = time()
for i in range(NITER):
    b2 = expr.eval()
print("Time for internal compiler eval:", round((time() - t0) / NITER, 3))
print(f"CRatio for result: {b1.cratio:.3f}")
b2_n = ia.iarray2numpy(b2)
print(b2_n)

ia.cmp_arrays(bn, b2_n, success="OK. Results are the same.")
