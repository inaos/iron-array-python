# Calling scalar UDFs from expressions.
# This is for 1-dim arrays.

from time import time

import numpy as np

import iarray as ia
from iarray import udf


# Define array params
shape = [20_000_000]
dtype = np.float64
# Let's favor speed during computations
ia.set_config_defaults(favor=ia.Favor.SPEED, dtype=dtype)
# ia.set_config_defaults(clevel=0, btune=False)


@udf.scalar(verbose=0)
def fsum(a: udf.float64, b: udf.float64) -> float:
    return a + b

@udf.scalar(verbose=0)
def fmult(a: udf.float64, b: udf.float64) -> float:
    return a * b


libs = ia.UdfLibraries()
libs["lib"].register_func(fsum)
libs["lib2"].register_func(fmult)

# Create initial containers
a1 = ia.linspace(shape, 0, 10)
a2 = np.linspace(0, 10, shape[0], dtype=dtype).reshape(shape)

print("pure expr evaluation ...")
expr = "4 * (x * x)"
expr = ia.expr_from_string(expr, {"x": a1})
t0 = time()
b1 = expr.eval()
print("Time for pure expr eval:", round((time() - t0), 3))
print(f"cratio for result: {b1.cratio:.3f}")
b1_n = b1.data
print(b1_n)

print("scalar udf evaluation ...")
# expr = "lib.f(a1, a1)"  # segfault.  fix it by propagating errors correctly!
# expr = "4 * lib.fsum(x, x) + lib2.fmult(x, x)"  # segfaults too
expr = "4 * lib2.fmult(x, x)"
expr = ia.expr_from_string(expr, {"x": a1})
t0 = time()
b1 = expr.eval()
print("Time for UDF eval:", round((time() - t0), 3))
print(f"cratio for result: {b1.cratio:.3f}")
b1_n = b1.data
print(b1_n)

# import numexpr as ne
# print("numexpr evaluation ...")
# expr = "4 * x * x"
# a1_n = a1.data
# t0 = time()
# b1 = ne.evaluate(expr, {"x": a1_n})
# print("Time for pure expr eval:", round((time() - t0), 3))
# print(b1)
