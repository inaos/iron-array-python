# Calling scalar UDFs from expressions.
# This is for 1-dim arrays.

from time import time

import numpy as np

import iarray as ia
from iarray import udf


# Define array params
shape = [100_000_000]


@udf.scalar(lib="lib")
def fsum(a: udf.float64, b: udf.float64) -> float:
    if a < 0:
        return -a + b
    else:
        return a + b

# With the new mechanism for registering functions in the udf.scalar decorator,
# it is not necessary to register functions manually anymore.
print("Registered UDF funcs:", tuple(ia.udf_registry.iter_all_func_names()))

# Create initial containers
a1 = ia.linspace(shape, -5, 5)

print("** scalar udf evaluation ...")
expr = "4 * lib.fsum(x, y)"
expr = ia.expr_from_string(expr, {"x": a1, "y": 1})
t0 = time()
b1 = expr.eval()
print("Time:", round((time() - t0), 3))
print(f"cratio for result: {b1.cratio:.3f}")
b1_n = b1.data
print(b1_n)

# Calling scalar UDFs inside another UDF
@udf.jit
def udf_sum(out: udf.Array(udf.float64, 1), x: udf.Array(udf.float64, 1), y: udf.float64):
    for i in range(out.shape[0]):
        out[i] = 4 * lib.fsum(x[i], y)
    return 0

expr2 = ia.expr_from_udf(udf_sum, [a1], [1])
print("** udf evaluation ...")
t0 = time()
b2 = expr2.eval()
print("Time:", round((time() - t0), 3))
print(f"cratio for result: {b2.cratio:.3f}")
print(b.data)

# In case we want to clear the UDF registry explicitly
ia.udf_registry.clear()
