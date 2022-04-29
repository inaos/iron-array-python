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
    return a + b

@udf.scalar(lib="lib2")
def fmult(a: udf.float64, b: udf.float64) -> float:
    return a * b


# With the new mechanism for registering functions in the udf.scalar decorator,
# it is not necessary to register functions manually anymore.
# ia.udf_registry["lib"] = fsum
# ia.udf_registry["lib2"] = fmult
print("Registered UDF funcs:", tuple(ia.udf_registry.iter_all_func_names()))

# Create initial containers
a1 = ia.linspace(shape, 0, 10)
a2 = np.linspace(0, 10, shape[0]).reshape(shape)

print("** pure expr evaluation ...")
expr = "4 * (x * y)"
expr = ia.expr_from_string(expr, {"x": a1, "y": 1})
t0 = time()
b1 = expr.eval()
print("Time:", round((time() - t0), 3))
print(f"cratio for result: {b1.cratio:.3f}")
b1_n = b1.data
print(b1_n)

print("** scalar udf evaluation ...")
# expr = "4 * lib.fsum(x, y)"
expr = "4 * lib2.fmult(x, y)"
expr = ia.expr_from_string(expr, {"x": a1, "y": 1})
t0 = time()
b1 = expr.eval()
print("Time:", round((time() - t0), 3))
print(f"cratio for result: {b1.cratio:.3f}")
b1_n = b1.data
print(b1_n)

# In case we want to clear the UDF registry explicitly
ia.udf_registry.clear()
