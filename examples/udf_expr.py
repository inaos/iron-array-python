# Calling scalar UDFs from expressions.
# This is for 1-dim arrays.

import math
from time import time

import numpy as np

import iarray as ia
from iarray import udf


# Define array params
#shape = [20_000_000]
shape = [10]
dtype = np.float64
# Let's favor speed during computations
ia.set_config_defaults(favor=ia.Favor.SPEED, dtype=dtype)

@udf.scalar(verbose=0)
def f(a: udf.float64, b: udf.float64) -> float:
    return a * b

cfg = ia.get_config_defaults()
registry = ia.UdfRegistry(cfg)
lib = ia.UdfLibrary(registry, "lib")
bc = f.bc
print("argtypes -->", f.argtypes)
# TODO: is there a way to get the signature types more automatically?
types = [ia.DataType.IARRAY_DATA_TYPE_DOUBLE, ia.DataType.IARRAY_DATA_TYPE_DOUBLE]
lib.register(len(bc), bc, ia.DataType.IARRAY_DATA_TYPE_DOUBLE, 2, types, "f")

# Create initial containers
a1 = ia.linspace(shape, 0, 10)
a2 = np.linspace(0, 10, shape[0], dtype=dtype).reshape(shape)

print("iarray evaluation ...")
# expr = "lib.f(a1, a1)"  # segfault.  fix it by propagating errors correctly!
expr = "lib.f(x, x)"
expr = ia.expr_from_string(expr, {"x": a1}, registry=registry)
t0 = time()
b1 = expr.eval()
print("Time for UDF eval:", round((time() - t0), 3))
print(f"CRatio for result: {b1.cratio:.3f}")
b1_n = ia.iarray2numpy(b1)
print(b1_n)
