# Comparison of complex expressions performance when using 1 or 3 variables

import os
from time import time

import numexpr as ne
import numpy as np

import iarray as ia
from iarray import udf
from iarray.udf import Array
from iarray.py2llvm import float64, int64


NITER = 10  # number of iterations per benchmark
PROFILE = False
NVARS = 3  # number of variables in expression (only 1 or 3)
assert NVARS in (1, 3)

expr1 = "(x - 1.35) * (x - 4.45) * (x - 8.5)"
expr3 = "(x - 1.35) * (y - 4.45) * (z - 8.5)"

# Define array params
shape = [20_000_000]
chunkshape, blockshape = [4_000_000], [20_000]
# chunkshape, blockshape = None, None   # uncomment for automatic partitioning
dtype = np.float64
nthreads = 8
dtshape = ia.DTShape(shape=shape, dtype=dtype)

# Set global defaults
ia.set_config(clevel=9, nthreads=nthreads, chunkshape=chunkshape, blockshape=blockshape)
# Output required precision (in significant bits for the mantissa)
out_prec = 20


@udf.jit(verbose=0)
def f1(out: Array(float64, 1), x: Array(float64, 1)) -> int64:
    n = out.shape[0]
    for i in range(n):
        out[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


# Version with 3 parameters
@udf.jit(verbose=0)
def f3(
    out: Array(float64, 1), x: Array(float64, 1), y: Array(float64, 1), z: Array(float64, 1)
) -> int64:
    n = out.shape[0]
    for i in range(n):
        out[i] = (x[i] - 1.35) * (y[i] - 4.45) * (z[i] - 8.5)

    return 0


# Create initial containers
if PROFILE:
    a1_storage = None  # avoid a warning
    a1_fname = "a1.iarray"
    if not os.path.isfile(a1_fname):
        print(f"Creating {a1_fname}")
        a1_storage = ia.Storage(urlpath=a1_fname)
        a1 = ia.linspace(dtshape, 0, 10, storage=a1_storage)
    else:
        print(f"Reading {a1_fname}")
        a1 = ia.load(a1_fname)
else:
    a1 = ia.linspace(dtshape, 0, 10)


a2 = np.linspace(0, 10, shape[0], dtype=dtype).reshape(shape)

x = a2.copy()
y = a2.copy()
z = a2.copy()
t0 = time()
for i in range(NITER):
    if NVARS == 1:
        bn = eval(expr1, {"x": x})
    else:
        bn = eval(expr3, {"x": x, "y": y, "z": z})
print("Time for numpy eval:", round((time() - t0) / NITER, 3))
# print(bn)

ne.set_num_threads(nthreads)
t0 = time()
for i in range(NITER):
    if NVARS == 1:
        bne = ne.evaluate(expr1, {"x": x})
    else:
        bne = ne.evaluate(expr3, {"x": x, "y": y, "z": z})
print("Time for numexpr eval:", round((time() - t0) / NITER, 3))
# print(bne)

print("iarray evaluation starts...")
print("Operands cratio:", round(a1.cratio, 2))

iax = a1.copy()
iay = a1.copy()
iaz = a1.copy()

if NVARS == 1:
    expr = f1.create_expr([iax], fp_mantissa_bits=out_prec)
else:
    expr = f3.create_expr([iax, iay, iaz], fp_mantissa_bits=out_prec)
b1 = None  # avoid a warning
t0 = time()
for i in range(NITER):
    b1 = expr.eval()
print("Time for udf eval engine:", round((time() - t0) / NITER, 3))
b1_n = ia.iarray2numpy(b1)
# print(b1_n)

b2 = None  # avoid a warning
t0 = time()
if NVARS == 1:
    expr = ia.expr_from_string(expr1, {"x": iax}, fp_mantissa_bits=out_prec)
else:
    expr = ia.expr_from_string(expr3, {"x": iax, "y": iay, "z": iaz}, fp_mantissa_bits=out_prec)
for i in range(NITER):
    b2 = expr.eval()
print("Time for internal eval engine:", round((time() - t0) / NITER, 3))
b2_n = ia.iarray2numpy(b2)
# print(b2_n)
print("Output cratio:", round(b2.cratio, 2))

try:
    np.testing.assert_almost_equal(bn, b1_n, decimal=4)
    np.testing.assert_almost_equal(bn, b2_n, decimal=4)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
