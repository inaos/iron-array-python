import os
from time import time

import numexpr as ne
import numpy as np

import iarray as ia
from iarray import udf
from iarray.udf import Array
from iarray.py2llvm import float64, int64


# Number of iterations per benchmark
NITER = 10
PROFILE = False
NTHREADS = 20

cparams = dict(clib=ia.LZ4, clevel=5, nthreads=NTHREADS)

# Define array params
shape = [20 * 1000 * 1000]
chunkshape = [4000 * 1000]
blockshape = [20 * 1000]
dtype = np.float64


@udf.jit(verbose=0)
def f(out: Array(float64, 1), x: Array(float64, 1)) -> int64:
    n = out.shape[0]
    for i in range(n):
        out[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0

# Version with 3 parameters
# @udf.jit(verbose=0)
# def f2(out: Array(float64, 1), x: Array(float64, 1), y: Array(float64, 1), z: Array(float64, 1)) -> int64:
#     n = out.shape[0]
#     for i in range(n):
#         out[i] = (x[i] - 1.35) * (y[i] - 4.45) * (z[i] - 8.5)
#
#     return 0


# Create initial containers
if PROFILE:
    a1_fname = "a1.iarray"
    if not os.path.isfile(a1_fname):
        print(f"Creating {a1_fname}")
        a1_storage = ia.StorageProperties("blosc", chunkshape, blockshape, True, a1_fname)
        a1 = ia.linspace(ia.dtshape(shape, dtype), 0, 10, storage=a1_storage, **cparams)
    else:
        print(f"Reading {a1_fname}")
        a1 = ia.load(a1_fname, load_in_mem=True)
else:
    a1_storage = ia.StorageProperties("blosc", chunkshape, blockshape)
    a1 = ia.linspace(ia.dtshape(shape, dtype), 0, 10, storage=a1_storage, **cparams)

a2 = np.linspace(0, 10, shape[0], dtype=dtype).reshape(shape)

print("numpy evaluation...")
x = a2.copy()
y = a2.copy()
z = a2.copy()
t0 = time()
for i in range(NITER):
    bn = eval("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": x})
    # bn = eval("(x - 1.35) * (y - 4.45) * (z - 8.5)", {"x": x, "y": y, "z": z})
print("Time for numpy eval:", round((time() - t0) / NITER, 3))
print(bn)

print("numexpr evaluation...")
ne.set_num_threads(NTHREADS)
t0 = time()
for i in range(NITER):
    bne = ne.evaluate("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": x})
    # bne = ne.evaluate("(x - 1.35) * (y - 4.45) * (z - 8.5)", {"x": x, "y": y, "z": z})
print("Time for numexpr eval:", round((time() - t0) / NITER, 3))
print(bne)


a1_storage = ia.StorageProperties("blosc", chunkshape, blockshape)
eval_method = ia.EVAL_ITERBLOSC
iax = a1.copy(view=False, storage=a1_storage, **cparams)
iay = a1.copy(view=False, storage=a1_storage, **cparams)
iaz = a1.copy(view=False, storage=a1_storage, **cparams)

print("iarray evaluation...")
cparams2 = cparams.copy()
# cparams2.update(dict(fp_mantissa_bits=3, clevel=5))
# cparams2.update(dict(clevel=5))
expr = f.create_expr([iax], ia.dtshape(shape, dtype), ia.EVAL_ITERBLOSC, storage=a1_storage, **cparams2)
# expr = f2.create_expr([iax, iay, iaz], **cparams2)
# And now, the expression
t0 = time()
for i in range(NITER):
    b1 = expr.eval()
print("Time for llvm eval:", round((time() - t0) / NITER, 3))
b1_n = ia.iarray2numpy(b1)
print(b1_n)

t0 = time()
expr = ia.Expr(eval_method=eval_method, **cparams2)
# expr.bind('x', a1)
# expr.compile('(x - 1.35) * (x - 4.45) * (x - 8.5)')
expr.bind('x', iax)
expr.bind('y', iay)
expr.bind('z', iaz)
expr.bind_out_properties(ia.dtshape(shape, dtype), storage=a1_storage)
expr.compile('(x - 1.35) * (y - 4.45) * (z - 8.5)')
for i in range(NITER):
    b2 = expr.eval()
print("Time for internal compiler eval engine:", round((time() - t0) / NITER, 3))
b2_n = ia.iarray2numpy(b2)
print(b2_n)

try:
    # np.testing.assert_almost_equal(bn, b1_n)
    np.testing.assert_almost_equal(bn, b2_n)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
