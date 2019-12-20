import iarray as ia
from iarray import udf
import numpy as np

import py2llvm as llvm
from py2llvm import Array, float64, int64

# Number of iterations per benchmark
NITER = 10

#@llvm.jit(verbose=0)
#def inner(array: Array(float64, 1), out: Array(float64, 1)) -> int64:
#    for i in range(array.shape[0]):
#        x = array[i]
#        out[i] = (x - 1.35) * (x - 4.45) * (x - 8.5)
#
#    return 0
#
#@llvm.jit(verbose=0)
#def f(params: udf.params_type) -> int64:
#    n = params.out_size / params.out_typesize
#    return inner(params.inputs[0], n, params.out, n)


@llvm.jit(verbose=0)
def f(params: udf.params_type) -> int64:
    n = params.out_size / params.out_typesize

    for i in range(n):
        x = params.inputs[0][i]
        params.out[i] = (x - 1.35) * (x - 4.45) * (x - 8.5)

    return 0


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
expr.compile_udf(f)
for i in range(NITER):
    b1 = expr.eval(shape, pshape, dtype)
b1_n = ia.iarray2numpy(b1)
print(b1_n)

print("numpy evaluation...")
for i in range(NITER):
    b2 = eval("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": a2})
print(b2)

assert b2.shape == b1_n.shape

try:
    np.testing.assert_almost_equal(b2, b1_n)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
