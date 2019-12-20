import iarray as ia
from iarray import udf
import numpy as np

import py2llvm as llvm
#from py2llvm import Array, float64, int64
from py2llvm import int64


#@llvm.jit(verbose=0)
#def expr_func(array: Array(float64, 1), out: Array(float64, 1)) -> int64:
#    for i in range(array.shape[0]):
#        x = array[i]
#        out[i] = (x - 1.35) * (x - 4.45) * (x - 8.5)
#
#    return 0

# XXX The name MUST be 'expr_func' (required by iron-array)
@llvm.jit(verbose=0)
def expr_func(params: udf.params_type) -> int64:
    n = params.out_size / params.out_typesize

    for i in range(n):
        x = params.inputs[0][i]
        params.out[i] = (x - 1.35) * (x - 4.45) * (x - 8.5)

    return 0


# Define array params
shape = [10000, 2000]
pshape = [1000, 200]

# Create initial containers
a1 = ia.linspace(ia.dtshape(shape, pshape, np.float64), 0, 10)
a2 = ia.iarray2numpy(a1)

print("iarray evaluation...")

# And now, the expression
expr = ia.Expr(eval_flags="iterblosc", blocksize=0)
expr.bind("x", a1)
expr.compile_udf(expr_func.bc)
print(f'DEBUG expr.eval({shape}, {pshape}, {np.float64})')
b2 = expr.eval(shape, pshape, np.float64)

b2_n = ia.iarray2numpy(b2)
assert b2.shape == b2_n.shape
print(b2_n) # numpy
print(b2)   # iarray

print("numpy evaluation...")
b2 = eval("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": a2})

try:
    np.testing.assert_almost_equal(b2, b2_n)
    print("OK.  Results are the same.")
except AssertionError:
    print("ERROR. Results are different.")
