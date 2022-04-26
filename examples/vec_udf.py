import numpy as np
import iarray as ia
from iarray import udf


@udf.jit
def _fcond(out: udf.Array(udf.bool, 1),
           a: udf.Array(udf.float64, 1)) -> int:
    n = out.window_shape[0]
    for i in range(n):
        out[i] = 1 if a[i] > 0 else 0
    return 0


@udf.jit
def _fcond(out: udf.Array(udf.float64, 1),
          a: udf.Array(udf.float32, 1)) -> int:
    n = out.window_shape[0]
    for i in range(n):
        out[i] = 1 if a[i] > 0 else 0
    return 0


@udf.jit
def fcond(out: udf.Array(udf.bool, 1),
          a: udf.Array(udf.float32, 1),
          b: udf.Array(udf.float64, 1)) -> int:
    n = out.window_shape[0]
    for i in range(n):
        out[i] = 1 if (a[i] + b[i]) > 3 else 0
        # out[i] = a[i] + b[i]
    return 0


print("** vector udf evaluation ...")
a1 = ia.linspace([10], 0, 10, dtype=np.float32)
a2 = ia.linspace([10], 0, 10, dtype=np.float64)
# a2 = ia.ones([10], dtype=np.float64)  # this makes the code break with a BLOSC FAILED error!
expr = ia.expr_from_udf(fcond, [a1, a2])
#b1 = expr.eval().astype(np.bool_)
b1 = expr.eval()
print(b1.dtype)
print(b1.data)

print("** numpy evaluation ...")
b2 = (a1.data + a2.data) > 3
print(b2)
#np.testing.assert_array_almost_equal(b1.data, b2)
