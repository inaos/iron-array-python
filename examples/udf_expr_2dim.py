import iarray as ia
from iarray.expr_udf import expr_udf
from iarray import udf
import numpy as np


ia.set_config_defaults(dtype=np.float64)
a = ia.arange([10, 10])
b = ia.arange([10, 10])


# Using a UDF clip
@udf.scalar(lib="mylib")
def clip(a: udf.float64, amin: udf.float64, amax: udf.float64) -> udf.float64:
    if a < amin:
        return amin
    if a > amax:
        return amax
    return a


expr = expr_udf(
    "mylib.clip(a, 4, 13)",
    {"a": a},
    debug=1,
)
out = expr.eval()
print(out.data)


# Syntactic sugar on where()
expr = expr_udf(
    "a[b > 5 and not (a < 8 or b > 42)]",
    {"a": a, "b": b},
    debug=1,
)
out = expr.eval()
print(out.data)
