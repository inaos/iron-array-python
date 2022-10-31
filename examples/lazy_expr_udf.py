import iarray as ia
from iarray import udf
import numpy as np


@udf.scalar()
def clip(a: udf.float32, amin: udf.float32, amax: udf.float32) -> udf.float32:
    if a < amin:
        return amin
    if a > amax:
        return amax
    return a


x = ia.arange(10_000, shape=[10, 1_000], dtype=np.float32)

lazyexpr = ia.ulib.clip(x, 4, 13)
res = lazyexpr.eval()
print(res.data)

# check
xnp = x.data
resnp = np.clip(xnp, 4, 13)
np.testing.assert_almost_equal(resnp, res.data, decimal=3)
