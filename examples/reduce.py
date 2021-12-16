# matmul comparison against numpy.

import iarray as ia
import numpy as np
import os

ia.remove_urlpath("red.iarr")
a = ia.arange([10, 20, 10, 14], clevel=9, btune=False)

b = a[0]

an = ia.iarray2numpy(a)

cn2 = np.mean(an, axis=(2, 3, 1))
cn = ia.mean(a, axis=(2, 3, 1), chunks=(4,), blocks=(4,), urlpath="red.iarr")
d = ia.open("red.iarr")

np.testing.assert_allclose(d.data, cn2)
print("Matrix reduction is working!")
ia.remove_urlpath("red.iarr")
