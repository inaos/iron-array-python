# matmul comparison against numpy.

import iarray as ia
import numpy as np
import os

a = ia.arange(ia.DTShape([10, 20, 10, 14], np.float32), clevel=9)

b = a[0]

an = ia.iarray2numpy(a)

cn2 = np.mean(an, axis=(2, 3, 1))
cn = ia.mean(a, axis=(2, 3, 1), chunks=(4,), blocks=(4,), urlpath="red.iarray")
d = ia.open("red.iarray")

np.testing.assert_allclose(d.data, cn2)
print("Matrix reduction is working!")
os.remove("red.iarray")
