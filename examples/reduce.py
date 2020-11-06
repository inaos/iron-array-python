# matmul comparsion with numpy.

import iarray as ia
import numpy as np


a = ia.arange(ia.DTShape([10, 10, 10, 10], np.float32), clevel=9)
an = ia.iarray2numpy(a)


cn2 = np.max(an, axis=2)

c = ia.max(a, axis=2)
cn = ia.iarray2numpy(c)

np.testing.assert_allclose(cn, cn2)

print("Matrix reduction is working!")
