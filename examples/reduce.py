# matmul comparison against numpy.

import iarray as ia
import numpy as np


a = ia.arange([20, 20, 20, 20], btune=False)
print(a.info)

a_data = a.data
cn2 = np.std(a_data, axis=(3,))
cn = ia.std(a, axis=3)

np.testing.assert_allclose(cn.data, cn2)
print("Matrix reduction is working!")
