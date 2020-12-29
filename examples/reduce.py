# matmul comparison against numpy.

import iarray as ia
import numpy as np


a = ia.arange(ia.DTShape([10, 20, 10, 14], np.float32), clevel=9)

b = a[0]

an = ia.iarray2numpy(a)

cn2 = np.mean(an, axis=(2, 3, 1))
cn = ia.mean(a, axis=(2, 3, 1))

print("Matrix reduction is working!")
