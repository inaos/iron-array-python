# matmul comparison against numpy.

import iarray as ia
import numpy as np


a = ia.arange(ia.DTShape([10, 10, 10, 10], np.float32), clevel=9)
an = ia.iarray2numpy(a)

print(np.isscalar(an[1, 1, 1, 1]))

cn2 = np.max(an)
print(ia.Reduce.MAX.__str__().lower()[7:])
cn = ia.reduce(a, method=ia.Reduce.MAX)

np.testing.assert_allclose(cn, cn2)

print("Matrix reduction is working!")
