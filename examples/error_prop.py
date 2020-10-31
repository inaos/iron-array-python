# Exercises ia.matmul().  Notice that the differences in precision are due because the order of computations
# is different than in numpy (or BLAS).  This is because the algorithm uses different blocks in iarray.

import iarray as ia
import numpy as np

ashape = [100, 100]
bshape = [100, 100]

dtype = np.float64

a = ia.linspace(ia.DTShape(ashape, dtype), -1, 1)
an = ia.iarray2numpy(a)

b = ia.linspace(ia.DTShape(bshape, dtype), -1, 1)
bn = ia.iarray2numpy(b)

c = ia.matmul(a, b)
cn = np.matmul(an, bn)

cn_2 = ia.iarray2numpy(c)

rtol = 1e-6 if dtype == np.float32 else 1e-15

np.testing.assert_allclose(cn, cn_2, rtol=rtol)
