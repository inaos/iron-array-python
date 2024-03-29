# Exercises ia.matmul().  Notice that the differences in precision are due because the order of computations
# is different than in numpy (or BLAS).  This is because the algorithm uses different blocks in iarray.

import iarray as ia
import numpy as np

ashape = [100, 100]
bshape = [100, 100]
anum = int(np.prod(ashape))
bnum = int(np.prod(bshape))

dtype = np.float64

a = ia.linspace(-1, 1, anum, shape=ashape, dtype=dtype)
an = ia.iarray2numpy(a)

b = ia.linspace(-1, 1, bnum, shape=bshape, dtype=dtype)
bn = ia.iarray2numpy(b)

c = ia.matmul(a, b)
cn = np.matmul(an, bn)

cn_2 = ia.iarray2numpy(c)

rtol = 1e-6 if dtype == np.float32 else 1e-14
atol = 1e-6 if dtype == np.float32 else 1e-13


np.testing.assert_allclose(cn, cn_2, rtol=rtol, atol=atol)
