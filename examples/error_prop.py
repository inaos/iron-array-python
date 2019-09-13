import iarray as ia
import numpy as np

ashape = [100, 100]
abshape = [100, 100]
bshape = [100, 100]
bbshape = [100, 100]

dtype = np.float64

a = ia.linspace(ia.dtshape(ashape, None, dtype), -1, 1)
an = ia.iarray2numpy(a)

b = ia.linspace(ia.dtshape(bshape, None, dtype), -1, 1)
bn = ia.iarray2numpy(b)

c = ia.matmul(a, b, abshape, bbshape)
cn = np.matmul(an, bn)

cn_2 = ia.iarray2numpy(c)

rtol = 1e-6 if dtype == np.float32 else 1e-14

np.testing.assert_allclose(cn, cn_2, rtol=rtol)
