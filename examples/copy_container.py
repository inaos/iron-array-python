from time import time

import iarray as ia
import numpy as np


shape, pshape = [2000, 2000], [200, 200]
dtype = np.float64

a = ia.linspace(ia.dtshape(shape, pshape, dtype), -10, 10)

t0 = time()
b = a.copy()
t1 = time()

print(f"Time to make a copy (cont -> cont): {t1 - t0:.5f}")

bn = ia.iarray2numpy(b)

t0 = time()
c = a.copy(view=True)
t1 = time()

print(f"Time to make a copy (cont -> view): {t1 - t0:.5f}")

cn = ia.iarray2numpy(c)

np.testing.assert_allclose(bn, cn, rtol=1e-12)


v = a[100:1100, 200:1200]

t0 = time()
d = v.copy()
t1 = time()

print(f"Time to make a copy (view -> cont): {t1 - t0:.5f}")

dn = ia.iarray2numpy(d)

t0 = time()
e = v.copy(view=True)
t1 = time()

print(v)

print(f"Time to make a copy (view -> view): {t1 - t0:.5f}")

en = ia.iarray2numpy(e)

np.testing.assert_allclose(dn, en, rtol=1e-12)
