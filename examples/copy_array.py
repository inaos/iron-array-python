# Exercises the copy() with actual arrays and views of them too.

from time import time
import iarray as ia
import numpy as np


# Define the properties of an array
shape, cshape, bshape = [2000, 2000], [200, 200], [100, 100]
dtype = np.float64
store = ia.StorageProperties("blosc", chunkshape=cshape, blockshape=bshape)
a = ia.linspace(ia.dtshape(shape, dtype), -10, 10, storage=store)

# Do a regular copy of an array
t0 = time()
b = a.copy()
t1 = time()
print(f"Time to make a copy (cont -> cont): {t1 - t0:.5f}")

# Do a regular copy of an array
bn = ia.iarray2numpy(b)
t0 = time()
c = a.copy(view=True)
t1 = time()
print(f"Time to make a copy (cont -> view): {t1 - t0:.5f}")

cn = ia.iarray2numpy(c)
np.testing.assert_allclose(bn, cn)

# Get a slice (view)
v = a[100:1100, 200:1200]
t0 = time()
d = v.copy()
t1 = time()
print(f"Time to make a copy (view -> cont): {t1 - t0:.5f}")

# Do a copy of the slice (view)
dn = ia.iarray2numpy(d)
t0 = time()
e = v.copy(view=True)
t1 = time()
print(f"Time to make a copy (view -> view): {t1 - t0:.5f}")

en = ia.iarray2numpy(e)
np.testing.assert_allclose(dn, en)
