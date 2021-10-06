# Store an array persistently and load it back.

import iarray as ia
import numpy as np


urlpath = "arange.iarr"
shape = (70, 130)
size = int(np.prod(shape))
a = np.arange(size, dtype=np.float64).reshape(shape)

print(f"Creating {urlpath}")
store = ia.Store(urlpath=urlpath)
b = ia.numpy2iarray(a, store=store)

print(f"Reading {urlpath}")
c = ia.open(urlpath)
d = ia.iarray2numpy(c)

np.testing.assert_array_equal(a, d)
print("Correct values read!")
