# Store an array persistently and load it back.

import iarray as ia
import numpy as np


filename = 'arange.iarray'
shape = (7, 13)
size = int(np.prod(shape))
a = np.arange(size, dtype=np.float64).reshape(shape)

print(f"Creating {filename}")
store = ia.StorageProperties("blosc", chunkshape=(2,3), blockshape=(1,3), filename=filename)
b = ia.numpy2iarray(a, storage=store)

print(f"Reading {filename}")
c = ia.load(filename)
d = ia.iarray2numpy(c)

np.testing.assert_array_equal(a, d)
print("Correct values read!")
