import iarray as ia
import numpy as np
import os.path


filename = 'arange.iarray'
shape = (7, 13)
pshape = (2, 3)
size = int(np.prod(shape))

a = np.arange(size, dtype=np.float64).reshape(shape)

if not os.path.isfile(filename):
    print(f"Creating {filename}")
    b = ia.numpy2iarray(a, pshape, filename=filename)
else:
    print(f"Reading {filename}")
    c = ia.load(filename)
    d = ia.iarray2numpy(c)

    np.testing.assert_array_equal(a, d)
    print("Correct values read!")
