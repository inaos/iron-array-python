# Example for creating a random distribution array and store it on a file

import iarray as ia
import numpy as np

size = 10000
shape = [size]

dtshape = ia.DTShape(shape, dtype=np.float32)
a1 = ia.irandom.uniform(dtshape, 0, 1)
a2 = ia.iarray2numpy(a1)
print(a2[:10])

a1 = ia.irandom.uniform(dtshape, 0, 1)
a2 = ia.iarray2numpy(a1)
print(a2[:10])

b1 = np.random.uniform(0, 1, size).astype(np.float32)

storage = ia.Store(urlpath="random_dist.iarray")
b2 = ia.numpy2iarray(b1, storage=storage)

# Check that distributions are equal
print(ia.irandom.kstest(a1, b2))
