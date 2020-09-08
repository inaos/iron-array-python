# Example

import iarray as ia
import numpy as np

size = 10000
shape = [size]
cshape = [100]
bshape = [20]

ia.random_set_seed(1)

a1 = ia.random_uniform(ia.dtshape(shape, dtype=np.float32), 0, 1)
a2 = ia.iarray2numpy(a1)
print(a2[:10])

a1 = ia.random_uniform(ia.dtshape(shape, dtype=np.float32), 0, 1)
a2 = ia.iarray2numpy(a1)
print(a2[:10])

b1 = np.random.uniform(0, 1, size).astype(np.float32)

storage = ia.StorageProperties("blosc", cshape, bshape, True, "test_poisson_f_06.iarray")
b2 = ia.numpy2iarray(b1, storage=storage)

# Check that distributions are equal
print(ia.random_kstest(a1, b2))
