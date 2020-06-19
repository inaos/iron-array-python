import iarray as ia
import numpy as np

size = 10000
shape = [size]
pshape = [100]

a1 = ia.random_uniform(ia.dtshape(shape, pshape, dtype=np.float32), 0, 1)
a2 = ia.iarray2numpy(a1)

b1 = np.random.uniform(0, 1, size).astype(np.float32)

storage = ia.StorageProperties("blosc", True, "test_poisson_f_06.iarray")
b2 = ia.numpy2iarray(b1, pshape=pshape, storage=storage)

# Check that distributions are equal
print(ia.random_kstest(a1, b2))
