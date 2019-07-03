import iarray as ia
import numpy as np

size = 10000
shape = [size]
pshape = [100]

a1 = ia.random_uniform(0, 1, shape, pshape, dtype="float")
a2 = ia.iarray2numpy(a1)

b1 = np.random.uniform(0, 1, size).astype(np.float32)

b2 = ia.numpy2iarray(b1, pshape=pshape, filename="test_poisson_f_06.iarray")

# Check that distributions are equal
print(ia.random_kstest(a1, b2))
