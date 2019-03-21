import iarray as ia
import numpy as np


cfg = ia.Config()
ctx = ia.Context(cfg)
rnd_ctx = ia.RandomContext(ctx, seed=123)
np.random.seed(456)

shape = [10, 10]
pshape = [4, 5]

size = int(np.prod(shape))

c1 = ia.empty(ctx, shape, pshape)

for (elem_ind, part) in c1.iter_write():
    part[:] = np.ones(part.shape, dtype=np.float64)

c2 = ia.iarray2numpy(ctx, c1)

print(c2)
