import iarray as ia
import numpy as np
from itertools import zip_longest as izip


cfg = ia.Config()
ctx = ia.Context(cfg)
rnd_ctx = ia.RandomContext(ctx, seed=123)
np.random.seed(456)

shape = [10, 10]
pshape = [4, 5]
blockshape = [2, 2]

size = int(np.prod(shape))

c1 = ia.empty(ctx, shape, pshape)
c2 = ia.arange(ctx, size, shape=shape, pshape=pshape)

for i, ((e1, p1), (e2, p2)) in enumerate(izip(c2.iter_read_block(pshape), c1.iter_write_block())):
   p2[:] = p1

c3 = ia.iarray2numpy(ctx, c1)

print(c3)
