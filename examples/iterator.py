import iarray as ia
import numpy as np
from itertools import zip_longest as izip


shape = [10, 10]
pshape = [4, 5]
blockshape = [2, 2]

size = int(np.prod(shape))

c1 = ia.empty(shape, pshape)
c2 = ia.arange(size, shape=shape, pshape=pshape)

for i, ((info1, p1), (info2, p2)) in enumerate(izip(c2.iter_read_block(pshape), c1.iter_write_block())):
   p2[:] = p1

c3 = ia.iarray2numpy(c1)

print(c3)
