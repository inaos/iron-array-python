import iarray as ia
import numpy as np
from itertools import zip_longest as izip


cfg = ia.Config()
ctx = ia.Context(cfg)

shape = [10, 10]
pshape = [2, 2]

a = ia.arange(ctx, 100, shape=shape, pshape=pshape)
a2 = a[1:5, 1:5]
an = ia.iarray2numpy(ctx, a2)

b = ia.arange(ctx, 100, shape=shape, pshape=pshape)
b2 = b[6:10, 3:7]
bn = ia.iarray2numpy(ctx, b2)

c = ia.matmul(ctx, a2, b2, pshape, pshape)
cn = ia.iarray2numpy(ctx, c)

cn2 = np.matmul(an, bn)

print(cn2)
