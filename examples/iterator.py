import iarray as ia
import numpy as np

cfg = ia.Config()
ctx = ia.Context(cfg)

shape = [4, 5]
pshape = [2, 2]
size = int(np.prod(shape))

a = ia.arange(ctx, size, shape=shape, pshape=pshape)

for index, elem in a:
    print(f"{index}: {elem}")
