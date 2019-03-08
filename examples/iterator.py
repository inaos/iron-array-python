import iarray as ia
import numpy as np

cfg = ia.Config()
ctx = ia.Context(cfg)

shape = [4, 4]
pshape = [2, 3]
block = [2, 2]

size = int(np.prod(shape))

a = ia.arange(ctx, size, shape=shape, pshape=pshape)

b = ia.iarray2numpy(ctx, a)

print("Element-wise")

for index, elem in a.iter_elem():
    print(f"{index}: {elem}")

print("Block-wise")

for index, bl in a.iter_block(block):
    print(f"{index}: {bl}")
