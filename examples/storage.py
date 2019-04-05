import iarray as ia
from time import time
import numpy as np


# Vector sizes and partitions
N = 2 * 1000 * 1000
shape = [N]
pshape = [N // 10]
blockshape = [N // 100]

clevel = 0   # compression level
clib = ia.IARRAY_LZ4  # compression codec



# Create iarray context
cfg = ia.Config(eval_flags="iterblock", compression_codec=clib, compression_level=clevel, blocksize=0)
ctx = ia.Context(cfg)

t0 = time()
c = ia.linspace(ctx, N, 0, 1, shape, pshape)
t1 = time()

print(f"Time for creating blosc container: {round(t1 - t0, 3)}")

t0 = time()
for (i, x) in c.iter_block(blockshape):
    print(x)
t1 = time()

print("-" * 30)
