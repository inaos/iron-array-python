import iarray as ia
from itertools import zip_longest as izip
from time import time
import ctypes
import sys

nthreads = int(sys.argv[1])

cfg = ia.Config(max_num_threads=nthreads, compression_level=0)
ctx = ia.Context(cfg)

shape_a = [2000, 2000]
size_a = shape_a[0] * shape_a[1]
pshape_a = None
bshape_a = [2000, 2000]

shape_b = [2000, 2000]
size_b = shape_b[0] * shape_b[1]
pshape_b = None
bshape_b = [2000, 2000]

pshape = None

a = ia.arange(ctx, size_a, shape=shape_a, pshape=pshape_a)

b = ia.arange(ctx, size_b, shape=shape_b, pshape=pshape_b)

nrep = 10

c = None
cn2 = None


for i in range(nrep):
    t0 = time()
    c = ia.matmul(ctx, a, b, bshape_a, bshape_b)
    t1 = time()
    print(t1 - t0)


