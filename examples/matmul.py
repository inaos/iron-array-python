import iarray as ia
import numpy as np
from itertools import zip_longest as izip
from time import time
import ctypes

mkl_rt = ctypes.CDLL('libmkl_rt.dylib')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

mkl_set_num_threads(1)
print(f"Numpy max threads: {mkl_get_max_threads()}")

cfg = ia.Config()
ctx = ia.Context(cfg)

shape_a = [2000, 2000]
size_a = np.prod(shape_a)
shape_b = [2000]
size_b = np.prod(shape_b)

pshape = None

a = ia.arange(ctx, size_a, shape=shape_a, pshape=pshape)
an = ia.iarray2numpy(ctx, a)

b = ia.arange(ctx, size_b, shape=shape_b, pshape=pshape)
bn = ia.iarray2numpy(ctx, b)

nrep = 10

c = None
cn2 = None

t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()

print(f"Time to compute matmul with numpy: {(t1-t0)/nrep} s")

t0 = time()
for i in range(nrep):
    c = ia.matmul(ctx, a, b, pshape, pshape)
t1 = time()
print(f"Time to compute matmul with iarray: {(t1-t0)/nrep} s")

cn = ia.iarray2numpy(ctx, c)

np.testing.assert_almost_equal(cn, cn2)

print("Las multiplicaciones son iguales!")
