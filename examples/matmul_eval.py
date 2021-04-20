# matmul comparsion with numpy.

import iarray as ia
import numpy as np
from time import time
import ctypes

mkl_rt = ctypes.CDLL("libmkl_rt.dylib")
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

nthreads = 8
dtype = np.float64
shape = [2000, 2000]
ia.set_config(nthreads=nthreads, dtype=dtype)

a = ia.arange(shape)
an = ia.iarray2numpy(a)

b = ia.arange(shape)
bn = ia.iarray2numpy(b)

nrep = 10

c = None
cn2 = None

mkl_set_num_threads(nthreads)
t0 = time()
for _ in range(nrep):
    cn2 = np.matmul(an, bn)
t1 = time()

print(f"Time to compute matmul with numpy: {(t1-t0)/nrep:0.3f} s")

mkl_set_num_threads(1)
t0 = time()
for i in range(nrep):
    c = ia.matmul(a, b)
t1 = time()
print(f"Time to compute matmul with iarray: {(t1-t0)/nrep:0.3f} s")
print(f"CRatio for result: {c.cratio:.3f}")

cn = ia.iarray2numpy(c)

np.allclose(cn, cn2)

print("Matrix multiplication is working!")
