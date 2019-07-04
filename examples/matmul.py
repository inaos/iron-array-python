import iarray as ia
import numpy as np
from time import time
import ctypes

mkl_rt = ctypes.CDLL('libmkl_rt.dylib')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

mkl_set_num_threads(1)
print(f"Numpy max threads: {mkl_get_max_threads()}")

dtshape_a = ia.dtshape([2000, 2000], [200, 200])
bshape_a = [200, 200]

dtshape_b = ia.dtshape([2000, 2000], [200, 200])
bshape_b = [200, 200]

a = ia.arange(dtshape_a, clevel=0)
an = ia.iarray2numpy(a)

b = ia.arange(dtshape_b, clevel=0)
bn = ia.iarray2numpy(b)

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
    c = ia.matmul(a, b, bshape_a, bshape_b, clevel=0, max_num_threads=1)
t1 = time()
print(f"Time to compute matmul with iarray: {(t1-t0)/nrep} s")

cn = ia.iarray2numpy(c)

np.testing.assert_almost_equal(cn, cn2)

print("Matrix multiplication is working!")
