# matmul comparsion with numpy.

import iarray as ia
import numpy as np
import ctypes

mkl_rt = ctypes.CDLL("libmkl_rt.dylib")
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads

nthreads = 4

a = ia.arange([2000, 1000], clevel=0)
b = ia.arange([2000, 1000], clevel=0)

an = ia.iarray2numpy(a)

ia.cmp_arrays(a.T, ia.transpose(b))
ia.cmp_arrays(a.transpose().copy(), ia.transpose(b))

cn = ia.iarray2numpy(a.T)
cn2 = an.T

np.allclose(cn, cn2)

print("Matrix transposition is working!")
