import iarray as ia
import scipy.io
from time import time
import numpy as np


_ = ia.set_config(clevel=1, codec=ia.Codecs.LZ4, btune=False, nthreads=8, dtype=np.float64, seed=0)

shape = (25000, 13859)
# wia = ia.random.normal(shape, 3, 2, chunks=(1024, 1024), blocks=(128, 128), enforce_frame=True, urlpath="dense.iarray", fp_mantissa_bits=3)

wia = ia.load("dense.iarray")
print(wia.info)

w = wia.data

bia = ia.linspace((wia.shape[1],), 2, 10, chunks=(wia.chunks[1],), blocks=(wia.blocks[1],))

# b[256:512] = 0
t0 = time()
c = w.dot(bia.data)
t1 = time()
print(f"numpy: {t1 - t0:.5f} s")

t0 = time()
cia = ia.opt_gemv(wia, bia, use_mkl=False, chunks=(wia.chunks[0],), blocks=(wia.blocks[0],))
t1 = time()

print(f"iarray (gemv): {t1 - t0:.5f} s")

t0 = time()
cia2 = ia.matmul(wia, bia, chunks=(wia.chunks[0],), blocks=(wia.blocks[0],))
t1 = time()

print(f"iarray: {t1 - t0:.5f} s")


np.testing.assert_allclose(cia.data, c, rtol=1e-10 if w.dtype == np.float64 else 1e-4)
