import iarray as ia
import scipy.io
from time import time
import numpy as np


_ = ia.set_config(clevel=1, codec=ia.Codec.LZ4, btune=False, nthreads=8, dtype=np.float64, seed=0)

shape = (50_000, 13_859)

params = ia.matmul_gemv_params(shape[0], shape[1], l2_size=512 * 1024)
a_chunks, a_blocks, b_chunks, b_blocks = params

aia = ia.random.normal(
    shape,
    3,
    2,
    chunks=a_chunks,
    blocks=a_blocks,
    contiguous=True,
    urlpath="dense.iarr",
    mode="w",
    fp_mantissa_bits=3,
)

aia = ia.load("dense.iarr")
print(aia.info)

a = aia.data

bia = ia.linspace((aia.shape[1],), 2, 10, chunks=b_chunks, blocks=b_blocks)
b = bia.data

t0 = time()
c = a.dot(b)
t1 = time()
print(f"numpy: {t1 - t0:.5f} s")

t0 = time()
cia2 = ia.matmul(aia, bia)
t1 = time()

print(f"iarray: {t1 - t0:.5f} s")


np.testing.assert_allclose(cia2.data, c, rtol=1e-10 if a.dtype == np.float64 else 1e-4)
