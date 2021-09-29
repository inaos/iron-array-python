import iarray as ia
from time import time
import numpy as np

M = 10_000
K = 10_000
N = 5_000

nreps = 3

dtype = np.dtype(np.float32)

params = ia.matmul_params(M, K, N, itemsize=dtype.itemsize, l2_size=512 * 1024)
a_chunks, a_blocks, b_chunks, b_blocks = params


# a_chunks = (1000, 1000)
# b_chunks = (1000, 1000)
# a_blocks = (200, 200)
# b_blocks = (200, 200)
# c_chunks = (1000, 1000)
# c_blocks = (200, 200)

a = ia.random.random_sample((M, K), dtype=dtype, chunks=a_chunks, blocks=b_blocks)

b = ia.random.random_sample(
    (K, N),
    dtype=dtype,
    chunks=b_chunks,
    blocks=b_blocks,
    urlpath="c.iarr",
    mode="w",
    contiguous=False,
)


t0 = time()
for i in range(nreps):
    c2 = ia.matmul(a, b)
t1 = time()
print(f"Time iarray: {(t1 - t0) / nreps:.4f} s")

a_np = a.data
b_np = b.data

t0 = time()
for _ in range(nreps):
    c_np = a_np @ b_np
t1 = time()
print(f"Time numpy: {(t1 - t0) / nreps:.4f} s")

np.testing.assert_allclose(c2.data, c_np, rtol=1e-5)
