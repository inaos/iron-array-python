import iarray as ia
from time import time
import numpy as np

M = 250_000
K = 5_000
N = 1_000

algorithm = "gemm3"
nreps = 1

dtype = np.dtype(np.float32)

params = ia.opt_gemm_params(M, K, N, itemsize=dtype.itemsize, l2_size=512 * 1024)
print(params)
a_chunks = params["a_chunks"]
a_blocks = params["a_blocks"]
b_chunks = params["b_chunks"]
b_blocks = params["b_blocks"]

a_chunks = (1000, 1000)
b_chunks = (1000, 1000)
a_blocks = (200, 200)
b_blocks = (200, 200)
c_chunks = (1000, 1000)
c_blocks = (200, 200)

a = ia.random.random_sample((M, K), dtype=dtype, chunks=a_chunks, blocks=b_blocks)

b = ia.random.random_sample(
    (K, N),
    dtype=dtype,
    chunks=b_chunks,
    blocks=b_blocks,
)


t0 = time()
for i in range(nreps):
    c2 = ia.opt_gemm(a, b, ott_b=True, chunks=c_chunks, blocks=c_blocks)
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
