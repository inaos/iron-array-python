import iarray as ia
from time import time
import numpy as np

M = 1
K = 3
N = 15_000

dtype = np.dtype(np.float32)

params = ia.opt_gemm2_params(M, K, N, itemsize=dtype.itemsize, l2_size=256 * 1024)
print(params)
t = np.ones((M, K), dtype=dtype)
t = np.tril(t)
a = ia.numpy2iarray(t, chunks=params["a_chunks"], blocks=params["a_blocks"])
# a = ia.random.random_sample(
#     (M, K), dtype=dtype, chunks=params["a_chunks"], blocks=params["a_blocks"]
# )

b = ia.random.random_sample(
    (K, N),
    dtype=dtype,
    chunks=params["b_chunks"],
    blocks=params["b_blocks"],
)

t0 = time()
c2 = ia.opt_gemm2(a, b)
t1 = time()
print(f"Time iarray_gemm2: {t1 - t0:.4f} s")

a_np = a.data
b_np = b.data
t0 = time()
c_np = a_np @ b_np
t1 = time()
print(f"Time numpy: {t1 - t0:.4f} s")

np.testing.assert_allclose(c2.data, c_np, rtol=1e-5)