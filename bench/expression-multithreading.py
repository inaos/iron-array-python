import iarray as ia
import numpy as np
from time import time
import numexpr as ne
from iarray import udf
from iarray.py2llvm import float64
import numba as nb
import os

# Numba uses OpemMP, and this collides with the libraries in ironArray.
# Using the next envvar seems to fix the issue (bar a small printed info line).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

max_num_threads = 4
nrep = 5


@nb.jit(nopython=True, cache=True, parallel=True)
def poly_numba(x):
    y = np.empty(x.shape, x.dtype)
    for i in nb.prange(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return y


@udf.jit
def poly_udf(x: udf.Array(float64, 1), y: udf.Array(float64, 1)):
    n = x.shape[0]
    for i in range(n):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return 0


# Define array params
shape = [32 * 1024 * 1024]
chunks = [4 * 1024 * 1024]
blocks = [16 * 1024]
size = int(np.prod(shape))
sizeMB = int(np.prod(shape)) * 8 / 2**20

bcfg = ia.Config(chunks=chunks, blocks=blocks)


def time_expr(expr, result):
    t = []
    for _ in range(nrep):
        t0 = time()
        expr.eval()
        t1 = time()
        t.append(round(sizeMB / (t1 - t0), 2))
    t.remove(max(t))
    result.append(np.mean(t))


res = []
for num_threads in range(1, max_num_threads + 1):
    print(f"Num. threads: {num_threads}")
    # omp_set_num_threads(num_threads)
    res_i = []

    # Numpy
    a1 = np.linspace(0, 10, size).reshape(shape)
    t = []
    for _ in range(nrep):
        t0 = time()
        eval("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": a1})
        t1 = time()
        t.append(round(sizeMB / (t1 - t0), 2))
    t.remove(max(t))
    res_i.append(np.mean(t))

    # numba
    t = []
    for _ in range(nrep):
        t0 = time()
        poly_numba(a1)
        t1 = time()
        t.append(round(sizeMB / (t1 - t0), 2))
    t.remove(max(t))
    res_i.append(np.mean(t))

    # Numexpr
    t = []
    ne.set_num_threads(num_threads)
    for _ in range(nrep):
        t0 = time()
        ne.evaluate("(x - 1.35) * (x - 4.45) * (x - 8.5)", local_dict={"x": a1})
        t1 = time()
        t.append(round(sizeMB / (t1 - t0), 2))
    t.remove(max(t))
    res_i.append(np.mean(t))

    # Superchunk with compression and UDF
    with ia.config(cfg=bcfg, nthreads=num_threads):
        a1 = ia.linspace(
            0,
            10,
            int(np.prod(shape)),
            shape=shape,
        )
        expr = ia.expr_from_udf(poly_udf, [a1])
        time_expr(expr, res_i)

    res.append(res_i)

print("Speed in MB/s")
np.set_printoptions(formatter={"numpystr": lambda x: "{:>15}".format(x)})
print(np.array([["NumPy", "Numba", "numexpr", "ironArray"]]))
np.set_printoptions(formatter={"float": lambda x: "{0:15.1f}".format(x)})
print(np.array(res))
