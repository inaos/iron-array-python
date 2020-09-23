import iarray as ia
import numpy as np
from time import time
import ctypes
import matplotlib.pyplot as plt
import numexpr as ne
from iarray import udf
from iarray.py2llvm import float64
from numba import config, njit, prange

#omp = ctypes.CDLL('libiomp5.so')
#omp_set_num_threads = omp.omp_set_num_threads

max_num_threads = 4
nrep = 5

@njit(parallel=True)
def poly_numba(x):
    y = np.empty(x.shape, x.dtype)
    for i in prange(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return y

#config.THREADING_LAYER = 'omp'

@udf.jit
def poly_udf(x: udf.Array(float64, 1), y: udf.Array(float64, 1)):
    n = x.shape[0]
    for i in range(n):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return 0


# Define array params
shape = [32 * 512 * 1024]
chunkshape = [32 * 1024]
blockshape = [16 * 1024]
dtshape = ia.dtshape(shape)
size = int(np.prod(shape))

bstorage = ia.StorageProperties(ia.BACKEND_BLOSC, chunkshape, blockshape)
pstorage = ia.StorageProperties(ia.BACKEND_PLAINBUFFER, None, None)

eval_method = ia.EVAL_AUTO

res = []

for num_threads in range(1, max_num_threads + 1):
    print(f"Num. threads: {num_threads}")
    # omp_set_num_threads(num_threads)
    res_i = []
    kwargs = dict(nthreads=num_threads)

    # Numpy
    a1 = np.linspace(0, 10, size).reshape(shape)
    t = []
    for _ in range(nrep):
        t0 = time()
        b1 = eval("(x - 1.35) * (x - 4.45) * (x - 8.5)", {"x": a1})
        t1 = time()
        t.append(round(size / 2 ** 20 * 8 / (t1 - t0), 2))
    t.remove(max(t))
    res_i.append(np.mean(t))

    # numba
    a1 = np.linspace(0, 10, size).reshape(shape)
    t = []
    for _ in range(nrep):
        t0 = time()
        b1 = poly_numba(a1)
        t1 = time()
        t.append(round(size / 2 ** 20 * 8 / (t1 - t0), 2))
    t.remove(max(t))
    res_i.append(np.mean(t))

    # Numexpr
    a1 = np.linspace(0, 10, size).reshape(shape)
    t = []
    ne.set_num_threads(num_threads)
    for _ in range(nrep):
        t0 = time()
        b1 = ne.evaluate("(x - 1.35) * (x - 4.45) * (x - 8.5)", local_dict={'x': a1})
        t1 = time()
        t.append(round(size / 2 ** 20 * 8 / (t1 - t0), 2))
    t.remove(max(t))
    res_i.append(np.mean(t))

    # Plainbuffer
    a1 = ia.linspace(dtshape, 0, 10, storage=pstorage, nthreads=num_threads)
    expr = ia.Expr(eval_method=eval_method, nthreads=num_threads)
    expr.bind("x", a1)
    expr.bind_out_properties(dtshape, storage=pstorage)
    expr.compile("(x - 1.35) * (x - 4.45) * (x - 8.5)")
    t = []
    for _ in range(nrep):
        t0 = time()
        b1 = expr.eval()
        t1 = time()
        t.append(round(size / 2 ** 20 * 8 / (t1 - t0), 2))
    t.remove(max(t))
    res_i.append(np.mean(t))

    # Superchunk without compression
    a1 = ia.linspace(dtshape, 0, 10, storage=bstorage, clevel=0, **kwargs)
    expr = ia.Expr(eval_method=eval_method, clevel=0, **kwargs)
    expr.bind("x", a1)
    expr.bind_out_properties(dtshape, storage=bstorage)
    expr.compile("(x - 1.35) * (x - 4.45) * (x - 8.5)")
    t = []
    for _ in range(nrep):
        t0 = time()
        b1 = expr.eval()
        t1 = time()
        t.append(round(size / 2 ** 20 * 8 / (t1 - t0), 2))
    t.remove(max(t))
    res_i.append(np.mean(t))

    # Superchunk with compression
    a1 = ia.linspace(dtshape, 0, 10, storage=bstorage, clevel=9, **kwargs)
    expr = ia.Expr(eval_method=eval_method, clevel=9, **kwargs)
    expr.bind("x", a1)
    expr.bind_out_properties(dtshape, storage=bstorage)
    expr.compile("(x - 1.35) * (x - 4.45) * (x - 8.5)")
    t = []
    for _ in range(nrep):
        t0 = time()
        b1 = expr.eval()
        t1 = time()
        t.append(round(size / 2 ** 20 * 8 / (t1 - t0), 2))
    t.remove(max(t))
    res_i.append(np.mean(t))

    # Superchunk with compression and UDF
    a1 = ia.linspace(dtshape, 0, 10, storage=bstorage, clevel=9, **kwargs)
    expr = poly_udf.create_expr([a1], dtshape, storage=bstorage, method=ia.EVAL_ITERBLOSC, clevel=9, **kwargs)
    t = []
    for _ in range(nrep):
        t0 = time()
        b1 = expr.eval()
        t1 = time()
        t.append(round(size / 2 ** 20 * 8 / (t1 - t0), 2))
    t.remove(max(t))
    res_i.append(np.mean(t))

    res.append(res_i)

print(res)
