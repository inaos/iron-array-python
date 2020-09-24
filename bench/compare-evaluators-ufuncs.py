from itertools import zip_longest
import math
from time import time

from numba import jit
import numexpr as ne
import numpy as np

import iarray as ia
from iarray import udf
from iarray.udf import float64, int64


# Number of iterations per benchmark
NITER = 10

# Vector and sizes and chunking
shape = [20 * 1000 * 1000]
N = int(np.prod(shape))

chunkshape, blockshape = None, None  # use automatic partition advice
# chunkshape, blockshape = [400 * 1000], [16 * 1000]  # user-defined partitions
itershape_ = [400 * 1000]

expression = '(cos(x) - 1.35) * (x - 4.45) * (sin(x) - 8.5)'
expression_np = '(np.cos(x) - 1.35) * (x - 4.45) * (np.sin(x) - 8.5)'
clevel = 9   # compression level
clib = ia.LZ4  # compression codec
nthreads = 8  # number of threads for the evaluation and/or compression


def poly_python(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (math.cos(x[i]) - 1.35) * (x[i] - 4.45) * (math.sin(x[i]) - 8.5)
    return y


# @jit(nopython=True, cache=True, parallel=True)
@jit(nopython=True, cache=True)
def poly_numba(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (math.cos(x[i]) - 1.35) * (x[i] - 4.45) * (math.sin(x[i]) - 8.5)
    return y


@jit(nopython=True, cache=True)
def poly_numba2(x, y):
    for i in range(len(x)):
        y[i] = (math.cos(x[i]) - 1.35) * (x[i] - 4.45) * (math.sin(x[i]) - 8.5)


@udf.jit(verbose=0)
def poly_llvm(out: udf.Array(float64, 1), x: udf.Array(float64, 1)) -> int64:
    n = out.shape[0]
    for i in range(n):
        out[i] = (math.cos(x[i]) - 1.35) * (x[i] - 4.45) * (math.sin(x[i]) - 8.5)

    return 0


def do_regular_evaluation():
    print("Regular evaluation of the expression:", expression, "with %d elements" % N)

    x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)

    # Reference to compare to
    y0 = eval(expression_np)
    # print(y0, y0.shape)

    if N <= 2e6:
        t0 = time()
        y1 = poly_python(x)
        print("Regular evaluate via python:", round(time() - t0, 4))
        np.testing.assert_almost_equal(y0, y1)

    y1 = None  # shut-up warnings about variables possibly referenced before assignment
    t0 = time()
    for i in range(NITER):
        y1 = eval(expression_np)
    print("Regular evaluate via numpy:", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    ne.set_num_threads(1)
    for i in range(NITER):
        y1 = ne.evaluate(expression, local_dict={'x': x})
    print("Regular evaluate via numexpr:", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    ne.set_num_threads(nthreads)
    for i in range(NITER):
        y1 = ne.evaluate(expression, local_dict={'x': x})
    print("Regular evaluate via numexpr (multi-thread):", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        y1 = poly_numba(x)
    print("Regular evaluate via numba:", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        y1 = poly_numba(x)
    print("Regular evaluate via numba (II):", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)


def do_block_evaluation(backend):
    if backend is ia.BACKEND_PLAINBUFFER:
        storage = ia.StorageProperties(ia.BACKEND_PLAINBUFFER)
    else:
        storage = ia.StorageProperties(ia.BACKEND_BLOSC, chunkshape, blockshape)

    print(f"Block ({storage.backend}) evaluation of the expression:", expression, "with %d elements" % N)
    cparams = dict(clib=clib, clevel=clevel, nthreads=nthreads)

    x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)
    xa = ia.linspace(ia.dtshape(shape=shape), 0., 10., storage=storage, **cparams)

    if backend is ia.BACKEND_BLOSC:
        print("Operand cratio:", round(xa.cratio, 2))

    # Reference to compare to
    y0 = eval(expression_np)

    # itershape has to be the same than chunkshape for iter_write when using blosc backends
    ya = ia.empty(ia.dtshape(shape=shape), storage=storage, **cparams)
    itershape = ya.chunkshape if backend is ia.BACKEND_BLOSC else itershape_

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape), storage=storage, **cparams)
        for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(itershape), ya.iter_write_block(itershape)):
            y[:] = eval(expression_np)
    print("Block evaluate via numpy:", round((time() - t0) / NITER, 4))

    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape), storage=storage, **cparams)
        for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(itershape), ya.iter_write_block(itershape)):
            ne.evaluate(expression, local_dict={'x': x}, out=y)
    print("Block evaluate via numexpr:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape), storage=storage, **cparams)
        for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(itershape), ya.iter_write_block(itershape)):
            # y[:] = poly_numba(x)
            poly_numba2(x, y)
    print("Block evaluate via numba (II):", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    for engine in ("internal", "udf"):
        # TODO: the line below should be removed after UDF can evaluate things with plain buffers
        # See https://github.com/inaos/iron-array/issues/347
        if backend is ia.BACKEND_PLAINBUFFER and engine == "udf":
            continue

        t0 = time()
        if engine == "internal":
            expr = ia.Expr(**cparams)
            expr.bind('x', xa)
            expr.bind_out_properties(ia.dtshape(shape), storage)
            expr.compile(expression)
        else:
            # For some reason, the UDF engine does not work when backend is PLAINBUFFER
            expr = poly_llvm.create_expr([xa], ia.dtshape(shape), storage=storage,  **cparams)
        for i in range(NITER):
            ya = expr.eval()
        avg = round((time() - t0) / NITER, 4)
        print(f"Block evaluate via iarray.eval (backend: {backend}, engine: {engine}): {avg:.4f}")
        y1 = ia.iarray2numpy(ya)
        np.testing.assert_almost_equal(y0, y1)

    # TODO: support math ufuncs for lazy expressions
    # t0 = time()
    # x = xa
    # for i in range(NITER):
    #     ya = eval(expression_np, {"x": x})
    #     ya = ya.eval(storage=storage, **cparams)
    # avg = round((time() - t0) / NITER, 4)
    # print(f"Block evaluate via iarray.LazyExpr.eval (backend: {backend}, engine: internal): {avg:.4f}")
    # y1 = ia.iarray2numpy(ya)
    # np.testing.assert_almost_equal(y0, y1)

    if backend is ia.BACKEND_BLOSC:
        print("Result cratio:", round(ya.cratio, 2))


if __name__ == "__main__":
    do_regular_evaluation()
    print("-*-" * 10)
    do_block_evaluation(ia.BACKEND_PLAINBUFFER)
    print("-*-" * 10)
    do_block_evaluation(ia.BACKEND_BLOSC)
