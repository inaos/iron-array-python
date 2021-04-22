# Comparison of different array evaluators (numpy, numexpr, numba, iarray...)
# It looks like you need to set envvar KMP_DUPLICATE_LIB_OK=TRUE manually in order to run this.

from itertools import zip_longest
from time import time
import os

import numba as nb
import numexpr as ne
import numpy as np

import iarray as ia
from iarray import udf
from iarray.udf import float64, int64

# Numba uses OpemMP, and this collides with the libraries in ironArray.
# Using the next envvar seems to fix the issue (bar a small printed info line).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Number of iterations per benchmark
NITER = 10

# Vector sizes and chunking
shape = [2 * 1000 * 1000]
N = int(np.prod(shape))
chunks, blocks = None, None  # use automatic partition advice
# chunks, blocks = [400 * 1000], [16 * 1000]  # user-defined partitions

expression = "(x - 1.35) * (x - 4.45) * (x - 8.5)"
clevel = 9  # compression level
nthreads = 8  # number of threads for the evaluation and/or compression


def poly_python(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return y


@nb.jit(nopython=True, cache=True, parallel=True)
def poly_numba(x):
    y = np.empty(x.shape, x.dtype)
    for i in nb.prange(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return y


@nb.jit(nopython=True, cache=True, parallel=True)
def poly_numba2(x, y):
    for i in nb.prange(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)


@udf.jit(verbose=0)
def poly_llvm(out: udf.Array(float64, 1), x: udf.Array(float64, 1)) -> int64:
    n = out.shape[0]
    for i in range(n):
        out[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


def do_regular_evaluation():
    print(f"Regular evaluation of the expression: {expression} with {N} elements")

    x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)

    # Reference to compare to
    y0 = eval(expression)
    # print(y0, y0.shape)

    if N <= 2e6:
        t0 = time()
        y1 = poly_python(x)
        print("Regular evaluate via python:", round(time() - t0, 4))
        np.testing.assert_almost_equal(y0, y1)

    y1 = None  # shut-up warnings about variables possibly referenced before assignment
    t0 = time()
    for i in range(NITER):
        y1 = eval(expression)
    print("Regular evaluate via numpy:", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    ne.set_num_threads(1)
    for i in range(NITER):
        y1 = ne.evaluate(expression, local_dict={"x": x})
    print("Regular evaluate via numexpr:", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    ne.set_num_threads(nthreads)
    for i in range(NITER):
        y1 = ne.evaluate(expression, local_dict={"x": x})
    print("Regular evaluate via numexpr (multi-thread):", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    nb.set_num_threads(1)
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

    nb.set_num_threads(nthreads)
    t0 = time()
    for i in range(NITER):
        y1 = poly_numba(x)
    print("Regular evaluate via numba (II, multi-thread):", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        y1 = ia.ext.poly_cython(x)
    print("Regular evaluate via cython:", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        y1 = ia.ext.poly_cython_nogil(x)
    print("Regular evaluate via cython (nogil):", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)


def do_block_evaluation(plainbuffer):
    print(f"Block evaluation (plainbuffer={plainbuffer})")
    if plainbuffer:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunks, blocks, plainbuffer=False)

    ia.set_config(clevel=clevel, nthreads=nthreads, store=store)
    print(ia.get_config())

    x = np.linspace(0, 10, N).reshape(shape)
    xa = ia.linspace(shape, 0.0, 10.0)

    if not plainbuffer:
        print("Operand cratio:", round(xa.cratio, 2))

    # Reference to compare to
    y0 = eval(expression)

    ya = ia.empty(shape)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(shape)
        for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
            y[:] = eval(expression)
    print("Block evaluate via numpy:", round((time() - t0) / NITER, 4))

    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    ne.set_num_threads(1)
    t0 = time()
    for i in range(NITER):
        ya = ia.empty(shape)
        for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
            ne.evaluate(expression, local_dict={"x": x}, out=y)
    print("Block evaluate via numexpr:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    ne.set_num_threads(nthreads)
    t0 = time()
    for i in range(NITER):
        ya = ia.empty(shape)
        for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
            ne.evaluate(expression, local_dict={"x": x}, out=y)
    print("Block evaluate via numexpr (multi-thread):", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    nb.set_num_threads(1)
    t0 = time()
    for i in range(NITER):
        ya = ia.empty(shape)
        for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
            # y[:] = poly_numba(x)
            poly_numba2(x, y)
    print("Block evaluate via numba (II):", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    nb.set_num_threads(nthreads)
    t0 = time()
    for i in range(NITER):
        ya = ia.empty(shape)
        for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
            # y[:] = poly_numba(x)
            poly_numba2(x, y)
    print("Block evaluate via numba (II, multi-thread):", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(shape)
        for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
            y[:] = ia.ext.poly_cython(x)
    print("Block evaluate via cython:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(shape)
        for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
            y[:] = ia.ext.poly_cython_nogil(x)
    print("Block evaluate via cython (nogil):", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    for engine in ("internal", "udf"):
        t0 = time()
        if engine == "internal":
            expr = ia.expr_from_string(expression, {"x": xa})
        else:
            expr = ia.expr_from_udf(poly_llvm, [xa])
        for i in range(NITER):
            ya = expr.eval()
        avg = round((time() - t0) / NITER, 4)
        print(f"Block evaluate via iarray.eval (engine: {engine}): {avg:.4f}")
        y1 = ia.iarray2numpy(ya)
        np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    x = xa
    for i in range(NITER):
        ya = eval(expression, {"x": x})
        ya = ya.eval()
    avg = round((time() - t0) / NITER, 4)
    print(f"Block evaluate via iarray.LazyExpr.eval (engine: internal): {avg:.4f}")
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    if not plainbuffer:
        print("Result cratio:", round(ya.cratio, 2))


if __name__ == "__main__":
    do_regular_evaluation()
    print("-*-" * 10)
    do_block_evaluation(plainbuffer=True)
    print("-*-" * 10)
    do_block_evaluation(plainbuffer=False)
