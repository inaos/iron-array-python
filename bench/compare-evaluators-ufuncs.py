# Comparison of different array evaluators (numpy, numexpr, numba, iarray...)

import math
from time import time
from itertools import zip_longest

import numba as nb
import numexpr as ne
import numpy as np

import iarray as ia
from iarray import udf
from iarray.udf import float64, int64


# Number of iterations per benchmark
NITER = 4

# Vector and sizes and chunking
shape = [100 * 1000 * 1000]
N = int(np.prod(shape))

# chunks, blocks = None, None  # use automatic partition advice
# chunks, blocks = [400 * 1000], [16 * 1000]  # user-defined partitions
chunks, blocks = [1 * 1000 * 1000], [20 * 1000]  # user-defined partitions

expression = "(cos(x) - 1.35) * (x - 4.45) * (sin(x) - 8.5)"
lazy_expression = "(ia.cos(x) - 1.35) * (x - 4.45) * (ia.sin(x) - 8.5)"
numpy_expression = "(np.cos(x) - 1.35) * (x - 4.45) * (np.sin(x) - 8.5)"
clevel = 9  # compression level
nthreads = 8  # number of threads for the evaluation and/or compression


def poly_python(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (math.cos(x[i]) - 1.35) * (x[i] - 4.45) * (math.sin(x[i]) - 8.5)
    return y


@nb.jit(nopython=True, cache=True, parallel=True)
def poly_numba(x):
    y = np.empty(x.shape, x.dtype)
    for i in nb.prange(len(x)):
        y[i] = (math.cos(x[i]) - 1.35) * (x[i] - 4.45) * (math.sin(x[i]) - 8.5)
    return y


@nb.jit(nopython=True, cache=True, parallel=True)
def poly_numba2(x, y):
    for i in nb.prange(len(x)):
        y[i] = (math.cos(x[i]) - 1.35) * (x[i] - 4.45) * (math.sin(x[i]) - 8.5)


@udf.jit(verbose=0)
def poly_llvm(out: udf.Array(float64, 1), x: udf.Array(float64, 1)) -> int64:
    n = out.shape[0]
    for i in range(n):
        out[i] = (math.cos(x[i]) - 1.35) * (x[i] - 4.45) * (math.sin(x[i]) - 8.5)

    return 0


def do_regular_evaluation():
    print(f"Regular evaluation of the expression: {expression} with {N} elements")

    x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)

    # Reference to compare to
    y0 = eval(numpy_expression)
    # print(y0, y0.shape)

    if N <= 2e6:
        t0 = time()
        y1 = poly_python(x)
        print("Regular evaluate via python:", round(time() - t0, 4))
        np.testing.assert_almost_equal(y0, y1)

    y1 = None  # shut-up warnings about variables possibly referenced before assignment
    t0 = time()
    for i in range(NITER):
        y1 = eval(numpy_expression)
    print("Regular evaluate via numpy:", round((time() - t0) / NITER, 4))
    # np.testing.assert_almost_equal(y0, y1)

    # ne.set_num_threads(1)
    # t0 = time()
    # for i in range(NITER):
    #     y1 = ne.evaluate(expression, local_dict={"x": x})
    # print("Regular evaluate via numexpr:", round((time() - t0) / NITER, 4))
    # np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    # ne.set_num_threads(nthreads)
    for i in range(NITER):
        y1 = ne.evaluate(expression, local_dict={"x": x})
    print("Regular evaluate via numexpr (multi-thread):", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    # nb.set_num_threads(1)
    # t0 = time()
    # for i in range(NITER):
    #     y1 = poly_numba(x)
    # print("Regular evaluate via numba:", round((time() - t0) / NITER, 4))
    # np.testing.assert_almost_equal(y0, y1)
    #
    # t0 = time()
    # for i in range(NITER):
    #     y1 = poly_numba(x)
    # print("Regular evaluate via numba (II):", round((time() - t0) / NITER, 4))
    # np.testing.assert_almost_equal(y0, y1)

    # nb.set_num_threads(nthreads)
    t0 = time()
    for i in range(NITER):
        y1 = poly_numba(x)
    print("Regular evaluate via numba (multi-thread):", round((time() - t0) / NITER, 4))
    # np.testing.assert_almost_equal(y0, y1)


def do_block_evaluation():
    print(f"Block evaluation")
    cfg = ia.Config(chunks=chunks, blocks=blocks)
    # ia.set_config_defaults(codec=ia.Codec.LZ4, clevel=clevel, nthreads=nthreads, chunks=chunks, blocks=blocks)
    # The latest versions of BTune work much better for 1-dim arrays
    ia.set_config_defaults(cfg=cfg, favor=ia.Favor.SPEED)

    xa = ia.linspace(shape, 0.0, 10.0)
    # x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)
    #
    # print("Operand cratio:", round(xa.cratio, 2))
    #
    # # Reference to compare to
    # y0 = eval(numpy_expression)
    # ya = ia.empty(shape)
    #
    # t0 = time()
    # for i in range(NITER):
    #     ya = ia.empty(shape)
    #     for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
    #         y[:] = eval(numpy_expression)
    # print("Block evaluate via numpy:", round((time() - t0) / NITER, 4))
    #
    # y1 = ia.iarray2numpy(ya)
    # np.testing.assert_almost_equal(y0, y1)
    #
    # ne.set_num_threads(1)
    # t0 = time()
    # for i in range(NITER):
    #     ya = ia.empty(shape)
    #     for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
    #         ne.evaluate(expression, local_dict={"x": x}, out=y)
    # print("Block evaluate via numexpr:", round((time() - t0) / NITER, 4))
    # y1 = ia.iarray2numpy(ya)
    # np.testing.assert_almost_equal(y0, y1)
    #
    # ne.set_num_threads(nthreads)
    # t0 = time()
    # for i in range(NITER):
    #     ya = ia.empty(shape)
    #     for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
    #         ne.evaluate(expression, local_dict={"x": x}, out=y)
    # print("Block evaluate via numexpr (multi-thread):", round((time() - t0) / NITER, 4))
    # y1 = ia.iarray2numpy(ya)
    # np.testing.assert_almost_equal(y0, y1)
    #
    # nb.set_num_threads(1)
    # t0 = time()
    # for i in range(NITER):
    #     ya = ia.empty(shape)
    #     for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
    #         # y[:] = poly_numba(x)
    #         poly_numba2(x, y)
    # print("Block evaluate via numba (II):", round((time() - t0) / NITER, 4))
    # y1 = ia.iarray2numpy(ya)
    # np.testing.assert_almost_equal(y0, y1)
    #
    # nb.set_num_threads(nthreads)
    # t0 = time()
    # for i in range(NITER):
    #     ya = ia.empty(shape)
    #     for ((j, x), (k, y)) in zip_longest(xa.iter_read_block(), ya.iter_write_block()):
    #         # y[:] = poly_numba(x)
    #         poly_numba2(x, y)
    # print("Block evaluate via numba (II, multi-thread):", round((time() - t0) / NITER, 4))
    # y1 = ia.iarray2numpy(ya)
    # np.testing.assert_almost_equal(y0, y1)

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
        # np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    x = xa
    for i in range(NITER):
        ya = eval(lazy_expression, {"x": x, "ia": ia})
        ya = ya.eval()
    avg = round((time() - t0) / NITER, 4)
    print(f"Block evaluate via iarray.LazyExpr.eval (engine: internal): {avg:.4f}")
    y1 = ia.iarray2numpy(ya)
    # np.testing.assert_almost_equal(y0, y1)

    print("Result cratio:", round(ya.cratio, 2))


if __name__ == "__main__":
    do_regular_evaluation()
    print("-*-" * 10)
    do_block_evaluation()
