from itertools import zip_longest as zip
from time import time

from numba import jit
import numexpr as ne
import numpy as np

import iarray as ia
from iarray import udf
from iarray.udf import Array
from iarray.py2llvm import float64, int64


# Number of iterations per benchmark
NITER = 10

# Vector sizes and partitions
shape = [20 * 1000 * 1000]
N = int(np.prod(shape))
pshape = [4000 * 1000]

block_size = pshape
expression = '(x - 1.35) * (x - 4.45) * (x - 8.5)'
clevel = 5   # compression level
clib = ia.LZ4  # compression codec
nthreads = 20  # number of threads for the evaluation and/or compression


# Make this True if you want to test the pre-compilation in Numba (not necessary, really)
NUMBA_PRECOMP = False

if NUMBA_PRECOMP:
    from numba.pycc import CC
    cc = CC('numba_prec')
    # Uncomment the following line to print out the compilation steps
    cc.verbose = True

    @cc.export('poly_double', 'f8[:](f8[:])')
    def poly_numba_prec(x):
        y = np.empty(x.shape, x.dtype)
        for i in range(len(x)):
            y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
        return y

    cc.compile()
    import numba_prec  # for pre-compiled numba code


def poly_python(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return y


# @jit(nopython=True, cache=True, parallel=True)
@jit(nopython=True, cache=True)
def poly_numba(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return y


@jit(nopython=True, cache=True)
def poly_numba2(x, y):
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)


@udf.jit(verbose=0)
def poly_llvm(out: Array(float64, 1), x: Array(float64, 1)) -> int64:
    n = out.shape[0]
    for i in range(n):
        out[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)

    return 0


def do_regular_evaluation():
    print("Regular evaluation of the expression:", expression, "with %d elements" % N)

    x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)

    # Reference to compare to
    y0 = (x - 1.35) * (x - 4.45) * (x - 8.5)
    # print(y0, y0.shape)

    if N <= 2e6:
        t0 = time()
        y1 = poly_python(x)
        print("Regular evaluate via python:", round(time() - t0, 4))
        np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        y1 = (x - 1.35) * (x - 4.45) * (x - 8.5)
    print("Regular evaluate via numpy:", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    nthreads = ne.set_num_threads(1)
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

    if NUMBA_PRECOMP:
        t0 = time()
        for i in range(NITER):
            y1 = numba_prec.poly_double(x)
        print("Regular evaluate via pre-compiled numba:", round((time() - t0) / NITER, 4))
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


def do_block_evaluation(pshape_):
    storage = "superchunk" if pshape_ is not None else "plain buffer"
    print(f"Block ({storage}) evaluation of the expression:", expression, "with %d elements" % N)
    cparams = dict(clib=clib, clevel=clevel, nthreads=nthreads)

    x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)
    # TODO: looks like nelem is not in the same position than numpy
    xa = ia.linspace(ia.dtshape(shape=shape, pshape=pshape_), 0., 10., **cparams)

    if (pshape is not None):
        print("Operand cratio:", round(xa.cratio, 2))

    # Reference to compare to
    y0 = (x - 1.35) * (x - 4.45) * (x - 8.5)

    block_write = None if pshape_ == pshape else block_size

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape, pshape=pshape_), **cparams)
        for ((j, x), (k, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            y[:] = (x - 1.35) * (x - 4.45) * (x - 8.5)
    print("Block evaluate via numpy:", round((time() - t0) / NITER, 4))

    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape, pshape=pshape_), **cparams)
        for ((j, x), (k, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            ne.evaluate(expression, local_dict={'x': x}, out=y)
    print("Block evaluate via numexpr:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape, pshape=pshape_), **cparams)
        for ((j, x), (k, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            # y[:] = poly_numba(x)
            poly_numba2(x, y)
    print("Block evaluate via numba (II):", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    if NUMBA_PRECOMP:
        t0 = time()
        for i in range(NITER):
            ya = ia.empty(ia.dtshape(shape=shape, pshape=pshape_), **cparams)
            for ((j, x), (k, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
                y[:] = numba_prec.poly_double(x)
        print("Block evaluate via pre-compiled numba:", round((time() - t0) / NITER, 4))
        y1 = ia.iarray2numpy(ya)
        np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape, pshape=pshape_), **cparams)
        for ((j, x), (k, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            y[:] = ia.ext.poly_cython(x)
    print("Block evaluate via cython:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape, pshape=pshape_), **cparams)
        for ((j, x), (k, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            y[:] = ia.ext.poly_cython_nogil(x)
    print("Block evaluate via cython (nogil):", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    if pshape_ is None:
        # eval_method = "iterchunk"
        eval_method = "iterblock"
    else:
        eval_method = "iterblosc"

    t0 = time()
    if pshape_ is None:
        expr = ia.Expr(eval_flags=eval_method, **cparams)
        expr.bind('x', xa)
        expr.compile('(x - 1.35) * (x - 4.45) * (x - 8.5)')
    else:
        # expr.compile_udf(poly_llvm)
        expr = poly_llvm.create_expr([xa], **cparams)
    for i in range(NITER):
        ya = expr.eval(shape, pshape_, np.float64)
    avg = round((time() - t0) / NITER, 4)
    print(f"Block evaluate via iarray.eval (method: {eval_method}): {avg:.4f}")
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    x = xa
    for i in range(NITER):
        ya = ((x - 1.35) * (x - 4.45) * (x - 8.5))
        ya = ya.eval(method="iarray_eval", pshape=pshape_, eval_flags=eval_method, **cparams)
    avg = round((time() - t0) / NITER, 4)
    print(f"Block evaluate via iarray.LazyExpr.eval (method: {eval_method}): {avg:.4f}")
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    if (pshape is not None):
        print("Result cratio:", round(ya.cratio, 2))


if __name__ == "__main__":
    do_regular_evaluation()
    print("-*-" * 10)
    do_block_evaluation(pshape)
    print("-*-" * 10)
    do_block_evaluation(None)
