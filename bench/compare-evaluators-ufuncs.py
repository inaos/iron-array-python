import iarray as ia
from time import time
import numpy as np
import numexpr as ne
import numba as nb
from itertools import zip_longest as zip
import math


# Number of iterations per benchmark
NITER = 10

# Vector sizes and partitions
shape = [20 * 1000 * 1000]
N = int(np.prod(shape))

chunkshape = [400 * 1000]
blockshape = [16 * 1000]

itershape = chunkshape

expression = '(cos(x) - 1.35) * (x - 4.45) * (sin(x) - 8.5)'
clevel = 6   # compression level
clib = ia.LZ4  # compression codec
nthreads = 8  # number of threads for the evaluation and/or compression


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
            y[i] = (np.cos(x[i]) - 1.35) * (x[i] - 4.45) * (np.sin(x[i]) - 8.5)
        return y

    cc.compile()
    import numba_prec  # for pre-compiled numba code


def poly_python(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (math.cos(x[i]) - 1.35) * (x[i] - 4.45) * (math.sin(x[i]) - 8.5)
    return y


# @nb.jit(nopython=True, cache=True, parallel=True)
@nb.jit(nopython=True, cache=True)
def poly_numba(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (np.cos(x[i]) - 1.35) * (x[i] - 4.45) * (np.sin(x[i]) - 8.5)
    return y


@nb.jit(nopython=True, cache=True)
def poly_numba2(x, y):
    for i in range(len(x)):
        y[i] = (np.cos(x[i]) - 1.35) * (x[i] - 4.45) * (np.sin(x[i]) - 8.5)


def do_regular_evaluation():
    print("Regular evaluation of the expression:", expression, "with %d elements" % N)

    x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)

    # Reference to compare to
    y0 = (np.cos(x) - 1.35) * (x - 4.45) * (np.sin(x) - 8.5)
    # print(y0, y0.shape)

    if N <= 2e6:
        t0 = time()
        y1 = poly_python(x)
        print("Regular evaluate via python:", round(time() - t0, 4))
        np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        y1 = (np.cos(x) - 1.35) * (x - 4.45) * (np.sin(x) - 8.5)
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

    nb.set_num_threads(1)
    y = x.copy()
    t0 = time()
    for i in range(NITER):
        #y1 = poly_numba(x)
        poly_numba2(x, y)
    print("Regular evaluate via numba:", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    nb.set_num_threads(nthreads)
    y = x.copy()
    t0 = time()
    for i in range(NITER):
        #y1 = poly_numba(x)
        poly_numba2(x, y)
    print("Regular evaluate via numba (multi-thread):", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    if NUMBA_PRECOMP:
        t0 = time()
        for i in range(NITER):
            #y1 = numba_prec.poly_double(x)
            y1[:] = numba_prec.poly_double(x)
        print("Regular evaluate via pre-compiled numba:", round((time() - t0) / NITER, 4))
        np.testing.assert_almost_equal(y0, y1)


def do_block_evaluation(chunkshape_):
    if chunkshape_ is None:
        storage = ia.StorageProperties(backend="plainbuffer")
    else:
        storage = ia.StorageProperties(backend="blosc", chunkshape=chunkshape_, blockshape=blockshape)
    print(f"Block ({storage.backend}) evaluation of the expression:", expression, "with %d elements" % N)
    cparams = dict(clib=clib, clevel=clevel, nthreads=nthreads, storage=storage)

    x = np.linspace(0, 10, N, dtype=np.double).reshape(shape)
    # TODO: looks like nelem is not in the same position than numpy
    xa = ia.linspace(ia.dtshape(shape=shape), 0., 10., **cparams)

    if chunkshape_ is not None:
        print("Operand cratio:", round(xa.cratio, 2))

    # Reference to compare to
    y0 = (np.cos(x) - 1.35) * (x - 4.45) * (np.sin(x) - 8.5)

    block_write = None if chunkshape_ == chunkshape else itershape

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape), **cparams)
        for ((j, x), (k, y)) in zip(xa.iter_read_block(itershape), ya.iter_write_block(block_write)):
            y[:] = (np.cos(x) - 1.35) * (x - 4.45) * (np.sin(x) - 8.5)
    print("Block evaluate via numpy:", round((time() - t0) / NITER, 4))

    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape), **cparams)
        for ((j, x), (k, y)) in zip(xa.iter_read_block(itershape), ya.iter_write_block(block_write)):
            ne.evaluate(expression, local_dict={'x': x}, out=y)
    print("Block evaluate via numexpr:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    nb.set_num_threads(1)
    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape), **cparams)
        for ((j, x), (k, y)) in zip(xa.iter_read_block(itershape), ya.iter_write_block(block_write)):
            # y[:] = poly_numba(x)
            poly_numba2(x, y)
    print("Block evaluate via numba:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    nb.set_num_threads(nthreads)
    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ia.dtshape(shape=shape), **cparams)
        for ((j, x), (k, y)) in zip(xa.iter_read_block(itershape), ya.iter_write_block(block_write)):
            # y[:] = poly_numba(x)
            poly_numba2(x, y)
    print("Block evaluate via numba (multi-thread):", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    if NUMBA_PRECOMP:
        t0 = time()
        for i in range(NITER):
            ya = ia.empty(ia.dtshape(shape=shape), **cparams)
            for ((j, x), (k, y)) in zip(xa.iter_read_block(itershape), ya.iter_write_block(block_write)):
                y[:] = numba_prec.poly_double(x)
        print("Block evaluate via pre-compiled numba:", round((time() - t0) / NITER, 4))
        y1 = ia.iarray2numpy(ya)
        np.testing.assert_almost_equal(y0, y1)

    if chunkshape_ is None:
        eval_method = ia.EVAL_ITERCHUNK
    else:
        eval_method = ia.EVAL_ITERBLOSC

    t0 = time()
    expr = ia.Expr(eval_method=eval_method, **cparams)
    expr.bind('x', xa)
    expr.bind_out_properties(ia.dtshape(shape, np.float64), storage=storage)
    expr.compile(expression)
    for i in range(NITER):
        ya = expr.eval()
    avg = round((time() - t0) / NITER, 4)
    print(f"Block evaluate via iarray.eval (method: {eval_method}): {avg:.4f}")
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    x = xa
    for i in range(NITER):
        ya = ((x.cos() - 1.35) * (x - 4.45) * (x.sin() - 8.5)).eval(
            method="iarray_eval", eval_method=eval_method, **cparams)
    avg = round((time() - t0) / NITER, 4)
    print(f"Block evaluate via iarray.LazyExpr.eval (method: {eval_method}): {avg:.4f}")
    y1 = ia.iarray2numpy(ya)
    np.testing.assert_almost_equal(y0, y1)

    if (chunkshape_ is not None):
        print("Result cratio:", round(ya.cratio, 2))


if __name__ == "__main__":
    do_regular_evaluation()
    print("-*-" * 10)
    do_block_evaluation(chunkshape)
    print("-*-" * 10)
    do_block_evaluation(None)
