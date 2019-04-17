import iarray as ia
from time import time
import numpy as np
import numexpr as ne
from numba import jit
from itertools import zip_longest as zip


# Number of iterations per benchmark
NITER = 10

# Vector sizes and partitions
N = 10 * 1000 * 1000
shape = [N]
pshape = [100 * 1000]

block_size = pshape
expression = '(x - 1.35) * (x - 4.45) * (x - 8.5) * (x + 1.5) * (x + 4.6)'
clevel = 0   # compression level
clib = ia.IARRAY_LZ4  # compression codec

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
            y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5) * (x[i] + 1.5) * (x[i] + 4.6)
        return y

    cc.compile()
    import numba_prec  # for pre-compiled numba code


def poly_python(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5) * (x[i] + 1.5) * (x[i] + 4.6)
    return y


# @jit(nopython=True, cache=True, parallel=True)
@jit(nopython=True, cache=True)
def poly_numba(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5) * (x[i] + 1.5) * (x[i] + 4.6)
    return y


@jit(nopython=True, cache=True)
def poly_numba2(x, y):
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5) * (x[i] + 1.5) * (x[i] + 4.6)


def do_regular_evaluation():
    print("Regular evaluation of the expression:", expression, "with %d elements" % N)

    x = np.linspace(0, 10, N, dtype=np.double)

    # Reference to compare to
    y0 = (x - 1.35) * (x - 4.45) * (x - 8.5) * (x + 1.5) * (x + 4.6)
    # print(y0, y0.shape)

    if N <= 2e6:
        t0 = time()
        y1 = poly_python(x)
        print("Regular evaluate via python:", round(time() - t0, 4))
        np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        y1 = (x - 1.35) * (x - 4.45) * (x - 8.5) * (x + 1.5) * (x + 4.6)
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
        y1 = ia.poly_cython(x)
    print("Regular evaluate via cython:", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        y1 = ia.poly_cython_nogil(x)
    print("Regular evaluate via cython (nogil):", round((time() - t0) / NITER, 4))
    np.testing.assert_almost_equal(y0, y1)


def do_block_evaluation(pshape_):
    storage = "superchunk" if pshape_ is not None else "plain buffer"
    print(f"Block ({storage}) evaluation of the expression:", expression, "with %d elements" % N)
    cfg = ia.Config(eval_flags="iterblock", compression_codec=clib, compression_level=clevel,
                    blocksize=0, # block_size[0],
                    max_num_threads=1)
    ctx = ia.Context(cfg)

    x = np.linspace(0, 10, N, dtype=np.double)
    # TODO: looks like nelem is not in the same position than numpy
    xa = ia.linspace(ctx, N, 0., 10., shape=shape, pshape=pshape_)

    # Reference to compare to
    y0 = (x - 1.35) * (x - 4.45) * (x - 8.5) * (x + 1.5) * (x + 4.6)

    block_write = None if pshape_ == pshape else block_size

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ctx, shape=shape, pshape=pshape_)
        for ((i, x), (j, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            y[:] = (x - 1.35) * (x - 4.45) * (x - 8.5) * (x + 1.5) * (x + 4.6)
    print("Block evaluate via numpy:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ctx, ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ctx, shape=shape, pshape=pshape_)
        for ((i, x), (j, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            ne.evaluate(expression, local_dict={'x': x}, out=y)
    print("Block evaluate via numexpr:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ctx, ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ctx, shape=shape, pshape=pshape_)
        for ((i, x), (j, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            # y[:] = poly_numba(x)
            poly_numba2(x, y)
    print("Block evaluate via numba:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ctx, ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ctx, shape=shape, pshape=pshape_)
        for ((i, x), (j, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            # y[:] = poly_numba(x)
            poly_numba2(x, y)
    print("Block evaluate via numba (II):", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ctx, ya)
    np.testing.assert_almost_equal(y0, y1)

    if NUMBA_PRECOMP:
        t0 = time()
        for i in range(NITER):
            ya = ia.empty(ctx, shape=shape, pshape=pshape_)
            for ((i, x), (j, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
                y[:] = numba_prec.poly_double(x)
        print("Block evaluate via pre-compiled numba:", round((time() - t0) / NITER, 4))
        y1 = ia.iarray2numpy(ctx, ya)
        np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ctx, shape=shape, pshape=pshape_)
        for ((i, x), (j, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            y[:] = ia.poly_cython(x)
    print("Block evaluate via cython:", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ctx, ya)
    np.testing.assert_almost_equal(y0, y1)

    t0 = time()
    for i in range(NITER):
        ya = ia.empty(ctx, shape=shape, pshape=pshape_)
        for ((i, x), (j, y)) in zip(xa.iter_read_block(block_size), ya.iter_write_block(block_write)):
            y[:] = ia.poly_cython_nogil(x)
    print("Block evaluate via cython (nogil):", round((time() - t0) / NITER, 4))
    y1 = ia.iarray2numpy(ctx, ya)
    np.testing.assert_almost_equal(y0, y1)

    if pshape_ is not None:
        t0 = time()
        expr = ia.Expression(ctx)
        expr.bind(b'x', xa)
        expr.compile(b'(x - 1.35) * (x - 4.45) * (x - 8.5) * (x + 1.5) * (x + 4.6)')
        for i in range(NITER):
            ya = expr.eval(shape, pshape_, "double")
        print("Block evaluate via iarray.eval:", round((time() - t0) / NITER, 4))
        y1 = ia.iarray2numpy(ctx, ya)
        np.testing.assert_almost_equal(y0, y1)

        # TODO: This currently crashes.  Investigate...
        # t0 = time()
        # x = xa
        # for i in range(NITER):
        #     ya = ((x - 1.35) * (x - 4.45) * (x - 8.5)).eval()
        # print("Block evaluate via iarray.LazyExpr.eval('numexpr'):", round((time() - t0) / NITER, 4))
        # y1 = ia.iarray2numpy(ctx, ya)
        # np.testing.assert_almost_equal(y0, y1)

        t0 = time()
        x = xa
        for i in range(NITER):
            ya = ((x - 1.35) * (x - 4.45) * (x - 8.5) * (x + 1.5) * (x + 4.6)).eval(method="iarray.eval")
        print("Block evaluate via iarray.LazyExpr.eval('iarray.eval')):", round((time() - t0) / NITER, 4))
        y1 = ia.iarray2numpy(ctx, ya)
        np.testing.assert_almost_equal(y0, y1)


if __name__ == "__main__":
    #do_regular_evaluation()
    print("-*-"*10)
    do_block_evaluation(pshape)
    print("-*-" * 10)
    #do_block_evaluation(None)
