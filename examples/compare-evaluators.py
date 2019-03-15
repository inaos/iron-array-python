from time import time
import numpy as np
import numexpr as ne
from numba import jit

import iarray as ia


N = 1000 * 1000 * 10
shape = [N]
pshape = [2 * 100 * 1000]
block_size = pshape
expression = '(x - 1.35) * (x - 4.45) * (x - 8.5)'


@jit(nopython=True)
def poly(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return y


def do_regular_evaluation():
    print("Regular evaluation of the expression:", expression, "with %d elements" % N)

    x = np.linspace(0, 10, N, dtype=np.double)

    # Reference to compare to
    y0 = (x - 1.35) * (x - 4.45) * (x - 8.5)
    # print(y0, y0.shape)

    t0 = time()
    y1 = (x - 1.35) * (x - 4.45) * (x - 8.5)
    print("Regular evaluate via numpy:", round(time() - t0, 3))

    t0 = time()
    y2 = ne.evaluate(expression, local_dict={'x': x})
    print("Regular evaluate via numexpr:", round(time() - t0, 3))

    t0 = time()
    y3 = poly(x)
    print("Regular evaluate via numba:", round(time() - t0, 3))

    np.testing.assert_almost_equal(y0, y1)
    np.testing.assert_almost_equal(y0, y2)
    np.testing.assert_almost_equal(y0, y3)


def do_block_evaluation():
    print("Block evaluation of the expression:", expression, "with %d elements" % N)

    x = np.linspace(0, 10, N, dtype=np.double)
    # TODO: looks like nelem is not in the same position than numpy
    xa = ia.linspace(ctx, N, 0., 10., shape=shape, pshape=pshape)

    # Reference to compare to
    y0 = (x - 1.35) * (x - 4.45) * (x - 8.5)

    t0 = time()
    ya = ia.empty(ctx, shape=shape, pshape=pshape)
    for ((i, x), (j, y)) in zip(xa.iter_block(block_size), ya.iter_write()):
        y[:] = (x - 1.35) * (x - 4.45) * (x - 8.5)
    print("Block evaluate via numpy:", round(time() - t0, 3))
    y1 = ia.iarray2numpy(ctx, ya)

    t0 = time()
    ya = ia.empty(ctx, shape=shape, pshape=pshape)
    for ((i, x), (j, y)) in zip(xa.iter_block(block_size), ya.iter_write()):
        y[:] = ne.evaluate(expression, local_dict={'x': x})
    print("Block evaluate via numexpr:", round(time() - t0, 3))
    y2 = ia.iarray2numpy(ctx, ya)

    t0 = time()
    ya = ia.empty(ctx, shape=shape, pshape=pshape)
    for ((i, x), (j, y)) in zip(xa.iter_block(block_size), ya.iter_write()):
        y[:] = poly(x)
    print("Block evaluate via numba:", round(time() - t0, 3))
    y3 = ia.iarray2numpy(ctx, ya)

    np.testing.assert_almost_equal(y0, y1)
    np.testing.assert_almost_equal(y0, y2)
    np.testing.assert_almost_equal(y0, y3)


if __name__ == "__main__":
    cfg = ia.Config()
    ctx = ia.Context(cfg)
    do_regular_evaluation()
    print("-*-"*10)
    do_block_evaluation()
