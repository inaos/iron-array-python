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


def do_compare():
    print("Evaluating expression:", expression, "with %d elements" % N)
    cfg = ia.Config()
    ctx = ia.Context(cfg)

    x = np.linspace(0, 10, N, dtype=np.double)
    xa = ia.linspace(ctx, N, 0., 10., shape=shape, pshape=pshape)

    # Reference to compare to
    y0 = (x - 1.35) * (x - 4.45) * (x - 8.5)

    t0 = time()
    ya = ia.empty(ctx, shape=shape, pshape=pshape)
    for ((i, x), (j, y)) in zip(xa.iter_block(block_size), ya.iter_write()):
        y[:] = (x - 1.35) * (x - 4.45) * (x - 8.5)
    print("Evaluate via numpy:", round(time() - t0, 3))
    y1 = ia.iarray2numpy(ctx, ya)
    # print(y1, y1.shape)

    t0 = time()
    ya = ia.empty(ctx, shape=shape, pshape=pshape)
    for ((i, x), (j, y)) in zip(xa.iter_block(block_size), ya.iter_write()):
        y[:] = ne.evaluate(expression, local_dict={'x': x})
    print("Evaluate via numexpr:", round(time() - t0, 3))
    y2 = ia.iarray2numpy(ctx, ya)

    t0 = time()
    ya = ia.empty(ctx, shape=shape, pshape=pshape)
    for ((i, x), (j, y)) in zip(xa.iter_block(block_size), ya.iter_write()):
        y[:] = poly(x)
    print("Evaluate via numba:", round(time() - t0, 3))
    y3 = ia.iarray2numpy(ctx, ya)

    np.testing.assert_almost_equal(y0, y1)
    np.testing.assert_almost_equal(y0, y2)
    np.testing.assert_almost_equal(y0, y3)


if __name__ == "__main__":
    do_compare()
