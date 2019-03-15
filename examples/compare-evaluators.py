from time import time
import numpy as np
import numexpr as ne
from numba import jit

import iarray as ia


@jit(nopython=True)
def poly(x):
    y = np.empty(x.shape, x.dtype)
    for i in range(len(x)):
        y[i] = (x[i] - 1.35) * (x[i] - 4.45) * (x[i] - 8.5)
    return y


cfg = ia.Config()
ctx = ia.Context(cfg)

N = 1000 * 1000 *10

shape = [N]
pshape = [2 * 100 * 1000]
block_size = pshape

# x = ia.arange(ctx, N, shape=shape, pshape=pshape)
xa = ia.linspace(ctx, N, 0., 10., shape=shape, pshape=pshape)
ya = ia.empty(ctx, shape=shape, pshape=pshape)

t0 = time()
for ((i, x), (j, y)) in zip(xa.iter_block(block_size), ya.iter_write()):
    print("->", y.shape, x.shape, i, j)
    assert(i == j)
    # print(f"{index}: {x}")
    y[:] = (x - 1.35) * (x - 4.45) * (x - 8.5)
print("Evaluate via numpy:", round(time() - t0, 3))
ya1 = ia.iarray2numpy(ctx, ya)
print(ya1)

# t0 = time()
# for index, x in xa.iter_block(block_size):
#     y = ne.evaluate('(x - 1.35) * (x - 4.45) * (x - 8.5)')
#     ya[index[0]:index[0] + len(y)] = y
# print("Evaluate via numexpr:", round(time() - t0, 3))
# ya2 = ya.copy()
#
# t0 = time()
# for index, x in xa.iter_block(block_size):
#     y = poly(x)
#     ya[index[0]:index[0] + len(y)] = y
# print("Evaluate via numba:", round(time() - t0, 3))
# ya3 = ya.copy()

#np.testing.assert_almost_equal(ya1, ya2)
#np.testing.assert_almost_equal(ya1, ya3)
