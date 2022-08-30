#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This computes the sum of a diagonal matrix.  This is to measure ultimate
# write and read performance, without the media (either memory or disk) being a
# bottleneck.

import iarray as ia
from iarray.udf import jit, Array, float64
from time import time


# N = 20_000  # around 3 GB
N = 80_000  # around 50 GB
size = N * N * 8 / 2**30  # size in GB

@jit
def eye(out: Array(float64, 2)) -> int:
    n = out.window_shape[0]
    m = out.window_shape[1]
    row_start = out.window_start[0]
    col_start = out.window_start[1]
    for i in range(n):
        for j in range(m):
            if row_start + i == col_start + j:
                out[i, j] = 1
            else:
                out[i, j] = 0
    return 0


if True:
    ia.set_config_defaults(favor=ia.Favor.CRATIO)
    t0 = time()
    expr = ia.expr_from_udf(eye, [], shape=(N, N), mode='w', urlpath="reduce-eye.iarr")
    # expr = ia.expr_from_udf(eye, [], shape=(N, N))  # in-memory
    iax = expr.eval()
    t = time() - t0
    print(f"time for storing array: {t:.3f}s ({size / t:.2g} GB/s)")

iax = ia.open("reduce-eye.iarr")

t0 = time()
total = iax.sum(axis=(1,0))
t = time() - t0
print(f"time for reducing array: {t:.3f}s ({size / t:.2g} GB/s)")

print("Total sum:", total)
