#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import iarray as ia
import numpy as np
import zarr
from time import time

shape = (1_000, 1_000, 1_000)
chunks = (100, 100, 100)
blocks = (25, 25, 25)


# print(selection)

dtype = np.float64
itemsize = np.dtype(dtype).itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a caterva array from a numpy array
a = ia.numpy2iarray(nparray, chunks=chunks, blocks=blocks)
b = zarr.array(nparray, shape=shape, chunks=chunks, dtype=dtype)

n_iter = 1

selection = (
    np.random.choice(np.arange(shape[0]), 2),
    np.arange(shape[1]),
    np.random.choice(np.arange(shape[2]), 2),
)

shape = [len(s) for s in selection]
value = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

# Set selection

t0 = time()
for _ in range(n_iter):
    b.set_orthogonal_selection(selection, value)
t1 = time()
print(f"Set zarr: {(t1 - t0)/n_iter:.4f} s")

t0 = time()
for _ in range(n_iter):
    a.set_orthogonal_selection(selection, value)
t1 = time()
print(f"Set iarray: {(t1 - t0)/n_iter:.4f} s")


# Get selection
t0 = time()
for _ in range(n_iter):
    buffer2 = b.get_orthogonal_selection(selection)
t1 = time()
print(f"get zarr: {(t1 - t0)/n_iter:.4f} s")

t0 = time()
for _ in range(n_iter):
    buffer = a.get_orthogonal_selection(selection).view(dtype)
t1 = time()
print(f"get iarray: {(t1 - t0)/n_iter:.4f} s")

np.testing.assert_equal(buffer, buffer2)
