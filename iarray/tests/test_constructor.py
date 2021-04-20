import os
import pytest
import numpy as np
import iarray as ia


# linspace
@pytest.mark.parametrize(
    "start, stop, shape, chunkshape, blockshape, dtype",
    [
        (0, 10, [100, 120, 50], [33, 21, 34], [12, 13, 7], np.float64),
        (-0.1, -0.2, [40, 39, 52, 12], [12, 17, 6, 5], [5, 4, 6, 5], np.float32),
        (0, 10, [55, 24, 31], None, None, np.float64),
        (-0.1, -0.2, [4, 3, 5, 2], None, None, np.float32),
    ],
)
def test_linspace(start, stop, shape, chunkshape, blockshape, dtype):
    if blockshape is None or chunkshape is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunkshape, blockshape)
    size = int(np.prod(shape))
    a = ia.linspace(shape, start, stop, dtype=dtype, store=store)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    c = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)


# arange
@pytest.mark.parametrize(
    "start, stop, shape, chunkshape, blockshape, dtype",
    [
        (0, 10, [22, 21, 51], [12, 14, 22], [5, 3, 6], np.float64),
        (0, 1, [12, 12, 15, 13, 18, 19], [6, 5, 4, 7, 7, 5], [2, 2, 1, 2, 3, 3], np.float32),
        (0, 10, [10, 12, 5], None, None, np.float64),
        (-0.1, -0.2, [4, 3, 5, 2], None, None, np.float32),
    ],
)
def test_arange(start, stop, shape, chunkshape, blockshape, dtype):
    if blockshape is None or chunkshape is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunkshape, blockshape)
    size = int(np.prod(shape))
    step = (stop - start) / size
    a = ia.arange(shape, start, stop, step, dtype=dtype, store=store)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    c = np.arange(start, stop, step, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)


# from_file
@pytest.mark.parametrize(
    "start, stop, shape, chunkshape, blockshape, dtype, urlpath",
    [
        (0, 10, [1234], [123], [21], np.float64, "test.fromfile0.iarray"),
        (
            -0.1,
            -0.10,
            [10, 12, 21, 31, 11],
            [4, 3, 5, 5, 2],
            [2, 3, 2, 3, 2],
            np.float32,
            "test.fromfile1.iarray",
        ),
    ],
)
def test_from_file(start, stop, shape, chunkshape, blockshape, dtype, urlpath):
    size = int(np.prod(shape))
    npdtype = np.float64 if dtype == np.float64 else np.float32
    a = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    store = ia.Store(chunkshape, blockshape, urlpath, enforce_frame=True)
    ia.numpy2iarray(a, store=store)
    c = ia.load(urlpath)
    d = ia.iarray2numpy(c)
    np.testing.assert_almost_equal(a, d)
    os.remove(urlpath)


# get_slice
@pytest.mark.parametrize(
    "start, stop, slice, shape, chunkshape, blockshape, dtype",
    [
        (
            0,
            10,
            (slice(2, 4), slice(5, 10), slice(1, 2)),
            [21, 31, 21],
            [10, 12, 5],
            [3, 6, 2],
            np.float64,
        ),
        (-0.1, -0.2, (slice(2, 4), slice(7, 12)), [55, 123], [12, 16], [5, 7], np.float32),
        (0, 10, (slice(2, 4), slice(5, 10), slice(1, 2)), [10, 12, 5], None, None, np.float64),
        (-0.1, -0.2, (slice(2, 4), slice(7, 12)), [12, 16], None, None, np.float32),
    ],
)
def test_slice(start, stop, slice, shape, chunkshape, blockshape, dtype):
    if blockshape is None or chunkshape is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunkshape, blockshape)
    size = int(np.prod(shape))
    step = (stop - start) / size
    a = ia.arange(shape, start, stop, step, dtype=dtype, store=store)
    b = a[slice]
    c = ia.iarray2numpy(b)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    d = np.arange(start, stop, step, dtype=npdtype).reshape(shape)[slice]
    np.testing.assert_almost_equal(c, d)


# empty  # TODO: make it work properly
@pytest.mark.parametrize(
    "shape, chunkshape, blockshape, dtype",
    [
        ([55, 123, 72], [10, 12, 25], [2, 3, 7], np.float64),
        ([10, 12, 5], None, None, np.float32),
    ],
)
def test_empty(shape, chunkshape, blockshape, dtype):
    if blockshape is None or chunkshape is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunkshape, blockshape)
    with ia.config(store=store):
        a = ia.empty(shape, dtype=dtype)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    assert b.dtype == npdtype
    assert b.shape == tuple(shape)


# zeros
@pytest.mark.parametrize(
    "shape, chunkshape, blockshape, dtype",
    [
        ([134, 1234, 238], [10, 25, 35], [2, 7, 12], np.float64),
        ([456, 431], [102, 16], [12, 7], np.float32),
        ([10, 12, 5], None, None, np.float64),
        ([12, 16], None, None, np.float32),
    ],
)
def test_zeros(shape, chunkshape, blockshape, dtype):
    if blockshape is None or chunkshape is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunkshape, blockshape)
    a = ia.zeros(shape, dtype=dtype, store=store)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    c = np.zeros(shape, dtype=npdtype)
    np.testing.assert_almost_equal(b, c)


# ones
@pytest.mark.parametrize(
    "shape, chunkshape, blockshape, dtype",
    [
        ([456, 12, 234], [55, 6, 21], [12, 3, 5], np.float64),
        ([1024, 55], [66, 22], [12, 3], np.float32),
        ([10, 12, 5], None, None, np.float64),
        ([12, 16], None, None, np.float32),
    ],
)
def test_ones(shape, chunkshape, blockshape, dtype):
    if blockshape is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunkshape, blockshape, enforce_frame=True)
    a = ia.ones(shape, dtype=dtype, store=store)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    c = np.ones(shape, dtype=npdtype)
    np.testing.assert_almost_equal(b, c)


# full
@pytest.mark.parametrize(
    "fill_value, shape, chunkshape, blockshape, dtype",
    [
        (8.34, [123, 432, 222], [24, 31, 15], [6, 6, 6], np.float64),
        (2.00001, [567, 375], [52, 16], [9, 7], np.float32),
        (8.34, [10, 12, 5], None, None, np.float64),
        (2.00001, [12, 16], None, None, np.float32),
    ],
)
def test_full(fill_value, shape, chunkshape, blockshape, dtype):
    if blockshape is None:
        store = ia.Store(plainbuffer=True)
    else:
        store = ia.Store(chunkshape, blockshape)
    a = ia.full(shape, fill_value, dtype=dtype, store=store)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    c = np.full(shape, fill_value, dtype=npdtype)
    np.testing.assert_almost_equal(b, c)
