import os
import pytest
import iarray as ia
import numpy as np


# linspace
@pytest.mark.parametrize("start, stop, shape, pshape, dtype",
                         [
                             (0, 10, [10, 12, 5], [2, 3, 2], np.float64),
                             (-0.1, -0.2, [4, 3, 5, 2], [2, 2, 2, 2], np.float32),
                             (0, 10, [10, 12, 5], None, np.float64),
                             (-0.1, -0.2, [4, 3, 5, 2], None, np.float32)
                         ])
def test_linspace(start, stop, shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.linspace(ia.dtshape(shape, pshape, dtype), start, stop)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    c = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)


# arange
@pytest.mark.parametrize("start, stop, shape, pshape, dtype",
                         [
                             (0, 10, [10, 12, 5], [2, 3, 2], np.float64),
                             (-0.1, -0.2, [4, 3, 5, 2], [2, 3, 2, 2], np.float32),
                             (0, 10, [10, 12, 5], None, np.float64),
                             (-0.1, -0.2, [4, 3, 5, 2], None, np.float32)
                         ])
def test_arange(start, stop, shape, pshape, dtype):
    size = int(np.prod(shape))
    step = (stop - start) / size
    a = ia.arange(ia.dtshape(shape=shape, pshape=pshape, dtype=dtype), start, stop, step)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    c = np.arange(start, stop, step, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)


# from_file
@pytest.mark.parametrize("start, stop, shape, pshape, dtype, filename",
                         [
                             (0, 10, [9], [2], np.float64, "test.fromfile0.iarray"),
                             (-0.1, -0.10, [4, 3, 5, 5, 2], [2, 3, 2, 3, 2], np.float32, "test.fromfile1.iarray")
                         ])
def test_from_file(start, stop, shape, pshape, dtype, filename):
    size = int(np.prod(shape))
    npdtype = np.float64 if dtype == np.float64 else np.float32
    a = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    b = ia.numpy2iarray(a, pshape, filename=filename)
    c = ia.from_file(filename)
    d = ia.iarray2numpy(c)
    np.testing.assert_almost_equal(a, d)
    os.remove(filename)


# get_slice
@pytest.mark.parametrize("start, stop, slice, shape, pshape, dtype",
                         [
                             (0, 10, (slice(2, 4), slice(5, 10), slice(1, 2)), [10, 12, 5], [2, 3, 2], np.float64),
                             (-0.1, -0.2, (slice(2, 4), slice(7, 12)), [12, 16], [2, 7], np.float32),
                             (0, 10, (slice(2, 4), slice(5, 10), slice(1, 2)), [10, 12, 5], None, np.float64),
                             (-0.1, -0.2, (slice(2, 4), slice(7, 12)), [12, 16], None, np.float32)
                         ])
def test_slice(start, stop, slice, shape, pshape, dtype):
    size = int(np.prod(shape))
    step = (stop - start) / size
    a = ia.arange(ia.dtshape(shape=shape, pshape=pshape, dtype=dtype), start, stop, step)
    b = a[slice]
    c = ia.iarray2numpy(b)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    d = np.arange(start, stop, step, dtype=npdtype).reshape(shape)[slice]
    np.testing.assert_almost_equal(c, d)


# empty  # TODO: make it work properly
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([10, 12, 5], [2, 3, 2], np.float64),
                             ([10, 12, 5], None, np.float32),
                         ])
def test_empty(shape, pshape, dtype):
    a = ia.empty(ia.dtshape(shape, pshape, dtype))
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    assert b.dtype == npdtype
    assert b.shape == tuple(shape)


# zeros
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([10, 12, 5], [2, 3, 2], np.float64),
                             ([12, 16], [2, 7], np.float32),
                             ([10, 12, 5], None, np.float64),
                             ([12, 16], None, np.float32)
                         ])
def test_zeros(shape, pshape, dtype):
    a = ia.zeros(ia.dtshape(shape, pshape, dtype))
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    c = np.zeros(shape, dtype=npdtype)
    np.testing.assert_almost_equal(b, c)


# ones
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([10, 12, 5], [2, 3, 2], np.float64),
                             ([12, 16], [2, 7], np.float32),
                             ([10, 12, 5], None, np.float64),
                             ([12, 16], None, np.float32)
                         ])
def test_ones(shape, pshape, dtype):
    a = ia.ones(ia.dtshape(shape, pshape, dtype))
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    c = np.ones(shape, dtype=npdtype)
    np.testing.assert_almost_equal(b, c)


# full
@pytest.mark.parametrize("fill_value, shape, pshape, dtype",
                         [
                             (8.34, [10, 12, 5], [2, 3, 2], np.float64),
                             (2.00001, [12, 16], [2, 7], np.float32),
                             (8.34, [10, 12, 5], None, np.float64),
                             (2.00001, [12, 16], None, np.float32)
                         ])
def test_full(fill_value, shape, pshape, dtype):
    a = ia.full(ia.dtshape(shape, pshape, dtype), fill_value)
    b = ia.iarray2numpy(a)
    npdtype = np.float64 if dtype == np.float64 else np.float32
    c = np.full(shape, fill_value, dtype=npdtype)
    np.testing.assert_almost_equal(b, c)
