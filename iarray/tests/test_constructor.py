import os
import pytest
import iarray as ia
import numpy as np


# linspace
@pytest.mark.parametrize("start, stop, shape, pshape, dtype",
                         [
                             (0, 10, [10, 12, 5], [2, 3, 2], "double"),
                             (-0.1, -0.2, [4, 3, 5, 2], [2, 2, 2, 2], "float"),
                             (0, 10, [10, 12, 5], None, "double"),
                             (-0.1, -0.2, [4, 3, 5, 2], None, "float")
                         ])
def test_linspace(start, stop, shape, pshape, dtype):
    size = int(np.prod(shape))
    a = ia.linspace2(size, start, stop, shape, pshape, dtype)
    b = ia.iarray2numpy2(a)
    npdtype = np.float64 if dtype == "double" else np.float32
    c = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)


# arange
@pytest.mark.parametrize("start, stop, shape, pshape, dtype",
                         [
                             (0, 10, [10, 12, 5], [2, 3, 2], "double"),
                             (-0.1, -0.2, [4, 3, 5, 2], [2, 3, 2, 2], "float"),
                             (0, 10, [10, 12, 5], None, "double"),
                             (-0.1, -0.2, [4, 3, 5, 2], None, "float")
                         ])
def test_arange(start, stop, shape, pshape, dtype):
    size = int(np.prod(shape))
    step = (stop - start) / size
    a = ia.arange2(start, stop, step, shape=shape, pshape=pshape, dtype=dtype)
    b = ia.iarray2numpy2(a)
    npdtype = np.float64 if dtype == "double" else np.float32
    c = np.arange(start, stop, step, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)


# from_file
@pytest.mark.parametrize("start, stop, shape, pshape, dtype, filename",
                         [
                             (0, 10, [9], [2], "double", "test.fromfile0.iarray"),
                             (-0.1, -0.10, [4, 3, 5, 5, 2], [2, 3, 2, 3, 2], "float", "test.fromfile1.iarray")
                         ])
def test_from_file(start, stop, shape, pshape, dtype, filename):
    size = int(np.prod(shape))
    npdtype = np.float64 if dtype == "double" else np.float32
    a = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    b = ia.numpy2iarray2(a, pshape, filename)
    c = ia.from_file2(filename)
    d = ia.iarray2numpy2(c)
    np.testing.assert_almost_equal(a, d)
    os.remove(filename)


# get_slice
@pytest.mark.parametrize("start, stop, slice, shape, pshape, dtype",
                         [
                             (0, 10, (slice(2, 4), slice(5, 10), slice(1, 2)), [10, 12, 5], [2, 3, 2], "double"),
                             (-0.1, -0.2, (slice(2, 4), slice(7, 12)), [12, 16], [2, 7], "float"),
                             (0, 10, (slice(2, 4), slice(5, 10), slice(1, 2)), [10, 12, 5], None, "double"),
                             (-0.1, -0.2, (slice(2, 4), slice(7, 12)), [12, 16], None, "float")
                         ])
def test_slice(start, stop, slice, shape, pshape, dtype):
    size = int(np.prod(shape))
    step = (stop - start) / size
    a = ia.arange2(start, stop, step, shape=shape, pshape=pshape, dtype=dtype)
    b = a[slice]
    c = ia.iarray2numpy2(b)
    npdtype = np.float64 if dtype == "double" else np.float32
    d = np.arange(start, stop, step, dtype=npdtype).reshape(shape)[slice]
    np.testing.assert_almost_equal(c, d)


# empty  # TODO: make it work properly
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([10, 12, 5], [2, 3, 2], "double"),
                             ([10, 12, 5], None, "float"),
                         ])
def test_empty(shape, pshape, dtype):
    a = ia.empty2(shape, pshape, dtype)
    b = ia.iarray2numpy2(a)
    npdtype = np.float64 if dtype == "double" else np.float32
    assert b.dtype == npdtype
    assert b.shape == tuple(shape)


# zeros
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([10, 12, 5], [2, 3, 2], "double"),
                             ([12, 16], [2, 7], "float"),
                             ([10, 12, 5], None, "double"),
                             ([12, 16], None, "float")
                         ])
def test_zeros(shape, pshape, dtype):
    cfg = ia.Config()
    ctx = ia.Context(cfg)

    a = ia.zeros(ctx, shape, pshape, dtype)
    b = ia.iarray2numpy(ctx, a)
    npdtype = np.float64 if dtype == "double" else np.float32
    c = np.zeros(shape, dtype=npdtype)
    np.testing.assert_almost_equal(b, c)


# ones
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([10, 12, 5], [2, 3, 2], "double"),
                             ([12, 16], [2, 7], "float"),
                             ([10, 12, 5], None, "double"),
                             ([12, 16], None, "float")
                         ])
def test_ones(shape, pshape, dtype):
    cfg = ia.Config()
    ctx = ia.Context(cfg)

    a = ia.ones(ctx, shape, pshape, dtype)
    b = ia.iarray2numpy(ctx, a)
    npdtype = np.float64 if dtype == "double" else np.float32
    c = np.ones(shape, dtype=npdtype)
    np.testing.assert_almost_equal(b, c)


# full
@pytest.mark.parametrize("fill_value, shape, pshape, dtype",
                         [
                             (8.34, [10, 12, 5], [2, 3, 2], "double"),
                             (2.00001, [12, 16], [2, 7], "float"),
                             (8.34, [10, 12, 5], None, "double"),
                             (2.00001, [12, 16], None, "float")
                         ])
def test_full(fill_value, shape, pshape, dtype):
    cfg = ia.Config()
    ctx = ia.Context(cfg)

    a = ia.full(ctx, fill_value, shape, pshape, dtype)
    b = ia.iarray2numpy(ctx, a)
    npdtype = np.float64 if dtype == "double" else np.float32
    c = np.full(shape, fill_value, dtype=npdtype)
    np.testing.assert_almost_equal(b, c)
