import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize("start, stop, shape, pshape, dtype",
                         [
                             (0, 10, [10, 12, 5], [2, 3, 2], "double"),
                             (-0.1, -0.2, [4, 3, 5, 2], None, "float")
                         ])
def test_linspace(start, stop, shape, pshape, dtype):
    cfg = ia.Config()
    ctx = ia.Context(cfg)
    size = int(np.prod(shape))
    a = ia.linspace(ctx, size, start, stop, shape, pshape, dtype)
    b = ia.iarray2numpy(ctx, a)
    npdtype = np.float64 if dtype == "double" else np.float32
    c = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)


@pytest.mark.parametrize("start, stop, shape, pshape, dtype",
                         [
                             (0, 10, [10, 12, 5], [2, 3, 2], "double"),
                             (-0.1, -0.2, [4, 3, 5, 2], [2, 3, 2, 2], "float")
                         ])
def test_arange(start, stop, shape, pshape, dtype):
    cfg = ia.Config()
    ctx = ia.Context(cfg)
    size = int(np.prod(shape))
    step = (stop - start)/size
    a = ia.arange(ctx, start, stop, step, shape=shape, pshape=pshape, dtype=dtype)
    b = ia.iarray2numpy(ctx, a)
    npdtype = np.float64 if dtype == "double" else np.float32
    c = np.arange(start, stop, step, dtype=npdtype).reshape(shape)
    np.testing.assert_almost_equal(b, c)


@pytest.mark.parametrize("start, stop, shape, pshape, dtype, filename",
                         [
                             (0, 10, [9], [2], "double", "test.fromfile0.iarray"),
                             (-0.1, -0.10, [4, 3, 5, 5, 2], [2, 3, 2, 3, 2], "float", "test.fromfile1.iarray")
                         ])
def test_from_file(start, stop, shape, pshape, dtype, filename):
    cfg = ia.Config()
    ctx = ia.Context(cfg)
    size = int(np.prod(shape))
    npdtype = np.float64 if dtype == "double" else np.float32
    a = np.linspace(start, stop, size, dtype=npdtype).reshape(shape)
    b = ia.numpy2iarray(ctx, a, pshape, filename)
    c = ia.from_file(ctx, filename)
    d = ia.iarray2numpy(ctx, c)
    np.testing.assert_almost_equal(a, d)
