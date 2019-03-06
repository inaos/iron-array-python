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
