import pytest
import iarray as ia
import numpy as np


params_names = "shape, chunkshape, blockshape, axis, dtype"
params_data = [
    ([100, 100], [50, 50], [20, 20], 0, np.float32),
    ([20, 100, 30, 50], [10, 40, 10, 11], [4, 5, 3, 7], 1, np.float64),
    ([10, 13, 12, 14, 12, 10], [5, 4, 6, 2, 3, 7], [2, 2, 2, 2, 2, 2], 4, np.float64),
]


@pytest.mark.parametrize(params_names, params_data)
def test_max(shape, chunkshape, blockshape, axis, dtype):

    storage = ia.Storage(chunkshape, blockshape)
    a1 = ia.linspace(ia.DTShape(shape, dtype), -10, 10, storage=storage)
    a2 = ia.iarray2numpy(a1)

    b1 = ia.max(a1, axis=axis)
    b2 = np.max(a2, axis=axis)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol)


@pytest.mark.parametrize(param_names, params_data)
def test_min(shape, chunkshape, blockshape, axis, dtype):

    storage = ia.Storage(chunkshape, blockshape)
    a1 = ia.linspace(ia.DTShape(shape, dtype), -10, 10, storage=storage)
    a2 = ia.iarray2numpy(a1)

    b1 = ia.min(a1, axis=axis)
    b2 = np.min(a2, axis=axis)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol)


@pytest.mark.parametrize(param_names, params_data)
def test_sum(shape, chunkshape, blockshape, axis, dtype):
    storage = ia.Storage(chunkshape, blockshape)
    a1 = ia.linspace(ia.DTShape(shape, dtype), -10, 10, storage=storage)
    a2 = ia.iarray2numpy(a1)

    b1 = ia.sum(a1, axis=axis)
    b2 = np.sum(a2, axis=axis)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol)


@pytest.mark.parametrize(param_names, params_data)
def test_prod(shape, chunkshape, blockshape, axis, dtype):
    storage = ia.Storage(chunkshape, blockshape)
    a1 = ia.linspace(ia.DTShape(shape, dtype), -10, 10, storage=storage)
    a2 = ia.iarray2numpy(a1)

    b1 = ia.prod(a1, axis=axis)
    b2 = np.prod(a2, axis=axis)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol)


@pytest.mark.parametrize(param_names, params_data)
def test_mean(shape, chunkshape, blockshape, axis, dtype):
    storage = ia.Storage(chunkshape, blockshape)
    a1 = ia.linspace(ia.DTShape(shape, dtype), -10, 10, storage=storage)
    a2 = ia.iarray2numpy(a1)

    b1 = ia.mean(a1, axis=axis)
    b2 = np.mean(a2, axis=axis)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol)


@pytest.mark.parametrize(param_names, params_data)
def test_std(shape, chunkshape, blockshape, axis, dtype):
    storage = ia.Storage(chunkshape, blockshape)
    a1 = ia.linspace(ia.DTShape(shape, dtype), -10, 10, storage=storage)
    a2 = ia.iarray2numpy(a1)

    b1 = ia.std(a1, axis=axis)
    b2 = np.std(a2, axis=axis)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(ia.iarray2numpy(b1), b2, rtol=rtol)
