import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize(
    "shape, chunkshape, blockshape, dtype",
    [
        ([100, 100], [50, 50], [20, 20], np.float32),
        ([100, 100], None, None, np.float64),
        ([100, 500], None, None, np.float32),
        ([1453, 266], [100, 200], [30, 20], np.float64),
    ],
)
def test_transpose(shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties(plainbuffer=True)
    else:
        storage = ia.StorageProperties(chunkshape, blockshape)
    a = ia.linspace(ia.dtshape(shape, dtype), -10, 10, storage=storage)

    b = ia.iarray2numpy(a)
    bn = b.T

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    at = a.T
    an = ia.iarray2numpy(at)
    np.testing.assert_allclose(an, bn, rtol=rtol)

    at = a.transpose(view=False)
    an = ia.iarray2numpy(at)
    np.testing.assert_allclose(an, bn, rtol=rtol)

    at = ia.transpose(a, view=False, storage=ia.StorageProperties(plainbuffer=True))
    an = ia.iarray2numpy(at)
    np.testing.assert_allclose(an, bn, rtol=rtol)
