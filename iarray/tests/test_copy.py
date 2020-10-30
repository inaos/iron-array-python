import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize(
    "shape, chunkshape, blockshape, dtype",
    [
        ([100, 100], [50, 50], [20, 20], np.float32),
        ([100, 100], None, None, np.float64),
        ([100, 100, 100], None, None, np.float32),
        ([20, 100, 30, 50], [10, 40, 10, 11], [4, 5, 3, 7], np.float64),
        ([11, 12, 14, 15, 16], None, None, np.float32),
        ([10, 13, 12, 14, 12, 10], [5, 4, 6, 2, 3, 7], [2, 2, 2, 2, 2, 2], np.float64),
        ([2, 3, 4, 5, 6, 7, 8, 9], None, None, np.float32),
    ],
)
def test_copy(shape, chunkshape, blockshape, dtype):
    if chunkshape is None:
        storage = ia.Storage(plainbuffer=True)
    else:
        storage = ia.Storage(chunkshape, blockshape)
    a_ = ia.linspace(ia.DTShape(shape, dtype), -10, 10, storage=storage)
    sl = tuple([slice(0, s - 1) for s in shape])
    a = a_[sl]
    b = a.copy()
    an = ia.iarray2numpy(a)
    bn = ia.iarray2numpy(b)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(an, bn, rtol=rtol)
