import pytest
import iarray as ia
import numpy as np


# Slice
@pytest.mark.parametrize(
    "shape, chunkshape, blockshape, start, stop, dtype",
    [
        ([200], [70], [30], [20], [30], np.float64),
        ([100, 100], [20, 20], [10, 13], [5, 10], [30, 40], np.float32),
        ([100, 100], None, None, [5, 10], [30, 40], np.float64),
        ([100, 100, 100], None, None, [5, 46, 10], [30, 77, 40], np.float32),
    ],
)
def test_slice(shape, chunkshape, blockshape, start, stop, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties(plainbuffer=True)
    else:
        storage = ia.StorageProperties(chunkshape, blockshape, enforce_frame=True)

    slices = tuple(slice(start[i], stop[i]) for i in range(len(start)))
    if len(start) == 1:
        slices = slices[0]

    a = ia.linspace(ia.dtshape(shape, dtype), -10, 10, storage=storage)
    an = ia.iarray2numpy(a)

    b = a[slices]
    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(an[slices], bn)
