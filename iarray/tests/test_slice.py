import pytest
import iarray as ia
import numpy as np


# Slice
@pytest.mark.parametrize("shape, pshape, start, stop, dtype",
                         [
                             ([100, 100], [20, 20], [0, 0], [10, 10], np.float64),
                         ])
def test_slice(shape, pshape, start, stop, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", True)

    slices = tuple(slice(start[i], stop[i]) for i in range(len(start)))
    if len(start) == 1:
        slices = slices[0]

    a = ia.linspace(ia.dtshape(shape, pshape, dtype), -10, 10, storage=storage)
    an = ia.iarray2numpy(a)

    b = a[slices]
    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(an[slices], bn)
