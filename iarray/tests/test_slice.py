import pytest
import iarray as ia
import numpy as np


# Slice
@pytest.mark.parametrize("shape, pshape, start, stop, dtype",
                         [
                             ([100], [20], [20], [30], "double"),
                             ([100, 100], [20, 20], [5, 10], [30, 40], "float"),
                             ([100, 100], None, [5, 10], [30, 40], "double"),
                             ([100, 100, 100], None, [5, 46, 10], [30, 77, 40], "float")

                         ])
def test_slice(shape, pshape, start, stop, dtype):
    slices = tuple(slice(start[i], stop[i]) for i in range(len(start)))
    if len(start) == 1:
        slices = slices[0]

    a = ia.linspace(ia.dtshape(shape, pshape, dtype), -10, 10)
    an = ia.iarray2numpy(a)

    b = a[slices]
    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(an[slices], bn)
