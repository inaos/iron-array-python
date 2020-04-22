import pytest
import iarray as ia
import numpy as np


# Slice
@pytest.mark.parametrize("shape, pshape, start, stop, dtype",
                         [
                             ([100, 100], [20, 20], [0, 0], [10, 10], np.float64),
                         ])
def test_slice(shape, pshape, start, stop, dtype):
    print("Create storage properties")
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)

    slices = tuple(slice(start[i], stop[i]) for i in range(len(start)))
    if len(start) == 1:
        slices = slices[0]
    print("Create a linspace")
    a = ia.linspace(ia.dtshape(shape, pshape, dtype), -10, 10, storage=storage)
    print("Convert an array to numpy")
    an = ia.iarray2numpy(a)
    print("Create a caterva slice")
    b = a[slices]
    print("Convert another to numpy")
    bn = ia.iarray2numpy(b)
    print("Assert slices")
    np.testing.assert_almost_equal(an[slices], bn)
    print("Finish")
