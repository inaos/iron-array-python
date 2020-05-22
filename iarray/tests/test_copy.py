import pytest
import iarray as ia
import numpy as np


@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([100], [20], np.float64),
                             ([100, 100], [20, 20], np.float32),
#                            ([100, 100], None, np.float64),
#                            ([100, 100, 100], None, np.float32),
                             ([20, 100, 30, 50], [10, 40, 10, 11], np.float64),
#                            ([11, 12, 14, 15, 16], None, np.float32),
                             ([10, 13, 12, 14, 12, 10], [5, 4, 6, 2, 3, 7], np.float64),
#                            ([2, 3, 4, 5, 6, 7, 8, 9], None, np.float32),
                         ])
def test_copy_old(shape, pshape, dtype):
    a = ia.linspace(ia.dtshape(shape, pshape, dtype), -10, 10)
    b = a.copy()
    c = a.copy(view=True)
    bn = ia.iarray2numpy(b)
    cn = ia.iarray2numpy(c)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(bn, cn, rtol=rtol)


@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([100, 100], [20, 20], np.float32),
                             ([100, 100], None, np.float64),
                             ([100, 100, 100], None, np.float32),
                             ([20, 100, 30, 50], [10, 40, 10, 11], np.float64),
                             ([11, 12, 14, 15, 16], None, np.float32),
                             ([10, 13, 12, 14, 12, 10], [5, 4, 6, 2, 3, 7], np.float64),
                             ([2, 3, 4, 5, 6, 7, 8, 9], None, np.float32)
                         ])
def test_copy(shape, pshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)
    a_ = ia.linspace(ia.dtshape(shape, pshape, dtype), -10, 10, storage=storage)
    sl = tuple([slice(0, s-1) for s in shape])
    a = a_[sl]
    b = a.copy()
    c = a.copy(view=True)
    bn = ia.iarray2numpy(b)
    cn = ia.iarray2numpy(c)

    rtol = 1e-6 if dtype == np.float32 else 1e-14

    np.testing.assert_allclose(bn, cn, rtol=rtol)
