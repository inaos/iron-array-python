import pytest
import iarray as ia
import numpy as np
from itertools import zip_longest as izip


# Expression
@pytest.mark.parametrize("shape, pshape, bshape, dtype",
                         [
                             ([100, 100], [20, 20], [20, 20], np.float64),
                             ([100, 100], [15, 15], [15, 15], np.float32),
                             ([10, 10, 10], [4, 5, 6], [4, 5, 6], np.float64),
                             ([10, 10, 10, 10], [3, 4, 3, 4], [3, 4, 3, 4], np.float32),
                             ([100, 100], None, [30, 30], np.float64),
                             ([100, 100], None, [15, 15], np.float32),
                             ([10, 10, 10], None, [4, 5, 6], np.float64),
                             ([10, 10, 10, 10], None, [3, 4, 3, 4], np.float32)
                         ])
def test_iterator(shape, pshape, bshape, dtype):
    if pshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", False)
    a = ia.linspace(ia.dtshape(shape, pshape, dtype), -10, 10, storage=storage)
    an = ia.iarray2numpy(a)

    b = ia.empty(ia.dtshape(shape, pshape), storage=storage)

    for i, ((ainfo, aslice), (_, bslice)) in enumerate(izip(a.iter_read_block(bshape), b.iter_write_block(bshape))):
        bslice[:] = aslice
        start = ainfo.elemindex
        stop = tuple(ainfo.elemindex[i] + ainfo.shape[i] for i in range(len(ainfo.elemindex)))
        slices = tuple(slice(start[i], stop[i]) for i in range(len(start)))
        np.testing.assert_almost_equal(aslice, an[slices])

    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(bn, an)
