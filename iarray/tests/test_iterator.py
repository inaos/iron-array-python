import pytest
import iarray as ia
import numpy as np
from itertools import zip_longest as izip


# Expression
@pytest.mark.parametrize("shape, chunkshape, blockshape, itershape, dtype",
                         [
                             ([100, 100], [20, 20], [10, 10], [20, 20], np.float64),
                             ([100, 100], [15, 15], [7, 8], [15, 15], np.float32),
                             ([10, 10, 10], [4, 5, 6], [2, 3, 6], [4, 5, 6], np.float64),
                             ([10, 10, 10, 10], [3, 4, 3, 4], [2, 2, 2, 2], [3, 4, 3, 4], np.float32),
                             ([100, 100], None, None, [30, 30], np.float64),
                             ([100, 100], None, None, [15, 15], np.float32),
                             ([10, 10, 10], None, None, [4, 5, 6], np.float64),
                             ([10, 10, 10, 10], None, None, [3, 4, 3, 4], np.float32)
                         ])
def test_iterator(shape, chunkshape, blockshape, itershape, dtype):
    if chunkshape is None:
        storage = ia.StorageProperties("plainbuffer")
    else:
        storage = ia.StorageProperties("blosc", chunkshape, blockshape, False)
    a = ia.linspace(ia.dtshape(shape, dtype), -10, 10, storage=storage)
    an = ia.iarray2numpy(a)

    b = ia.empty(ia.dtshape(shape, dtype), storage=storage)

    zip = izip(a.iter_read_block(itershape), b.iter_write_block(itershape))
    for i, ((ainfo, aslice), (_, bslice)) in enumerate(zip):
        bslice[:] = aslice
        start = ainfo.elemindex
        stop = tuple(ainfo.elemindex[i] + ainfo.shape[i] for i in range(len(ainfo.elemindex)))
        slices = tuple(slice(start[i], stop[i]) for i in range(len(start)))
        np.testing.assert_almost_equal(aslice, an[slices])

    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(bn, an)
