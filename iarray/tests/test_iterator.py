import pytest
import iarray as ia
import numpy as np
import numexpr as ne
from itertools import zip_longest as izip


# Expression
@pytest.mark.parametrize("shape, pshape, dtype",
                         [
                             ([100, 100], [20, 20], "double"),
                             ([100, 100], [15, 15], "float"),
                             ([10, 10, 10], [4, 5, 6], "double"),
                             ([10, 10, 10, 10], [3, 4, 3, 4], "float")
                         ])
def test_iterator(shape, pshape, dtype):
    cfg = ia.Config()
    ctx = ia.Context(cfg)

    size = int(np.prod(shape))
    a = ia.linspace(ctx, size, -10, 10, shape, pshape, dtype)
    an = ia.iarray2numpy(ctx, a)

    b = ia.empty(ctx, shape, pshape)

    for i, ((ainfo, aslice), (binfo, bslice)) in enumerate(izip(a.iter_read_block(pshape), b.iter_write_block())):
        bslice[:] = aslice
        start = ainfo.elemindex
        stop = tuple(ainfo.elemindex[i] + ainfo.shape[i] for i in range(len(ainfo.elemindex)))
        slices = tuple(slice(start[i], stop[i]) for i in range(len(start)))
        np.testing.assert_almost_equal(aslice, an[slices])

    bn = ia.iarray2numpy(ctx, b)

    np.testing.assert_almost_equal(bn, an)

