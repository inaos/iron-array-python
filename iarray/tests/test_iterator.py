import pytest
import iarray as ia
import numpy as np
from itertools import zip_longest as izip


# Expression
@pytest.mark.parametrize(
    "shape, chunks, blocks, itershape, dtype, acontiguous, aurlpath, bcontiguous, burlpath",
    [
        ([100, 100], [20, 20], [10, 10], [20, 20], np.float64, True, None, True, None),
        (
            [100, 100],
            [15, 15],
            [7, 8],
            [15, 15],
            np.float32,
            True,
            "test_iter_acontiguous.iarr",
            True,
            "test_iter_bcontiguous.iarr",
        ),
        (
            [10, 10, 10],
            [4, 5, 6],
            [2, 3, 6],
            [4, 5, 6],
            np.float64,
            False,
            "test_iter_asparse.iarr",
            False,
            "test_iter_bsparse.iarr",
        ),
        (
            [10, 10, 10, 10],
            [3, 4, 3, 4],
            [2, 2, 2, 2],
            [3, 4, 3, 4],
            np.float32,
            False,
            None,
            False,
            None,
        ),
        (
            [100, 100],
            [50, 50],
            [20, 20],
            [50, 50],
            np.float64,
            False,
            "test_iter_asparse.iarr",
            True,
            None,
        ),
        (
            [100, 100],
            [23, 35],
            [21, 33],
            [23, 35],
            np.float32,
            False,
            None,
            True,
            "test_iter_bcontiguous.iarr",
        ),
        (
            [10, 10, 10],
            [10, 10, 10],
            [5, 5, 5],
            [10, 10, 10],
            np.float64,
            True,
            "test_iter_asparse.iarr",
            True,
            "test_iter_bsparse.iarr",
        ),
        (
            [10, 10, 10, 10],
            [3, 4, 3, 4],
            [3, 4, 3, 4],
            [3, 4, 3, 4],
            np.float32,
            True,
            None,
            False,
            None,
        ),
    ],
)
def test_iterator(
    shape, chunks, blocks, itershape, dtype, acontiguous, aurlpath, bcontiguous, burlpath
):
    acfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath)
    bcfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=bcontiguous, urlpath=burlpath)

    ia.remove_urlpath(aurlpath)
    ia.remove_urlpath(burlpath)
    a = ia.linspace(shape, -10, 10, dtype=dtype, cfg=acfg)
    an = ia.iarray2numpy(a)

    b = ia.empty(shape, dtype=dtype, cfg=bcfg)

    zip = izip(a.iter_read_block(itershape), b.iter_write_block(itershape))
    for i, ((ainfo, aslice), (_, bslice)) in enumerate(zip):
        bslice[:] = aslice
        start = ainfo.elemindex
        stop = tuple(ainfo.elemindex[i] + ainfo.shape[i] for i in range(len(ainfo.elemindex)))
        slices = tuple(slice(start[i], stop[i]) for i in range(len(start)))
        np.testing.assert_almost_equal(aslice, an[slices])

    bn = ia.iarray2numpy(b)

    np.testing.assert_almost_equal(bn, an)

    ia.remove_urlpath(aurlpath)
    ia.remove_urlpath(burlpath)
