import pytest
import iarray as ia
import numpy as np
from itertools import zip_longest as izip


# Expression
@pytest.mark.parametrize(
    "shape, chunks, blocks, itershape, dtype, np_dtype, acontiguous, aurlpath, bcontiguous, burlpath, mode",
    [
        ([100, 100], [20, 20], [10, 10], [20, 20], np.float64, "f4", True, None, True, None, "r"),
        (
            [100, 100],
            [15, 15],
            [7, 8],
            [15, 15],
            np.float32,
            None,
            True,
            "test_iter_acontiguous.iarr",
            True,
            "test_iter_bcontiguous.iarr",
            "w-",
        ),
        (
            [10, 10, 10],
            [4, 5, 6],
            [2, 3, 6],
            [4, 5, 6],
            np.float64,
            ">f8",
            False,
            "test_iter_asparse.iarr",
            False,
            "test_iter_bsparse.iarr",
            "w",
        ),
        pytest.param(
            [10, 10, 10, 10],
            [3, 4, 3, 4],
            [2, 2, 2, 2],
            [3, 4, 3, 4],
            np.int32,
            None,
            False,
            None,
            False,
            None,
            "r",
            marks=pytest.mark.heavy,
        ),
        (
            [100, 100],
            [50, 50],
            [20, 20],
            [50, 50],
            np.uint64,
            "m8[ms]",
            False,
            "test_iter_asparse.iarr",
            True,
            None,
            "r+",
        ),
        (
            [100, 100],
            [23, 35],
            [21, 33],
            [23, 35],
            np.uint16,
            None,
            False,
            None,
            True,
            "test_iter_bcontiguous.iarr",
            "a",
        ),
        pytest.param(
            [10, 10, 10],
            [10, 10, 10],
            [5, 5, 5],
            [10, 10, 10],
            np.int16,
            None,
            True,
            "test_iter_asparse.iarr",
            True,
            "test_iter_bsparse.iarr",
            "a",
            marks=pytest.mark.heavy,
        ),
        (
            [10, 10, 10, 10],
            [3, 4, 3, 4],
            [3, 4, 3, 4],
            [3, 4, 3, 4],
            np.float32,
            None,
            True,
            None,
            False,
            None,
            "w",
        ),
    ],
)
def test_iterator(
    shape,
    chunks,
    blocks,
    itershape,
    dtype,
    np_dtype,
    acontiguous,
    aurlpath,
    bcontiguous,
    burlpath,
    mode,
):
    acfg = ia.Config(chunks=chunks, blocks=blocks, contiguous=acontiguous, urlpath=aurlpath)
    bcfg = ia.Config(
        chunks=chunks, blocks=blocks, contiguous=bcontiguous, urlpath=burlpath, mode="a"
    )

    ia.remove_urlpath(aurlpath)
    ia.remove_urlpath(burlpath)
    max = 1
    out_dtype = dtype if np_dtype is None else np.dtype(np_dtype)
    if out_dtype not in [np.float64, np.float32]:
        for i in range(len(shape)):
            max *= shape[i]
    a = ia.linspace(0, max, int(np.prod(shape)), shape=shape, dtype=dtype, np_dtype=np_dtype, cfg=acfg)
    an = ia.iarray2numpy(a)

    b = ia.empty(shape, dtype=dtype, np_dtype=np_dtype, cfg=bcfg)
    mode2 = b.cfg.mode
    b.cfg.mode = mode
    if mode in ["r"]:
        with pytest.raises(IOError):
            izip(a.iter_read_block(itershape), b.iter_write_block(itershape))
        b.cfg.mode = mode2

    zip = izip(a.iter_read_block(itershape), b.iter_write_block(itershape))
    for i, ((ainfo, aslice), (_, bslice)) in enumerate(zip):
        bslice[:] = aslice
        start = ainfo.elemindex
        stop = tuple(ainfo.elemindex[i] + ainfo.shape[i] for i in range(len(ainfo.elemindex)))
        slices = tuple(slice(start[i], stop[i]) for i in range(len(start)))
        if out_dtype in [np.float64, np.float32]:
            np.testing.assert_almost_equal(aslice, an[slices])
        else:
            np.testing.assert_array_equal(aslice, an[slices])

    bn = ia.iarray2numpy(b)

    if out_dtype in [np.float64, np.float32]:
        np.testing.assert_almost_equal(bn, an)
    else:
        np.testing.assert_array_equal(bn, an)

    ia.remove_urlpath(aurlpath)
    ia.remove_urlpath(burlpath)
